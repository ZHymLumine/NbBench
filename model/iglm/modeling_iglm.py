from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from warnings import warn

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.autograd as autograd
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import RobertaForMaskedLM, AutoModel, GPT2LMHeadModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, 
    BertModel, 
    BertLMPredictionHead, 
    ModelOutput
)

from .configuration_iglm import IgLMConfig


logger = logging.get_logger(__name__)

def exists(x):
    return x is not None

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
        return score

# Copied from transformers.models.bert.modeling_bert.BertPooler
class IgLMPooler(nn.Module):
    def __init__(self, config: IgLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class IgLMHeads(nn.Module):
    """
    Classification heads for IgLM model.
    """
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.species = nn.Linear(config.hidden_size, 6)
        self.chain = nn.Linear(config.hidden_size, 2)
        self.graft = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        species_score = self.species(pooled_output)
        chain_score = self.chain(pooled_output)
        graft_score = self.graft(pooled_output)
        return prediction_scores, species_score, chain_score, graft_score


@dataclass
class IgLMOutput(ModelOutput):
    """
    Output type of for IgLM model.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    species_logits: torch.FloatTensor = None
    chain_logits: torch.FloatTensor = None
    graft_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class IgLM(BertPreTrainedModel):
    """
    BERT model for antibody sequences, with classification heads
    for species, chain type, and presence of grafting
    """
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = IgLMHeads(config)

        self.init_weights()

        self.num_species = 6
        self.num_chains = 2
        self.num_grafts = 2

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        species_label=None,
        chain_label=None,
        graft_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if exists(
            return_dict) else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, species_score, chain_score, graft_score = self.cls(
            sequence_output, pooled_output)

        b = input_ids.shape[0]

        total_loss, masked_lm_loss, species_loss, chain_loss, graft_loss = None, None, None, None, None
        if exists(labels):
            mlm_loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = mlm_loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))

        if exists(species_label):
            species_freqs = torch.bincount(species_label,
                                           minlength=self.num_species)
            species_weights = b / (species_freqs * self.num_species)
            species_loss_fct = nn.CrossEntropyLoss(weight=species_weights)
            species_loss = species_loss_fct(
                species_score.view(-1, self.num_species),
                species_label.view(-1))

        if exists(chain_label):
            chain_freqs = torch.bincount(chain_label,
                                         minlength=self.num_chains)
            species_weights = b / (chain_freqs * self.num_chains)
            chain_loss_fct = nn.CrossEntropyLoss(weight=species_weights)
            chain_loss = chain_loss_fct(chain_score.view(-1, 2),
                                        chain_label.view(-1))

        if exists(graft_label):
            graft_freqs = torch.bincount(graft_label,
                                         minlength=self.num_grafts)
            graft_weights = b / (graft_freqs * self.num_grafts)
            graft_loss_fct = nn.CrossEntropyLoss(weight=graft_weights)
            graft_loss = graft_loss_fct(graft_score.view(-1, 2),
                                        graft_label.view(-1))

        total_loss = \
            masked_lm_loss if exists(masked_lm_loss) else 0 \
            + species_loss if exists(species_loss) else 0 \
            + chain_loss if exists(chain_loss) else 0 \
            + graft_loss if exists(graft_loss) else 0

        if not return_dict:
            output = (prediction_scores, species_score, chain_score,
                      graft_score) + outputs[2:]
            return ((total_loss, ) + output) if exists(total_loss) else output

        return IgLMOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            species_logits=species_score,
            chain_logits=chain_score,
            graft_logits=graft_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class IgLMPooler(nn.Module):
    def __init__(self, config: IgLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class IgLMForSequenceClassification(PreTrainedModel):
    config_class = IgLMConfig
    base_model_prefix = "iglm"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.iglm = GPT2LMHeadModel.from_pretrained(config._name_or_path, config=config)
        self.pooler = IgLMPooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.iglm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        print(f"outputs: {outputs}")

        hidden_states = outputs.hidden_states[-1]
        cls_token = hidden_states[:, 0]
        pooled_output = cls_token
        logits = self.classifier(pooled_output)

        # pooled_output = self.pooler(hidden_states) if self.pooler is not None else None
        # logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class IgLMForAminoAcidLevel(PreTrainedModel):
    config_class = IgLMConfig
    base_model_prefix = "iglm"
    supports_gradient_checkpointing = True
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.iglm = GPT2LMHeadModel.from_pretrained(config._name_or_path, config=config)

        self.tokenizer = tokenizer

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        outputs = self.iglm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # final_input= outputs[0]
        final_input = outputs.hidden_states[-1]
        logits = self.classifier(final_input)

        loss = None
        if labels is not None:
            # logits = logits[:, 1:1+labels.size(1), :]
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MCRMSELoss()
                
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1).long())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class IgLMForBindingSequenceClassification(PreTrainedModel):
    config_class = IgLMConfig
    base_model_prefix = "iglm"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.iglm = GPT2LMHeadModel.from_pretrained(config._name_or_path, config=config)
        
        esm_config = EsmConfig.from_pretrained('facebook/esm2_t33_650M_UR50D')
        self.classifier = nn.Linear(config.hidden_size + esm_config.hidden_size, config.num_labels)

        # print(f"self.iglm: {self.iglm}")
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        antigen_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get nanobody outputs
        outputs = self.iglm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Combine embeddings
        hidden_states = outputs.hidden_states[-1][:, 0]

        # print(f"hidden_states: {hidden_states.shape}")
        # print(f"antigen_embedding: {antigen_embedding.shape}")
        # Perform pooling on antigen_hidden_states
        antigen_embedding = antigen_embedding.squeeze(1)
        ag_min_pooled = torch.min(antigen_embedding, dim=1).values
        ag_mean_pooled = torch.mean(antigen_embedding, dim=1)
        ag_max_pooled = torch.max(antigen_embedding, dim=1).values

        combined_min = torch.cat((hidden_states, ag_min_pooled), dim=-1)
        combined_mean = torch.cat((hidden_states, ag_mean_pooled), dim=-1)
        combined_max = torch.cat((hidden_states, ag_max_pooled), dim=-1)

        logits_min = self.classifier(combined_min)
        logits_mean = self.classifier(combined_mean)
        logits_max = self.classifier(combined_max)

        # Voting mechanism
        logits = (logits_min + logits_mean + logits_max) / 3

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class IgLMForParatope(PreTrainedModel):
    config_class = IgLMConfig
    base_model_prefix = "iglm"
    supports_gradient_checkpointing = True
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.iglm = GPT2LMHeadModel.from_pretrained(config._name_or_path, config=config)

        self.tokenizer = tokenizer
        esm_config = EsmConfig.from_pretrained('facebook/esm2_t33_650M_UR50D')

        self.residual_layers = nn.Sequential(
            ResidualBlock(config.hidden_size + esm_config.hidden_size),
            ResidualBlock(config.hidden_size + esm_config.hidden_size),
            ResidualBlock(config.hidden_size + esm_config.hidden_size)
        )
        self.classifier = nn.Linear(config.hidden_size + esm_config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        antigen_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.iglm(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # final_input= outputs[0]
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        antigen_embedding = antigen_embedding.squeeze()
        
        ag_min_pooled = torch.min(antigen_embedding, dim=1).values  # [batch_size, hidden_size]
        ag_mean_pooled = torch.mean(antigen_embedding, dim=1)      # [batch_size, hidden_size]
        ag_max_pooled = torch.max(antigen_embedding, dim=1).values # [batch_size, hidden_size]
        

        batch_size, seq_len, hidden_size = hidden_states.shape
        ag_min_pooled = ag_min_pooled.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        ag_mean_pooled = ag_mean_pooled.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        ag_max_pooled = ag_max_pooled.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        combined_min = torch.cat([hidden_states, ag_min_pooled], dim=-1)
        combined_mean = torch.cat([hidden_states, ag_mean_pooled], dim=-1)
        combined_max = torch.cat([hidden_states, ag_max_pooled], dim=-1)
        
        final_input_min = self.residual_layers(combined_min)
        final_input_mean = self.residual_layers(combined_mean)
        final_input_max = self.residual_layers(combined_max)
        
        logits_min = self.classifier(final_input_min)
        logits_mean = self.classifier(final_input_mean)
        logits_max = self.classifier(final_input_max)

        logits = (logits_min + logits_mean + logits_max) / 3

        loss = None
        if labels is not None:
            # logits = logits[:, 1:1+labels.size(1), :]
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MCRMSELoss()
                
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1).long())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )