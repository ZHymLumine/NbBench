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
from transformers import RobertaForMaskedLM, BertForMaskedLM
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
from transformers import EsmModel, EsmConfig
from .configuration_protbert import ProtBertConfig
from ..modeling_utils import FocalLoss


logger = logging.get_logger(__name__)

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

class ProtBertPooler(nn.Module):
    def __init__(self, config: ProtBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = hidden_states.mean(dim=1)
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ProtBertForSequenceClassification(PreTrainedModel):
    config_class = ProtBertConfig
    base_model_prefix = "protbert"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.protbert = RobertaForMaskedLM.from_pretrained(config._name_or_path, config=config)
        self.protbert = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        self.pooler = ProtBertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if config.freeze:
            for param in self.protbert.parameters():
                param.requires_grad = False 
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.protbert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states[-1]
        # hidden_states = outputs.last_hidden_state
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
        )


class ProtBertForAminoAcidLevel(PreTrainedModel):
    config_class = ProtBertConfig
    base_model_prefix = "protbert"
    supports_gradient_checkpointing = True
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        print(f"config: {config}")
        self.num_labels = config.num_labels
        self.config = config
    
        # self.protbert = RobertaForMaskedLM.from_pretrained(config._name_or_path, config=config) 
        self.protbert = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        if config.freeze:
            for param in self.protbert.parameters():
                param.requires_grad = False 

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
        output_hidden_states: Optional[bool] = True,
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
        
        outputs = self.protbert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # final_input= outputs[0]
        final_input = outputs.hidden_states[-1]
        # final_input = outputs.last_hidden_state
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
        )

class ProtBertForBindingSequenceClassification(PreTrainedModel):
    config_class = ProtBertConfig
    base_model_prefix = "protbert"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.protbert = RobertaForMaskedLM.from_pretrained(config._name_or_path, config=config)
        self.protbert = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        if config.freeze:
            for param in self.protbert.parameters():
                param.requires_grad = False 
        esm_config = EsmConfig.from_pretrained('facebook/esm2_t33_650M_UR50D')
        self.pooler = ProtBertPooler(config)
        self.classifier = nn.Linear(config.hidden_size + esm_config.hidden_size, config.num_labels)

        if  self.num_labels == 2:
            self.classifier = nn.Linear(config.hidden_size + esm_config.hidden_size, 1)
        else:
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
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get nanobody outputs
        outputs = self.protbert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.pooler(outputs.hidden_states[-1])
        # hidden_states = self.pooler(outputs.last_hidden_state)

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
        # logits = (logits_min + logits_mean + logits_max) / 3
        logits = logits_mean

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
                loss_fct = FocalLoss(gamma=2, alpha=0.25, task_type='binary')
                loss = loss_fct(logits, labels.view(-1, 1).float())
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

class ProtBertForParatope(PreTrainedModel):
    config_class = ProtBertConfig
    base_model_prefix = "protbert"
    supports_gradient_checkpointing = True
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        # self.protbert = RobertaForMaskedLM.from_pretrained(config._name_or_path, config=config)
        self.protbert = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        if config.freeze:
            for param in self.protbert.parameters():
                param.requires_grad = False 

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
        output_hidden_states: Optional[bool] = True,
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
        
        outputs = self.protbert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # final_input= outputs[0]
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # hidden_states = outputs.last_hidden_state
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

        # logits = (logits_min + logits_mean + logits_max) / 3
        logits = logits_mean


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
        )