from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaForMaskedLM
from typing import Optional
import torch

class RobertaEmbeddingsV2(RobertaEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0) # here padding_idx is always 0

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_ids = self.create_position_ids_from_input_ids(input_ids)  
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return self.dropout(self.LayerNorm(embeddings))
    
    def create_position_ids_from_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        mask = input_ids.ne(self.pad_token_id).int()
        return torch.cumsum(mask, dim=1).long() * mask
    

class RobertaModelV2(RobertaModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = RobertaEmbeddingsV2(config)


class RobertaForMaskedLMV2(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModelV2(config, add_pooling_layer=False)
        self.post_init()