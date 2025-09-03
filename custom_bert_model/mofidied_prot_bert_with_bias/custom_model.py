import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertEncoder, BertModel
from transformers import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import sys
import os
from .custom_self_attention import *

class CustomModel_With_Modified_Attention(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        print(" CUSTOM MODEL BIAS")
        for layer in self.encoder.layer:
            old_self_attn = layer.attention.self
            new_self_attn = SelfAttentionWithContact(config)

            # Copy pretrained weights
            new_self_attn.query.load_state_dict(old_self_attn.query.state_dict())
            new_self_attn.key.load_state_dict(old_self_attn.key.state_dict())
            new_self_attn.value.load_state_dict(old_self_attn.value.state_dict())

            layer.attention.self = new_self_attn


 
        




    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cmap=None):
  
        # Determine input shape for attention mask extension
        output_attentions    = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict if return_dict is not None else self.config.use_return_dict

        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = (1.0 - extended_mask) * -1e4
        # cmap = cmap.unsqueeze(1)
        # cmap = (1.0 - cmap) * -1e4

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # 3) Encoder loop (no extra layernorm!)
        all_hidden_states = () if output_hidden_states else None
        all_attentions    = () if output_attentions else None
        hidden_states     = embedding_output
        for i, layer_module in enumerate(self.encoder.layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 3a) Self-attention with sim_matrix
            attn_outputs = layer_module.attention.self(
                hidden_states,
                attention_mask=extended_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                output_attentions=output_attentions,
                cmap=cmap, 
            )
            attn_output = layer_module.attention.output(attn_outputs[0], hidden_states)

            # 3b) Feed-forward
            intermediate_output = layer_module.intermediate(attn_output)
            hidden_states       = layer_module.output(intermediate_output, attn_output)

            if output_attentions:
                all_attentions += (attn_outputs[1],)


        pooled_output = self.pooler(hidden_states)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )