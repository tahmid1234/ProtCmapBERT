#we just added cmap as bias without any learner params,


import math
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention


class SelfAttentionWithContact(BertSelfAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        past_key_value: tuple = None,
        output_attentions: bool = False,
        cmap: torch.FloatTensor = None, 
        **kwargs
    ):

        batch_size, seq_len, hidden_size = hidden_states.size()
        
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states) 

        
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        if Q.shape[1] !=16 or Q.shape[3]!=64:
            print("Issue with shape",Q.shape)
        
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            print("NaN or Inf detected in Q!")
        if torch.isnan(K).any() or torch.isinf(K).any():
            print("NaN or Inf detected in K!")
        if torch.isnan(V).any() or torch.isinf(V).any():
            print("NaN or Inf detected in V!")

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.attention_head_size)  # (batch_size, seq_len, seq_len)
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
            print("NaN  attn_scored!")

        if cmap is not None:
            
            attn_scores = attn_scores+cmap  

            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                print("NaN or Inf detected in cmap * attn_scored!")
        
        if attention_mask is not None:
            attn_scores = attn_scores +attention_mask

       
        attention_probs = nn.functional.softmax(attn_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs,V)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), context_layer.size(1), -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

     
        if torch.isnan(context_layer).any() or torch.isinf(context_layer).any():
            print("NaN or Inf detected in output!")

        return outputs
