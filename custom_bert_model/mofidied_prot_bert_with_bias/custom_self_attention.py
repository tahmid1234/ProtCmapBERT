import math
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertConfig

#per head alpha to infuence its cmap
class SelfAttentionWithContact(BertSelfAttention):
    def __init__(self,config:BertConfig):
        super().__init__(config)
        # We are using a scalar alpha learnable parameter to influence our cmap as a bias.
        # Initialize to 0 so it doesn't disrupt pre-trained weights initially.
        print("learnable parameter")
        self.cmap_alpha_per_head = nn.Parameter(torch.zeros(self.num_attention_heads))


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
            cmap = cmap.to(dtype = attn_scores.dtype) #so we can take advantage of AMP
            cmap_expanded = cmap.unsqueeze(1) # Shape: (batch_size, 1, seq_len, seq_len)
            rehspade_apha = self.cmap_alpha_per_head.view(1,self.num_attention_heads,1,1) #shape changes from head_num to 1,head_num,1,1 
            # This is a soft bias. It adds a scaled contact map to heads of the attention scores.
            # For non-contacting residues (cmap=0), it adds 0, causing no change.
            # For contacting residues (cmap=1), it adds `cmap_alpha`, providing a positive bias.
            # each of the num_heads can learn its own distinct strength (alpha_h) for biasing attention towards contacts. 
            alpha_bias = rehspade_apha * cmap_expanded
            attn_scores = attn_scores + alpha_bias
            #rehspade_apha * cmap_expanded  = [1,head_num,1,1]*[batch,1,seq,seq] = [batch,head_num,seq,seq] = shape of attention score


            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                print("NaN or Inf detected in cmap * attn_scored!")
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype = attn_scores.dtype)
            attn_scores = attn_scores +attention_mask

       
        attention_probs = nn.functional.softmax(attn_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs,V)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

     
        #if torch.isnan(context_layer).any() or torch.isinf(context_layer).any():
            #print("NaN or Inf detected in output!")

        return outputs
