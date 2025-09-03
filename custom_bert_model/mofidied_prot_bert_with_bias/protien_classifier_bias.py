import torch
import torch.nn as nn
from transformers import BertConfig
import sys
import os
from .custom_model import CustomModel_With_Modified_Attention
def get_pooling(output,pooling="first"):
        if pooling == "mean":
            output = torch.mean(output, dim=1)
        elif pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        return output
class ProtienClassifierBias(nn.Module):
    def __init__(self, pretrained_model_name,args):
        super(ProtienClassifierBias, self).__init__()
        print(f"Protien Classifier Biass WITHOUT CNN dropout rate = {args.drop_out_rate}")
        config = BertConfig.from_pretrained(pretrained_model_name)
        config.attention_probs_dropout_prob = args.drop_out_rate
        config.hidden_dropout_prob = args.drop_out_rate    
        self.model  = CustomModel_With_Modified_Attention.from_pretrained(pretrained_model_name, config=config)
        self.mf_head = nn.Linear(config.hidden_size, args.output_dim)
    
    def forward(self,input_ids,attention_mask,cmap):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, cmap=cmap)

        return  self.mf_head(get_pooling(output.last_hidden_state,"mean"))

