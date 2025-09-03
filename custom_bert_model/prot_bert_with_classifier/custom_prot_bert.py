import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertConfig
    




class CustomBert_Prot_Emb(nn.Module):
    def __init__(self, pretrained_model_name,args):
        super(CustomBert_Prot_Emb, self).__init__()
        print(pretrained_model_name," model", " BASICC", "droprate changed",args.drop_out_rate)
        config = BertConfig.from_pretrained(pretrained_model_name)
        config.attention_probs_dropout_prob = args.drop_out_rate
        config.hidden_dropout_prob = args.drop_out_rate
        print("Hidden Size", config.hidden_size)

        self.pretrained_model = BertModel.from_pretrained(pretrained_model_name,config=config)
        # Output heads for each task
        self.mf_head = nn.Linear(config.hidden_size, args.output_dim)



    def forward(self, input_ids,mask,cmap=None):

        pretrained_output = self.pretrained_model(input_ids,attention_mask=mask)

        pooled_output = pretrained_output.pooler_output
        
        logits = self.mf_head(pooled_output)

        return logits