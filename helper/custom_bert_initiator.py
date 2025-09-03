import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))
from custom_bert_model.prot_bert_with_classifier.custom_prot_bert import CustomBert_Prot_Emb
from custom_bert_model.mofidied_prot_bert_with_bias.protien_classifier_bias import ProtienClassifierBias


                                              


def get_custom_prot_model_instance(args):
    pretrained_model_name = args.pretrained_model_name
    if args.extra_layer =='cmap_bias' :
        print("Protien Classifier Bias")
        return ProtienClassifierBias(pretrained_model_name,args)
    elif args.extra_layer =='basic':
        print("Basic Pretrained Model")
        return CustomBert_Prot_Emb(pretrained_model_name,args)


