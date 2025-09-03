import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import gc 
import pickle
from performance_matrix import calculate_protein_centric_fmax,calculate_term_centric_aupr
from eval_helper import map_terms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.sample_stat import load_GO_annot, load_EC_annot
from helper.custom_bert_initiator import get_custom_prot_model_instance
from helper.tokenizer import load_pretrained_tokenizer,load_new_tokenizer
from data_loading.tf_data_loading import get_data_loader


def wrap_the_model(model):
    gpu_count = 0
        #multi gpu configuration
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Wrap the model for multi-GPU
        gpu_count = torch.cuda.device_count()
    return model,gpu_count


def evaluate_model(ont,model, dataloader, device, filename, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename + "_results.pkl")

    # To compare our model with the newest DeepFRI model, we removed the missing terms that were not included in either dataset.
    _, missing_terms = map_terms(ont)
    missing_indices = [index for index, _ in missing_terms]
    
    COUNT = 0
    # To check if all unique samples are processed
    empty_set = set()
    results = {}

    for batch_idx, batch in enumerate( dataloader):

        input_ids, attention_mask, cmap, label,prot_ids = batch  
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        cmap = cmap.to(device)
        label = label.to(device)
        for prot_id in prot_ids:
            empty_set.add(prot_id)
        
        COUNT+= cmap.shape[0]



        with torch.no_grad(),torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            logits = model(input_ids, attention_mask, cmap)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = label.detach().cpu().numpy()
        probs[:, missing_indices] = 0
        labels[:, missing_indices] = 0
        
       
        print("batch id######", batch_idx,"probs shape ", probs.shape, "count ", COUNT," prot set ",len(empty_set))

        for i, prot_id in enumerate(prot_ids):
            results[prot_id] = {
                "probs": probs[i],
                "labels": labels[i]
            }

    # Immediately save to file to avoid memory overflow
    with open(output_path, "wb") as f:
        pickle.dump(results, f)



  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'ec'], help="Ontology.")
    parser.add_argument('--fasta_seq', type=str, help="All our testing files contain sequence data")
    parser.add_argument('--test_file_path', type=str, default="/mmfs1/projects/changhui.yan/mdtahmid.islam/ProtCmapBERT_Refined/ProtCmapBERT/evaluation_ds/seq_td_clean_go", help="File with test PDB chains.")
    parser.add_argument( '--extra_layer', type=str,default='cmap_bias' , choices=['cmap_bias','basic'], help="choose cmap_bias to use our model and basic to evaluate a finetuned model")
    parser.add_argument('--tokenizer_path', type=str, default=None,  help="tokenizer_path")
    parser.add_argument('--output_dir',type=str, default="/mmfs1/projects/changhui.yan/mdtahmid.islam/ProtCmapBERT_Refined/ProtCmapBERT/evaluation/eval_results",help="Saving path for thr raw logits")
    parser.add_argument('--pretrained_model_name', type=str, default='Rostlab/prot_bert',  help="Pretrained Model Name")
    parser.add_argument('--drop_out_rate',type=float,default=0.01)    
    parser.add_argument('--m_path',type=str, default='./ProtCmapBERT/all_models/ProtCmapBERT/dr_01_mf_lr_7e-06_cmap_bias_per_head_alpha_clipping_1_.pt',help="model_path")
    parser.add_argument('--file_pattern',type=str, default='test_datatest_protein_data*.tfrecord',help = "tf file pattern")
    
    parser.add_argument('--file_id',type = str,default = 1,help="  if gor the raw logit file")
    args = parser.parse_args()
    args.pad_len =  1002
    print(args)

    
    if args.ontology == 'ec':
        prot2annot, goterms, gonames, counts = load_EC_annot("/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_annot.tsv")  
    else:
        prot2annot, goterms, gonames, counts = load_GO_annot("/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv")
        
    goterms = goterms[args.ontology]
    gonames = gonames[args.ontology]
    args.output_dim = len(goterms)
    if args.tokenizer_path is not None:
        tokenizer = load_pretrained_tokenizer(args.tokenizer_path)
    else:
        tokenizer = load_new_tokenizer()
    args.vocab_size = tokenizer.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    test_loader =  get_data_loader(device,args.file_pattern,tokenizer,args.ontology,args.fasta_seq,args.test_file_path,args.pad_len)
 
    model = get_custom_prot_model_instance(args)

    model,gpu_count = wrap_the_model(model)
    model.to(device)

    model_path  = args.m_path
   
    print("Model Path",model_path)
    #model loading
    model.load_state_dict(torch.load(model_path))

    model = torch.compile(model)

    print("Evaluation Strated")
    # Complete file path for saving the predicted logits
    file_name = f'{args.ontology}_{args.extra_layer}_{args.file_id}'
    model.eval()
    evaluate_model(args.ontology,model,test_loader,device,file_name,args.output_dir)
    print("Evaluation Done")


    
  








   

