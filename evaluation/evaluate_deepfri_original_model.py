import numpy as np
import json
import sys 
import os
import pandas as pd
from performance_matrix import *
import argparse
from eval_helper import map_terms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test_data_processing.test_data_creation_scripts.annotation_processing import load_all_go_terms,load_all_ec_numbers
# Load GO term lists
#this annotaion recod provides true lable



def evaluate_performance(args):

    filterd_out_chains = 0
    label_name = args.label_name
    index_mapping,missing_terms = map_terms(label_name)
    # /mmfs1/home/mdtahmid.islam/deepfri_bert/evaluation/eval_results/mf_all_preds_raw_given_pdb.npz
    # /mmfs1/home/mdtahmid.islam/deepfri_bert/evaluation/eval_results/{label_name}_all_preds_raw.npz
    data = np.load(f'/mmfs1/home/mdtahmid.islam/deepfri_bert/evaluation/eval_results/ec_final_all_preds_raw.npz')
    all_remapped_preds = []
    all_mf_labels = []
    common_keys = []
    filtered_pdb = None
    if args.percentage != "100":
        filtered_pdb = get_filtered_pdb(args.file_path_suffix,args.percentage)

    for chain_id in data.files:
        # Skip if we don't have ground-truth label
        if filtered_pdb and chain_id not in filtered_pdb:
            # print(" filtered out : ", chain_id)
            filterd_out_chains+=1
            continue
        
        terms  = args.annotations.get(chain_id.lower())
        # if chain_id =='5XTC-L' or chain_id=='5XTC-l':
        # print(terms," annotation",chain_id.lower())
        if terms is None:
            # print(" this chain id was in the new model's file but not in the annotation", chain_id)
            filterd_out_chains+=1
            continue
        # else:
        #     print(chain_id," okay")
        # break


        binary_preds = data[chain_id]  # shape (989,)
        remapped_preds = np.zeros(len(index_mapping))

        for i, latest_idx in enumerate(index_mapping):
            if latest_idx is not None:
                remapped_preds[i] = binary_preds[latest_idx]
            else:
                remapped_preds[i] = 0.0  # or np.nan

        # Get ground-truth MF label
        mf_label = args.annotations[chain_id.lower()][label_name]
       

        all_remapped_preds.append(remapped_preds)
        all_mf_labels.append(mf_label)
        common_keys.append(chain_id)  # optional: for tracking

    # # Optional: convert to NumPy arrays
    all_remapped_preds = np.array(all_remapped_preds)
    all_labels = np.array(all_mf_labels)

    # print("Total samples processed:", len(all_remapped_preds))
    # print("Shape of remapped predictions:", all_remapped_preds.shape)
    # print(f'Shape of {label_name} labels:', all_labels.shape)
    # print("Total chains:", len(data.files), "filterd_out_chains ",filterd_out_chains)

    fmax = calculate_protein_centric_fmax(all_labels,all_remapped_preds)
    # print(fmax," fmax")
    micro_aupr,macro_aupr = calculate_term_centric_aupr(all_labels,all_remapped_preds)
    print(f"{label_name}, at {args.percentage}%, Fmax={fmax:.3f}, MACRO_AUPR={macro_aupr:.3f},  MICRO_AUPR={micro_aupr:.3f}")

def get_filtered_pdb(file_path,percentage):
    df = pd.read_csv(f'/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/{file_path}.csv')

    filtered_pdb =  df[df["<"+percentage+"%"] == 1]["PDB-chain"].tolist()
    return filtered_pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--label_name', type=str, default='ec', choices=['mf', 'bp', 'cc', 'ec'], help="Ontology.")
    parser.add_argument( '--percentage', type=str, default='100', help="Identity Similarity.")
    args = parser.parse_args()
    if args.label_name == 'ec':
        args.file_path_suffix = "nrPDB-EC_2020.04_test"
        args.annotations = load_all_ec_numbers('/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_annot.tsv')

    else:
        args.file_path_suffix = "nrPDB-GO_2019.06.18_test"
        args.annotations = load_all_go_terms('/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv')
    args.percentage = '30'
    evaluate_performance(args)
    args.percentage = '40'
    evaluate_performance(args)
    args.percentage = '50'
    evaluate_performance(args)
    args.percentage = '70'
    evaluate_performance(args)
    args.percentage = '95'
    evaluate_performance(args)
    

        
            


