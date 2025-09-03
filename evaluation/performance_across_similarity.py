import pandas as pd
import pickle
import numpy as np
from performance_matrix import calculate_protein_centric_fmax,calculate_term_centric_aupr
def load_and_filter_by_similarity( threshold='<30%',ont="ec",model="rs"):

    if ont=='ec':
        file_path_suffix = "nrPDB-EC_2020.04_test"
    else:
        file_path_suffix = "nrPDB-GO_2019.06.18_test"

    similarity_csv_path = f'/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/{file_path_suffix}.csv'
    # Load the results
    results_path = f'/mmfs1/home/mdtahmid.islam/deepfri_bert/evaluation/eval_results/{ont}_cmap_bias_per_head_alpha_no_cnn_finetuned_final_evaluation_results.pkl'
    if model == 'rs':  
        print("RS model****")                                                                                         
        results_path = f'/mmfs1/home/mdtahmid.islam/deepfri_bert/evaluation/eval_results/{ont}_basic_prot_final_prot_results.pkl'
   
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    print(len(results))
    # Load the similarity CSV
    df = pd.read_csv(similarity_csv_path, index_col="PDB-chain")

    
    # Subset keys by threshold
    filtered_probs = []
    filtered_labels = []

    for prot_id, content in results.items():


        if prot_id in df.index and df.loc[prot_id, threshold] == 1:
            filtered_probs.append(content['probs'])
            filtered_labels.append(content['labels'])
        

    if not filtered_probs:
        raise ValueError(f"No proteins matched the {threshold} filter.")
    final_labels, final_probs = np.stack(filtered_labels), np.stack(filtered_probs)
    print(" Shape of final_probs:", final_probs.shape, end=" ")
    print("Shape of final labels",final_labels.shape)
    fmax = calculate_protein_centric_fmax(final_labels,final_probs)
    micro_aupr,macro_aupr = calculate_term_centric_aupr(final_labels,final_probs)
    print(f'model = {model} {ont}, at {threshold}, Fmax={fmax:.3f}, MACRO_AUPR={macro_aupr:.3f},  MICRO_AUPR={micro_aupr:.3f} ')

o = 'cc'
m = "rs"
load_and_filter_by_similarity(threshold='<30%',ont=o,model=m)
load_and_filter_by_similarity(threshold='<40%',ont=o,model=m)
load_and_filter_by_similarity(threshold='<50%',ont=o,model=m)
load_and_filter_by_similarity(threshold='<70%',ont=o,model=m)
load_and_filter_by_similarity(threshold='<95%',ont=o,model=m)