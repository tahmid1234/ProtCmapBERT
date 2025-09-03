import numpy as np
import json
import sys 
import os

def map_terms(label_name):
    looking_for ="goterms"
    keywords = {'mf':'DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_molecular_function_model_params.json',
                'ec':'DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_enzyme_commission_model_params.json',
                'bp':'DeepFRI-MERGED_MultiGraphConv_3x512_fcd_2048_ca_10A_biological_process_model_params.json',
                'cc':'DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_cellular_component_model_params.json'}
    old_model_json = keywords[label_name]
    new_model_json = f"DeepFRI-MERGED_GraphConv_gcd_512-512-512_fcd_1024_ca_10.0_{label_name}_model_params.json"
    # if label_name == 'mf':
    #         old_model_json = "DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_molecular_function_model_params.json"
    #         new_model_json = "DeepFRI-MERGED_GraphConv_gcd_512-512-512_fcd_1024_ca_10.0_mf_model_params.json"
    # elif label_name == 'ec':
    #         old_model_json = "DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_enzyme_commission_model_params.json"
    #         new_model_json = "DeepFRI-MERGED_GraphConv_gcd_512-512-512_fcd_1024_ca_10.0_ec_model_params.json"
    file_path = '/mmfs1/projects/changhui.yan/mdtahmid.islam/ProtCmapBERT_Refined/ProtCmapBERT/data'
    with open(f'{file_path}/{new_model_json}') as f1, open(f'{file_path}/{old_model_json}') as f2:
        latest_go_terms = json.load(f1)[looking_for]   # 989 terms
        old_go_terms = json.load(f2)[looking_for]      # 489 terms

    # Build index map from old terms to index in latest model
    go_term_to_index_latest = {term: idx for idx, term in enumerate(latest_go_terms)}

    # Create the index mapping: old index -> new index
    index_mapping = []
    missing_terms = []
    
    for old_idx, term in enumerate(old_go_terms):
        if term in go_term_to_index_latest:
            index_mapping.append(go_term_to_index_latest[term])
        else:
            index_mapping.append(None)  # This term is missing in the latest model
            missing_terms.append((old_idx, term))
            

    # print(f"Total terms in old: {len(old_go_terms)}")
    # print(f"Total matched terms: {len([i for i in index_mapping if i is not None])}")
    # print(f"Missing terms: {len(missing_terms)}")
    # print(missing_terms)
    return index_mapping,missing_terms