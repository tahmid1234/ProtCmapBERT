"""
* This script aligns GO terms between our model and the latest DeepFRI model 
  to ensure a fair comparison.

* Context:
    - Our model was trained on a dataset with a slightly different set of GO terms.
    - The latest DeepFRI model was trained on a dataset containing more GO terms.
    - To compare fairly, we only consider the common terms present in both models.

* Function: map_terms(label_name)
    - Args:
        label_name (str): One of {"mf", "ec", "bp", "cc"} indicating the ontology type.
    - Process:
        1. Loads GO terms from our model and the latest DeepFRI model parameter files.
        2. Builds an index mapping from our model’s GO terms to their corresponding indices 
           in the latest DeepFRI model (if they exist).
        3. Marks missing terms (those not found in the latest DeepFRI model).
    - Returns:
        index_mapping (list[int | None]): Maps each of our model’s term indices to 
                                          the latest DeepFRI model’s indices, or None if missing.
        missing_terms (list[tuple[int, str]]): List of (index, term) pairs from our model 
                                               that are not present in the latest DeepFRI model.
"""


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

    file_path = './data'
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