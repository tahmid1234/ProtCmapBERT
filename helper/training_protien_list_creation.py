# Purpose of this file is to make sure whether the datase sourced from DeepFRI Github Repository
# has consistent sequence lenght with the ca-ca dist matrix

import tensorflow as tf
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np
import functools
import sys 
import os
import csv




# --- Configuration ---
ont = 'EC' #or go
if ont =='EC':
    TF_DATA_DIR = '/mmfs1/home/mdtahmid.islam/EC_Files/PDB-EC'
else:
    TF_DATA_DIR = '/mmfs1/home/mdtahmid.islam/pdb_files'
# '/mmfs1/home/mdtahmid.islam/deepfri_bert/test_data_processing/test_data_creation_scripts/seq_td_clean_go'

file_pattern = os.path.join(TF_DATA_DIR,f"PDB_{ont}_train_*.tfrecords")







# --------- PARSE FUNCTION ---------
def parse_example(example_proto):
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
      
        
    }
    return tf.io.parse_single_example(example_proto, feature_description)

# --------- PROCESSING ---------
mismatched_ids = []


max_length = 0

dataset = tf.data.Dataset.list_files(file_pattern)
dataset = dataset.interleave(
    lambda filepath: tf.data.TFRecordDataset(filepath),
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

i =0

train_ids = []
j = 0
for sample in dataset:
    i+=1
    prot_id = sample['prot_id'].numpy().decode('utf-8')
    train_ids.append(prot_id)
np.savez(f"/mmfs1/home/mdtahmid.islam/deepfri_bert/data/{ont}_train_protein_ids.npz", ids=train_ids)
print(f'totla_count = {i} total0s = {j}') 
   
    

