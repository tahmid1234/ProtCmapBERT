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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.fasta import read_fasta



# --- Configuration ---
TF_DATA_DIR = '/mmfs1/home/mdtahmid.islam/pdb_files'
# '/mmfs1/home/mdtahmid.islam/deepfri_bert/test_data_processing/test_data_creation_scripts/seq_td_clean_go'

file_pattern = os.path.join(TF_DATA_DIR,"PDB_GO_train_*.tfrecords")
# os.path.join(TF_DATA_DIR,"test_datatest_protein_data*.tfrecord")

# TF_DATA_DIR = '/mmfs1/home/mdtahmid.islam/pdb_files'

# file_pattern = os.path.join(TF_DATA_DIR,"PDB_GO_train_*.tfrecords")

# --------- SETTINGS ---------
ontology = "mf"  # or 'bp', 'cc' etc. for label key



fasta_sequences = read_fasta('/mmfs1/home/mdtahmid.islam/nrPDB-GO_2019.06.18_sequences.fasta')

def get_sequence(prot_id,fasta_sequences):
    if isinstance(prot_id, bytes):
        prot_id = prot_id.decode('utf-8')
    return fasta_sequences.get(prot_id, "")

# --------- PARSE FUNCTION ---------
def parse_example(example_proto):
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'ca_dist_matrix': tf.io.VarLenFeature(tf.float32),
        ontology + "_labels": tf.io.VarLenFeature(tf.int64),
        'seq_1hot': tf.io.VarLenFeature(tf.float32),
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64,default_value=0),
        'sequence': tf.io.FixedLenFeature([], tf.string, default_value=''),  
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
j = 0
es = set()
count_1 = 0
count_2 = 0
min_count = 2000
max_count = 0
for sample in dataset:
    i+=1
    prot_id = sample['prot_id'].numpy()
    L = sample['L'][0]
    min_count = min(L,min_count)
    max_count = max(L,max_count)
    seq = get_sequence(prot_id,fasta_sequences)           
    es.add(prot_id)
    sequence = sample['sequence'].numpy()
    ca_dist_len = sample['ca_dist_matrix'].values.shape[0]
    seq_1hot_len = sample['seq_1hot'].values.shape[0]
    max_length = max(len(seq),max_length)
    if len(seq)>550:
        count_1+=1
    if len(seq)==1000: 
        count_2+=1
    print(f'seq length = {len(sequence)}  ca_dis shape is {ca_dist_len}, prot_id ,{ len(es)}')
    j+=1
    

print(len(es)," total sample count is ",i," ",j)
print(f'min_count {min_count}, maxCoint {max_count}')
print(max_length, " sequence max lenght",count_1," 400 count 300 ",count_2)
# --------- SAVE TO CSV ---------
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["prot_id"])
#     writer.writerows(mismatched_ids)

# print(f"Saved {len(mismatched_ids)} mismatched prot_ids to {output_csv}")