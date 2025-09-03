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


fasta_sequences = read_fasta('/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_sequences.fasta')
#ec_fasta_sequence = read_fasta('/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_sequences.fasta')
# --- Configuration ---
# TF_DATA_DIR = '/mmfs1/home/mdtahmid.islam/EC_Files/PDB-EC/'
TF_DATA_DIR = '/mmfs1/home/mdtahmid.islam/pdb_files'
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 700
DISTANCE_THRESHOLD = 10.0 
file_pattern = os.path.join(TF_DATA_DIR,"PDB_GO_train_*.tfrecords")






# --------- SETTINGS ---------
ontology = "mf"  # or 'bp', 'cc' etc. for label key
output_csv = "mismatched_prot_ids.csv"




# --------- GET SEQUENCE ---------
def get_sequence(prot_id, fasta_sequences):
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
    }
    return tf.io.parse_single_example(example_proto, feature_description)

# --------- PROCESSING ---------
mismatched_ids = []



dataset = tf.data.Dataset.list_files(file_pattern)
dataset = dataset.interleave(
    lambda filepath: tf.data.TFRecordDataset(filepath),
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
#delete this line after verification
int i =0

for sample in dataset:
    prot_id = sample['prot_id'].numpy()
    sequence = get_sequence(prot_id, fasta_sequences)
    seq_len = len(sequence)

    ca_dist_len = sample['ca_dist_matrix'].values.shape[0]
    seq_1hot_len = sample['seq_1hot'].values.shape[0]

    seq_1hot_length_check = (seq_1hot_len % 26 == 0)
    if seq_1hot_length_check:
        inferred_seq_len = seq_1hot_len // 26
    else:
        inferred_seq_len = -1  # Invalid
    print(inferred_seq_len," seq hot  cadist ", ca_dist_len," seq ", seq_len," port ", prot_id.decode("utf-8"))
    if inferred_seq_len != seq_len or ca_dist_len != (seq_len * seq_len):
        print(" inside if")
        mismatched_ids.append([prot_id.decode("utf-8")])
    


# --------- SAVE TO CSV ---------
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["prot_id"])
#     writer.writerows(mismatched_ids)

# print(f"Saved {len(mismatched_ids)} mismatched prot_ids to {output_csv}")