"""
* This script converts protein PDB files into TFRecord datasets.  
* For each protein, it stores:
    - Sequence
    - CA–CA distance matrix
    - Functional annotations (GO or EC terms)
* Supports filtering by sequence identity (e.g., <30%) to build test sets.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
from annotation_processing import load_all_go_terms,load_all_ec_numbers
import sys 
import argparse
from pdb_extractor import load_predicted_clean_PDB

# Function to create a TFRecord example
def create_example(prot_id, distance_matrix,seq, terms,label_name):
    if label_name =='ec':
        feature = {
        'prot_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[prot_id.encode()])),
        'ca_dist_matrix': tf.train.Feature(float_list=tf.train.FloatList(value=distance_matrix.flatten())),
        'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seq.encode()])),
        'ec_labels':tf.train.Feature(int64_list=tf.train.Int64List(value=terms['ec'])),
        'L': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(seq)])),
    }
    else:
        feature = {
            'prot_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[prot_id.encode()])),
            'ca_dist_matrix': tf.train.Feature(float_list=tf.train.FloatList(value=distance_matrix.flatten())),
            'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seq.encode()])),
            'mf_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=terms['mf'])),
            'bp_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=terms['bp'])),
            'cc_labels': tf.train.Feature(int64_list=tf.train.Int64List(value=terms['cc'])),
            'L': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(seq)])),
            
        }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--label_name', type=str, default='ec', choices=['mf', 'bp', 'cc', 'ec'], help="Ontology.")
    parser.add_argument( '--percentage', type=str, default='30', choices=['30', '100'], help="Identity Similarity.")
    args = parser.parse_args()
    
    filtered_pdb = None
    pdb_ext = 'go'
    #label decides which file to utilize as source of test samples
    if args.label_name == 'ec':
        file_path_suffix = "nrPDB-EC_2020.04_test"
        annotations = load_all_ec_numbers('data/nrPDB-EC_2020.04_annot.tsv')
        pdb_ext = 'ec'
    else:
        file_path_suffix = "nrPDB-GO_2019.06.18_test"
        annotations = load_all_go_terms('data/nrPDB-GO_2019.06.18_annot.tsv'
        )
    df = pd.read_csv(f'data/{file_path_suffix}.csv')
    #pdb id and chain chain id are splitted
    df[['PDB_ID', 'Chain_ID']] = df['PDB-chain'].str.split('-', expand=True)
    output_dir_suffix = "clean_"+pdb_ext
    #if perecentage is 30, we create and array containing only pid with 30 or less similarity identity percentage
    if args.percentage == '30':
        print(" Percentage ", args.percentage)
        filtered_pdb =  df[df["<30%"] == 1]["PDB-chain"].tolist()
        output_dir_suffix = output_dir_suffix+"_less_30"
    output_dir = f'data/{output_dir_suffix}'
    os.makedirs(output_dir, exist_ok=True) 
    base_filename = 'test_datatest_protein_data'
    entries_per_file = 1000

    file_index = 0
    record_count = 0
    writer = None
   
    pdb_folder = f"data/clean_td_{pdb_ext}_pdb"
    pdb_files = glob.glob(os.path.join(pdb_folder, "*.pdb")) 
    # Create a TFRecord file
    for idx, pdb_file in enumerate(pdb_files):
        prot_id = os.path.basename(pdb_file).replace(".pdb", "")



        #comment this if don't want to create a dataset with only <30% identity similariyt
        if filtered_pdb and prot_id not in filtered_pdb:
            print(" filtered out")
            continue
        
    
        # Load GO term labels
        
        terms  = annotations.get(prot_id.lower())
        



        # calling the the deepfri's util's function for sequence and ca-ca dist matrix
        distance_matrix, seq = load_predicted_clean_PDB(f'{pdb_folder}/{prot_id}.pdb')
        if len(seq) != distance_matrix.shape[0]:
            print(f'{prot_id} has a mis match. Lenght of the sequence is {len(seq)} and the shape of the matrix is  {distance_matrix.shape} ')
            continue
        if terms is None:
            print(f"{prot_id} is none for ",args.label_name)
            continue
        example = create_example(prot_id, distance_matrix,seq, terms,args.label_name)
        print("example has been created")
        # Create new writer every 1000 records
        if record_count % entries_per_file == 0:
            if writer:
                writer.close()
            print("in side if")
            file_path = os.path.join(output_dir, f'{base_filename}{file_index:02d}.tfrecord')
            writer = tf.io.TFRecordWriter(file_path)
            file_index += 1
            print(f"Writing to {file_path}")

        writer.write(example.SerializeToString())
        record_count += 1
        print("Data entry processed #",idx)


    # Close the last writer
    if writer:
        writer.close()

    print(f"Finished writing {record_count} records into {file_index} files.")