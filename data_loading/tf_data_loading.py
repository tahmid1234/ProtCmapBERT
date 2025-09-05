import sys
import os
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np
import re
from transformers import BertTokenizer
import functools
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))
from helper.fasta import read_fasta

# --- Configuration ---
BATCH_SIZE = 16
DISTANCE_THRESHOLD = 10.0



def parse_tfrecord_fn(example,ont):
    feature_description = {
        'prot_id': tf.io.FixedLenFeature([], tf.string),
        'ca_dist_matrix': tf.io.VarLenFeature(tf.float32),
        ont+"_labels": tf.io.VarLenFeature(tf.int64),
        'seq_1hot': tf.io.VarLenFeature(tf.float32),
        'sequence': tf.io.FixedLenFeature([], tf.string, default_value=''),  
        
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    prot_id = example['prot_id']
    ca_dist_matrix = tf.sparse.to_dense(example['ca_dist_matrix'])
    label = tf.sparse.to_dense(example[ont+'_labels'])
    sequence = example['sequence']  # This will be empty string if not present

    return prot_id, ca_dist_matrix, label,sequence

def get_sequence(prot_id,fasta_sequences):
    if isinstance(prot_id, bytes):
        prot_id = prot_id.decode('utf-8')
    return fasta_sequences.get(prot_id, "")

def get_token(tokenizer,sequence,pad_len):

    sequence = ' '.join(list(sequence))
    sequence = re.sub(r"[UZOB]", "X",  sequence)
    return tokenizer(
        sequence,
        max_length=pad_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def process_sample(prot_id, ca_dist, label,sequence,tokenizer,fasta_sequences,pad_len):
    # Get sequence
    #if sequence is not provided in the dataset, use fasta file
    if not sequence:
        sequence = get_sequence(prot_id,fasta_sequences)
        
    if not sequence:
        print(" no sequbnece for prot_id is ", prot_id)
        return None
    
    # Tokenize sequence
    #padded_sequence = sequence.ljust(pad_len, 'X')[:pad_len]
    seq_len = len(sequence)
    if seq_len> pad_len-2:
        print(f'{prot_id} has a sequence {seq_len}')
        return None
    

    
 
    tokenized = get_token(tokenizer,sequence,pad_len)
    #return_tensors='pt', the tokenizer always returns batch-formatted tensors
    #tokenized['input_ids'] = tensor([[2, 21, 24, ..., 3]])  # Shape [1, 512]
    input_ids = tokenized['input_ids'].squeeze(0)
    attention_mask = tokenized['attention_mask'].squeeze(0)
    

    # Process contact map
    
    if seq_len > 0:
        # Ensure ca_dist is numpy array
        ca_dist = ca_dist.numpy() if hasattr(ca_dist, 'numpy') else ca_dist
        # ca_dist = np.insert(ca_dist, 0, ca_dist[0])
        # Handle case where ca_dist might be empty
        if ca_dist.size == 0:
            expanded_binary_cmap = np.zeros((pad_len, pad_len), dtype=np.float32)
        else:
            # Reshape distance matrix
            try:
                dist_matrix = ca_dist.reshape(seq_len, seq_len)
                binary_cmap = (dist_matrix <= DISTANCE_THRESHOLD).astype(np.float32)
                #The follwoing code snipped to align the contact map with tokens. After tokenization there is 2 extra token CLS and SEP. We can handle SEP later while padding.
                expanded_binary_cmap = np.zeros((seq_len+1, seq_len+1), dtype=binary_cmap.dtype)
                expanded_binary_cmap[1:, 1:] = binary_cmap
                num_ones = attention_mask.sum().item()
                # print(expanded_binary_cmap.shape," extended CMAP SIZE binary ", binary_cmap.shape," seq len ", seq_len, " attention mask - ", num_ones," input ids lenght ",input_ids.shape)
                # Pad or truncate contact map
                if expanded_binary_cmap.shape[0] > pad_len or expanded_binary_cmap.shape[1] > pad_len:
                    expanded_binary_cmap = expanded_binary_cmap[:pad_len, :pad_len]
                else:
                    pad_height = pad_len - expanded_binary_cmap.shape[0]
                    pad_width = pad_len - expanded_binary_cmap.shape[1]
                    expanded_binary_cmap = np.pad(expanded_binary_cmap, ((0, pad_height), (0, pad_width)), 'constant')
            except ValueError:
                expanded_binary_cmap = np.zeros((pad_len, pad_len), dtype=np.float32)
    else:
        expanded_binary_cmap = np.zeros((pad_len, pad_len), dtype=np.float32)
    
    # Convert to PyTorch tensors
    cmap_tensor = torch.from_numpy(expanded_binary_cmap).float()
    label_tensor = torch.from_numpy(label.numpy() if hasattr(label, 'numpy') else label).float()
    if isinstance(prot_id, bytes):
        prot_id = prot_id.decode('utf-8')
    return input_ids, attention_mask, cmap_tensor, label_tensor,prot_id

class TFRecordIterableDataset(IterableDataset):
    def __init__(self, file_pattern,tokenizer,ont,fasta_file_path,pad_len):
        self.file_pattern = file_pattern
        self.tokenizer = tokenizer
        self.ont = ont
        self.pad_len = pad_len
        self.fasta_sequences = read_fasta(fasta_file_path) if fasta_file_path else None

        
    #When using PyTorch’s DataLoader with an IterableDataset and setting num_workers > 0, 
    ###there’s a risk that each worker may process the same data, leading to duplication.
    #To prevent data duplication across workers
    ###we used torch.utils.data.get_worker_info()
    # Single-process data loading if worker_info is None:
    #assigned_files = file_list
    #file_list is a list of all data files (e.g., TFRecord files) to be processed.
	#worker_id is the unique identifier for the current worker (ranging from 0 to num_workers - 1).
	#num_workers is the total number of worker processes.
    #since we have 30 files and 4 worker, the distribuition would look like this
    #•Worker 0 (worker_id = 0) processes: file_list[0::4] → ['file0', 'file4', 'file8', ..., 'file28']
	#•Worker 1 (worker_id = 1) processes: file_list[1::4] → ['file1', 'file5', 'file9', ..., 'file29']
	#•Worker 2 (worker_id = 2) processes: file_list[2::4] → ['file2', 'file6', 'file10', ..., 'file26']
	#•Worker 3 (worker_id = 3) processes: file_list[3::4] → ['file3', 'file7', 'file11', ..., 'file27']
    def __iter__(self):
      
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        all_files = sorted(tf.io.gfile.glob(self.file_pattern))
        files_for_this_worker = all_files[worker_id::num_workers]

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(files_for_this_worker, dtype=tf.string))
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
        print(worker_id, num_workers," worker ",end="")
        parse_fn_with_args = functools.partial(parse_tfrecord_fn, ont=self.ont)
        dataset = dataset.map(parse_fn_with_args, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Convert to PyTorch tensors
        for prot_id, ca_dist, label,sequence in dataset:
            processed_sample = process_sample(
                                prot_id.numpy(), 
                                ca_dist.numpy(), 
                                label.numpy(),
                                sequence.numpy().decode('utf-8'),
                                self.tokenizer,
                                self.fasta_sequences,
                                self.pad_len)
            if processed_sample is not None:
                yield processed_sample


def collate_fn(batch):
   
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    cmaps = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    prot_ids = [item[4] for item in batch]
    return input_ids, attention_masks, cmaps, labels,prot_ids


def get_data_loader(device,FILE_NAME_PATTERN,tokenizer,ont,fasta_file_path,directory,pad_len):
    print("BATCH_SIZE",BATCH_SIZE)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size = BATCH_SIZE * torch.cuda.device_count()
    batch_size = BATCH_SIZE 
    file_pattern = os.path.join(directory, FILE_NAME_PATTERN)
    dataset = TFRecordIterableDataset(file_pattern,tokenizer,ont,fasta_file_path,pad_len)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )

    return data_loader
    # Simple test to verify the data pipeline works
    # for batch_idx, (input_ids, attention_mask, cmap, label) in enumerate(data_loader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Input IDs shape: {input_ids.shape}")
    #     print(f"  Attention mask shape: {attention_mask.shape}")
    #     print(f"  Contact map shape: {cmap.shape}")
    #     print(f"  Labels shape: {label.shape}")
    #     print("Cmap")
    #     print(cmap)
        
    #     if batch_idx == 0:  # Just check first few batches
    #         break