import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW

sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))
from helper.custom_bert_initiator import get_custom_prot_model_instance
from helper.sample_stat import load_GO_annot, load_EC_annot
from helper.tokenizer import load_pretrained_tokenizer,load_new_tokenizer
from data_loading.tf_data_loading import get_data_loader


def wrap_the_model(model):
    gpu_count = 0
        #multi gpu configuration
    if torch.cuda.device_count() > 0:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Wrap the model for multi-GPU
        gpu_count = torch.cuda.device_count()
    return model,gpu_count


def start_training(optimizer,model,pos_weigh,train_loader,val_loader,ep,device,m_path):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    best_val_loss = float('inf')
    #define patience
    patience = 10 
    patience_counter = 0
    #define scaler
    scaler = torch.cuda.amp.GradScaler()if device.type == 'cuda' else None
    for epoch in range(ep):  # Example: 10 epochs
        print(f"epoch-{epoch} has strated ")
        
        total_loss = 0
        batch_count = 0

        print("Training Loop")
        #this set is used to verify whether an internal unnoticed issue is repeating same samples
        train_prot_ids = set()
        val_prot_ids = set()
        #this is to check if all samples have been processed
        sample_count = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):

            input_ids, attention_mask, cmap, label,prot_ids = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)    
            cmap = cmap.to(device)
            label = label.to(device)

            #A loop to check if all the sample in an epoch are unique
            for prot_id in prot_ids:
                train_prot_ids.add(prot_id)
            
            if torch.isnan(cmap).any() or torch.isinf(cmap).any():
                print("NaN or Inf detected in cmap!")

            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(input_ids, attention_mask,cmap)
            loss = loss_fn(logits, label.float())
            total_loss += loss.item()

            
            optimizer.zero_grad()
            if scaler is not None:  # GPU path
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:       # CPU path
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            #batch_count+= input_ids.size(0)
            batch_count = batch_idx
            sample_count = cmap.shape[0]+sample_count

            print(f"train Epoch {epoch}, batchidx, {batch_idx}, batch count: {batch_count} ,Loss: {loss.item()}, port_ids_count, {len(train_prot_ids)}, sample count, {sample_count}")
            
            # if batch_idx == 0:  # Just check first few batches
            #     break

        print(total_loss," : training total loss batch_count: ", batch_count," prot count", len(train_prot_ids))
        training_avg_loss = total_loss / (batch_count+1)

        print("Validation") 
        model.eval()
        val_loss = 0.0
        batch_count = 0
        sample_count = 0
        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, batch in enumerate(val_loader):
       
                input_ids, attention_mask, cmap, label,prot_ids = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                cmap = cmap.to(device)
                label = label.to(device)

                for prot_id in prot_ids:
                    val_prot_ids.add(prot_id)

        
                # Wrap forward pass in autocast
    
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    outputs = model(input_ids,attention_mask, cmap)
                    loss = loss_fn(outputs, label.float())  # Calculate loss inside autocast
                   
                sample_count = cmap.shape[0]+sample_count
                val_loss += loss.item() 
                batch_count = batch_idx
                print(f"val - Epoch {epoch}, batchidx, {batch_idx}, batch count: {batch_count} ,Loss: {loss.item()}, port_ids, {len(val_prot_ids)}, Sample Count, {sample_count}")
        
        avg_val_loss = val_loss / (batch_count+1)
        print(val_loss," : validation total loss batch_count: ", batch_count," port count ", len(val_prot_ids))
        
        # --- DeepFRI-style Early Stopping ---
        print(f"Epoch {epoch+1}/{epoch} | Train Loss: {training_avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model (weights only)
            
            torch.save(model.state_dict(),m_path) 
            print("    Model has been updated at",epoch," epoch and loss is =",best_val_loss)
        else:
            patience_counter += 1
            print("patience_counter: ",patience_counter," epoch:",epoch)
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}! ",m_path)
                break


def get_class_weights(class_sizes, mean_class,clip_min):
  
    pos_weights = mean_class /class_sizes +1e-18
    
    # Clip weights to avoid extreme values (between 1 and 10)
    if clip_min >=1:
        print(f"Clip added {clip_min} to 10.0")
        pos_weights = np.clip(pos_weights,clip_min, 10.0)
    else:
        print("No Clipping")
        pos_weights = np.clip(pos_weights, a_min=None, a_max=10.0)
    # Convert to tensor
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    return pos_weights_tensor

if __name__ =='__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=5e-6, help="learning_rate.")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('-pd', '--pad_len', type=int, default = 1002, help="Padd length (max len of protein sequences in train set).")
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'ec'], help="Ontology.")
    parser.add_argument( '--extra_layer', type=str,  choices=['cnn','cmap','cnn_bias','cmap_bias','cmap_cnn_bias_alpha','cnn_mul_alpha','cnn_bias_per_head_alpha','cnn_mul_per_head_alpha','cmap_bias_per_head_alpha_no_cnn','no_cnn_mul_per_head_alpha','basic'], help="Do you want to add a new layer to your model?")
    parser.add_argument('--fasta_seq', type=str, default='/mmfs1/home/mdtahmid.islam/nrPDB-GO_2019.06.18_sequences.fasta',  help="sequence_path")
    parser.add_argument('--output_directory', type=str, default='/mmfs1/home/mdtahmid.islam/deepfri_bert/all_models',  help="model_saving_directory")
    parser.add_argument('--tokenizer_path', type=str,  help="tokenizer_path")
    parser.add_argument('--pretrained_model_name', type=str, default='Rostlab/prot_bert',  help="Pretrained Model Name")
    parser.add_argument('--drop_out_rate',type=float,default=1e-1)
    parser.add_argument('--clip_min',type=float,default=1.0)
    args = parser.parse_args()
    if args.ontology == 'ec':
        prot2annot, goterms, gonames, counts = load_EC_annot("/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_annot.tsv")
        args.directory = '/mmfs1/home/mdtahmid.islam/EC_Files/PDB-EC'
        args.fasta_seq = '/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_sequences.fasta'
        args.file_pattern_train = 'PDB_EC_train_*.tfrecords'
        args.file_pattern_valid = 'PDB_EC_valid_*.tfrecords'
        
    else:
        prot2annot, goterms, gonames, counts = load_GO_annot("/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv")
        args.directory = '/mmfs1/home/mdtahmid.islam/pdb_files'
        args.file_pattern_train = "PDB_GO_train_*.tfrecords"
        args.file_pattern_valid = "PDB_GO_valid_*.tfrecords"
    goterms = goterms[args.ontology]
    gonames = gonames[args.ontology]
    args.output_dim = len(goterms)
    print(f"Sample Count = {len(prot2annot)} \n Args: {args}")

    # computing weights for imbalanced selected lable classes
    
    class_sizes = counts[args.ontology]
    mean_class_size = np.mean(class_sizes)
    pos_weight = get_class_weights(class_sizes, np.mean(class_sizes),args.clip_min)


    #load token or use Prot-Bert's tokenizer
    if args.tokenizer_path is not None:
        tokenizer = load_pretrained_tokenizer(args.tokenizer_path)
    else:
        tokenizer = load_new_tokenizer()
    #load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #define data loader
    train_loader = get_data_loader(device,args.file_pattern_train,tokenizer,args.ontology,args.fasta_seq,args.directory,args.pad_len)
    valid_loader =  get_data_loader(device,args.file_pattern_valid,tokenizer,args.ontology,args.fasta_seq,args.directory,args.pad_len)

    #define encoder
    model = get_custom_prot_model_instance(args)
    model,gpu_count = wrap_the_model(model)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    m_path = f'{args.output_directory}/custom_trained_model/dr_01_{args.ontology}_lr_{args.lr}_{args.extra_layer}_{args.pad_len}_again_clipping_{int(args.clip_min)}_.pt'
    if args.extra_layer == 'basic':
        m_path = f'{args.output_directory}/{args.pretrained_model_name}_{args.ontology}_lr_{args.lr}_clip_{int(args.clip_min)}_.pt'
    print(m_path," before strating")

    start_training(
        optimizer,
        model,
        pos_weight.to(device),
        train_loader,
        valid_loader,
        args.epochs,
        device,
        m_path
    )
