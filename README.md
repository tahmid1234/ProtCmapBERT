# ProtCmapBERT

ProtCmapBERT is a framework that incorporates protein structural information into its attention mechanism for protein function prediction. It is a structure-aware model designed to leverage both sequence and structural data for improved accuracy.

---

## Installation

```bash
git clone https://github.com/tahmid1234/ProtCmapBERT.git
cd ProtCmapBERT
pip install -r requirements.txt
```

---

## Training

Run the following command to start training:

```bash
python trainer/trainer_for_custom_model.py \
    --ontology ec \
    --training_ds_directory 'your_training_dir_path' \
    --extra_layer cmap_bias \
    --lr 7e-6 \
    --epochs 200 \
    --drop_out_rate 0.01 \
    --clip_min 1
```

### Arguments
- `--ontology`: One of `[mf, bp, cc, ec]`. Each option will train a separate model.
- `--extra_layer`: One of `[cmap_bias, basic]`. 
  - `cmap_bias`: Trains a model with an attention layer influenced by the contact map.
  - `basic`: Fine-tunes a Prot-BERT model without considering structural information.
- Other hyperparameters: learning rate, epochs, dropout rate, and gradient clipping can be adjusted as needed.

Trained models will be stored in the `all_models` directory.

---

## Download ProtCmapBERT Models & Evaluation Dataset

### Download ProtCmapBERT models from Hugging Face Hub
```bash
python all_models/download_model_files.py
```

### Download evaluation dataset
```bash
python evaluation_ds/download_evaluation_ds.py
```

Training dataset can be provided upon request.

---

## Data Preprocessing

Convert PDB files into TFRecords format:

```bash
python pdb_tf_processing/tf_cration_with_clean_pdb.py
```

This will generate `.tfrecord` files required for model training and evaluation.

---

## Evaluation

Run the following command to evaluate a trained model:

```bash
python evaluation/evaluator.py \
    --ontology ec \
    --extra_layer cmap_bias \
    --test_file_path evaluation_ds/ec_test \
    --file_id test_evaluation \
    --m_path all_models/ProtCmapBERT/dr_01_ec_lr_7e-06_cmap_bias_per_head_alpha_clipping_1_.pt
```


