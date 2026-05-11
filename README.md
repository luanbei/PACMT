# PACMT

PACMT is a pretrained sequence model-based framework for viral identification and hierarchical taxonomic classification of metagenomic sequences.

## Overview

PACMT uses a two-stage serial workflow. First, a binary classifier screens input sequences as virus or non-virus. Sequences predicted as viral are then passed to a hierarchical classifier that predicts order, family, genus and species labels. A taxonomy-consistent decoding strategy is used to select a valid order-family-genus-species path.

## Repository structure

```text
PACMT/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── predict_binary_hierarchy.py
│   ├── predict_binary.py
│   ├── predict_hierarchy.py
│   ├── train_binary.py
│   └── train_hierarchy.py
├── examples/
│   ├── example.fasta
│   └── example.csv
└── docs/
    └── PACMT_usage.docx
```

## Installation

We recommend using a conda environment with Python 3.8.

```bash
conda create -n pacmt python=3.8 -y
conda activate pacmt
pip install -r requirements.txt
```

Test the installation:

```bash
python -c "import torch, transformers, pandas, sklearn; print('PACMT environment OK')"
```

## Model files

The complete PACMT prediction workflow requires the following model and taxonomy files:

```text
DNABERT-2 backbone directory
binary model directory
hierarchical model directory
label_taxonomy_mapping.csv
taxonomy_paths.csv
```

The model weights and taxonomy mapping files should be downloaded separately, for example from Hugging Face, and placed under:

```text
models/
├── backbone/
├── binary_model/
├── hierarchy_model/
└── taxonomy/
    ├── label_taxonomy_mapping.csv
    └── taxonomy_paths.csv
```

## Input format

PACMT supports FASTA or CSV input.

FASTA format:

```text
>seq1
ATCGCCGAATACGAATTC...
>seq2
GCTAGCTAGCTAGCTA...
```

CSV format:

```text
id,seq
seq1,ATCGCCGAATACGAATTC...
seq2,GCTAGCTAGCTAGCTA...
```

## Complete prediction workflow

The complete two-stage workflow first performs binary viral screening and then applies hierarchical classification to predicted viral sequences.

```bash
python scripts/predict_binary_hierarchy.py \
  --backbone_dir models/backbone \
  --binary_ckpt_dir models/binary_model \
  --hierarchy_ckpt_dir models/hierarchy_model \
  --mapping_csv models/taxonomy/label_taxonomy_mapping.csv \
  --taxonomy_path_csv models/taxonomy/taxonomy_paths.csv \
  --input_csv examples/example.csv \
  --seq_col seq \
  --id_col id \
  --seg_len 500 \
  --stride 250 \
  --max_length 512 \
  --batch_size 32 \
  --device cuda \
  --virus_threshold 0.5 \
  --tau 0.2 \
  --out_csv pacmt_predictions.csv
```

For FASTA input, replace the CSV input arguments with:

```bash
--input_fasta examples/example.fasta
```

## Binary viral screening only

```bash
python scripts/predict_binary.py \
  --backbone_dir models/backbone \
  --ckpt_dir models/binary_model \
  --input_csv examples/example.csv \
  --seq_col seq \
  --id_col id \
  --seg_len 500 \
  --stride 250 \
  --max_length 512 \
  --batch_size 32 \
  --device cuda \
  --tau 0.2 \
  --threshold 0.5 \
  --out_csv binary_predictions.csv
```

## Hierarchical classification only

This mode directly classifies all input sequences into order, family, genus and species levels without binary viral screening.

```bash
python scripts/predict_hierarchy.py \
  --backbone_dir models/backbone \
  --ckpt_dir models/hierarchy_model \
  --mapping_csv models/taxonomy/label_taxonomy_mapping.csv \
  --taxonomy_path_csv models/taxonomy/taxonomy_paths.csv \
  --input_csv examples/example.csv \
  --seq_col seq \
  --id_col id \
  --seg_len 500 \
  --stride 250 \
  --max_length 512 \
  --batch_size 32 \
  --device cuda \
  --tau 0.2 \
  --out_csv hierarchy_predictions.csv
```

## Output

The complete prediction workflow outputs a CSV file containing:

```text
id
seq_len
n_segments
is_virus
virus_confidence
order_id, order_name, order_conf
family_id, family_name, family_conf
genus_id, genus_name, genus_conf
species_id, species_name, species_conf
joint_score
log_joint_score
```

`is_virus=1` indicates that the sequence is predicted as viral. For non-viral predictions, hierarchical taxonomic fields are left empty.

## Training

Binary model training:

```bash
python scripts/train_binary.py \
  --model_name_or_path /path/to/DNABERT-2 \
  --data_path /path/to/binary_dataset \
  --output_dir /path/to/output_binary_model \
  --model_max_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --max_eval_samples 0
```

Hierarchical model training:

```bash
python scripts/train_hierarchy.py \
  --model_name_or_path /path/to/DNABERT-2 \
  --data_path /path/to/hierarchy_dataset \
  --output_dir /path/to/output_hierarchy_model \
  --model_max_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 100 \
  --write_per_sample_results True
```

## Citation

If you use PACMT, please cite:

Luan B, Li P, et al. PACMT: a pretrained language model-based framework for viral identification and hierarchical taxonomic classification of metagenomic data.
