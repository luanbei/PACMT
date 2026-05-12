# PACMT Usage Guide

PACMT is a pretrained sequence model-based framework for viral identification and hierarchical taxonomic classification of metagenomic sequences. It uses a two-stage workflow:

1. **Binary viral screening**: predicts whether an input sequence is viral or non-viral.
2. **Hierarchical viral classification**: assigns predicted viral sequences to **order**, **family**, **genus** and **species** levels with taxonomy-consistent path decoding.

PACMT supports both **FASTA** and **CSV** input files and outputs viral confidence scores together with hierarchical taxonomic predictions.

---

## 1. Repository structure

The recommended GitHub repository structure is:

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
    └── PACMT_usage.md
```

The model weights are not recommended to be stored directly in GitHub. They should be provided separately, for example through Hugging Face, Zenodo or an institutional repository.

---

## 2. Installation

We recommend creating a new conda environment with Python 3.8.

```bash
conda create -n pacmt python=3.8 -y
conda activate pacmt
pip install -r requirements.txt
```

A minimal environment test can be performed using:

```bash
python -c "import torch, transformers, pandas, sklearn; print('PACMT environment OK')"
```

If CUDA or PyTorch compatibility issues occur, reinstall PyTorch according to the CUDA version available on the target machine.

---

## 3. Required files

To run the complete PACMT workflow, the following files or directories are required:

```text
DNABERT-2 backbone directory
binary model directory
hierarchical model directory
label_taxonomy_mapping.csv
taxonomy_paths.csv
input FASTA or input CSV file
```

A recommended local model directory is:

```text
models/
├── backbone/
├── binary_model/
│   ├── pytorch_model.bin
│   ├── head_config.json
│   └── tokenizer files
├── hierarchy_model/
│   ├── pytorch_model.bin
│   ├── head_config.json
│   └── tokenizer files
└── taxonomy/
    ├── label_taxonomy_mapping.csv
    └── taxonomy_paths.csv
```

### 3.1 label_taxonomy_mapping.csv

This file maps internal label IDs to taxonomy names. It must contain at least the following columns:

```text
rank,label_id,taxonomy_name
```

Example:

```text
rank,label_id,taxonomy_name
order,0,Articulavirales
family,0,Orthomyxoviridae
genus,0,Alphainfluenzavirus
species,0,Influenza A virus
```

### 3.2 taxonomy_paths.csv

This file defines valid taxonomy paths and is used for taxonomy-consistent decoding. It must contain at least the following columns:

```text
order_id,family_id,genus_id,species_id
```

Each row represents one valid order-family-genus-species path. During hierarchical decoding, PACMT selects the valid path with the highest joint score.

---

## 4. Input formats

PACMT supports both FASTA and CSV input files.

### 4.1 FASTA input

```text
>seq1
ATCGCCGAATACGAATTC...
>seq2
GCTAGCTAGCTAGCTA...
```

### 4.2 CSV input

The CSV file must contain a sequence column. An ID column is recommended.

```text
id,seq
seq1,ATCGCCGAATACGAATTC...
seq2,GCTAGCTAGCTAGCTA...
```

Default column names used by the scripts:

```text
id column: id
sequence column: seq
```

If different column names are used, specify them with `--id_col` and `--seq_col`.

---

## 5. Complete two-stage prediction workflow

The recommended PACMT prediction mode is the complete two-stage workflow. It first performs binary viral screening and then applies hierarchical classification only to sequences predicted as viral.

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

For example:

```bash
python scripts/predict_binary_hierarchy.py \
  --backbone_dir models/backbone \
  --binary_ckpt_dir models/binary_model \
  --hierarchy_ckpt_dir models/hierarchy_model \
  --mapping_csv models/taxonomy/label_taxonomy_mapping.csv \
  --taxonomy_path_csv models/taxonomy/taxonomy_paths.csv \
  --input_fasta examples/example.fasta \
  --seg_len 500 \
  --stride 250 \
  --max_length 512 \
  --batch_size 32 \
  --device cuda \
  --virus_threshold 0.5 \
  --tau 0.2 \
  --out_csv pacmt_predictions.csv
```

### Important parameters

| Parameter | Description |
|---|---|
| `--seg_len` | Fragment length used during prediction. Recommended: `500`. |
| `--stride` | Sliding-window stride. Recommended: `250`. If omitted, the scripts use non-overlapping windows. |
| `--max_length` | Maximum tokenizer input length. Recommended: `512`. |
| `--batch_size` | Batch size for model inference. Recommended: `32`, adjust according to GPU memory. |
| `--virus_threshold` | Threshold for binary viral prediction. Default: `0.5`. |
| `--tau` | Temperature for softmax-weighted segment pooling. Default: `0.2`. |
| `--device` | Use `cuda` for GPU inference or `cpu` for CPU inference. |

---

## 6. Output format

The complete workflow outputs a CSV file with the following major columns:

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

Column descriptions:

| Column | Description |
|---|---|
| `id` | Sequence identifier. |
| `seq_len` | Length of the input sequence. |
| `n_segments` | Number of fragments generated from the input sequence. |
| `is_virus` | Binary prediction result. `1` indicates viral; `0` indicates non-viral. |
| `virus_confidence` | Sequence-level viral confidence score after segment-level aggregation. |
| `order_name`, `family_name`, `genus_name`, `species_name` | Predicted taxonomic names. |
| `order_conf`, `family_conf`, `genus_conf`, `species_conf` | Rank-specific confidence scores along the selected valid path. |
| `joint_score` | Joint probability score of the selected valid taxonomy path. |
| `log_joint_score` | Log-transformed joint score of the selected valid taxonomy path. |

If `is_virus=0`, the hierarchical taxonomic fields are left empty.

---

## 7. Binary viral screening only

Use this mode when only virus-versus-non-virus prediction is required.

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

For FASTA input:

```bash
python scripts/predict_binary.py \
  --backbone_dir models/backbone \
  --ckpt_dir models/binary_model \
  --input_fasta examples/example.fasta \
  --seg_len 500 \
  --stride 250 \
  --max_length 512 \
  --batch_size 32 \
  --device cuda \
  --tau 0.2 \
  --threshold 0.5 \
  --out_csv binary_predictions.csv
```

The output includes:

```text
id,seq_len,n_segments,is_virus,virus_confidence
```

---

## 8. Hierarchical classification only

Use this mode when input sequences are already considered viral and only taxonomic annotation is needed.

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

For FASTA input:

```bash
python scripts/predict_hierarchy.py \
  --backbone_dir models/backbone \
  --ckpt_dir models/hierarchy_model \
  --mapping_csv models/taxonomy/label_taxonomy_mapping.csv \
  --taxonomy_path_csv models/taxonomy/taxonomy_paths.csv \
  --input_fasta examples/example.fasta \
  --seg_len 500 \
  --stride 250 \
  --max_length 512 \
  --batch_size 32 \
  --device cuda \
  --tau 0.2 \
  --out_csv hierarchy_predictions.csv
```

This mode directly outputs order-, family-, genus- and species-level predictions for all input sequences.

---

## 9. Binary model training

The binary training script reads `train.csv`, `dev.csv` and `test.csv` from the specified data directory.

Required files:

```text
binary_dataset/
├── train.csv
├── dev.csv
└── test.csv
```

Required CSV columns:

```text
seq,label
```

Example:

```text
seq,label
ATCGCCGAATACGAATTC...,1
GCTAGCTAGCTAGCTA...,0
```

`label=1` indicates viral sequences, and `label=0` indicates non-viral sequences.

Training command:

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

Main output files:

```text
pytorch_model.bin
head_config.json
tokenizer files
test_metrics.json
```

---

## 10. Hierarchical model training

The hierarchical training script trains four classification heads corresponding to order, family, genus and species.

Required files:

```text
hierarchy_dataset/
├── train.csv
├── dev.csv
└── test.csv
```

Required CSV columns:

```text
seq,order_id,family_id,genus_id,species_id
```

Example:

```text
seq,order_id,family_id,genus_id,species_id
ATCGCCGAATACGAATTC...,0,0,0,0
GCTAGCTAGCTAGCTA...,1,3,12,105
```

Training command:

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

Main output files:

```text
pytorch_model.bin
head_config.json
test_metrics_all_heads.json
head_performance.json
head_performance_summary.json
head_performance.csv
test_order_results.csv
test_family_results.csv
test_genus_results.csv
test_species_results.csv
FINAL_METRICS.json
tokenizer files
```

The `head_config.json` file records the number of classes for each classification head, for example:

```json
{
  "num_order": 84,
  "num_family": 222,
  "num_genus": 1651,
  "num_species": 7566
}
```

These values are required when loading the hierarchical classifier for prediction.

---

## 11. Notes and limitations

- PACMT is intended for research use in viral sequence screening and hierarchical taxonomic annotation.
- Species-level prediction is generally more difficult than higher-rank prediction, especially for short sequences, divergent viruses or underrepresented taxa.
- `taxonomy_paths.csv` is required for taxonomy-consistent decoding and prevents biologically invalid order-family-genus-species combinations.
- For long input sequences, PACMT uses sliding-window segmentation followed by segment-level probability aggregation.
- Model weights and large datasets should not be uploaded directly to GitHub. Use Hugging Face, Zenodo or another model/data repository instead.
- For reproducibility, keep the same tokenizer files, `head_config.json`, `label_taxonomy_mapping.csv` and `taxonomy_paths.csv` together with the released model weights.

---

## 12. Citation

If you use PACMT, please cite:

```text
Luan B, Li P, et al. PACMT: a pretrained language model-based framework for viral identification and hierarchical taxonomic classification of metagenomic data.
```
