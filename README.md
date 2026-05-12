PACMT
PACMT is a pretrained sequence model-based framework for viral identification and hierarchical taxonomic classification of metagenomic sequences.
PACMT uses a two-stage serial workflow. First, a binary classifier screens input sequences as virus or non-virus. Sequences predicted as viral are then passed to a hierarchical classifier that predicts order, family, genus and species labels. A taxonomy-consistent decoding strategy is used to select a valid order-family-genus-species path.
Model and code availability
Source code and example files: https://github.com/luanbei/PACMT
Trained model weights and taxonomy files: https://huggingface.co/luanbei/PACMT
The GitHub repository contains the source code, example input files and usage documentation. The Hugging Face repository contains the trained model files, DNABERT-2 backbone files and taxonomy resources required for prediction.
Repository structure
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
Installation
We recommend using a conda environment with Python 3.8.
```bash
git clone https://github.com/luanbei/PACMT.git
cd PACMT

conda create -n pacmt python=3.8 -y
conda activate pacmt
pip install -r requirements.txt
```
Test the installation:
```bash
python -c "import torch, transformers, pandas, sklearn; print('PACMT environment OK')"
```
Download model files
The trained PACMT model files are available from Hugging Face:
```bash
pip install -U huggingface_hub
hf download luanbei/PACMT --local-dir models
```
After downloading, the local model directory should look like:
```text
models/
├── backbone/
├── binary_model/
├── hierarchy_model/
└── taxonomy/
    ├── label_taxonomy_mapping.csv
    └── taxonomy_paths.csv
```
Required files
The complete PACMT workflow requires:
```text
models/backbone/
models/binary_model/
models/hierarchy_model/
models/taxonomy/label_taxonomy_mapping.csv
models/taxonomy/taxonomy_paths.csv
```
The `label_taxonomy_mapping.csv` file maps internal label IDs to taxonomy names and should contain at least:
```text
rank,label_id,taxonomy_name
```
The `taxonomy_paths.csv` file defines valid hierarchical taxonomy paths and should contain at least:
```text
order_id,family_id,genus_id,species_id
```
Input format
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
If different column names are used in the CSV file, specify them with `--id_col` and `--seq_col`.
Complete two-stage prediction workflow
The complete workflow first performs binary viral screening and then applies hierarchical classification only to sequences predicted as viral.
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
Example:
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
Binary viral screening only
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
Hierarchical classification only
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
Output
The complete workflow outputs a CSV file containing:
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
`is_virus=1` indicates that the input sequence is predicted as viral. If `is_virus=0`, the hierarchical taxonomic fields are left empty.
Training
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
Notes and limitations
PACMT is intended for research use in viral sequence screening and hierarchical taxonomic annotation.
Species-level prediction is generally more difficult than higher-rank prediction, especially for short, divergent or underrepresented viral sequences.
`taxonomy_paths.csv` is required for taxonomy-consistent decoding and prevents biologically invalid order-family-genus-species combinations.
For long input sequences, PACMT uses sliding-window segmentation followed by segment-level probability aggregation.
Model weights and large datasets are hosted on Hugging Face rather than directly in this GitHub repository.
Citation
If you use PACMT, please cite:
```text
Luan B, Li P, et al. PACMT: a pretrained language model-based framework for viral identification and hierarchical taxonomic classification of metagenomic data.
```
