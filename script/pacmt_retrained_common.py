#!/usr/bin/env python3
"""Shared implementation for retrained PACMT inference scripts."""
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

RANKS = ("order", "family", "genus", "species")


def read_fasta(path):
    records, sid, pieces = [], None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    records.append((sid, "".join(pieces).upper()))
                sid, pieces = line[1:].split()[0], []
            else:
                pieces.append(line)
    if sid is not None:
        records.append((sid, "".join(pieces).upper()))
    return records


def load_inputs(input_fasta, input_csv, seq_col, id_col):
    if (input_fasta is None) == (input_csv is None):
        raise ValueError("Choose exactly one: --input_fasta or --input_csv")
    if input_fasta:
        recs = read_fasta(input_fasta)
        return [x[0] for x in recs], [x[1] for x in recs]
    df = pd.read_csv(input_csv)
    if seq_col not in df.columns:
        raise ValueError("CSV missing seq col: {}".format(seq_col))
    ids = df[id_col].astype(str).tolist() if id_col and id_col in df.columns else [str(i) for i in range(len(df))]
    return ids, df[seq_col].astype(str).str.upper().tolist()


def split_sequence(seq, seg_len=500, stride=None):
    seq = (seq or "").upper()
    stride = seg_len if stride is None else stride
    if seg_len <= 0 or stride <= 0:
        raise ValueError("seg_len and stride must be > 0")
    if len(seq) <= seg_len:
        return [seq]
    segments = [seq[start:start + seg_len] for start in range(0, len(seq) - seg_len + 1, stride)]
    return segments or [seq]


def get_device(value):
    if value == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(value)


class BinaryHeadModel(nn.Module):
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
        return self.classifier(hidden[:, 0, :])


def load_binary_model(backbone_dir, ckpt_dir, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(backbone_dir, trust_remote_code=True)
    model = BinaryHeadModel(backbone, num_labels=2)
    weights = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.isfile(weights):
        raise FileNotFoundError("Missing binary weights: " + weights)
    missing, unexpected = model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError("Binary checkpoint incompatible. missing={}, unexpected={}".format(missing[:12], unexpected[:12]))
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def pooled_pvirus_softmax(model, tokenizer, segments, max_length, batch_size, device, tau):
    if tau <= 0:
        raise ValueError("tau must be > 0")
    probs = []
    for i in range(0, len(segments), batch_size):
        enc = tokenizer(segments[i:i + batch_size], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        probs.append(torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=1)[:, 1].cpu())
    probs = torch.cat(probs)
    weights = torch.softmax(probs / float(tau), dim=0)
    return float((weights * probs).sum().item()), int(probs.numel())


class NewMultiHeadModel(nn.Module):
    def __init__(self, backbone, sizes):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({r: nn.Linear(backbone.config.hidden_size, sizes[r]) for r in RANKS})

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
        cls = hidden[:, 0, :]
        return {r: self.heads[r](cls) for r in RANKS}


class LegacyMultiHeadModel(nn.Module):
    def __init__(self, backbone, sizes):
        super().__init__()
        self.backbone = backbone
        for rank in RANKS:
            setattr(self, rank + "_head", nn.Linear(backbone.config.hidden_size, sizes[rank]))

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
        cls = hidden[:, 0, :]
        return tuple(getattr(self, r + "_head")(cls) for r in RANKS)


def _head_sizes(ckpt_dir):
    path = os.path.join(ckpt_dir, "head_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError("Missing hierarchy head_config.json: " + path)
    cfg = json.load(open(path, encoding="utf-8"))
    result = {}
    for rank in RANKS:
        value = cfg.get("num_" + rank, cfg.get(rank))
        if value is None and isinstance(cfg.get("label_sizes"), dict):
            value = cfg["label_sizes"].get(rank)
        if value is None:
            raise ValueError("head_config.json has no class count for " + rank)
        result[rank] = int(value)
    return result


def load_hierarchy_model(backbone_dir, ckpt_dir, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, trust_remote_code=True)
    path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.isfile(path):
        raise FileNotFoundError("Missing hierarchy weights: " + path)
    state = torch.load(path, map_location="cpu")
    backbone = AutoModel.from_pretrained(backbone_dir, trust_remote_code=True)
    new_format = any(str(k).startswith("heads.") for k in state)
    model = NewMultiHeadModel(backbone, _head_sizes(ckpt_dir)) if new_format else LegacyMultiHeadModel(backbone, _head_sizes(ckpt_dir))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("Hierarchy checkpoint incompatible. missing={}, unexpected={}".format(missing[:12], unexpected[:12]))
    model.to(device).eval()
    return model, tokenizer, new_format


def load_names(mapping_csv):
    df = pd.read_csv(mapping_csv)
    needed = {"rank", "label_id", "taxonomy_name"}
    if not needed.issubset(df.columns):
        raise ValueError("mapping_csv must contain " + ", ".join(sorted(needed)))
    return {rank: {int(x.label_id): str(x.taxonomy_name) for x in df[df["rank"] == rank].itertuples(index=False)} for rank in RANKS}


def load_paths(taxonomy_path_csv, device):
    cols = [r + "_id" for r in RANKS]
    df = pd.read_csv(taxonomy_path_csv)
    if any(c not in df.columns for c in cols):
        raise ValueError("taxonomy_path_csv must contain " + ", ".join(cols))
    paths = torch.tensor(df[cols].drop_duplicates().values, dtype=torch.long, device=device)
    if paths.numel() == 0:
        raise ValueError("No taxonomy paths loaded")
    return paths


@torch.inference_mode()
def pooled_hierarchy_probs(model, tokenizer, segments, max_length, batch_size, device, new_format, tau):
    if tau <= 0:
        raise ValueError("tau must be > 0")
    values = {rank: [] for rank in RANKS}
    for i in range(0, len(segments), batch_size):
        enc = tokenizer(segments[i:i + batch_size], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        output = model(enc["input_ids"], enc["attention_mask"])
        iterator = ((r, output[r]) for r in RANKS) if new_format else zip(RANKS, output)
        for rank, logits in iterator:
            values[rank].append(torch.softmax(logits, dim=1).cpu())
    result = {}
    for rank in RANKS:
        probs = torch.cat(values[rank])
        weights = torch.softmax(probs.max(dim=1).values / float(tau), dim=0).unsqueeze(1)
        result[rank] = (weights * probs).sum(dim=0).to(device)
    return result


def decode_valid_path(probs, paths, names):
    score = torch.zeros(paths.shape[0], device=paths.device)
    for i, rank in enumerate(RANKS):
        score += torch.log(probs[rank].clamp_min(1e-12))[paths[:, i]]
    best = int(score.argmax().item()); chosen = paths[best]
    row = {"joint_score": float(torch.exp(score[best]).item()), "log_joint_score": float(score[best].item())}
    for i, rank in enumerate(RANKS):
        label = int(chosen[i].item())
        row[rank + "_id"] = label
        row[rank + "_name"] = names[rank].get(label, "")
        row[rank + "_conf"] = float(probs[rank][label].item())
    return row
