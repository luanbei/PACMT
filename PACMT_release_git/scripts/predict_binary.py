#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""predict_binary_softmax_pool.py

Binary prediction with softmax-weighted segment pooling.

Per-segment:
  p_i = P(virus | segment_i)

Pooling:
  w_i = softmax(p_i / tau)
  P = Σ w_i * p_i

Decision:
  is_virus = 1 if P >= threshold else 0
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def read_fasta(path):
    records = []
    sid = None
    chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    records.append((sid, "".join(chunks).upper()))
                sid = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if sid is not None:
            records.append((sid, "".join(chunks).upper()))
    return records


def split_sequence(seq, seg_len=500, stride=None):
    seq = (seq or "").upper()
    stride = seg_len if stride is None else stride
    if seg_len <= 0 or stride <= 0:
        raise ValueError("seg_len and stride must be > 0")
    if len(seq) <= seg_len:
        return [seq]
    segs = []
    for start in range(0, len(seq) - seg_len + 1, stride):
        segs.append(seq[start:start + seg_len])
    return segs


class BinaryHeadModel(nn.Module):
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls = last_hidden[:, 0, :]
        return self.classifier(cls)


def load_num_labels(ckpt_dir):
    p = os.path.join(ckpt_dir, "head_config.json")
    if not os.path.exists(p):
        return 2
    try:
        cfg = json.load(open(p, "r", encoding="utf-8"))
        for k in ["num_labels", "n_labels", "num_class", "num_classes"]:
            if k in cfg:
                return int(cfg[k])
    except Exception:
        pass
    return 2


def load_model(backbone_dir, ckpt_dir, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, trust_remote_code=True)

    num_labels = load_num_labels(ckpt_dir)
    backbone = AutoModel.from_pretrained(backbone_dir, trust_remote_code=True)
    model = BinaryHeadModel(backbone, num_labels=num_labels)

    w = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(w):
        raise FileNotFoundError("Missing weights: {}".format(w))
    state = torch.load(w, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def pooled_pvirus_softmax(model, tokenizer, segments, max_length, batch_size, device, tau):
    if len(segments) == 0:
        raise ValueError("No segments to predict.")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    p_list = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pvirus = probs[:, 1]
        p_list.append(pvirus.detach().cpu())

    pvirus_all = torch.cat(p_list, dim=0)
    w = torch.softmax(pvirus_all / float(tau), dim=0)
    return float((w * pvirus_all).sum().item()), int(pvirus_all.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)

    ap.add_argument("--input_fasta", default=None)
    ap.add_argument("--input_csv", default=None)
    ap.add_argument("--seq_col", default="seq")
    ap.add_argument("--id_col", default=None)

    ap.add_argument("--seg_len", type=int, default=500)
    ap.add_argument("--stride", type=int, default=None)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--tau", type=float, default=0.2,
                    help="Softmax pooling temperature; smaller -> more like max, larger -> more like mean")
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    if (args.input_fasta is None) == (args.input_csv is None):
        raise ValueError("Choose exactly one: --input_fasta or --input_csv")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, tok = load_model(args.backbone_dir, args.ckpt_dir, device)

    if args.input_fasta:
        recs = read_fasta(args.input_fasta)
        ids = [x[0] for x in recs]
        seqs = [x[1] for x in recs]
    else:
        df = pd.read_csv(args.input_csv)
        if args.seq_col not in df.columns:
            raise ValueError("CSV missing seq col: {}".format(args.seq_col))
        seqs = df[args.seq_col].astype(str).str.upper().tolist()
        if args.id_col and args.id_col in df.columns:
            ids = df[args.id_col].astype(str).tolist()
        else:
            ids = [str(i) for i in range(len(df))]

    rows = []
    for sid, seq in zip(ids, seqs):
        segs = split_sequence(seq, seg_len=args.seg_len, stride=args.stride)
        pvirus, nseg = pooled_pvirus_softmax(model, tok, segs, args.max_length, args.batch_size, device, args.tau)
        is_virus = 1 if pvirus >= args.threshold else 0
        rows.append({
            "id": sid,
            "seq_len": int(len(seq)),
            "n_segments": int(nseg),
            "is_virus": int(is_virus),
            "virus_confidence": float(pvirus),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("[OK] saved:", args.out_csv)


if __name__ == "__main__":
    main()
