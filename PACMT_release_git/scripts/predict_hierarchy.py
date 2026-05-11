#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""predict_hierarchy_softmax_chain.py

Hierarchy prediction with:
- Softmax-weighted segment pooling (per-rank)
- Conditional-probability chain decoding + valid taxonomy path constraint

Pooling:
For each rank r:
  p_i^r = softmax(logits_i^r)
  s_i^r = max_k p_i^r[k]
  w_i^r = softmax(s_i^r / tau)
  P^r = Σ_i w_i^r * p_i^r

Decoding:
  (o*,f*,g*,s*) = argmax_{(o,f,g,s)∈𝒫} log P(o|x)+log P(f|x)+log P(g|x)+log P(s|x)

Inputs:
- FASTA or CSV with sequences
- taxonomy_paths.csv (order_id,family_id,genus_id,species_id)
- mapping_csv (rank,label_id,taxonomy_name)

Output CSV columns:
id, seq_len, n_segments,
order_id,order_name,order_conf,
family_id,family_name,family_conf,
genus_id,genus_name,genus_conf,
species_id,species_name,species_conf,
joint_score,log_joint_score
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


class MultiHeadModel(nn.Module):
    def __init__(self, backbone, num_order, num_family, num_genus, num_species):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.order_head = nn.Linear(hidden, num_order)
        self.family_head = nn.Linear(hidden, num_family)
        self.genus_head = nn.Linear(hidden, num_genus)
        self.species_head = nn.Linear(hidden, num_species)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls = last_hidden[:, 0, :]
        return (
            self.order_head(cls),
            self.family_head(cls),
            self.genus_head(cls),
            self.species_head(cls),
        )


def load_head_cfg(ckpt_dir):
    p = os.path.join(ckpt_dir, "head_config.json")
    if not os.path.exists(p):
        raise FileNotFoundError("Missing head_config.json: {}".format(p))
    cfg = json.load(open(p, "r", encoding="utf-8"))
    return int(cfg["num_order"]), int(cfg["num_family"]), int(cfg["num_genus"]), int(cfg["num_species"])


def load_model(backbone_dir, ckpt_dir, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, trust_remote_code=True)

    n_o, n_f, n_g, n_s = load_head_cfg(ckpt_dir)
    backbone = AutoModel.from_pretrained(backbone_dir, trust_remote_code=True)
    model = MultiHeadModel(backbone, n_o, n_f, n_g, n_s)

    w = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(w):
        raise FileNotFoundError("Missing weights: {}".format(w))

    state = torch.load(w, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tokenizer


def load_id2name_maps(mapping_csv):
    df = pd.read_csv(mapping_csv)
    required = set(["rank", "label_id", "taxonomy_name"])
    if not required.issubset(df.columns):
        raise ValueError("mapping_csv must contain {}, got columns={}".format(required, list(df.columns)))

    maps = {}
    for rank in ["order", "family", "genus", "species"]:
        sub = df[df["rank"] == rank]
        maps[rank] = {int(r.label_id): str(r.taxonomy_name) for r in sub.itertuples(index=False)}
    return maps


def _id2name(rank, tax_id, id2name):
    return id2name.get(rank, {}).get(int(tax_id), "")


def load_taxonomy_paths(taxonomy_path_csv):
    cols = ["order_id", "family_id", "genus_id", "species_id"]
    df = pd.read_csv(taxonomy_path_csv)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("taxonomy_path_csv missing columns {}. Need {}. Got {}".format(missing, cols, list(df.columns)))
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(int)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No taxonomy paths loaded (empty).")
    return out


@torch.inference_mode()
def predict_probs_softmax_pool(model, tokenizer, segments, device, max_length, batch_size, tau):
    if len(segments) == 0:
        raise ValueError("No segments to predict.")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    po_list, pf_list, pg_list, ps_list = [], [], [], []
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

        lo, lf, lg, ls = model(input_ids=input_ids, attention_mask=attention_mask)

        po_list.append(torch.softmax(lo, dim=1).detach().cpu())
        pf_list.append(torch.softmax(lf, dim=1).detach().cpu())
        pg_list.append(torch.softmax(lg, dim=1).detach().cpu())
        ps_list.append(torch.softmax(ls, dim=1).detach().cpu())

    po = torch.cat(po_list, dim=0)
    pf = torch.cat(pf_list, dim=0)
    pg = torch.cat(pg_list, dim=0)
    ps = torch.cat(ps_list, dim=0)

    w_o = torch.softmax(po.max(dim=1).values / float(tau), dim=0).unsqueeze(1)
    w_f = torch.softmax(pf.max(dim=1).values / float(tau), dim=0).unsqueeze(1)
    w_g = torch.softmax(pg.max(dim=1).values / float(tau), dim=0).unsqueeze(1)
    w_s = torch.softmax(ps.max(dim=1).values / float(tau), dim=0).unsqueeze(1)

    p_order = (w_o * po).sum(dim=0).to(device)
    p_family = (w_f * pf).sum(dim=0).to(device)
    p_genus = (w_g * pg).sum(dim=0).to(device)
    p_species = (w_s * ps).sum(dim=0).to(device)
    return p_order, p_family, p_genus, p_species


def decode_best_path(p_order, p_family, p_genus, p_species, paths_df):
    eps = 1e-12
    log_po = (p_order.clamp_min(eps)).log().detach().cpu()
    log_pf = (p_family.clamp_min(eps)).log().detach().cpu()
    log_pg = (p_genus.clamp_min(eps)).log().detach().cpu()
    log_ps = (p_species.clamp_min(eps)).log().detach().cpu()

    o_ids = torch.tensor(paths_df["order_id"].values, dtype=torch.long)
    f_ids = torch.tensor(paths_df["family_id"].values, dtype=torch.long)
    g_ids = torch.tensor(paths_df["genus_id"].values, dtype=torch.long)
    s_ids = torch.tensor(paths_df["species_id"].values, dtype=torch.long)

    log_scores = log_po[o_ids] + log_pf[f_ids] + log_pg[g_ids] + log_ps[s_ids]
    best_idx = int(torch.argmax(log_scores).item())

    best = {
        "order_id": int(o_ids[best_idx].item()),
        "family_id": int(f_ids[best_idx].item()),
        "genus_id": int(g_ids[best_idx].item()),
        "species_id": int(s_ids[best_idx].item()),
        "log_joint_score": float(log_scores[best_idx].item()),
    }
    best["joint_score"] = float(torch.exp(log_scores[best_idx]).item())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--mapping_csv", required=True)
    ap.add_argument("--taxonomy_path_csv", required=True)

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
    id2name = load_id2name_maps(args.mapping_csv)
    paths_df = load_taxonomy_paths(args.taxonomy_path_csv)

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

        p_order, p_family, p_genus, p_species = predict_probs_softmax_pool(
            model=model,
            tokenizer=tok,
            segments=segs,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            tau=args.tau,
        )

        best = decode_best_path(p_order, p_family, p_genus, p_species, paths_df)

        oid, fid, gid, sid2 = best["order_id"], best["family_id"], best["genus_id"], best["species_id"]

        row = {
            "id": sid,
            "seq_len": int(len(seq)),
            "n_segments": int(len(segs)),

            "order_id": oid,
            "order_name": _id2name("order", oid, id2name),
            "order_conf": float(p_order[oid].item()),

            "family_id": fid,
            "family_name": _id2name("family", fid, id2name),
            "family_conf": float(p_family[fid].item()),

            "genus_id": gid,
            "genus_name": _id2name("genus", gid, id2name),
            "genus_conf": float(p_genus[gid].item()),

            "species_id": sid2,
            "species_name": _id2name("species", sid2, id2name),
            "species_conf": float(p_species[sid2].item()),

            "joint_score": best["joint_score"],
            "log_joint_score": best["log_joint_score"],
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("[OK] saved:", args.out_csv)


if __name__ == "__main__":
    main()
