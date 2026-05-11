#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binary + Hierarchy prediction (DNABERT-2) with **conditional-chain taxonomy decoding**.

Classifier logic:
1) Binary head estimates P(virus|x) on each segment, then aggregates across segments with:
      C = (Σ c_i^(p+1)) / (Σ c_i^p)
   where c_i = P_virus(segment_i). If C >= virus_threshold -> is_virus = 1 else 0.

2) If is_virus==1, the hierarchy model outputs per-rank marginal probabilities:
      P(order|x), P(family|x), P(genus|x), P(species|x)
   (each pooled across segments with the same p-weighted pooling above, applied **per class**).

   Then we enforce hierarchical consistency by selecting the best **VALID** taxonomy path (o,f,g,s) from a
   provided path set 𝒫:
      (o*,f*,g*,s*) = argmax_{(o,f,g,s)∈𝒫}  log P(o|x)+log P(f|x)+log P(g|x)+log P(s|x)

   This avoids inconsistent outputs like genus not belonging to the predicted family.

Required for conditional-chain decoding:
- Provide --taxonomy_path_csv with columns: order_id,family_id,genus_id,species_id
  OR (fallback) a mapping_csv that includes those columns on species rows (rank=='species').

Outputs:
- id, seq_len, n_segments
- is_virus, virus_confidence
- (if virus) order/family/genus/species id + name + conf along selected path
- joint_score, log_joint_score
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# -----------------------
# Common IO
# -----------------------
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
    """Default stride=None -> stride=seg_len (non-overlap)."""
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


# -----------------------
# Binary model
# -----------------------
class BinaryHeadModel(nn.Module):
    """DNABERT-2 backbone + binary head"""
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


def load_binary_model(backbone_dir, binary_ckpt_dir, device):
    # tokenizer: prefer ckpt_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(binary_ckpt_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, trust_remote_code=True)

    num_labels = load_num_labels(binary_ckpt_dir)
    if num_labels != 2:
        print("[WARN] binary num_labels={} (expected 2). Script assumes label 1 = virus.".format(num_labels))

    backbone = AutoModel.from_pretrained(backbone_dir, trust_remote_code=True)
    model = BinaryHeadModel(backbone, num_labels=num_labels)

    w = os.path.join(binary_ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(w):
        raise FileNotFoundError("Missing binary weights: {}".format(w))

    state = torch.load(w, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] binary missing keys (first 10): {}".format(missing[:10]))
    if unexpected:
        print("[WARN] binary unexpected keys (first 10): {}".format(unexpected[:10]))

    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def predict_binary_softmax_pool_pvirus(model, tokenizer, segments, max_length, batch_size, device, tau):
    """Predict P(virus) per segment and softmax-weight pool:
         w_i = softmax(pvirus_i / tau),  P = Σ w_i * pvirus_i
       tau > 0 controls sharpness (smaller -> closer to max; larger -> closer to mean).
    """
    if len(segments) == 0:
        raise ValueError("No segments to predict (binary).")

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
        probs = torch.softmax(logits, dim=1)  # [B,2]
        pvirus = probs[:, 1]                  # label 1 = virus
        p_list.append(pvirus.detach().cpu())

    pvirus_all = torch.cat(p_list, dim=0)  # [nseg]
    if tau <= 0:
        raise ValueError("tau must be > 0 for softmax pooling")
    w = torch.softmax(pvirus_all / float(tau), dim=0)
    pooled = float((w * pvirus_all).sum().item())
    return pooled


# -----------------------
# Hierarchy model
# -----------------------
class MultiHeadModel(nn.Module):
    """DNABERT-2 backbone + 4 heads (order/family/genus/species)"""
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


def load_head_cfg(hierarchy_ckpt_dir):
    p = os.path.join(hierarchy_ckpt_dir, "head_config.json")
    if not os.path.exists(p):
        raise FileNotFoundError("Missing hierarchy head_config.json: {}".format(p))
    cfg = json.load(open(p, "r", encoding="utf-8"))
    return int(cfg["num_order"]), int(cfg["num_family"]), int(cfg["num_genus"]), int(cfg["num_species"])


def load_id2name_maps(mapping_csv):
    """mapping_csv columns: rank,label_id,taxonomy_name"""
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


def load_taxonomy_paths(taxonomy_path_csv, mapping_csv):
    cols = ["order_id", "family_id", "genus_id", "species_id"]

    if taxonomy_path_csv:
        df = pd.read_csv(taxonomy_path_csv)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError("taxonomy_path_csv missing columns {}. Need {}. Got {}".format(missing, cols, list(df.columns)))
        out = df[cols].copy()
    else:
        df = pd.read_csv(mapping_csv)
        if not set(cols).issubset(df.columns):
            raise ValueError(
                "For conditional-chain decoding you must provide --taxonomy_path_csv, "
                "OR your mapping_csv must include columns order_id,family_id,genus_id,species_id on species rows."
            )
        sub = df[df["rank"] == "species"].copy()
        out = sub[cols].copy()

    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(int)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No taxonomy paths loaded (empty).")
    return out


def load_hierarchy_model(backbone_dir, hierarchy_ckpt_dir, device):
    # tokenizer: prefer ckpt_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(hierarchy_ckpt_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, trust_remote_code=True)

    n_o, n_f, n_g, n_s = load_head_cfg(hierarchy_ckpt_dir)

    backbone = AutoModel.from_pretrained(backbone_dir, trust_remote_code=True)
    model = MultiHeadModel(backbone, n_o, n_f, n_g, n_s)

    w = os.path.join(hierarchy_ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(w):
        raise FileNotFoundError("Missing hierarchy weights: {}".format(w))

    state = torch.load(w, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] hierarchy missing keys (first 10): {}".format(missing[:10]))
    if unexpected:
        print("[WARN] hierarchy unexpected keys (first 10): {}".format(unexpected[:10]))

    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def predict_hierarchy_softmax_pool(model, tokenizer, segments, device, max_length, batch_size, tau):
    """Predict per-rank probabilities and softmax-weight pool across segments.

    For each rank r∈{order,family,genus,species}:
      - compute per-segment prob vector p_i^r
      - define segment confidence score s_i^r = max_k p_i^r[k]
      - weights w_i^r = softmax(s_i^r / tau)
      - pooled P^r = Σ_i w_i^r * p_i^r   (vector)

    tau > 0 controls sharpness.
    """
    if len(segments) == 0:
        raise ValueError("No segments to predict (hierarchy).")
    if tau <= 0:
        raise ValueError("tau must be > 0 for softmax pooling")

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

    po = torch.cat(po_list, dim=0)  # [nseg, n_order]
    pf = torch.cat(pf_list, dim=0)  # [nseg, n_family]
    pg = torch.cat(pg_list, dim=0)  # [nseg, n_genus]
    ps = torch.cat(ps_list, dim=0)  # [nseg, n_species]

    # weights per rank based on max prob per segment
    w_o = torch.softmax(po.max(dim=1).values / float(tau), dim=0).unsqueeze(1)  # [nseg,1]
    w_f = torch.softmax(pf.max(dim=1).values / float(tau), dim=0).unsqueeze(1)
    w_g = torch.softmax(pg.max(dim=1).values / float(tau), dim=0).unsqueeze(1)
    w_s = torch.softmax(ps.max(dim=1).values / float(tau), dim=0).unsqueeze(1)

    p_order = (w_o * po).sum(dim=0)  # [n_order]
    p_family = (w_f * pf).sum(dim=0)
    p_genus = (w_g * pg).sum(dim=0)
    p_species = (w_s * ps).sum(dim=0)

    return p_order.to(device), p_family.to(device), p_genus.to(device), p_species.to(device)


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
    ap.add_argument("--binary_ckpt_dir", required=True)
    ap.add_argument("--hierarchy_ckpt_dir", required=True)
    ap.add_argument("--mapping_csv", required=True)

    ap.add_argument("--taxonomy_path_csv", default=None)

    ap.add_argument("--input_fasta", default=None)
    ap.add_argument("--input_csv", default=None)
    ap.add_argument("--seq_col", default="seq")
    ap.add_argument("--id_col", default=None)

    ap.add_argument("--seg_len", type=int, default=500)
    ap.add_argument("--stride", type=int, default=None)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--virus_threshold", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.2, help="Softmax pooling temperature; smaller -> more like max, larger -> more like mean")

    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    if (args.input_fasta is None) == (args.input_csv is None):
        raise ValueError("Choose exactly one: --input_fasta or --input_csv")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    binary_model, binary_tok = load_binary_model(args.backbone_dir, args.binary_ckpt_dir, device)
    hier_model, hier_tok = load_hierarchy_model(args.backbone_dir, args.hierarchy_ckpt_dir, device)
    id2name = load_id2name_maps(args.mapping_csv)
    paths_df = load_taxonomy_paths(args.taxonomy_path_csv, args.mapping_csv)

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

        pvirus = predict_binary_softmax_pool_pvirus(
            model=binary_model,
            tokenizer=binary_tok,
            segments=segs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            tau=args.tau,
        )
        is_virus = 1 if pvirus >= args.virus_threshold else 0

        row = {
            "id": sid,
            "seq_len": int(len(seq)),
            "n_segments": int(len(segs)),
            "is_virus": int(is_virus),
            "virus_confidence": float(pvirus),
        }

        if is_virus == 1:
            p_order, p_family, p_genus, p_species = predict_hierarchy_softmax_pool(
                model=hier_model,
                tokenizer=hier_tok,
                segments=segs,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                tau=args.tau,
            )
            best = decode_best_path(p_order, p_family, p_genus, p_species, paths_df)

            oid, fid, gid, sid2 = best["order_id"], best["family_id"], best["genus_id"], best["species_id"]
            row.update({
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
            })
        else:
            row.update({
                "order_id": "",
                "order_name": "",
                "order_conf": "",

                "family_id": "",
                "family_name": "",
                "family_conf": "",

                "genus_id": "",
                "genus_name": "",
                "genus_conf": "",

                "species_id": "",
                "species_name": "",
                "species_conf": "",

                "joint_score": "",
                "log_joint_score": "",
            })

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("[OK] saved: {}".format(args.out_csv))


if __name__ == "__main__":
    main()
