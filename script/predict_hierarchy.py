#!/usr/bin/env python3
"""Hierarchy PACMT predictor; command-line options match the original script."""
import argparse
import pandas as pd
from pacmt_retrained_common import (decode_valid_path, get_device, load_hierarchy_model, load_inputs, load_names,
                                    load_paths, pooled_hierarchy_probs, split_sequence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_dir", required=True); ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--mapping_csv", required=True); ap.add_argument("--taxonomy_path_csv", required=True)
    ap.add_argument("--input_fasta", default=None); ap.add_argument("--input_csv", default=None)
    ap.add_argument("--seq_col", default="seq"); ap.add_argument("--id_col", default=None)
    ap.add_argument("--seg_len", type=int, default=500); ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=512); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--tau", type=float, default=0.2); ap.add_argument("--out_csv", required=True)
    args = ap.parse_args(); device = get_device(args.device)
    ids, seqs = load_inputs(args.input_fasta, args.input_csv, args.seq_col, args.id_col)
    model, tok, new_format = load_hierarchy_model(args.backbone_dir, args.ckpt_dir, device)
    names, paths = load_names(args.mapping_csv), load_paths(args.taxonomy_path_csv, device); rows = []
    for sid, seq in zip(ids, seqs):
        segs = split_sequence(seq, args.seg_len, args.stride)
        row = {"id": sid, "seq_len": len(seq), "n_segments": len(segs)}
        row.update(decode_valid_path(pooled_hierarchy_probs(model, tok, segs, args.max_length, args.batch_size, device, new_format, args.tau), paths, names)); rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False); print("[OK] saved:", args.out_csv)


if __name__ == "__main__": main()
