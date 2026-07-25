#!/usr/bin/env python3
"""Joint PACMT predictor; command-line options match predict_bina_hiera_o.py."""
import argparse
import pandas as pd
from pacmt_retrained_common import (decode_valid_path, get_device, load_binary_model, load_hierarchy_model, load_inputs,
                                    load_names, load_paths, pooled_hierarchy_probs, pooled_pvirus_softmax, split_sequence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_dir", required=True); ap.add_argument("--binary_ckpt_dir", required=True); ap.add_argument("--hierarchy_ckpt_dir", required=True)
    ap.add_argument("--mapping_csv", required=True); ap.add_argument("--taxonomy_path_csv", default=None)
    ap.add_argument("--input_fasta", default=None); ap.add_argument("--input_csv", default=None)
    ap.add_argument("--seq_col", default="seq"); ap.add_argument("--id_col", default=None)
    ap.add_argument("--seg_len", type=int, default=500); ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=512); ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--device", default="cuda")
    ap.add_argument("--virus_threshold", type=float, default=0.5); ap.add_argument("--tau", type=float, default=0.2); ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    if not args.taxonomy_path_csv: raise ValueError("New hierarchy model requires --taxonomy_path_csv for legal-path decoding")
    device = get_device(args.device); ids, seqs = load_inputs(args.input_fasta, args.input_csv, args.seq_col, args.id_col)
    binary, btok = load_binary_model(args.backbone_dir, args.binary_ckpt_dir, device)
    hierarchy, htok, new_format = load_hierarchy_model(args.backbone_dir, args.hierarchy_ckpt_dir, device)
    names, paths = load_names(args.mapping_csv), load_paths(args.taxonomy_path_csv, device); rows = []
    for sid, seq in zip(ids, seqs):
        segs = split_sequence(seq, args.seg_len, args.stride); pvirus, nseg = pooled_pvirus_softmax(binary, btok, segs, args.max_length, args.batch_size, device, args.tau)
        row = {"id": sid, "seq_len": len(seq), "n_segments": nseg, "is_virus": int(pvirus >= args.virus_threshold), "virus_confidence": pvirus}
        if row["is_virus"]:
            probs = pooled_hierarchy_probs(hierarchy, htok, segs, args.max_length, args.batch_size, device, new_format, args.tau); row.update(decode_valid_path(probs, paths, names))
        else:
            for r in ("order", "family", "genus", "species"):
                row.update({r + "_id": "", r + "_name": "", r + "_conf": ""})
            row.update({"joint_score": "", "log_joint_score": ""})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False); print("[OK] saved:", args.out_csv)


if __name__ == "__main__": main()
