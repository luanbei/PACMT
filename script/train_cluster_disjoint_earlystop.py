#!/usr/bin/env python3
"""Single-GPU DNABERT-2 multi-head training for the cluster-disjoint split.

Initializes only from a base DNABERT-2 model (never from the earlier
random-fragment PACMT checkpoint), selects the best checkpoint by dev loss,
and stops after a configurable number of non-improving dev evaluations.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import sklearn.metrics as sk_metrics
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import EarlyStoppingCallback, Trainer


RANKS = ("order", "family", "genus", "species")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base DNABERT-2 directory; must not be an old PACMT checkpoint."})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Directory containing train.csv, dev.csv, test.csv and label_maps.json."})


@dataclass
class PACMTTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=512)
    early_stopping_patience: int = field(default=5)
    early_stopping_threshold: float = field(default=0.0)
    write_per_sample_results: bool = field(default=True)


class MultiHeadDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        values = {rank: [] for rank in RANKS}
        seqs: List[str] = []
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                seqs.append(row["seq"].strip().upper())
                for rank in RANKS:
                    values[rank].append(int(row[f"{rank}_id"]))
        enc = tokenizer(seqs, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = {rank: torch.tensor(values[rank], dtype=torch.long) for rank in RANKS}
        logging.info("Loaded %d fragments: %s", len(seqs), path)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        item = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
        item.update({f"labels_{rank}": self.labels[rank][idx] for rank in RANKS})
        return item


class Collator:
    def __call__(self, features):
        return {key: torch.stack([x[key] for x in features]) for key in features[0]}


class MultiHeadModel(nn.Module):
    def __init__(self, backbone, nclasses: Dict[str, int]):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.heads = nn.ModuleDict({rank: nn.Linear(hidden, nclasses[rank]) for rank in RANKS})
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels_order=None, labels_family=None,
                labels_genus=None, labels_species=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls = sequence_output[:, 0, :]
        logits = {rank: self.heads[rank](cls) for rank in RANKS}
        labels = {"order": labels_order, "family": labels_family,
                  "genus": labels_genus, "species": labels_species}
        loss = None
        if labels_order is not None:
            loss = sum(self.loss_fct(logits[rank], labels[rank]) for rank in RANKS)
        return {"loss": loss, **{f"logits_{rank}": logits[rank] for rank in RANKS}}


def evaluate_test(model, dataset, batch_size: int, output_dir: str, write_rows: bool) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    true, pred = ({r: [] for r in RANKS}, {r: [] for r in RANKS})
    rows = {r: [] for r in RANKS}
    index = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator()):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            n = batch["input_ids"].size(0)
            for rank in RANKS:
                y = batch[f"labels_{rank}"].numpy()
                p = out[f"logits_{rank}"].detach().cpu().numpy().argmax(axis=1)
                true[rank].extend(y.tolist()); pred[rank].extend(p.tolist())
                if write_rows:
                    rows[rank].extend((index+i, int(y[i]), int(p[i]), int(y[i] == p[i])) for i in range(n))
            index += n
    metrics: Dict[str, Any] = {}
    for rank in RANKS:
        y, p = np.asarray(true[rank]), np.asarray(pred[rank])
        metrics[rank] = {
            "accuracy": float(sk_metrics.accuracy_score(y, p)),
            "precision_macro": float(sk_metrics.precision_score(y, p, average="macro", zero_division=0)),
            "recall_macro": float(sk_metrics.recall_score(y, p, average="macro", zero_division=0)),
            "f1_macro": float(sk_metrics.f1_score(y, p, average="macro", zero_division=0)),
            "precision_weighted": float(sk_metrics.precision_score(y, p, average="weighted", zero_division=0)),
            "recall_weighted": float(sk_metrics.recall_score(y, p, average="weighted", zero_division=0)),
            "f1_weighted": float(sk_metrics.f1_score(y, p, average="weighted", zero_division=0)),
            "mcc": float(sk_metrics.matthews_corrcoef(y, p)),
            "n_samples": int(len(y)),
            "n_observed_classes": int(len(set(y.tolist()))),
        }
        if write_rows:
            with open(os.path.join(output_dir, f"test_{rank}_results.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["index", "true_label", "pred_label", "correct"]); w.writerows(rows[rank])
    metrics["overall_mean"] = {k: float(np.mean([metrics[r][k] for r in RANKS]))
                               for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted", "mcc")}
    return metrics


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, PACMTTrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one visible GPU even if the launch command omits it
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This script is configured for one GPU.")
    if not os.path.isfile(os.path.join(model_args.model_name_or_path, "pytorch_model.bin")):
        raise FileNotFoundError("model_name_or_path must be the original DNABERT-2 model directory with pytorch_model.bin")
    if args.evaluation_strategy == "no" or args.save_strategy == "no":
        raise ValueError("Use --evaluation_strategy steps and --save_strategy steps for early stopping.")
    if not args.load_best_model_at_end:
        raise ValueError("Use --load_best_model_at_end True so the final test uses the best dev checkpoint.")
    if args.metric_for_best_model not in ("eval_loss", "loss") or args.greater_is_better is not False:
        raise ValueError("Use --metric_for_best_model eval_loss --greater_is_better False.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=args.model_max_length,
                                                           padding_side="right", use_fast=True, trust_remote_code=True)
    train = MultiHeadDataset(os.path.join(data_args.data_path, "train.csv"), tokenizer, args.model_max_length)
    dev = MultiHeadDataset(os.path.join(data_args.data_path, "dev.csv"), tokenizer, args.model_max_length)
    test = MultiHeadDataset(os.path.join(data_args.data_path, "test.csv"), tokenizer, args.model_max_length)
    with open(os.path.join(data_args.data_path, "label_maps.json"), encoding="utf-8") as f:
        maps = json.load(f)["maps"]
    nclasses = {rank: len(maps[rank]) for rank in RANKS}
    for rank in RANKS:
        observed_max = int(train.labels[rank].max())
        if observed_max + 1 != nclasses[rank]:
            raise ValueError(f"{rank}: label map/class count mismatch")
    backbone = transformers.AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = MultiHeadModel(backbone, nclasses)

    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=Collator(), tokenizer=tokenizer,
                      callbacks=[EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)])
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "head_config.json"), "w") as f:
        json.dump(nclasses, f, indent=2)
    for name in ("label_maps.json", "mapping.csv", "fragment_summary.tsv"):
        src = os.path.join(data_args.data_path, name)
        if os.path.isfile(src): shutil.copy2(src, os.path.join(args.output_dir, name))
    metrics = evaluate_test(model, test, args.per_device_eval_batch_size, args.output_dir, args.write_per_sample_results)
    for name in ("test_metrics_all_heads.json", "head_performance.json", "head_performance_summary.json", "FINAL_METRICS.json"):
        with open(os.path.join(args.output_dir, name), "w") as f: json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "head_performance.csv"), "w", newline="") as f:
        keys = list(metrics["overall_mean"]); w = csv.writer(f); w.writerow(["head"] + keys)
        for head in (*RANKS, "overall_mean"): w.writerow([head] + [metrics[head][k] for k in keys])
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump({"base_model": model_args.model_name_or_path, "data_path": data_args.data_path,
                   "best_model_checkpoint": trainer.state.best_model_checkpoint,
                   "best_metric": trainer.state.best_metric, "nclasses": nclasses,
                   "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}, f, indent=2)
    logging.info("Finished. Best dev checkpoint: %s; best eval loss: %s", trainer.state.best_model_checkpoint, trainer.state.best_metric)


if __name__ == "__main__":
    main()
