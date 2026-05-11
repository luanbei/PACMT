#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, csv, json, logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

import transformers
from transformers import Trainer
import sklearn.metrics as sk_metrics


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    init_weights_from: Optional[str] = field(default=None, metadata={"help": "从已有 pytorch_model.bin 加载权重（只加载模型，不恢复optimizer）"})


@dataclass
class DataArguments:
    data_path: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=512)
    max_eval_samples: int = field(default=50000, metadata={"help": "0=全量；>0=抽样数量（更快）"})


class BinaryDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int):
        seqs, labels = [], []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seqs.append(row["seq"].strip().upper())
                labels.append(int(row["label"]))
        enc = tokenizer(seqs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_labels = int(self.labels.max().item()) + 1

    def __len__(self): return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}


@dataclass
class Collator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([f[k] for f in features], dim=0) for k in features[0].keys()}


class Model(nn.Module):
    def __init__(self, backbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = out[0] if isinstance(out, tuple) else out.last_hidden_state
        cls = seq[:, 0, :]
        logits = self.classifier(cls)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = softmax_np(logits)
    scores = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]

    tn, fp, fn, tp = sk_metrics.confusion_matrix(labels, preds).ravel()
    out = {
        "accuracy": float(sk_metrics.accuracy_score(labels, preds)),
        "precision": float(sk_metrics.precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(sk_metrics.recall_score(labels, preds, average="binary", zero_division=0)),
        "f1": float(sk_metrics.f1_score(labels, preds, average="binary", zero_division=0)),
        "mcc": float(sk_metrics.matthews_corrcoef(labels, preds)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    try:
        out["auroc"] = float(sk_metrics.roc_auc_score(labels, scores))
    except Exception:
        out["auroc"] = None
    try:
        out["aupr"] = float(sk_metrics.average_precision_score(labels, scores))
    except Exception:
        out["aupr"] = None
    return out


def main():
    logging.basicConfig(level=logging.INFO)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.data_path is None:
        raise ValueError("必须提供 --data_path")

    os.makedirs(training_args.output_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    train_ds = BinaryDataset(os.path.join(data_args.data_path, "train.csv"), tokenizer, training_args.model_max_length)
    dev_ds   = BinaryDataset(os.path.join(data_args.data_path, "dev.csv"), tokenizer, training_args.model_max_length)
    test_ds  = BinaryDataset(os.path.join(data_args.data_path, "test.csv"), tokenizer, training_args.model_max_length)

    backbone = transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir, trust_remote_code=True
    )
    model = Model(backbone, num_labels=train_ds.num_labels)

    if model_args.init_weights_from:
        logging.info(f"Loading finetuned weights from {model_args.init_weights_from}")
        sd = torch.load(model_args.init_weights_from, map_location="cpu")
        model.load_state_dict(sd, strict=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=Collator(),
        compute_metrics=compute_metrics,
    )

    # 这里不 resume_from_checkpoint，避免 optimizer state 不兼容
    trainer.train()

    torch.save(model.state_dict(), os.path.join(training_args.output_dir, "pytorch_model.bin"))
    with open(os.path.join(training_args.output_dir, "head_config.json"), "w") as f:
        json.dump({"num_labels": train_ds.num_labels}, f, indent=2)
    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.max_eval_samples and training_args.max_eval_samples > 0:
        n = min(training_args.max_eval_samples, len(test_ds))
        test_eval = Subset(test_ds, list(range(n)))
        logging.info(f"Evaluate test subset: {n}/{len(test_ds)}")
    else:
        test_eval = test_ds
        logging.info(f"Evaluate FULL test: {len(test_ds)} (slow)")

    metrics = trainer.evaluate(eval_dataset=test_eval)
    with open(os.path.join(training_args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Done.")

if __name__ == "__main__":
    main()
