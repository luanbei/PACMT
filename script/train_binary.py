#!/usr/bin/env python3
"""Memory-safe 4-GPU DDP DNABERT-2 binary training for cluster-disjoint splits."""
from __future__ import print_function

import csv
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import sklearn.metrics as sk_metrics
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import EarlyStoppingCallback, Trainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Original DNABERT-2 base-model directory."})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Directory with two-column-or-more seq,label train.csv/dev.csv/test.csv."})


@dataclass
class BinaryTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=160)
    early_stopping_patience: int = field(default=3)
    early_stopping_threshold: float = field(default=0.0001)
    write_per_sample_results: bool = field(default=True)


class LazyBinaryDataset(Dataset):
    """Read only seq/label columns and tokenize at mini-batch time."""
    def __init__(self, path):
        self.seqs = []
        self.labels = []
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"seq", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError("{} missing {}".format(path, sorted(missing)))
            for line, row in enumerate(reader, start=2):
                sequence = row["seq"].strip().upper()
                label = row["label"].strip()
                if not sequence or label not in ("0", "1"):
                    raise ValueError("Invalid seq/label at {}:{}".format(path, line))
                self.seqs.append(sequence)
                self.labels.append(int(label))
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        logging.info("Loaded %d raw sequences from %s", len(self.seqs), path)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return {"seq": self.seqs[index], "labels": self.labels[index]}


class BatchTokenizerCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch = self.tokenizer([item["seq"] for item in features], padding="max_length",
                               truncation=True, max_length=self.max_length, return_tensors="pt")
        batch.pop("token_type_ids", None)
        batch["labels"] = torch.stack([item["labels"] for item in features])
        return batch


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def evaluate_binary(model, dataset, collator, batch_size, output_dir, write_rows):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    y_true, y_pred, y_prob, rows = [], [], [], []
    index = 0
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            truth = labels.numpy()
            y_true.extend(truth.tolist()); y_pred.extend(predictions.tolist()); y_prob.extend(probabilities.tolist())
            if write_rows:
                rows.extend((index + i, int(truth[i]), int(predictions[i]), float(probabilities[i]), int(truth[i] == predictions[i]))
                            for i in range(len(truth)))
            index += len(truth)
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    result = {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision_macro": float(sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(sk_metrics.matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(sk_metrics.roc_auc_score(y_true, y_prob)),
        "n_samples": int(len(y_true)),
    }
    if write_rows:
        with open(os.path.join(output_dir, "test_predictions.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "true_label", "pred_label", "virus_probability", "correct"])
            writer.writerows(rows)
    return result


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, BinaryTrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.evaluation_strategy == "no" or args.save_strategy == "no":
        raise ValueError("Early stopping requires --evaluation_strategy steps and --save_strategy steps")
    if not args.load_best_model_at_end or args.metric_for_best_model not in ("eval_loss", "loss") or args.greater_is_better is not False:
        raise ValueError("Use --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False")
    if args.save_steps != args.eval_steps:
        raise ValueError("--save_steps must equal --eval_steps when load_best_model_at_end is enabled")
    for split in ("train", "dev", "test"):
        if not os.path.isfile(os.path.join(data_args.data_path, split + ".csv")):
            raise FileNotFoundError("Missing {}.csv in {}".format(split, data_args.data_path))

    args.ddp_find_unused_parameters = False
    args.remove_unused_columns = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
        model_max_length=args.model_max_length, padding_side="right", use_fast=True, trust_remote_code=True)
    train = LazyBinaryDataset(os.path.join(data_args.data_path, "train.csv"))
    dev = LazyBinaryDataset(os.path.join(data_args.data_path, "dev.csv"))
    test = LazyBinaryDataset(os.path.join(data_args.data_path, "test.csv"))
    if set(train.labels.tolist()) != {0, 1}:
        raise ValueError("Train set must contain both binary labels")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=2, trust_remote_code=True)
    collator = BatchTokenizerCollator(tokenizer, args.model_max_length)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev, data_collator=collator,
                      tokenizer=tokenizer, callbacks=[EarlyStoppingCallback(args.early_stopping_patience,
                                                                             args.early_stopping_threshold)])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    barrier()
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        metrics = evaluate_binary(unwrapped, test, collator, args.per_device_eval_batch_size,
                                  args.output_dir, args.write_per_sample_results)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as handle:
            json.dump({"base_model": model_args.model_name_or_path, "data_path": data_args.data_path,
                       "best_model_checkpoint": trainer.state.best_model_checkpoint,
                       "best_dev_eval_loss": trainer.state.best_metric,
                       "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                       "model_max_length": args.model_max_length,
                       "early_stopping_patience": args.early_stopping_patience,
                       "early_stopping_threshold": args.early_stopping_threshold}, handle, indent=2)
        logging.info("Finished. Best checkpoint: %s; best dev loss: %s", trainer.state.best_model_checkpoint,
                     trainer.state.best_metric)
    barrier()


if __name__ == "__main__":
    main()
