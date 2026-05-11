#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DNABERT-2 多头分类（order/family/genus/species）训练 + 轻量评估脚本（无 AUROC/AUPR）

变化（相对旧版）：
- 测试集评估不再计算 AUROC/AUPR（避免大测试集 + 大类别数时耗时/爆内存）
- 其余指标保留：accuracy、precision/recall/f1（macro/weighted）、MCC
- 仍输出：
  - test_metrics_all_heads.json（每个头 + overall_mean 指标）
  - head_performance.json（同结构，单独文件）
  - head_performance_summary.json（同结构，固定文件名便于引用）
  - 可选 head_performance.csv
  - 可选 test_*_results.csv（index,true_label,pred_label,correct）
- 支持两种加载方式：
  1) --resume_from_checkpoint CKPT_DIR：真正“续训”（恢复 optimizer/scheduler/global_step 等），等价于 Trainer.train(resume_from_checkpoint=...)
  2) --init_from_checkpoint CKPT_DIR：只加载 CKPT_DIR/pytorch_model.bin 权重（不恢复训练状态），用于“从某个权重初始化重新训”

CSV 格式（必须包含列名）：
seq,order_id,family_id,genus_id,species_id
"""

import os
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import Trainer

from peft import LoraConfig, get_peft_model

import sklearn.metrics as sk_metrics

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------
# 1) 参数
# -------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "预训练模型路径或名称"})
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="query,value")

    # 只初始化权重（不恢复训练状态）
    init_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "只加载该目录下的 pytorch_model.bin 权重；不恢复optimizer/step（不是续训）"}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "包含 train.csv/dev.csv/test.csv 的目录"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)

    # 真正续训（恢复 optimizer/scheduler/step）
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "继续训练用的 checkpoint 目录（如 .../checkpoint-10000）"}
    )

    # 轻量评估输出文件
    head_performance_json: str = field(default="head_performance.json")
    head_performance_summary_json: str = field(default="head_performance_summary.json")
    head_performance_csv: str = field(default="head_performance.csv")
    write_head_performance_csv: bool = field(default=True)

    # 是否输出每样本结果（true/pred/correct）
    write_per_sample_results: bool = field(default=True)


# -------------------
# 2) 数据集
# -------------------

class MultiHeadDataset(Dataset):
    """CSV: seq,order_id,family_id,genus_id,species_id"""

    def __init__(self, csv_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super().__init__()
        seqs: List[str] = []
        order_labels: List[int] = []
        family_labels: List[int] = []
        genus_labels: List[int] = []
        species_labels: List[int] = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seqs.append(row["seq"].strip().upper())
                order_labels.append(int(row["order_id"]))
                family_labels.append(int(row["family_id"]))
                genus_labels.append(int(row["genus_id"]))
                species_labels.append(int(row["species_id"]))

        enc = tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels_order = torch.tensor(order_labels, dtype=torch.long)
        self.labels_family = torch.tensor(family_labels, dtype=torch.long)
        self.labels_genus = torch.tensor(genus_labels, dtype=torch.long)
        self.labels_species = torch.tensor(species_labels, dtype=torch.long)

        self.num_order = int(self.labels_order.max().item()) + 1
        self.num_family = int(self.labels_family.max().item()) + 1
        self.num_genus = int(self.labels_genus.max().item()) + 1
        self.num_species = int(self.labels_species.max().item()) + 1

        logging.info(
            f"Loaded {len(self.input_ids)} samples from {csv_path} "
            f"(order={self.num_order}, family={self.num_family}, genus={self.num_genus}, species={self.num_species})"
        )

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels_order": self.labels_order[idx],
            "labels_family": self.labels_family[idx],
            "labels_genus": self.labels_genus[idx],
            "labels_species": self.labels_species[idx],
        }


@dataclass
class DataCollatorForMultiHead:
    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            batch[k] = torch.stack([f[k] for f in features], dim=0)
        return batch


# -------------------
# 3) 模型
# -------------------

class MultiHeadModel(nn.Module):
    def __init__(self, backbone: transformers.PreTrainedModel,
                 num_order: int, num_family: int, num_genus: int, num_species: int):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size

        self.order_head = nn.Linear(hidden, num_order)
        self.family_head = nn.Linear(hidden, num_family)
        self.genus_head = nn.Linear(hidden, num_genus)
        self.species_head = nn.Linear(hidden, num_species)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask,
                labels_order=None, labels_family=None, labels_genus=None, labels_species=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        cls = seq_out[:, 0, :]

        lo = self.order_head(cls)
        lf = self.family_head(cls)
        lg = self.genus_head(cls)
        ls = self.species_head(cls)

        loss = None
        if labels_order is not None:
            loss = (
                self.loss_fct(lo, labels_order)
                + self.loss_fct(lf, labels_family)
                + self.loss_fct(lg, labels_genus)
                + self.loss_fct(ls, labels_species)
            )

        return {
            "loss": loss,
            "logits_order": lo,
            "logits_family": lf,
            "logits_genus": lg,
            "logits_species": ls,
        }


# -------------------
# 4) 评估（轻量：不落盘 logits；不算 AUC/AUPR）
# -------------------

def eval_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = sk_metrics.accuracy_score(y_true, y_pred)
    prec_macro = sk_metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = sk_metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_w = sk_metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = sk_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)

    mcc = sk_metrics.matthews_corrcoef(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "mcc": float(mcc),
    }


def write_head_performance(all_metrics: Dict[str, Any], out_dir: str,
                           json_name: str, csv_name: str, write_csv: bool):
    # JSON
    with open(os.path.join(out_dir, json_name), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # CSV（可选）
    if write_csv:
        keys = list(all_metrics["overall_mean"].keys())
        with open(os.path.join(out_dir, csv_name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["head"] + keys)
            for head in ["order", "family", "genus", "species", "overall_mean"]:
                w.writerow([head] + [all_metrics[head].get(k, None) for k in keys])


def write_results_csv(path: str, rows: List[List[int]]):
    # rows: [index,true,pred,correct]
    with open(path, "w") as f:
        f.write("index,true_label,pred_label,correct\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")


def light_eval_on_test(model: nn.Module, test_dataset: MultiHeadDataset,
                       out_dir: str, batch_size: int, write_per_sample: bool) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorForMultiHead())
    it = loader if tqdm is None else tqdm(loader, desc="Light eval on test")

    true_o, pred_o = [], []
    true_f, pred_f = [], []
    true_g, pred_g = [], []
    true_s, pred_s = [], []

    rows_o, rows_f, rows_g, rows_s = [], [], [], []
    idx = 0

    with torch.no_grad():
        for batch in it:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            lo_true = batch["labels_order"].cpu().numpy()
            lf_true = batch["labels_family"].cpu().numpy()
            lg_true = batch["labels_genus"].cpu().numpy()
            ls_true = batch["labels_species"].cpu().numpy()

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            lo = out["logits_order"].cpu().numpy()
            lf = out["logits_family"].cpu().numpy()
            lg = out["logits_genus"].cpu().numpy()
            ls = out["logits_species"].cpu().numpy()

            po = lo.argmax(axis=1)
            pf = lf.argmax(axis=1)
            pg = lg.argmax(axis=1)
            ps = ls.argmax(axis=1)

            true_o.extend(lo_true.tolist()); pred_o.extend(po.tolist())
            true_f.extend(lf_true.tolist()); pred_f.extend(pf.tolist())
            true_g.extend(lg_true.tolist()); pred_g.extend(pg.tolist())
            true_s.extend(ls_true.tolist()); pred_s.extend(ps.tolist())

            if write_per_sample:
                for j in range(len(po)):
                    rows_o.append([idx, int(lo_true[j]), int(po[j]), int(lo_true[j] == po[j])])
                    rows_f.append([idx, int(lf_true[j]), int(pf[j]), int(lf_true[j] == pf[j])])
                    rows_g.append([idx, int(lg_true[j]), int(pg[j]), int(lg_true[j] == pg[j])])
                    rows_s.append([idx, int(ls_true[j]), int(ps[j]), int(ls_true[j] == ps[j])])
                    idx += 1
            else:
                idx += len(po)

    metrics_order = eval_from_pred(np.array(true_o), np.array(pred_o))
    metrics_family = eval_from_pred(np.array(true_f), np.array(pred_f))
    metrics_genus = eval_from_pred(np.array(true_g), np.array(pred_g))
    metrics_species = eval_from_pred(np.array(true_s), np.array(pred_s))

    overall = {}
    for k in metrics_order.keys():
        vals = [metrics_order[k], metrics_family[k], metrics_genus[k], metrics_species[k]]
        overall[k] = float(np.mean(vals))

    all_metrics = {
        "order": metrics_order,
        "family": metrics_family,
        "genus": metrics_genus,
        "species": metrics_species,
        "overall_mean": overall,
    }

    # 写每样本结果（可选）
    if write_per_sample:
        write_results_csv(os.path.join(out_dir, "test_order_results.csv"), rows_o)
        write_results_csv(os.path.join(out_dir, "test_family_results.csv"), rows_f)
        write_results_csv(os.path.join(out_dir, "test_genus_results.csv"), rows_g)
        write_results_csv(os.path.join(out_dir, "test_species_results.csv"), rows_s)

    # 写 metrics（主文件）
    with open(os.path.join(out_dir, "test_metrics_all_heads.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


# -------------------
# 5) 主函数
# -------------------

def main():
    logging.basicConfig(level=logging.INFO)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    train_csv = os.path.join(data_args.data_path, "train.csv")
    dev_csv = os.path.join(data_args.data_path, "dev.csv")
    test_csv = os.path.join(data_args.data_path, "test.csv")

    train_dataset = MultiHeadDataset(train_csv, tokenizer, training_args.model_max_length)
    dev_dataset = MultiHeadDataset(dev_csv, tokenizer, training_args.model_max_length)
    test_dataset = MultiHeadDataset(test_csv, tokenizer, training_args.model_max_length)

    backbone = transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    model = MultiHeadModel(
        backbone,
        num_order=train_dataset.num_order,
        num_family=train_dataset.num_family,
        num_genus=train_dataset.num_genus,
        num_species=train_dataset.num_species,
    )

    # 只加载权重（不恢复 optimizer/step）
    if model_args.init_from_checkpoint:
        ckpt_bin = os.path.join(model_args.init_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(ckpt_bin):
            raise FileNotFoundError(f"[init_from_checkpoint] 找不到 {ckpt_bin}")
        state = torch.load(ckpt_bin, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info("[INIT] Loaded weights from %s", ckpt_bin)
        if missing:
            logging.warning("[INIT] Missing keys (up to 20): %s", missing[:20])
        if unexpected:
            logging.warning("[INIT] Unexpected keys (up to 20): %s", unexpected[:20])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForMultiHead(),
        tokenizer=tokenizer,
    )

    # 续训：恢复训练状态（optimizer/scheduler/global_step 等）
    if training_args.resume_from_checkpoint:
        if not os.path.isdir(training_args.resume_from_checkpoint):
            raise FileNotFoundError(f"[resume_from_checkpoint] 目录不存在: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # 保存模型（多头 state_dict）+ tokenizer + head_config
    os.makedirs(training_args.output_dir, exist_ok=True)
    trainer.save_state()

    torch.save(model.state_dict(), os.path.join(training_args.output_dir, "pytorch_model.bin"))
    with open(os.path.join(training_args.output_dir, "head_config.json"), "w") as f:
        json.dump({
            "num_order": train_dataset.num_order,
            "num_family": train_dataset.num_family,
            "num_genus": train_dataset.num_genus,
            "num_species": train_dataset.num_species,
        }, f, indent=2)
    tokenizer.save_pretrained(training_args.output_dir)

    # 轻量测试评估（不保存 logits/embeddings；不算 AUC/AUPR）
    metrics = light_eval_on_test(
        model=model,
        test_dataset=test_dataset,
        out_dir=training_args.output_dir,
        batch_size=training_args.per_device_eval_batch_size or 32,
        write_per_sample=training_args.write_per_sample_results,
    )

    # 额外输出 head_performance.json / csv
    write_head_performance(
        all_metrics=metrics,
        out_dir=training_args.output_dir,
        json_name=training_args.head_performance_json,
        csv_name=training_args.head_performance_csv,
        write_csv=training_args.write_head_performance_csv,
    )

    # 固定再写一份汇总 JSON（内容同 metrics）
    with open(os.path.join(training_args.output_dir, training_args.head_performance_summary_json), "w") as f:
        json.dump(metrics, f, indent=2)

    # 运行摘要（可选）
    with open(os.path.join(training_args.output_dir, "FINAL_METRICS.json"), "w") as f:
        json.dump({
            "run_name": training_args.run_name,
            "output_dir": training_args.output_dir,
            "model_name_or_path": model_args.model_name_or_path,
            "model_max_length": training_args.model_max_length,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "resume_from_checkpoint": training_args.resume_from_checkpoint,
            "write_per_sample_results": training_args.write_per_sample_results,
            "metrics": metrics,
        }, f, indent=2)

    logging.info("Done. Light evaluation finished. Output -> %s", training_args.output_dir)


if __name__ == "__main__":
    main()
