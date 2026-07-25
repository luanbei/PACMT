#!/usr/bin/env python3
"""DDP launcher-compatible DNABERT-2 multi-head training.

Launch with torchrun.  The companion train_cluster_disjoint_earlystop.py must
be in the same directory: it supplies the identical model, dataset, loss and
metrics used for the single-GPU experiment.
"""
from __future__ import annotations

import json
import logging
import os
import shutil

import torch
import torch.distributed as dist
import transformers
from transformers import EarlyStoppingCallback, Trainer

from train_cluster_disjoint_earlystop import (
    ModelArguments, DataArguments, PACMTTrainingArguments,
    MultiHeadDataset, MultiHeadModel, Collator, RANKS, evaluate_test,
)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, PACMTTrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # DNABERT-2 exposes pooler parameters, while this classifier deliberately
    # uses the first last-hidden-state token.  The pooler therefore receives no
    # gradient. DDP must detect such parameters rather than assuming every
    # parameter is used in every backward pass.
    if args.ddp_find_unused_parameters is False:
        logging.warning("Overriding --ddp_find_unused_parameters False: DNABERT-2 pooler is intentionally unused.")
    args.ddp_find_unused_parameters = True

    # Do not set CUDA_VISIBLE_DEVICES here: torchrun must see all requested GPUs.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if not os.path.isfile(os.path.join(model_args.model_name_or_path, "pytorch_model.bin")):
        raise FileNotFoundError("model_name_or_path must be the original DNABERT-2 model directory with pytorch_model.bin")
    if args.evaluation_strategy == "no" or args.save_strategy == "no":
        raise ValueError("Use --evaluation_strategy steps and --save_strategy steps for early stopping.")
    if not args.load_best_model_at_end:
        raise ValueError("Use --load_best_model_at_end True so the final test uses the best dev checkpoint.")
    if args.metric_for_best_model not in ("eval_loss", "loss") or args.greater_is_better is not False:
        raise ValueError("Use --metric_for_best_model eval_loss --greater_is_better False.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=args.model_max_length,
        padding_side="right", use_fast=True, trust_remote_code=True,
    )
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
    trainer = Trainer(
        model=model, args=args, train_dataset=train, eval_dataset=dev,
        data_collator=Collator(), tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    barrier()

    # Only rank 0 writes model artefacts and performs the custom full-test pass.
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        torch.save(unwrapped.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "head_config.json"), "w") as f:
            json.dump(nclasses, f, indent=2)
        for name in ("label_maps.json", "mapping.csv", "fragment_summary.tsv"):
            src = os.path.join(data_args.data_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.output_dir, name))
        metrics = evaluate_test(unwrapped, test, args.per_device_eval_batch_size, args.output_dir, args.write_per_sample_results)
        for name in ("test_metrics_all_heads.json", "head_performance.json", "head_performance_summary.json", "FINAL_METRICS.json"):
            with open(os.path.join(args.output_dir, name), "w") as f:
                json.dump(metrics, f, indent=2)
        with open(os.path.join(args.output_dir, "head_performance.csv"), "w", newline="") as f:
            import csv
            keys = list(metrics["overall_mean"]); w = csv.writer(f); w.writerow(["head"] + keys)
            for head in (*RANKS, "overall_mean"):
                w.writerow([head] + [metrics[head][k] for k in keys])
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
            json.dump({"base_model": model_args.model_name_or_path, "data_path": data_args.data_path,
                       "best_model_checkpoint": trainer.state.best_model_checkpoint,
                       "best_metric": trainer.state.best_metric, "nclasses": nclasses,
                       "world_size": world_size, "per_device_train_batch_size": args.per_device_train_batch_size}, f, indent=2)
        logging.info("Finished. Best dev checkpoint: %s; best eval loss: %s", trainer.state.best_model_checkpoint, trainer.state.best_metric)
    barrier()


if __name__ == "__main__":
    main()
