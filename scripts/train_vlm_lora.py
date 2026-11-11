"""LoRA SFT of a small VLM on InspectEdge reports (Colab T4/L4/A100).

usage: python scripts/train_vlm_lora.py --config configs/train_vlm.yaml [--smoke] [--resume]
--smoke: overfit 8 examples for 30 steps and assert loss drops (plan 8 "one-batch overfit").
Validation selection metric is computed by notebook 04 with scripts/eval_vlm.py, not loss alone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback,
                          TrainingArguments)

from inspectedge.paths import ROOT
from inspectedge.vlm.sft_data import Collator, SFTDataset, build_epoch_plan


class NanGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs and not math.isfinite(logs["loss"]):
            raise RuntimeError(f"non-finite loss at step {state.global_step}: {logs['loss']}")


class EpochSetter(TrainerCallback):
    def __init__(self, ds):
        self.ds = ds

    def on_epoch_begin(self, args, state, control, **kw):
        self.ds.epoch = int(state.epoch or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_vlm.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override: key=value")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    torch.manual_seed(cfg["seed"])
    man = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    if cfg.get("categories"):
        man = man[man.category.isin(cfg["categories"])]
    train_recs = man[man.vlm_split == "train"].to_dict("records")
    items = build_epoch_plan(train_recs, cfg["normal_ratio"], cfg["mismatch_frac"], cfg["seed"])
    if a.smoke:
        items = items[:8]
    ds = SFTDataset(items, cfg["longest_edge"], 0.0 if a.smoke else cfg["aug_prob"], cfg["seed"])

    proc = AutoProcessor.from_pretrained(cfg["model_id"])
    proc.image_processor.size = {"longest_edge": cfg["longest_edge"]}
    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(cfg["model_id"], dtype=dtype).to("cuda")

    if cfg.get("freeze_vision", True):
        for n, p in model.named_parameters():
            if "vision" in n:
                p.requires_grad = False
    if cfg["method"] == "lora":
        lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                          target_modules=cfg["lora_targets"], init_lora_weights="gaussian")
        model = get_peft_model(model, lcfg)
        # keep LoRA weights in fp32 for stable fp16 mixed precision
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.print_trainable_parameters()

    out = ROOT / cfg["output_dir"] / ("smoke" if a.smoke else cfg["run_name"])
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=1 if a.smoke else cfg["epochs"],
        max_steps=30 if a.smoke else -1,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=1 if a.smoke else cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=5,
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=4,
        fp16=fp16,
        bf16=not fp16,
        gradient_checkpointing=cfg.get("grad_ckpt", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.get("workers", 2),
        remove_unused_columns=False,
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(proc),
                      callbacks=[NanGuard(), EpochSetter(ds)])
    t0 = time.time()
    resume = a.resume and any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(out / "final"))
    proc.save_pretrained(str(out / "final"))

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    meta = {
        "config": cfg, "n_items": len(items), "train_seconds": time.time() - t0, "losses": losses,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": platform.python_version(),
        "pip_freeze": subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.splitlines(),
        "manifest_split_hashes": json.loads((ROOT / "data/manifests/split_hashes.json").read_text()),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if a.smoke:
        first, last = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
        print(f"SMOKE loss {first:.3f} -> {last:.3f}")
        assert last < 0.5 * first, "overfit smoke test failed: loss did not halve on 8 examples"
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
