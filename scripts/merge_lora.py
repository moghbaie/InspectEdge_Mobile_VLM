"""Merge a LoRA adapter into its base model and save a standalone checkpoint (bf16).

    python scripts/merge_lora.py --model google/gemma-3n-E2B-it --adapter checkpoints/<run>/final --out checkpoints/<run>/merged

The merged folder is what gets quantized: evaluate it with
    python scripts/eval_vlm.py --model checkpoints/<run>/merged --load-4bit --split val --per-cell 0 --tag <run>_merged_nf4
(4-bit on top of a separate adapter is a different model from the one you would ship).
Skips the work if --out already holds a complete merged checkpoint (use --force to redo).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from inspectedge.paths import ROOT


def resolve(p: str) -> Path:
    return Path(p) if p.startswith("/") else ROOT / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id the adapter was trained on")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    adapter, out = resolve(a.adapter), resolve(a.out)
    marker = out / "merge_meta.json"
    if marker.exists() and not a.force:
        print(f"already merged: {out} (pass --force to redo)")
        return

    t0 = time.perf_counter()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoProcessor.from_pretrained(str(adapter)).save_pretrained(str(out))
    n_params = sum(p.numel() for p in model.parameters())
    # Written last: its presence means the checkpoint is complete (a disconnect mid-save leaves no marker).
    marker.write_text(json.dumps({"base_model": a.model, "adapter": str(adapter), "dtype": "bfloat16",
                                  "params": n_params, "seconds": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"merged {n_params / 1e9:.2f}B params -> {out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
