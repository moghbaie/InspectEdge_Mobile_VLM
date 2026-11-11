"""Evaluate a (pretrained or LoRA fine-tuned) VLM on the structured-report task.

B2 baseline : python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --split val --per-cell 10
Fine-tuned  : python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --adapter checkpoints/<run>/final --split val
Robustness  : add --corrupt low_light --severity 3
Locked test : --split test  (run ONCE per finalist, after checkpoint/threshold selection on val)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from inspectedge.augmentation.corruptions import apply as corrupt
from inspectedge.evaluation.metrics import grouped_bootstrap_ci, report_metrics
from inspectedge.paths import ROOT
from inspectedge.schemas.report import parse_model_output
from inspectedge.vlm.prompting import build_messages
from inspectedge.vlm.sft_data import load_image


def select(df, per_cell, seed):
    if per_cell <= 0:
        return df.reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby(["category", "label"]).head(per_cell).sort_values(["category", "label"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-4bit", action="store_true", help="for 3B+ challengers (bitsandbytes)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-cell", type=int, default=10, help="images per (category,label); 0 = full split")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--longest-edge", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--corrupt", default=None)
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    man = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    df = man[man.vlm_split == a.split]
    if a.categories:
        df = df[df.category.isin(a.categories)]
    sample = select(df, a.per_cell, a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32)
    proc = AutoProcessor.from_pretrained(a.adapter or a.model)
    if hasattr(proc, "image_processor") and "longest_edge" in getattr(proc.image_processor, "size", {}):
        proc.image_processor.size = {"longest_edge": a.longest_edge}
    kw = {"dtype": dtype}
    if a.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    if not a.load_4bit:
        model = model.to(dev)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / a.adapter) if not a.adapter.startswith("/") else a.adapter)
    model.eval()

    tag = a.tag or (a.model.split("/")[-1] + ("+" + a.adapter.strip("/").split("/")[-2] if a.adapter else ""))
    tag += f"_{a.split}" + (f"_{a.corrupt}{a.severity}" if a.corrupt else "")
    out_dir = ROOT / "reports/vlm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, lat = [], []
    with open(out_dir / f"{tag}_predictions.jsonl", "w", encoding="utf-8") as fo:
        for k, r in sample.iterrows():
            arr, _ = load_image(r.to_dict(), a.longest_edge)
            if a.corrupt:
                arr = corrupt(arr, a.corrupt, a.severity, seed=10_000 + k)[0]
            img = Image.fromarray(arr)
            prompt = proc.apply_chat_template(build_messages(r.category), add_generation_prompt=True)
            inputs = proc(text=prompt, images=[img], return_tensors="pt").to(model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            if dev == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            pr = parse_model_output(text)
            p = pr.parsed if pr.json_ok else {}
            finds = [f for f in (p.get("findings") or []) if isinstance(f, dict)] if isinstance(p.get("findings"), list) else []
            boxes = [f["region_xyxy_normalized"] for f in finds
                     if isinstance(f.get("region_xyxy_normalized"), list) and len(f["region_xyxy_normalized"]) == 4
                     and all(isinstance(v, (int, float)) for v in f["region_xyxy_normalized"])]
            row = dict(image_id=r.image_id, group_id=r.group_id, category=r.category,
                       gt_anomaly=r.label == "anomaly", gt_types=list(r.defect_types), gt_boxes=list(r.boxes),
                       json_ok=pr.json_ok, schema_ok=pr.schema_ok,
                       pred_status=p.get("inspection_status") if pr.json_ok else None,
                       pred_types=[f.get("type") for f in finds], pred_boxes=boxes,
                       errors=pr.errors[:3], raw=text[:1000], latency_s=lat[-1])
            rows.append(row)
            fo.write(json.dumps(row) + "\n")
            if k % 10 == 0:
                print(f"[{k + 1}/{len(sample)}] json={pr.json_ok} schema={pr.schema_ok} status={row['pred_status']} {lat[-1]:.2f}s", flush=True)

    m = report_metrics(rows)
    for key in ["false_reassurance_rate", "unsupported_finding_rate_on_normals", "schema_valid_rate"]:
        m[key + "_ci95"] = grouped_bootstrap_ci(rows, lambda rs, kk=key: report_metrics(rs)[kk], "group_id", n_boot=300)
    per_cat = {c: report_metrics([x for x in rows if x["category"] == c]) for c in sorted({x["category"] for x in rows})}
    m.update(model=a.model, adapter=a.adapter, split=a.split, corrupt=a.corrupt, severity=a.severity if a.corrupt else None,
             longest_edge=a.longest_edge, latency_p50_s=float(np.percentile(lat, 50)),
             latency_p95_s=float(np.percentile(lat, 95)),
             gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
             peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None,
             per_category=per_cat)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
