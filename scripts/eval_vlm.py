"""Evaluate a (pretrained or LoRA fine-tuned) VLM on the structured-report task.

B2 baseline : python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --split val --per-cell 10
Fine-tuned  : python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --adapter checkpoints/<run>/final --split val
Robustness  : add --corrupt low_light --severity 3
Locked test : --split test  (run ONCE per finalist, after checkpoint/threshold selection on val)
Faster      : add --batch-size 8 (batched greedy generation; latency is then re-measured
              unbatched on --latency-samples images so it stays comparable with batch-1 runs)
Long runs   : add --resume --out-dir /content/drive/MyDrive/InspectEdge/reports/vlm_eval [--max-items 300]
              predictions are appended + fsynced per image, so a crash/disconnect loses at most one image;
              re-run the same command to continue. Metrics are written only once every image is predicted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
    ap.add_argument("--out-dir", default=None, help="default: reports/vlm_eval (use a Drive path on Colab)")
    ap.add_argument("--resume", action="store_true", help="skip images already in <tag>_predictions.jsonl")
    ap.add_argument("--max-items", type=int, default=0, help="predict at most N new images this call, then exit (0 = all)")
    ap.add_argument("--batch-size", type=int, default=1, help="images per generate() call (halved automatically on OOM)")
    ap.add_argument("--latency-samples", type=int, default=30, help="batch>1 only: unbatched images timed for latency_p50/p95")
    a = ap.parse_args()

    man = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    df = man[man.vlm_split == a.split]
    if a.categories:
        df = df[df.category.isin(a.categories)]
    sample = select(df, a.per_cell, a.seed)

    tag = a.tag or (a.model.split("/")[-1] + ("+" + a.adapter.strip("/").split("/")[-2] if a.adapter else ""))
    tag += f"_{a.split}" + (f"_{a.corrupt}{a.severity}" if a.corrupt else "")
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "reports/vlm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{tag}_predictions.jsonl"

    # Resume: keep every fully written row (a crash mid-write leaves at most one truncated last line).
    rows = []
    if a.resume and pred_path.exists():
        wanted, seen = set(sample.image_id), set()
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("image_id") in wanted and row["image_id"] not in seen:
                seen.add(row["image_id"])
                rows.append(row)
        pred_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    done = {r["image_id"] for r in rows}
    todo = sample[~sample.image_id.isin(done)]
    if a.max_items > 0:
        todo = todo.iloc[:a.max_items]
    print(f"{tag}: {len(done)}/{len(sample)} already predicted, {len(todo)} to run this call -> {pred_path}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32)
    proc = AutoProcessor.from_pretrained(a.adapter or a.model)
    if hasattr(proc, "image_processor") and "longest_edge" in getattr(proc.image_processor, "size", {}):
        proc.image_processor.size = {"longest_edge": a.longest_edge}
    kw = {"dtype": dtype}
    if a.load_4bit:
        # Gemma 3n's AltUp clamps correction_coefs.weight in place during forward (and prediction_coefs
        # in train mode); bnb would store those tiny Linears as uint8 and the clamp crashes
        # ("result type Float can't be cast to ... unsigned char"). Keep them, the router and lm_head
        # in bf16 -- a few KB, no effect on the 4-bit size. Passing a list replaces bnb's default
        # skip list, hence lm_head is restated.
        skip = ["lm_head", "correction_coefs", "prediction_coefs", "modality_router"]
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype,
                                                       llm_int8_skip_modules=skip)
        kw["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    if not a.load_4bit:
        model = model.to(dev)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / a.adapter) if not a.adapter.startswith("/") else a.adapter)
    model.eval()

    if getattr(proc, "tokenizer", None) is not None:
        proc.tokenizer.padding_side = "left"   # batched generation with a decoder-only LM needs left padding

    def generate(items):
        """items: list of (k, row). Returns (decoded texts, wall seconds for the whole batch)."""
        imgs, prompts = [], []
        for k, r in items:
            arr, _ = load_image(r.to_dict(), a.longest_edge)
            if a.corrupt:
                arr = corrupt(arr, a.corrupt, a.severity, seed=10_000 + k)[0]
            imgs.append([Image.fromarray(arr)])
            prompts.append(proc.apply_chat_template(build_messages(r.category), add_generation_prompt=True))
        inputs = proc(text=prompts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True), dt

    def generate_safe(items):
        """Like generate(), but splits the batch in half on CUDA OOM. Returns [(text, amortised s, batch_n)]."""
        try:
            texts, dt = generate(items)
            return [(t, dt / len(items), len(items)) for t in texts]
        except torch.cuda.OutOfMemoryError:
            if len(items) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(items) // 2
            print(f"OOM at batch {len(items)}: retrying as {h} + {len(items) - h}", flush=True)
            return generate_safe(items[:h]) + generate_safe(items[h:])

    todo_items = list(todo.iterrows())
    bs = max(1, a.batch_size)
    t_start = time.perf_counter()
    with open(pred_path, "a" if a.resume else "w", encoding="utf-8") as fo:
        for b0 in range(0, len(todo_items), bs):
            batch = todo_items[b0:b0 + bs]
            for (k, r), (text, dt, bn) in zip(batch, generate_safe(batch)):
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
                           errors=pr.errors[:3], raw=text[:1000], latency_s=dt, batch_n=bn)
                rows.append(row)
                fo.write(json.dumps(row) + "\n")
            fo.flush()
            os.fsync(fo.fileno())   # checkpoint per batch: survives a runtime reset when out_dir is on Drive
            done_now = b0 + len(batch)
            rate = done_now / (time.perf_counter() - t_start)
            print(f"[{len(rows)}/{len(sample)}] {rate:.2f} img/s, ~{(len(sample) - len(rows)) / rate / 60:.0f} min left | "
                  f"last: json={pr.json_ok} schema={pr.schema_ok} status={row['pred_status']}", flush=True)

    if len(rows) < len(sample):
        print(f"PARTIAL: {len(rows)}/{len(sample)} predicted. Re-run the same command with --resume to continue.", flush=True)
        sys.exit(0)
    order = {iid: i for i, iid in enumerate(sample.image_id)}
    rows.sort(key=lambda x: order[x["image_id"]])
    batched = any(x.get("batch_n", 1) > 1 for x in rows)
    throughput = len(rows) / sum(x["latency_s"] for x in rows)
    peak_batched = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None
    if batched:
        # Amortised batch time is not per-request latency: time a fixed subset one image at a time instead.
        lat_items = list(sample.iloc[:max(1, a.latency_samples)].iterrows())
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        generate(lat_items[:1])   # warm-up
        lat = [generate([it])[1] for it in lat_items]
        print(f"latency re-measured unbatched on {len(lat)} images", flush=True)
    else:
        lat = [x["latency_s"] for x in rows]

    m = report_metrics(rows)
    for key in ["false_reassurance_rate", "unsupported_finding_rate_on_normals", "schema_valid_rate"]:
        m[key + "_ci95"] = grouped_bootstrap_ci(rows, lambda rs, kk=key: report_metrics(rs)[kk], "group_id", n_boot=300)
    per_cat = {c: report_metrics([x for x in rows if x["category"] == c]) for c in sorted({x["category"] for x in rows})}
    m.update(model=a.model, adapter=a.adapter, split=a.split, corrupt=a.corrupt, severity=a.severity if a.corrupt else None,
             longest_edge=a.longest_edge, latency_p50_s=float(np.percentile(lat, 50)),
             latency_p95_s=float(np.percentile(lat, 95)), latency_n=len(lat),
             latency_mode="unbatched_subset" if batched else "per_image", eval_batch_size=a.batch_size,
             throughput_img_per_s=throughput, peak_gpu_mem_gb_batched=peak_batched if batched else None,
             gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
             peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None,
             per_category=per_cat)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
