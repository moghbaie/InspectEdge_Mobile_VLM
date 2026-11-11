"""B3: detector + rules/template report, evaluated on the SAME VLM split and metrics as the VLM.

Memory bank per category is fitted on VLM-train normals only (no val/test leakage). Image threshold
is selected on VLM-val (recall target 0.85) and frozen; --split test reuses the frozen thresholds.
The template report never names a defect type (detector has no taxonomy) -> type "unknown".
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import torch

from inspectedge.detection.patchcore import PatchCore
from inspectedge.evaluation.metrics import grouped_bootstrap_ci, report_metrics, threshold_for_recall
from inspectedge.paths import ROOT
from inspectedge.schemas.report import SCHEMA_VERSION, validate_report

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from baseline_patchcore import CROP, RES, batches, load_rgb, score  # noqa: E402

OFF = (RES - CROP) / 2


def amap_to_boxes(amap, thr, max_boxes=3):
    b = (amap >= thr).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    comps = sorted([st[i] for i in range(1, n)], key=lambda s: -s[4])[:max_boxes]
    # crop coords -> full-image normalized coords (resize RES then center-crop CROP)
    return [[round((x + OFF) / RES, 3), round((y + OFF) / RES, 3), round((x + w + OFF) / RES, 3),
             round((y + h + OFF) / RES, 3)] for x, y, w, h, _ in comps]


def template_report(cat, s, amap, thr):
    flagged = s >= thr
    findings = []
    if flagged:
        for b in amap_to_boxes(amap, thr) or [[0.0, 0.0, 1.0, 1.0]]:
            findings.append({"type": "unknown", "region_xyxy_normalized": b,
                             "visible_evidence": f"detector anomaly score {s:.2f} above threshold",
                             "review_required": True})
    return {"schema_version": SCHEMA_VERSION, "asset_category": cat,
            "image_quality": {"usable": True, "issues": []}, "findings": findings,
            "inspection_status": "visible_anomaly" if flagged else "no_visible_anomaly"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--categories", nargs="*", default=None)
    a = ap.parse_args()
    man = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    out_dir = ROOT / "reports/b3_detector_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_file = out_dir / "frozen_thresholds.json"
    frozen = json.loads(thr_file.read_text()) if thr_file.exists() else {}
    if a.split == "test" and not frozen:
        raise SystemExit("run --split val first to freeze thresholds")
    rows = []
    for cat in a.categories or sorted(man.category.unique()):
        d = man[man.category == cat]
        tr = d[(d.vlm_split == "train") & (d.label == "normal")]
        ev = d[d.vlm_split == a.split].reset_index(drop=True)
        pc = PatchCore().fit(batches([load_rgb(ROOT / p) for p in tr.image_path]))
        s, amaps = score(pc, [load_rgb(ROOT / p) for p in ev.image_path])
        y = (ev.label == "anomaly").values
        if a.split == "val":
            frozen[cat] = threshold_for_recall(y, s, 0.85)
        thr = frozen[cat]
        for i, r in ev.iterrows():
            rep = template_report(cat, float(s[i]), amaps[i], thr)
            assert not validate_report(rep), validate_report(rep)
            rows.append(dict(image_id=r.image_id, group_id=r.group_id, category=cat, gt_anomaly=bool(y[i]),
                             gt_types=list(r.defect_types), gt_boxes=list(r.boxes), json_ok=True, schema_ok=True,
                             pred_status=rep["inspection_status"], pred_types=[f["type"] for f in rep["findings"]],
                             pred_boxes=[f["region_xyxy_normalized"] for f in rep["findings"]], score=float(s[i])))
        print(cat, "done", flush=True)
        del pc
        torch.cuda.empty_cache()
    if a.split == "val":
        thr_file.write_text(json.dumps(frozen, indent=2))
    m = report_metrics(rows)
    m["false_reassurance_rate_ci95"] = grouped_bootstrap_ci(
        rows, lambda rs: report_metrics(rs)["false_reassurance_rate"], "group_id", n_boot=300)
    (out_dir / f"{a.split}_metrics.json").write_text(json.dumps(m, indent=2))
    with open(out_dir / f"{a.split}_predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
