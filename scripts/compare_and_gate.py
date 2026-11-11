"""Compare systems on matched images and check acceptance gates (plan 2.3, 6.3).

usage: python scripts/compare_and_gate.py --split test \
         --system B3=reports/b3_detector_report/test_predictions.jsonl \
         --system B2=reports/vlm_eval/SmolVLM-500M-Instruct_test_predictions.jsonl \
         --system FT=reports/vlm_eval/<ft-tag>_test_predictions.jsonl
Writes reports/comparison_<split>.md. Also exports a blinded human-review sheet for grounding.
"""
import argparse
import csv
import json
import random

import numpy as np
from scipy.stats import binomtest

from inspectedge.evaluation.metrics import grouped_bootstrap_ci, report_metrics
from inspectedge.paths import ROOT

GATES = [  # (metric, op, target) - only the gates measurable from model outputs
    ("schema_valid_rate", ">=", 0.995),
    ("anomaly_recall_flagged", ">=", 0.85),
    ("unsupported_finding_rate_on_normals", "<=", 0.10),  # FP gate on verified normals
    ("false_reassurance_rate", "<=", 0.15),
]


def load(p):
    return {r["image_id"]: r for r in map(json.loads, open(ROOT / p, encoding="utf-8"))}


def correct(r):  # status correctness used for paired tests; invalid reports count as wrong
    flagged = r["pred_status"] in ("visible_anomaly", "uncertain")
    return bool(r["schema_ok"]) and flagged == r["gt_anomaly"]


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    return {"a_only": b01, "b_only": b10, "p_value": binomtest(b01, n, 0.5).pvalue if n else 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--system", action="append", required=True, help="NAME=path")
    ap.add_argument("--review-n", type=int, default=100)
    a = ap.parse_args()
    systems = dict(s.split("=", 1) for s in a.system)
    preds = {k: load(v) for k, v in systems.items()}
    common = sorted(set.intersection(*(set(p) for p in preds.values())))
    lines = [f"# Comparison on `{a.split}` ({len(common)} matched images)\n",
             "| System | " + " | ".join(m for m, _, _ in GATES) + " | defect_type_macro_f1 | region IoU>=0.3 |",
             "|---" * (len(GATES) + 3) + "|"]
    metrics = {}
    for name, p in preds.items():
        rows = [p[i] for i in common]
        m = report_metrics(rows)
        m["recall_ci95"] = grouped_bootstrap_ci(rows, lambda rs: report_metrics(rs)["anomaly_recall_flagged"], "group_id", 300)
        metrics[name] = m
        cells = []
        for k, op, t in GATES:
            v = m.get(k, float("nan"))
            ok = (v >= t) if op == ">=" else (v <= t)
            cells.append(f"{v:.3f} {'PASS' if ok else 'FAIL'}")
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {m.get('defect_type_macro_f1', float('nan')):.3f} | {m.get('region_iou>=0.3_rate', float('nan')):.3f} |")
    lines.append("\n## Paired McNemar tests on status correctness\n")
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ca = [correct(preds[names[i]][k]) for k in common]
            cb = [correct(preds[names[j]][k]) for k in common]
            r = mcnemar(ca, cb)
            lines.append(f"- {names[i]} vs {names[j]}: acc {np.mean(ca):.3f} vs {np.mean(cb):.3f}; "
                         f"discordant {r['a_only']}/{r['b_only']}; p={r['p_value']:.4f}")
    out = ROOT / f"reports/comparison_{a.split}.md"
    out.write_text("\n".join(lines) + "\n\n```json\n" + json.dumps(metrics, indent=1, default=str) + "\n```\n")
    print("\n".join(lines))

    # Blinded human grounding review (plan 6.2): shuffle systems, hide names.
    rng = random.Random(0)
    sample = rng.sample(common, min(a.review_n, len(common)))
    key, sheet = [], []
    for img in sample:
        for name in rng.sample(names, len(names)):
            rid = f"r{len(sheet):05d}"
            r = preds[name][img]
            sheet.append({"review_id": rid, "image_path": img, "model_output": r.get("raw") or json.dumps(
                {"status": r["pred_status"], "types": r["pred_types"], "boxes": r["pred_boxes"]}),
                "rating(supported/partial/unsupported)": "", "false_reassurance(y/n)": "", "notes": ""})
            key.append({"review_id": rid, "system": name})
    with open(ROOT / f"reports/human_review_{a.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (ROOT / f"reports/human_review_{a.split}_KEY_do_not_share_with_raters.json").write_text(json.dumps(key))


if __name__ == "__main__":
    main()
