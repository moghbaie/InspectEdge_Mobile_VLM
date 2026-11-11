"""Evaluation metrics (plan section 6).

Detector / anomaly metrics follow the MVTec AD / VisA / anomalib conventions:
image AUROC, AUPRC, F1-max, pixel AUROC and AUPRO (PRO integrated to FPR 0.3).
Report metrics follow MMAD-style closed-form scoring: anomaly discrimination accuracy,
defect-type macro-F1, plus safety-oriented rates (false reassurance, unsupported findings).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


# ----------------------------------------------------------------------------- detector
def image_level(y_true, scores) -> dict:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return {
        "image_auroc": float(roc_auc_score(y_true, scores)),
        "image_auprc": float(average_precision_score(y_true, scores)),
        "image_f1_max": float(f1[i]),
        "f1_max_threshold": float(t[i]) if len(t) else float("nan"),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def at_threshold(y_true, scores, thr) -> dict:
    """Operating-point metrics; thr MUST be chosen on validation data, never on test."""
    y_true, pred = np.asarray(y_true).astype(bool), np.asarray(scores) >= thr
    tp, fp = (pred & y_true).sum(), (pred & ~y_true).sum()
    fn, tn = (~pred & y_true).sum(), (~pred & ~y_true).sum()
    return {
        "threshold": float(thr),
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fp_per_100_normals": float(100 * fp / max(fp + tn, 1)),
    }


def threshold_for_recall(y_true, scores, target_recall=0.85) -> float:
    """Highest threshold that achieves target recall (used on validation split)."""
    y_true, scores = np.asarray(y_true).astype(bool), np.asarray(scores)
    pos = np.sort(scores[y_true])[::-1]
    k = int(np.ceil(target_recall * len(pos))) - 1
    return float(pos[max(k, 0)])


def pixel_auroc(masks: np.ndarray, amaps: np.ndarray, max_pixels: int = 5_000_000, seed: int = 0) -> float:
    y, s = masks.reshape(-1) > 0, amaps.reshape(-1)
    if y.size > max_pixels:
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s))


def aupro(masks: np.ndarray, amaps: np.ndarray, fpr_limit: float = 0.3, steps: int = 200) -> float:
    """Per-Region-Overlap AUC up to FPR=fpr_limit, normalized to [0,1] (Bergmann et al.)."""
    masks = masks > 0
    lo, hi = float(amaps.min()), float(amaps.max())
    thresholds = np.linspace(hi, lo, steps)
    regions = []
    for m in masks:
        lab, n = ndimage.label(m)
        regions.append([(lab == i) for i in range(1, n + 1)])
    neg = ~masks
    n_neg = max(neg.sum(), 1)
    fprs, pros = [], []
    for t in thresholds:
        b = amaps >= t
        fpr = (b & neg).sum() / n_neg
        ov = [(b[k] & r).sum() / r.sum() for k, rs in enumerate(regions) for r in rs]
        fprs.append(fpr)
        pros.append(np.mean(ov) if ov else 0.0)
        if fpr > fpr_limit:
            break
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    f, p = fprs[keep], pros[keep]
    # interpolate the last segment to exactly fpr_limit
    if len(fprs) > keep.sum():
        j = keep.sum()
        f1_, f2_ = fprs[j - 1], fprs[j]
        p_lim = pros[j - 1] + (pros[j] - pros[j - 1]) * (fpr_limit - f1_) / max(f2_ - f1_, 1e-12)
        f, p = np.append(f, fpr_limit), np.append(p, p_lim)
    return float(np.trapezoid(p, f) / fpr_limit)


# ----------------------------------------------------------------------------- report
def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def report_metrics(rows: list[dict]) -> dict:
    """rows: dicts with keys
    gt_anomaly(bool), gt_types(list[str]), gt_boxes(list[box]),
    json_ok, schema_ok, pred_status(str|None), pred_types(list[str]), pred_boxes(list[box]).
    """
    n = len(rows)
    json_ok = np.mean([r["json_ok"] for r in rows])
    schema_ok = np.mean([r["schema_ok"] for r in rows])
    # status mapping: anything other than no_visible_anomaly counts as "flagged" (conservative)
    y = np.array([r["gt_anomaly"] for r in rows], bool)
    flagged = np.array([r["pred_status"] in ("visible_anomaly", "uncertain") for r in rows], bool)
    abstain = np.array([r["pred_status"] in ("uncertain", None) for r in rows], bool)
    said_clean = np.array([r["pred_status"] == "no_visible_anomaly" for r in rows], bool)
    hard_pred = np.array([r["pred_status"] == "visible_anomaly" for r in rows], bool)

    out = {
        "n": n,
        "json_parse_rate": float(json_ok),
        "schema_valid_rate": float(schema_ok),
        "anomaly_balanced_acc": float(balanced_accuracy_score(y, flagged)) if len(set(y)) > 1 else float("nan"),
        "anomaly_recall_flagged": float(flagged[y].mean()) if y.any() else float("nan"),
        # Safety: claims "no anomaly" on a verified defect
        "false_reassurance_rate": float(said_clean[y].mean()) if y.any() else float("nan"),
        # Proxy for unsupported defect assertions: concrete finding on a verified-normal image
        "unsupported_finding_rate_on_normals": float(hard_pred[~y].mean()) if (~y).any() else float("nan"),
        "abstain_or_invalid_rate": float(abstain.mean()),
        # Strict: an invalid/unparseable report is WRONG (otherwise a model that never answers
        # scores a perfect false-reassurance rate). Use this for model selection.
        "status_acc_strict": float(np.mean([(r["pred_status"] == "visible_anomaly") == r["gt_anomaly"]
                                            and r["schema_ok"] for r in rows])),
    }
    # defect type macro-F1 on true anomalies where the model produced a finding (primary type)
    gt_t, pr_t = [], []
    ious = []
    for r in rows:
        if r["gt_anomaly"] and r["gt_types"]:
            gt_t.append(r["gt_types"][0])
            pr_t.append(r["pred_types"][0] if r["pred_types"] else "none")
            if r["pred_boxes"] and r["gt_boxes"]:
                ious.append(max(box_iou(p, g) for p in r["pred_boxes"] for g in r["gt_boxes"]))
            elif r["gt_boxes"]:
                ious.append(0.0)
    if gt_t:
        labels = sorted(set(gt_t))
        out["defect_type_macro_f1"] = float(f1_score(gt_t, pr_t, labels=labels, average="macro", zero_division=0))
        out["defect_type_acc"] = float(np.mean([a == b for a, b in zip(gt_t, pr_t)]))
    if ious:
        out["region_best_iou_mean"] = float(np.mean(ious))
        out["region_iou>=0.3_rate"] = float(np.mean(np.array(ious) >= 0.3))
    return out


# ----------------------------------------------------------------------------- statistics
def grouped_bootstrap_ci(rows, metric_fn, group_key, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap CI resampling *groups* (asset/session), not images (plan 6.3)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for k in rng.choice(keys, len(keys), replace=True) for r in groups[k]]
        try:
            v = metric_fn(sample)
            if v == v:
                vals.append(v)
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
