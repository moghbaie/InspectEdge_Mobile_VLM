"""B1: PatchCore baseline on the official VisA 1-class split, plus corruption robustness.

usage: python scripts/baseline_patchcore.py --categories candle pcb1 --corrupt
Threshold for recall/FPR is chosen on a seeded 30% validation slice of the official test set
and reported on the remaining 70% (never chosen on the reported slice).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from inspectedge.augmentation.corruptions import apply as corrupt
from inspectedge.detection.patchcore import PatchCore
from inspectedge.evaluation import metrics as M

from inspectedge.paths import ROOT  # noqa: E402
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RES, CROP = 256, 224


def load_rgb(path, size=RES):
    im = Image.open(path)
    im.draft("RGB", (size * 2, size * 2))
    return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))


def center_crop(a, c=CROP):
    o = (a.shape[0] - c) // 2
    return a[o : o + c, o : o + c]


def to_tensor(imgs):
    x = (np.stack([center_crop(i) for i in imgs]).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def batches(arrs, bs=16):
    for i in range(0, len(arrs), bs):
        yield to_tensor(arrs[i : i + bs])


def load_mask(path):
    if not isinstance(path, str) or not path:
        return np.zeros((CROP, CROP), np.uint8)
    m = np.asarray(Image.open(ROOT / path).convert("L").resize((RES, RES), Image.NEAREST))
    return (center_crop(m) > 0).astype(np.uint8)


def recall_at_fpr(y, s, fpr=0.10):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def score(pc, imgs):
    s, a = [], []
    for x in batches(imgs):
        si, ai = pc.predict(x, CROP)
        s.append(si)
        a.append(ai.astype(np.float32))
    s, a = np.concatenate(s), np.concatenate(a)
    assert np.isfinite(s).all() and np.isfinite(a).all(), "non-finite anomaly scores"
    return s, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--coreset", type=float, default=0.01)
    ap.add_argument("--corrupt", action="store_true")
    ap.add_argument("--severity", type=int, default=3)
    args = ap.parse_args()

    man = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    cats = args.categories or sorted(man.category.unique())
    out_dir = ROOT / "reports/baseline_patchcore"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for cat in cats:
        d = man[man.category == cat]
        tr = d[d.official_split == "train"]
        te = d[d.official_split == "test"].reset_index(drop=True)
        t0 = time.time()
        train_imgs = [load_rgb(ROOT / p) for p in tr.image_path]
        test_imgs = [load_rgb(ROOT / p) for p in te.image_path]
        masks = np.stack([load_mask(p) for p in te.mask_path])
        load_s = time.time() - t0

        pc = PatchCore(coreset_ratio=args.coreset).fit(batches(train_imgs))
        torch.cuda.synchronize()
        t1 = time.time()
        s, amaps = score(pc, test_imgs)
        per_img_ms = 1000 * (time.time() - t1) / len(te)
        y = (te.label == "anomaly").astype(int).values

        r = {"n_train": len(tr), "n_test": len(te), "load_s": load_s, "fit_s": pc.fit_seconds,
             "bank_size": len(pc.bank), "infer_ms_per_img": per_img_ms}
        r.update(M.image_level(y, s))
        r["pixel_auroc"] = M.pixel_auroc(masks, amaps)
        r["aupro_0.3"] = M.aupro(masks, amaps)

        # operating point: choose on val slice, report on held slice
        idx_val, idx_rep = train_test_split(np.arange(len(te)), test_size=0.7, stratify=y, random_state=0)
        thr = M.threshold_for_recall(y[idx_val], s[idx_val], 0.85)
        r["op_val_threshold_recall85"] = M.at_threshold(y[idx_rep], s[idx_rep], thr)

        if args.corrupt:
            rob = {}
            clean_auc = r["image_auroc"]
            for fam in ["low_light", "defocus_blur", "motion_blur", "sensor_noise", "jpeg", "glare", "combined"]:
                ci = [corrupt(im, fam, args.severity, seed=1000 + k)[0] for k, im in enumerate(test_imgs)]
                sc, _ = score(pc, ci)
                auc = M.image_level(y, sc)["image_auroc"]
                op = M.at_threshold(y[idx_rep], sc[idx_rep], thr)
                rob[fam] = {"image_auroc": auc, "auroc_drop": clean_auc - auc,
                            "recall_at_clean_thr": op["recall"], "fpr_at_clean_thr": op["fpr"],
                            "relative_recall_drop": 1 - op["recall"] / max(r["op_val_threshold_recall85"]["recall"], 1e-9),
                            # gate needs BOTH: recall can rise to 1.0 when everything is flagged
                            "fpr_increase": op["fpr"] - r["op_val_threshold_recall85"]["fpr"],
                            # diagnostic only (oracle threshold per condition, never deployable)
                            "recall_at_fpr10_oracle": recall_at_fpr(y, sc, 0.10)}
            r["robustness_sev%d" % args.severity] = rob
        results[cat] = r
        print(cat, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                               if not isinstance(v, dict)}), flush=True)
        del pc
        torch.cuda.empty_cache()

    keys = ["image_auroc", "image_auprc", "image_f1_max", "pixel_auroc", "aupro_0.3"]
    results["_mean"] = {k: float(np.mean([results[c][k] for c in cats])) for k in keys}
    results["_meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "coreset": args.coreset, "res": RES, "crop": CROP}
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["_mean"], indent=2))


if __name__ == "__main__":
    main()
