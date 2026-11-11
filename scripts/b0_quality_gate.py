"""B0: classical image-quality gate (blur / exposure / noise / compression / glare).

Trains a logistic regression on hand-crafted features: clean vs corrupted (sev 3-5) VLM-train images,
evaluates on VLM-val: AUROC and per-family detection rate at 5% false-reject on clean images.
"""
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inspectedge.augmentation.corruptions import PHOTOMETRIC, apply
from inspectedge.paths import ROOT
from inspectedge.vlm.sft_data import load_image


def features(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    hi = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = np.abs(np.diff(g, axis=1))
    block = gx[:, 7::8].mean() / (gx.mean() + 1e-6)  # 8x8 JPEG blockiness
    return [np.log1p(lap.var()), g.mean(), g.std(), (g > 250).mean(), (g < 10).mean(),
            np.log1p(np.median(np.abs(hi))), block, np.percentile(g, 99) - np.percentile(g, 1)]


def build(df, rng, n):
    X, y, fam = [], [], []
    for k, r in enumerate(df.sample(n, random_state=int(rng.integers(1e9))).to_dict("records")):
        img, _ = load_image(r, 512)
        X.append(features(img)); y.append(0); fam.append("clean")
        f = PHOTOMETRIC[k % len(PHOTOMETRIC)]
        out, _, _ = apply(img, f, int(rng.integers(3, 6)), seed=k)
        X.append(features(out)); y.append(1); fam.append(f)
    return np.array(X), np.array(y), np.array(fam)


def main():
    m = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    rng = np.random.default_rng(0)
    Xtr, ytr, _ = build(m[m.vlm_split == "train"], rng, 600)
    Xva, yva, fva = build(m[m.vlm_split == "val"], rng, 300)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    thr = np.quantile(p[yva == 0], 0.95)  # 5% false-reject of clean captures
    res = {"auroc": float(roc_auc_score(yva, p)), "threshold_clean_reject5": float(thr),
           "detect_rate_by_family": {f: float((p[fva == f] > thr).mean()) for f in PHOTOMETRIC}}
    (ROOT / "reports/b0_quality_gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
