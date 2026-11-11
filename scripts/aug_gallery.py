"""Visual QA gallery for augmentation families (plan 4.2): rows=family, cols=severity 1,3,5; red=boxes."""
import sys

import cv2
import numpy as np
import pandas as pd

from inspectedge.augmentation.corruptions import FAMILIES, apply, mask_to_boxes
from inspectedge.paths import ROOT
from inspectedge.vlm.sft_data import load_image

cat = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
m = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
rec = m[(m.category == cat) & (m.label == "anomaly") & (m.vlm_split == "train")].iloc[0].to_dict()
img, mask = load_image(rec, 384)
rows = []
for fam in FAMILIES:
    tiles = []
    for s in (1, 3, 5):
        out, mk, ar = apply(img, fam, s, seed=7, mask=mask)
        t = out.copy()
        h, w = t.shape[:2]
        for b in mask_to_boxes(mk, 4):
            cv2.rectangle(t, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (255, 0, 0), 2)
        cv2.putText(t, f"{fam} s{s} vis={ar.defect_visible}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
g = np.concatenate(rows, 0)
out = ROOT / f"reports/aug_gallery_{cat}.jpg"
cv2.imwrite(str(out), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
print(out, g.shape)
