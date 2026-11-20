"""Copy a handful of VAL images (never train, never test) into examples/ for the demo notebook (06).

    python scripts/make_examples.py

Images are resized to the eval longest edge (1024) and saved as JPEG, with ground truth in
examples/examples.json, so notebook 06 runs without downloading VisA (~2 GB).
VisA is CC BY 4.0: attribution is written to examples/README.md.
"""
from __future__ import annotations

import json

import pandas as pd
from PIL import Image

from inspectedge.paths import ROOT

LONGEST_EDGE = 1024
# (category, label): a defective + a normal PCB (the MVP family), plus a spread of other categories.
PICKS = [("pcb1", "anomaly"), ("pcb1", "normal"), ("pcb4", "anomaly"), ("pcb3", "normal"),
         ("capsules", "anomaly"), ("candle", "anomaly"), ("macaroni1", "normal"), ("cashew", "anomaly")]
SEED = 0


def main():
    man = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
    val = man[man.vlm_split == "val"]
    out = ROOT / "examples"
    out.mkdir(exist_ok=True)
    meta = []
    for cat, label in PICKS:
        r = val[(val.category == cat) & (val.label == label)].sample(1, random_state=SEED).iloc[0]
        im = Image.open(ROOT / r.image_path).convert("RGB")
        im.thumbnail((LONGEST_EDGE, LONGEST_EDGE))
        name = f"{cat}_{label}.jpg"
        im.save(out / name, quality=90)
        meta.append(dict(file=name, image_id=r.image_id, category=cat, label=label,
                         defect_types=list(r.defect_types), boxes=list(r.boxes), source_path=r.image_path))
    (out / "examples.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "README.md").write_text(
        "# Demo images\n\n"
        "A few images from the **val** split (never used for training or the locked test), resized to 1024 px.\n"
        "Ground truth (label, defect types, normalised boxes) is in `examples.json`. Regenerate with "
        "`python scripts/make_examples.py`.\n\n"
        "Source: VisA (Visual Anomaly) dataset, Zou et al., ECCV 2022, "
        "https://github.com/amazon-science/spot-diff — licensed CC BY 4.0. Images were resized; no other changes.\n",
        encoding="utf-8")
    print("wrote", len(meta), "examples to", out)


if __name__ == "__main__":
    main()
