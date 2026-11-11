"""Build the VisA manifest (plan 3.2) with checksums, near-duplicate groups and grouped splits.

usage: python -m inspectedge.data.build_manifest
Outputs data/manifests/visa_manifest.jsonl and data/manifests/split_hashes.json
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image

from inspectedge.augmentation.corruptions import mask_to_boxes

from inspectedge.paths import ROOT  # noqa: E402
VISA = ROOT / "data/raw/visa"
OUT = ROOT / "data/manifests"
SOURCE_ID = "visa_20220922"
LICENSE_ID = "CC-BY-4.0"
# Near-duplicate rule (validated on VisA, see reports/dataset_audit.md): Pearson correlation of
# 64x64 grayscale thumbnails >= THUMB_CORR_DUP within a category, or identical SHA-256.
# pHash is recorded but NOT used for grouping: fixed-rig photos of different units collide at
# Hamming distance 0 (e.g. 7,278 pcb4 pairs), which chains whole categories into one group.
THUMB_CORR_DUP = 0.998
SPLIT_FRACS = (0.70, 0.15, 0.15)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(row: dict) -> dict:
    p = VISA / row["image"]
    im = Image.open(p)
    w, h = im.size
    im.draft("RGB", (512, 512))
    small = im.convert("RGB")
    gray = np.asarray(small.convert("L"), np.float32)
    rec = {
        "width": w,
        "height": h,
        "sha256": sha256(p),
        "phash": str(imagehash.phash(small)),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),  # sharpness proxy
        "mean_luma": float(gray.mean()),
    }
    t = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).ravel()
    t = t - t.mean()
    rec["_thumb"] = (t / (np.linalg.norm(t) + 1e-6)).astype(np.float32)
    boxes, mask_area = [], 0.0
    if row["mask"]:
        m = np.asarray(Image.open(VISA / row["mask"]).convert("L"))
        mask_area = float((m > 0).mean())
        boxes = mask_to_boxes(m, min_area=int(0.00002 * m.size) + 1) or mask_to_boxes(m, min_area=1)
    rec["boxes"] = boxes
    rec["mask_area_frac"] = mask_area
    return rec


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def near_duplicate_groups(thumbs: np.ndarray, cats: list[str], sha: list[str]) -> tuple[list[int], int]:
    n = len(thumbs)
    uf = UnionFind(n)
    cats_arr = np.array(cats)
    n_pairs = 0
    for c in sorted(set(cats)):
        idx = np.nonzero(cats_arr == c)[0]
        C = np.triu(thumbs[idx] @ thumbs[idx].T, 1)
        for a, b in zip(*np.nonzero(C >= THUMB_CORR_DUP)):
            uf.union(int(idx[a]), int(idx[b]))
            n_pairs += 1
    by_sha = defaultdict(list)
    for i, s in enumerate(sha):
        by_sha[s].append(i)
    for idx in by_sha.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
            n_pairs += 1
    return [uf.find(i) for i in range(n)], n_pairs


def grouped_split(records: list[dict], seed: int = 0) -> None:
    """70/15/15 by group, stratified by (category, label). Assigns rec['vlm_split'] in place."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    strata = defaultdict(list)
    for gid, rs in groups.items():
        key = (rs[0]["category"], "anomaly" if any(x["label"] == "anomaly" for x in rs) else "normal")
        strata[key].append(gid)
    for key in sorted(strata):
        gids = sorted(strata[key])
        rng.shuffle(gids)
        n = len(gids)
        n_tr, n_va = int(round(SPLIT_FRACS[0] * n)), int(round(SPLIT_FRACS[1] * n))
        for k, gid in enumerate(gids):
            split = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
            for r in groups[gid]:
                r["vlm_split"] = split


def main():
    tax = yaml.safe_load(open(ROOT / "configs/taxonomy_map.yaml"))
    mapping, families = tax["mapping"], tax["asset_family"]
    fam_of = {c: f for f, cs in families.items() for c in cs}
    official = {}
    with open(VISA / "split_csv/1cls.csv") as f:
        for r in csv.DictReader(f):
            official[r["image"]] = r["split"]

    rows = []
    for anno in sorted(VISA.glob("*/image_anno.csv")):
        with open(anno) as f:
            rows += list(csv.DictReader(f))

    with ThreadPoolExecutor(8) as ex:
        desc = list(ex.map(describe, rows))

    records = []
    for row, d in zip(rows, desc):
        cat = row["image"].split("/")[0]
        label = "normal" if row["label"] == "normal" else "anomaly"
        raw = [] if label == "normal" else [s.strip() for s in row["label"].split(",") if s.strip()]
        mapped = [mapping.get(s, "unknown") for s in raw]
        # primary type first: prefer a concrete class over "unknown"
        mapped = sorted(dict.fromkeys(mapped), key=lambda t: t == "unknown")
        records.append(
            {
                "image_id": f"{SOURCE_ID}/{row['image']}",
                "source_id": SOURCE_ID,
                "license_id": LICENSE_ID,
                "category": cat,
                "asset_family": fam_of.get(cat, "unsupported"),
                "label": label,
                "defect_raw": ", ".join(raw),
                "defect_types": mapped,
                "image_path": f"data/raw/visa/{row['image']}",
                "mask_path": f"data/raw/visa/{row['mask']}" if row["mask"] else "",
                "official_split": official.get(row["image"], "missing"),
                "capture_session_id": None,  # not provided by VisA
                "device_id": "visa_rig",  # controlled acquisition, not phone
                "annotation_version": "visa_20220922+taxonomy_v1",
                **d,
            }
        )
    thumbs = np.stack([r.pop("_thumb") for r in records])
    gids, n_pairs = near_duplicate_groups(thumbs, [r["category"] for r in records], [r["sha256"] for r in records])
    print(f"near-duplicate pairs merged: {n_pairs}")
    for r, g in zip(records, gids):
        r["group_id"] = f"g{g:05d}"
    grouped_split(records)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "visa_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    split_hash = {}
    for s in ("train", "val", "test"):
        ids = sorted(r["image_id"] for r in records if r["vlm_split"] == s)
        split_hash[s] = {"n": len(ids), "sha256_of_ids": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    lock = OUT / "split_hashes.json"
    if lock.exists():  # locked split: a rebuild (e.g. new Colab session) must reproduce it exactly
        locked = json.loads(lock.read_text())
        if locked != split_hash:
            raise SystemExit(f"SPLIT LOCK VIOLATION: rebuilt split differs from {lock}: {split_hash}")
        print("split lock verified")
    else:
        lock.write_text(json.dumps(split_hash, indent=2))
    print(json.dumps(split_hash, indent=2))


if __name__ == "__main__":
    main()
