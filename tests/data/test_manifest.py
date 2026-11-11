"""Data validation tests (plan section 8: missing files, label vocabulary, invalid regions,
duplicate leakage, license provenance, split integrity). Run after build_manifest."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from inspectedge.schemas.report import DEFECT_TYPES

from inspectedge.paths import ROOT  # noqa: E402
MAN = ROOT / "data/manifests/visa_manifest.jsonl"
pytestmark = [pytest.mark.data, pytest.mark.skipif(not MAN.exists(), reason="manifest not built")]


@pytest.fixture(scope="module")
def recs():
    return [json.loads(l) for l in open(MAN, encoding="utf-8")]


def test_expected_size(recs):
    assert len(recs) == 10821
    assert sum(r["label"] == "anomaly" for r in recs) == 1200


def test_unique_ids(recs):
    assert len({r["image_id"] for r in recs}) == len(recs)


def test_all_files_exist(recs):
    missing = [r["image_path"] for r in recs if not (ROOT / r["image_path"]).exists()]
    missing += [r["mask_path"] for r in recs if r["mask_path"] and not (ROOT / r["mask_path"]).exists()]
    assert not missing, missing[:5]


def test_checksums_sample(recs):
    rng = np.random.default_rng(0)
    for i in rng.choice(len(recs), 40, replace=False):
        r = recs[i]
        assert hashlib.sha256((ROOT / r["image_path"]).read_bytes()).hexdigest() == r["sha256"]


def test_mask_label_consistency(recs):
    for r in recs:
        if r["label"] == "anomaly":
            assert r["mask_path"] and r["mask_area_frac"] > 0, r["image_id"]
            assert len(r["boxes"]) >= 1, r["image_id"]
        else:
            assert not r["mask_path"] and r["boxes"] == [], r["image_id"]


def test_boxes_valid(recs):
    for r in recs:
        for b in r["boxes"]:
            assert len(b) == 4 and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1, (r["image_id"], b)


def test_label_vocabulary(recs):
    mapping = yaml.safe_load(open(ROOT / "configs/taxonomy_map.yaml"))["mapping"]
    for r in recs:
        assert set(r["defect_types"]) <= set(DEFECT_TYPES)
        for raw in filter(None, (s.strip() for s in r["defect_raw"].split(","))):
            assert raw in mapping, f"unmapped raw label {raw!r}"
        if r["label"] == "anomaly":
            assert r["defect_types"], r["image_id"]


def test_official_split(recs):
    assert all(r["official_split"] in ("train", "test") for r in recs)
    assert not any(r["official_split"] == "train" and r["label"] == "anomaly" for r in recs)


def test_no_group_crosses_splits(recs):
    g = defaultdict(set)
    for r in recs:
        g[r["group_id"]].add(r["vlm_split"])
    leaks = [k for k, v in g.items() if len(v) > 1]
    assert not leaks, leaks[:5]


def test_no_exact_duplicate_across_splits(recs):
    s = defaultdict(set)
    for r in recs:
        s[r["sha256"]].add(r["vlm_split"])
    assert all(len(v) == 1 for v in s.values())


def test_split_ratios_and_coverage(recs):
    cnt = defaultdict(lambda: defaultdict(int))
    for r in recs:
        cnt[(r["category"], r["label"])][r["vlm_split"]] += 1
    for key, c in cnt.items():
        n = sum(c.values())
        assert abs(c["train"] / n - 0.70) < 0.06, (key, dict(c))
        assert c["val"] > 0 and c["test"] > 0, (key, dict(c))


def test_split_hashes_locked(recs):
    locked = json.loads((ROOT / "data/manifests/split_hashes.json").read_text())
    for s, v in locked.items():
        ids = sorted(r["image_id"] for r in recs if r["vlm_split"] == s)
        assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == v["sha256_of_ids"]


def test_license_provenance(recs):
    assert {r["license_id"] for r in recs} == {"CC-BY-4.0"}
    text = (ROOT / "data/raw/visa/LICENSE-DATASET").read_text(errors="ignore")
    assert "Attribution 4.0 International" in text and "NonCommercial" not in text
