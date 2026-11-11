import json

import numpy as np
import pytest

from inspectedge.augmentation import corruptions as C
from inspectedge.evaluation import metrics as M
from inspectedge.schemas import report as R

GOOD = {
    "schema_version": "1.0",
    "asset_category": "pcb1",
    "image_quality": {"usable": True, "issues": ["low_light"]},
    "findings": [
        {
            "type": "scratch",
            "region_xyxy_normalized": [0.24, 0.31, 0.68, 0.44],
            "visible_evidence": "narrow linear surface mark",
            "confidence": 0.72,
            "review_required": True,
        }
    ],
    "inspection_status": "visible_anomaly",
}


# ---------------------------------------------------------------- schema / parser
def test_valid_report_passes():
    assert R.validate_report(GOOD) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version="0.9"),
        lambda r: r["findings"][0].update(type="rust"),
        lambda r: r["findings"][0].update(region_xyxy_normalized=[0.5, 0.5, 0.4, 0.6]),
        lambda r: r["findings"][0].update(region_xyxy_normalized=[0, 0, 1.2, 1]),
        lambda r: r.update(inspection_status="safe"),
        lambda r: r.update(findings=[]),  # visible_anomaly with no findings
        lambda r: r.update(extra_field=1),
    ],
)
def test_invalid_reports_rejected(mutate):
    r = json.loads(json.dumps(GOOD))
    mutate(r)
    assert R.validate_report(r)


def test_parser_handles_fences_and_prose():
    txt = "Sure! Here is the report:\n```json\n" + json.dumps(GOOD) + "\n```\nDone."
    res = R.parse_model_output(txt)
    assert res.json_ok and res.schema_ok


def test_parser_handles_braces_in_strings():
    g = json.loads(json.dumps(GOOD))
    g["findings"][0]["visible_evidence"] = "mark shaped like } and {"
    assert R.parse_model_output("x " + json.dumps(g)).schema_ok


def test_parser_reports_garbage():
    res = R.parse_model_output("The image shows a candle.")
    assert not res.json_ok and res.errors


def test_canonical_json_is_deterministic():
    assert R.canonical_json(GOOD) == R.canonical_json(json.loads(json.dumps(GOOD)))


# ---------------------------------------------------------------- augmentation
@pytest.fixture
def img_mask():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), np.uint8)
    mask[40:60, 50:90] = 255
    return img, mask


@pytest.mark.parametrize("family", C.FAMILIES)
@pytest.mark.parametrize("sev", [1, 3, 5])
def test_corruption_deterministic_and_shape(img_mask, family, sev):
    img, mask = img_mask
    a, ma, ra = C.apply(img, family, sev, seed=42, mask=mask)
    b, mb, rb = C.apply(img, family, sev, seed=42, mask=mask)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and ra.params == rb.params
    assert ma.shape == mask.shape


@pytest.mark.parametrize("family", C.PHOTOMETRIC)
def test_photometric_preserves_mask(img_mask, family):
    img, mask = img_mask
    _, m, rec = C.apply(img, family, 3, 1, mask)
    assert np.array_equal(m, mask) and rec.defect_visible is True


def test_severity_monotone_for_blur(img_mask):
    img, _ = img_mask
    diffs = [np.abs(C.apply(img, "defocus_blur", s, 0)[0].astype(int) - img).mean() for s in range(1, 6)]
    assert all(d2 >= d1 for d1, d2 in zip(diffs, diffs[1:]))


def test_occlusion_marks_invisible_defect():
    img = np.zeros((100, 100, 3), np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[45:55, 45:55] = 255
    # search seeds for an occluder covering the defect; visibility flag must flip
    flags = {C.apply(img, "crop_occlusion", 1, s, mask)[2].defect_visible for s in range(300)}
    assert flags == {True, False}


def test_mask_to_boxes():
    mask = np.zeros((100, 200), np.uint8)
    mask[10:20, 20:60] = 255
    mask[70:90, 150:160] = 255
    boxes = C.mask_to_boxes(mask)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx([0.1, 0.1, 0.3, 0.2])
    for b in boxes:
        assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1


# ---------------------------------------------------------------- metrics
def test_image_metrics_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert M.image_level(y, y.astype(float))["image_auroc"] == 1.0
    r = M.image_level(y, np.random.default_rng(0).random(100))
    assert 0.3 < r["image_auroc"] < 0.7


def test_threshold_for_recall():
    y = np.array([0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.5, 0.2, 0.6, 0.7, 0.9])
    t = M.threshold_for_recall(y, s, 0.75)
    assert M.at_threshold(y, s, t)["recall"] >= 0.75


def test_aupro_perfect():
    masks = np.zeros((2, 32, 32), np.uint8)
    masks[0, 5:10, 5:10] = 1
    masks[1, 20:30, 1:4] = 1
    assert M.aupro(masks, masks.astype(float)) == pytest.approx(1.0, abs=1e-6)


def test_box_iou():
    assert M.box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert M.box_iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]) == 0.0


def test_report_metrics_safety_rates():
    rows = [
        dict(gt_anomaly=True, gt_types=["crack"], gt_boxes=[[0, 0, 0.5, 0.5]], json_ok=True, schema_ok=True,
             pred_status="no_visible_anomaly", pred_types=[], pred_boxes=[]),
        dict(gt_anomaly=True, gt_types=["crack"], gt_boxes=[[0, 0, 0.5, 0.5]], json_ok=True, schema_ok=True,
             pred_status="visible_anomaly", pred_types=["crack"], pred_boxes=[[0, 0, 0.5, 0.5]]),
        dict(gt_anomaly=False, gt_types=[], gt_boxes=[], json_ok=True, schema_ok=True,
             pred_status="visible_anomaly", pred_types=["scratch"], pred_boxes=[[0, 0, 1, 1]]),
        dict(gt_anomaly=False, gt_types=[], gt_boxes=[], json_ok=False, schema_ok=False,
             pred_status=None, pred_types=[], pred_boxes=[]),
    ]
    m = M.report_metrics(rows)
    assert m["false_reassurance_rate"] == 0.5
    assert m["unsupported_finding_rate_on_normals"] == 0.5
    assert m["json_parse_rate"] == 0.75
    assert m["region_best_iou_mean"] == 0.5


def test_grouped_bootstrap_returns_interval():
    rows = [{"g": i // 5, "v": float(i % 2)} for i in range(100)]
    lo, hi = M.grouped_bootstrap_ci(rows, lambda rs: np.mean([r["v"] for r in rs]), "g", n_boot=200)
    assert lo <= 0.5 <= hi


def test_high_severity_photometric_is_unverified(img_mask):
    img, mask = img_mask
    for fam in C.PHOTOMETRIC + ["combined"]:
        assert C.apply(img, fam, 5, 0, mask)[2].defect_visible is None


def test_sft_never_uses_unverified_severity():
    import inspect
    from inspectedge.vlm import sft_data
    default = inspect.signature(sft_data.make_example).parameters["max_severity"].default
    assert default < C.UNVERIFIED_SEVERITY


def test_strict_accuracy_penalises_invalid_outputs():
    rows = [dict(gt_anomaly=True, gt_types=["crack"], gt_boxes=[], json_ok=False, schema_ok=False,
                 pred_status=None, pred_types=[], pred_boxes=[])] * 4
    m = M.report_metrics(rows)
    assert m["false_reassurance_rate"] == 0.0  # lenient metric looks perfect...
    assert m["status_acc_strict"] == 0.0       # ...strict metric does not
