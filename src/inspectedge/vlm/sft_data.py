"""SFT example construction + collator for SmolVLM / Idefics3-style processors (plan 5.3, 5.4).

- Training-only degradation augmentation; image_quality.issues in the target reflects it.
- Occlusion that hides the defect -> status "uncertain" (plan 4.2 label preservation).
- A small fraction of asset-mismatch examples -> "unsupported_asset".
- Loss is computed on assistant tokens only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from inspectedge.augmentation.corruptions import apply as corrupt, mask_to_boxes
from inspectedge.paths import ROOT
from inspectedge.schemas.report import SCHEMA_VERSION
from inspectedge.vlm.prompting import build_messages, build_target

FAMILY_TO_ISSUES = {
    "low_light": ["low_light"],
    "defocus_blur": ["blur"],
    "motion_blur": ["blur"],
    "sensor_noise": ["noise"],
    "jpeg": ["compression"],
    "glare": ["glare"],
    "crop_occlusion": ["occlusion"],
    "perspective": [],
}
AUG_FAMILIES = list(FAMILY_TO_ISSUES)


def load_image(rec, longest_edge):
    im = Image.open(ROOT / rec["image_path"])
    im.draft("RGB", (longest_edge * 2, longest_edge * 2))
    im = im.convert("RGB")
    im.thumbnail((longest_edge, longest_edge))
    mask = None
    if rec.get("mask_path"):
        mask = np.asarray(Image.open(ROOT / rec["mask_path"]).convert("L").resize(im.size, Image.NEAREST))
    return np.asarray(im), mask


def make_example(rec: dict, longest_edge: int, aug_prob: float = 0.0, max_severity: int = 3,
                 asset_override: str | None = None, rng: random.Random | None = None):
    """Returns (PIL image, messages-with-target, meta)."""
    rng = rng or random.Random(0)
    img, mask = load_image(rec, longest_edge)
    rec = dict(rec)
    issues, meta = [], {"aug": None}
    if aug_prob and rng.random() < aug_prob:
        fam = rng.choice(AUG_FAMILIES)
        sev = rng.randint(1, max_severity)
        img, mask2, ar = corrupt(img, fam, sev, seed=rng.randint(0, 2**31 - 1), mask=mask)
        meta["aug"] = ar.to_dict()
        issues = list(FAMILY_TO_ISSUES[fam])
        if mask is not None:
            rec["boxes"] = mask_to_boxes(mask2, min_area=4)
            if ar.defect_visible is False:
                rec["boxes"] = []  # defect hidden -> build_target yields "uncertain"
    target = build_target(rec, quality_issues=issues)
    asset = rec["category"]
    if asset_override and asset_override != rec["category"]:
        asset = asset_override
        target = {
            "schema_version": SCHEMA_VERSION,
            "asset_category": asset,
            "image_quality": {"usable": True, "issues": issues},
            "findings": [],
            "inspection_status": "unsupported_asset",
        }
        meta["asset_mismatch"] = True
    return Image.fromarray(img), build_messages(asset, target), meta


def build_epoch_plan(records: list[dict], normal_ratio: float = 2.0, mismatch_frac: float = 0.05, seed: int = 0):
    """Balance normals vs anomalies and add asset-mismatch items. Returns list of (rec, asset_override)."""
    rng = random.Random(seed)
    an = [r for r in records if r["label"] == "anomaly"]
    no = [r for r in records if r["label"] == "normal"]
    rng.shuffle(no)
    no = no[: int(len(an) * normal_ratio)]
    items = [(r, None) for r in an + no]
    cats = sorted({r["category"] for r in records})
    for r in rng.sample(no, int(mismatch_frac * len(items))):
        items.append((r, rng.choice([c for c in cats if c != r["category"]])))
    rng.shuffle(items)
    return items


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, items, longest_edge=1024, aug_prob=0.0, seed=0):
        self.items, self.le, self.aug, self.seed = items, longest_edge, aug_prob, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rec, override = self.items[i]
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        img, msgs, _ = make_example(rec, self.le, self.aug, asset_override=override, rng=rng)
        return {"image": img, "messages": msgs}


@dataclass
class Collator:
    processor: object
    max_length: int = 2048

    def __call__(self, batch):
        p = self.processor
        full = [p.apply_chat_template(b["messages"], add_generation_prompt=False) for b in batch]
        prompts = [p.apply_chat_template(b["messages"][:1], add_generation_prompt=True) for b in batch]
        images = [[b["image"]] for b in batch]
        enc = p(text=full, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        pad_id = p.tokenizer.pad_token_id
        for k, (pr, img) in enumerate(zip(prompts, images)):
            n_prompt = p(text=[pr], images=[img], return_tensors="pt")["input_ids"].shape[1]
            # padding side may be left or right; locate the first non-pad token
            nonpad = (enc["attention_mask"][k] == 1).nonzero()
            start = int(nonpad[0]) if len(nonpad) else 0
            labels[k, : start + n_prompt] = -100
        labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc
