"""Physically-motivated low-quality-photo corruptions with severity 1..5 (plan section 4).

Every function is deterministic given (image, severity, seed). Geometric transforms are applied
identically to the defect mask so labels can be re-derived and checked (section 4.2).
Images are uint8 HxWx3 RGB numpy arrays; masks are uint8 HxW {0,255}.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

PHOTOMETRIC = ["low_light", "defocus_blur", "motion_blur", "sensor_noise", "jpeg", "glare"]
GEOMETRIC = ["perspective", "crop_occlusion"]
FAMILIES = PHOTOMETRIC + GEOMETRIC + ["combined"]


@dataclass
class AugRecord:
    family: str
    severity: int
    seed: int
    params: dict
    mask_area_before: int
    mask_area_after: int
    defect_visible: bool | None  # None for normal images (no mask)

    def to_dict(self):
        return asdict(self)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def low_light(img, s, rng):
    gamma = [1.3, 1.6, 2.0, 2.5, 3.0][s - 1]
    gain = [0.8, 0.65, 0.5, 0.38, 0.28][s - 1]
    x = (img.astype(np.float32) / 255.0) ** gamma * gain
    # low light raises relative noise
    x = x + rng.normal(0, 0.006 * s, x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8), {"gamma": gamma, "gain": gain}


def defocus_blur(img, s, rng):
    r = [1, 2, 3, 5, 7][s - 1]
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1, -1)
    k /= k.sum()
    return cv2.filter2D(img, -1, k), {"radius": r}


def motion_blur(img, s, rng):
    length = [5, 9, 13, 19, 27][s - 1]
    angle = float(rng.uniform(0, 180))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k), {"length": length, "angle": angle}


def sensor_noise(img, s, rng):
    # Poisson-Gaussian model: shot noise scales with signal, read noise is constant.
    photons = [120, 60, 30, 15, 8][s - 1]
    read = [0.01, 0.02, 0.03, 0.05, 0.07][s - 1]
    x = img.astype(np.float32) / 255.0
    x = rng.poisson(x * photons) / photons + rng.normal(0, read, x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8), {"photons": photons, "read_sigma": read}


def jpeg(img, s, rng):
    q = [60, 40, 25, 15, 8][s - 1]
    scale = [1.0, 0.9, 0.75, 0.6, 0.5][s - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(dec, (w, h), interpolation=cv2.INTER_LINEAR), {"quality": q, "scale": scale}


def glare(img, s, rng):
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    rx, ry = rng.uniform(0.1, 0.3) * w, rng.uniform(0.05, 0.2) * h
    strength = [0.25, 0.4, 0.55, 0.7, 0.85][s - 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))[..., None] * strength
    x = img.astype(np.float32) / 255.0
    x = x * (1 - g) + g  # specular highlight saturates toward white
    return np.clip(x * 255, 0, 255).astype(np.uint8), {"center": [cx / w, cy / h], "strength": strength}


def perspective(img, s, rng, mask=None):
    h, w = img.shape[:2]
    d = [0.02, 0.04, 0.06, 0.08, 0.10][s - 1]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-d, d, (4, 2)) * np.float32([w, h])
    dst = (src + jitter).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    m = None if mask is None else cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return out, m, {"max_shift": d}


def crop_occlusion(img, s, rng, mask=None):
    """Random rectangular occluder (e.g. glove/tool) of increasing size. Defect pixels under it
    are removed from the mask so visibility can be re-evaluated."""
    h, w = img.shape[:2]
    frac = [0.05, 0.1, 0.15, 0.22, 0.3][s - 1]
    ow, oh = int(w * np.sqrt(frac)), int(h * np.sqrt(frac))
    x0, y0 = int(rng.uniform(0, w - ow)), int(rng.uniform(0, h - oh))
    out = img.copy()
    color = rng.integers(20, 80, 3)
    out[y0 : y0 + oh, x0 : x0 + ow] = color
    m = None
    if mask is not None:
        m = mask.copy()
        m[y0 : y0 + oh, x0 : x0 + ow] = 0
    return out, m, {"box": [x0 / w, y0 / h, (x0 + ow) / w, (y0 + oh) / h]}


_PHOTO_FNS = {
    "low_light": low_light,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "sensor_noise": sensor_noise,
    "jpeg": jpeg,
    "glare": glare,
}
_GEO_FNS = {"perspective": perspective, "crop_occlusion": crop_occlusion}

# a defect is considered still visible if >= this fraction of its mask pixels survive
MIN_VISIBLE_FRACTION = 0.5
UNVERIFIED_SEVERITY = 4


def apply(img: np.ndarray, family: str, severity: int, seed: int, mask: np.ndarray | None = None):
    """Returns (image, mask, AugRecord)."""
    assert family in FAMILIES, family
    assert 1 <= severity <= 5
    rng = _rng(seed)
    area0 = int((mask > 0).sum()) if mask is not None else 0
    params: dict = {}
    out, m = img, mask
    if family == "combined":
        fams = list(rng.choice(PHOTOMETRIC, size=2, replace=False))
        for i, f in enumerate(fams):
            out, p = _PHOTO_FNS[f](out, max(1, severity - 1), _rng(seed + 1 + i))
            params[f] = p
    elif family in _PHOTO_FNS:
        out, params = _PHOTO_FNS[family](img, severity, rng)
    else:
        out, m, params = _GEO_FNS[family](img, severity, rng, mask)
    area1 = int((m > 0).sum()) if m is not None else 0
    visible = None
    if mask is not None and area0 > 0:
        visible = area1 >= MIN_VISIBLE_FRACTION * area0
        # Visual QA (reports/aug_gallery_*.jpg): severity>=4 blur/noise/jpeg erase sub-1% defects
        # while the mask is unchanged. Visibility is then unverified -> never a positive SFT target.
        if visible and family not in _GEO_FNS and severity >= UNVERIFIED_SEVERITY:
            visible = None
    rec = AugRecord(family, severity, seed, params, area0, area1, visible)
    return out, m, rec


def mask_to_boxes(mask: np.ndarray, min_area: int = 16) -> list[list[float]]:
    """Connected components -> normalized xyxy boxes (sorted by area, desc)."""
    if mask is None:
        return []
    h, w = mask.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a >= min_area:
            boxes.append((a, [x / w, y / h, (x + bw) / w, (y + bh) / h]))
    boxes.sort(key=lambda t: -t[0])
    return [b for _, b in boxes]
