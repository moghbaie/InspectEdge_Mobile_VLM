"""Compact PatchCore (Roth et al., CVPR 2022) for baseline B1.

WideResNet50 layer2+layer3 features, 3x3 local aggregation, channel reduction,
greedy k-center coreset memory bank, nearest-neighbour patch scoring.
Sized to run on a 4 GB laptop GPU and on Colab T4.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter


class FeatureExtractor(torch.nn.Module):
    def __init__(self, out_dim: int = 384):
        super().__init__()
        m = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.out_dim = out_dim
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)  # B,1536,H,W
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, C)
        f = F.adaptive_avg_pool1d(f.unsqueeze(1), self.out_dim).squeeze(1)  # channel reduction
        return f.reshape(B, H, W, self.out_dim)


def greedy_coreset(feats: torch.Tensor, n: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=feats.device).manual_seed(seed)
    P = torch.randn(feats.shape[1], proj_dim, device=feats.device, generator=g) / proj_dim**0.5
    z = feats.float() @ P
    idx = [int(torch.randint(len(z), (1,), generator=g, device=feats.device))]
    mind = torch.cdist(z, z[idx[-1]].unsqueeze(0)).squeeze(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        idx.append(i)
        mind = torch.minimum(mind, torch.cdist(z, z[i].unsqueeze(0)).squeeze(1))
    return torch.tensor(idx, device=feats.device)


class PatchCore:
    def __init__(self, device="cuda", coreset_ratio=0.01, out_dim=384, seed=0):
        self.device = device
        self.fe = FeatureExtractor(out_dim).to(device).half()
        self.ratio, self.seed = coreset_ratio, seed
        self.bank: torch.Tensor | None = None
        self.fp32_fallback = False

    def _features(self, x):
        """fp16 features with fp32 fallback. GTX 16xx cuDNN returns NaN in fp16 for some batch
        shapes (observed: batch 6); any non-finite output is recomputed in fp32."""
        dt = next(self.fe.parameters()).dtype
        f = self.fe(x.to(self.device, dt))
        if not torch.isfinite(f).all():
            self.fe = self.fe.float()
            self.fp32_fallback = True
            f = self.fe(x.to(self.device, torch.float32))
        assert torch.isfinite(f).all(), "non-finite PatchCore features in fp32"
        return f

    def fit(self, loader):
        t0 = time.time()
        feats = []
        for x in loader:
            feats.append(self._features(x).half().reshape(-1, self.fe.out_dim))
        feats = torch.cat(feats)
        n = max(1000, int(len(feats) * self.ratio))
        idx = greedy_coreset(feats, min(n, len(feats)), seed=self.seed)
        self.bank = feats[idx].float().contiguous()
        self.fit_seconds = time.time() - t0
        self.n_patches = len(feats)
        return self

    @torch.no_grad()
    def predict(self, x, out_size: int, sigma: float = 4.0):
        f = self._features(x).float()  # B,H,W,D
        B, H, W, D = f.shape
        d = torch.cdist(f.reshape(-1, D), self.bank).min(1).values.reshape(B, 1, H, W)
        amap = F.interpolate(d, size=(out_size, out_size), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        amap = np.stack([gaussian_filter(a, sigma) for a in amap])
        return amap.reshape(B, -1).max(1), amap
