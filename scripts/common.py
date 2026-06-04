"""
Shared building blocks for the DINOv3 PatchCore pipeline:

  * deterministic seeding
  * image loading / preprocessing
  * backbone factory (frozen DINOv3 ViT-S/16, random-init ViT, ResNet-50)
  * patch-token + CLS feature extraction with local neighbourhood aggregation
  * greedy k-center coreset subsampling (PatchCore)
  * a context manager that captures the last block's self-attention map
  * small metric helpers
"""
from __future__ import annotations
import os
import math
import random
import contextlib
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #
_MEAN = torch.tensor(C.NORM_MEAN).view(3, 1, 1)
_STD = torch.tensor(C.NORM_STD).view(3, 1, 1)


def load_image(path: str, size: int = C.IMG_SIZE) -> torch.Tensor:
    """Load an image -> normalised CHW float tensor (resized, square)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    chw = arr.permute(2, 0, 1)
    return (chw - _MEAN) / _STD


def load_image_raw(path: str, size: int = C.IMG_SIZE) -> np.ndarray:
    """Load an image as an un-normalised HWC uint8 array (for overlays)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Backbone factory
# --------------------------------------------------------------------------- #
class Backbone:
    """Wraps a timm model and exposes a uniform feature interface.

    `extract(batch)` returns:
        patches : (B, H, W, D)  spatial grid of patch features
        cls     : (B, D)        image-level descriptor
    """

    def __init__(self, name: str):
        import timm
        self.name = name
        self.kind = "vit" if name in ("dinov3", "randvit") else "cnn"

        if name == "dinov3":
            self.model = timm.create_model(
                C.TIMM_DINOV3, pretrained=True, num_classes=0)
            self.n_prefix = self.model.num_prefix_tokens
        elif name == "randvit":
            # identical architecture, randomly initialised (no pretraining)
            self.model = timm.create_model(
                C.TIMM_DINOV3, pretrained=False, num_classes=0)
            self.n_prefix = self.model.num_prefix_tokens
        elif name == "resnet50":
            # ImageNet-pretrained CNN, the original PatchCore backbone.
            self.model = timm.create_model(
                C.TIMM_RESNET, pretrained=True, features_only=True,
                out_indices=(2, 3))
        else:
            raise ValueError(f"unknown backbone {name}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # -- feature extraction -------------------------------------------------- #
    @torch.no_grad()
    def extract(self, batch: torch.Tensor):
        if self.kind == "vit":
            tokens = self.model.forward_features(batch)        # (B, N, D)
            cls = tokens[:, 0]                                 # CLS token
            patch = tokens[:, self.n_prefix:]                  # drop CLS+regs
            b, n, d = patch.shape
            g = int(round(math.sqrt(n)))
            grid = patch.reshape(b, g, g, d)                   # (B, H, W, D)
            return grid, cls
        else:
            feats = self.model(batch)                          # list of maps
            g = C.IMG_SIZE // 16                               # target grid (14)
            pooled = [F.adaptive_avg_pool2d(f, (g, g)) for f in feats]
            cat = torch.cat(pooled, dim=1)                     # (B, C, g, g)
            grid = cat.permute(0, 2, 3, 1).contiguous()        # (B, H, W, D)
            cls = F.adaptive_avg_pool2d(cat, (1, 1)).flatten(1)  # GAP descriptor
            return grid, cls


# --------------------------------------------------------------------------- #
# PatchCore feature post-processing
# --------------------------------------------------------------------------- #
def neighbourhood_aggregate(grid: torch.Tensor, k: int = C.NEIGHBOURHOOD):
    """k x k average pooling over the patch grid (locally aware features).

    grid : (B, H, W, D) -> (B, H, W, D)
    """
    if k is None or k <= 1:
        return grid
    b, h, w, d = grid.shape
    x = grid.permute(0, 3, 1, 2)                               # (B, D, H, W)
    x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.permute(0, 2, 3, 1).contiguous()


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


# --------------------------------------------------------------------------- #
# Greedy k-center coreset (PatchCore memory-bank subsampling)
# --------------------------------------------------------------------------- #
def greedy_coreset(pool: np.ndarray, n: int, seed: int = C.SEED) -> np.ndarray:
    """Return indices of `n` points greedily chosen to maximise min-distance.

    A standard farthest-point-sampling approximation of PatchCore's coreset.
    """
    rng = np.random.default_rng(seed)
    m = pool.shape[0]
    if n >= m:
        return np.arange(m)
    chosen = np.empty(n, dtype=np.int64)
    start = int(rng.integers(m))
    chosen[0] = start
    # squared distance of every point to the nearest chosen centre
    min_d = ((pool - pool[start]) ** 2).sum(1)
    for i in range(1, n):
        nxt = int(np.argmax(min_d))
        chosen[i] = nxt
        d = ((pool - pool[nxt]) ** 2).sum(1)
        min_d = np.minimum(min_d, d)
    return chosen


# --------------------------------------------------------------------------- #
# Attention capture (wraps scaled_dot_product_attention)
# --------------------------------------------------------------------------- #
class _AttnGrabber:
    def __init__(self):
        self.last = None


@contextlib.contextmanager
def capture_attention():
    """Context manager: temporarily wrap F.scaled_dot_product_attention so the
    most recent call's attention weights are recorded.  After the model's
    forward pass, `grabber.last` holds the final block's attention
    of shape (B, heads, N, N)."""
    grabber = _AttnGrabber()
    orig = F.scaled_dot_product_attention

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, **kw):
        s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        attn = (query.float() @ key.float().transpose(-2, -1)) * s
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn = attn + attn_mask.float()
        attn = attn.softmax(dim=-1)
        grabber.last = attn.detach()
        return orig(query, key, value, attn_mask=attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = wrapped
    try:
        yield grabber
    finally:
        F.scaled_dot_product_attention = orig


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def batched(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
