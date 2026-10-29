"""Frozen feature extractors (registry). Input (n_views, C, H, W) float32 -> output (n_views, D)."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from fdgmip.registry import BACKBONES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(Protocol):
    feature_dim: int
    def extract(self, views: np.ndarray) -> np.ndarray: ...


@BACKBONES.register("torchvision")
class TorchvisionBackbone:
    """Any torchvision classification model with its head replaced by Identity, weights frozen.

    arch:            e.g. 'resnet50', 'resnet18', 'densenet121', 'efficientnet_b0'
    weights:         torchvision weights enum name, e.g. 'IMAGENET1K_V1' (None = random init, tests only)
    input_size:      images are resized (bilinear) to input_size x input_size
    channel_adapter: 'repeat' (C=1 -> 3 copies) | 'first3' (take channels 0..2) | 'none' (C must be 3)
    intensity_max:   views are clipped to [0, intensity_max] and divided by it before ImageNet
                     normalisation. SUV scale -> TODO justify value in config.
    """
    def __init__(self, arch: str, weights: str | None, input_size: int, channel_adapter: str,
                 intensity_max: float, device: str = "cpu", batch_size: int = 32):
        model = getattr(torchvision.models, arch)(weights=weights)
        for head in ("fc", "classifier", "heads", "head"):
            if hasattr(model, head):
                setattr(model, head, torch.nn.Identity())
                break
        else:
            raise ValueError(f"don't know how to strip the head of '{arch}'")
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model, self.device, self.batch_size = model, device, batch_size
        self.input_size, self.channel_adapter, self.intensity_max = int(input_size), channel_adapter, float(intensity_max)
        with torch.no_grad():
            self.feature_dim = int(model(torch.zeros(1, 3, self.input_size, self.input_size, device=device)).shape[1])

    def _prepare(self, views: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(views, dtype=np.float32))
        if x.ndim != 4:
            raise ValueError(f"expected (n_views, C, H, W), got {tuple(x.shape)}")
        c = x.shape[1]
        if self.channel_adapter == "repeat":
            if c != 1:
                raise ValueError(f"channel_adapter='repeat' needs C=1, got C={c}")
            x = x.repeat(1, 3, 1, 1)
        elif self.channel_adapter == "first3":
            if c < 3:
                raise ValueError(f"channel_adapter='first3' needs C>=3, got C={c}")
            x = x[:, :3]
        elif self.channel_adapter == "none":
            if c != 3:
                raise ValueError(f"channel_adapter='none' needs C=3, got C={c}")
        else:
            raise ValueError(f"unknown channel_adapter '{self.channel_adapter}'")
        x = x.clamp(0, self.intensity_max) / self.intensity_max
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def extract(self, views: np.ndarray) -> np.ndarray:
        x = self._prepare(views)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                out.append(self.model(x[i:i + self.batch_size].to(self.device)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


@BACKBONES.register("foundation_stub")
class FoundationModelStub:
    """Interface placeholder for a medical foundation model (e.g. a PET/CT or radiology ViT).

    TODO: implement loading + preprocessing once a model is chosen; keep extract() contract.
    """
    def __init__(self, checkpoint: str | None = None, feature_dim: int = 0, **kwargs):
        self.checkpoint, self.feature_dim = checkpoint, feature_dim

    def extract(self, views: np.ndarray) -> np.ndarray:
        raise NotImplementedError("foundation model backbone not implemented")
