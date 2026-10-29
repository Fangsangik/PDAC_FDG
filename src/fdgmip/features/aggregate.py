"""View-level features (n_views, D) -> patient-level vector."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from fdgmip.registry import AGGREGATORS


class Aggregator(Protocol):
    def aggregate(self, feats: np.ndarray) -> np.ndarray: ...


@AGGREGATORS.register("mean")
class Mean:
    def aggregate(self, feats: np.ndarray) -> np.ndarray:
        return feats.mean(axis=0)


@AGGREGATORS.register("max")
class Max:
    def aggregate(self, feats: np.ndarray) -> np.ndarray:
        return feats.max(axis=0)


@AGGREGATORS.register("mean_max")
class MeanMax:
    def aggregate(self, feats: np.ndarray) -> np.ndarray:
        return np.concatenate([feats.mean(axis=0), feats.max(axis=0)])


@AGGREGATORS.register("concat")
class Concat:
    """Keeps view order; only meaningful when every patient has the same n_views."""
    def aggregate(self, feats: np.ndarray) -> np.ndarray:
        return feats.reshape(-1)
