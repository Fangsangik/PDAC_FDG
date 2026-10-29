from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """Every model (M0..M4) implements exactly this. X is a patient-indexed DataFrame."""
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Model": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...   # (n,) P(y=1)
