"""Batch-effect harmonisation interface. fit() must only ever see training-fold data.

Applied at `harmonization.stage`:
  feature : in run_train_eval, on the feature columns, per fold, before the model
  model   : inside TabularModel.fit/predict_proba (same contract, just a different call site)
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Harmonizer(Protocol):
    def fit(self, X: pd.DataFrame, batch: pd.Series) -> "Harmonizer": ...
    def transform(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame: ...
