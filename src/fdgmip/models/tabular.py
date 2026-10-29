"""One model class for M0..M4. The feature table has prefixed columns:
    pet_*   conventional PET metrics (suvmax, mtv_ml, tlg, ...)
    clin_*  clinical variables from the cohort table
    mip_*   aggregated MA-MIP backbone features
A model is: column glob patterns -> impute/scale (+one-hot for categoricals) -> [harmonizer] -> estimator.
Everything that learns from data lives inside the sklearn Pipeline, so fit() on a train fold is the
only place statistics are ever computed.
"""
from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer

from fdgmip.registry import ESTIMATORS, HARMONIZERS, MODELS

ESTIMATORS.register("logistic")(LogisticRegression)

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "none": lambda: FunctionTransformer()}


def select_columns(columns: list[str], patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        hits = fnmatch.filter(columns, pat)
        if not hits:
            raise KeyError(f"column pattern '{pat}' matched nothing in {list(columns)[:10]}...")
        out += [h for h in hits if h not in out]
    return out


@MODELS.register("tabular")
class TabularModel:
    def __init__(self, columns: list[str], estimator: dict, impute: str, scale: str,
                 harmonization: dict | None = None, batch_column: str | None = None):
        self.patterns, self.estimator_spec = list(columns), dict(estimator)
        self.impute, self.scale = impute, scale
        self.harmonization = dict(harmonization) if harmonization else None
        self.batch_column = batch_column
        self.pipeline: Pipeline | None = None
        self.harmonizer = None
        self.columns_: list[str] = []

    # ---- helpers --------------------------------------------------------
    def _split_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        cat = [c for c in self.columns_ if str(X[c].dtype) in ("category", "object", "bool")]
        num = [c for c in self.columns_ if c not in cat]
        return num, cat

    def _build(self, num: list[str], cat: list[str]) -> Pipeline:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy=self.impute)), ("scale", _SCALERS[self.scale]())])
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])
        ct = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], sparse_threshold=0)
        return Pipeline([("prep", ct), ("est", ESTIMATORS.build(self.estimator_spec))])

    def _harmonize(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.harmonization is None or self.harmonization.get("name", "none") == "none":
            return X
        if self.batch_column is None or self.batch_column not in X.columns:
            raise KeyError("harmonization at model stage needs batch_column present in X")
        num, _ = self._split_types(X)
        batch = X[self.batch_column]
        if fit:
            self.harmonizer = HARMONIZERS.build({k: v for k, v in self.harmonization.items() if k != "stage"})
            self.harmonizer.fit(X[num], batch)
        Xh = X.copy()
        Xh[num] = self.harmonizer.transform(X[num], batch)
        return Xh

    # ---- Model interface -----------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabularModel":
        self.columns_ = select_columns(list(X.columns), self.patterns)
        X = self._harmonize(X, fit=True)
        num, cat = self._split_types(X)
        self.pipeline = self._build(num, cat)
        self.pipeline.fit(X[self.columns_], np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("fit() first")
        X = self._harmonize(X, fit=False)
        return self.pipeline.predict_proba(X[self.columns_])[:, 1]
