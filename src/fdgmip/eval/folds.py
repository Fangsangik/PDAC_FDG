"""Patient-level repeated stratified k-fold. Generated once, saved, and read by every model."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)


def make_folds(cohort: pd.DataFrame, strata_columns: list[str], n_splits: int, n_repeats: int, seed: int) -> pd.DataFrame:
    """cohort: indexed by patient id. Returns long table (patient_id, repeat, fold)."""
    if not cohort.index.is_unique:
        raise ValueError("cohort index (patient id) must be unique")
    strata = cohort[strata_columns].astype(str).agg("|".join, axis=1)
    counts = strata.value_counts()
    small = counts[counts < n_splits]
    if len(small):
        log.warning("strata with fewer members than n_splits=%d: %s", n_splits, small.to_dict())
    rows = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (_, test_idx) in enumerate(skf.split(cohort.index, strata)):
            rows += [{"patient_id": pid, "repeat": r, "fold": k} for pid in cohort.index[test_idx]]
    return pd.DataFrame(rows)


def save_folds(folds: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)


def load_folds(path: str | Path) -> pd.DataFrame:
    f = pd.read_csv(path, dtype={"patient_id": str})
    if f.duplicated(["patient_id", "repeat"]).any():
        raise ValueError(f"{path}: a patient appears in more than one fold within a repeat")
    return f
