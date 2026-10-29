"""Clinical table + label loader. This is the ONLY entry point for label and clinical variables.

Leakage guard: columns are loaded by allowlist, and any header matching a forbidden pattern
(post-baseline information) raises instead of warning.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class LeakageError(ValueError):
    pass


def load_cohort(path: str | Path, cohort_cfg: dict) -> pd.DataFrame:
    """Return DataFrame indexed by patient id with label + strata + clinical columns only."""
    id_col = cohort_cfg["id_column"]
    label_col = cohort_cfg["label_column"]
    clinical = list(cohort_cfg["clinical_columns"])
    strata = list(cohort_cfg.get("strata_columns", [label_col]))
    categorical = list(cohort_cfg.get("categorical", []))
    forbidden = [re.compile(p, re.IGNORECASE) for p in cohort_cfg.get("forbidden_patterns", [])]

    header = list(pd.read_csv(path, nrows=0).columns)
    bad = [c for c in header if any(p.search(c) for p in forbidden)]
    if bad:
        raise LeakageError(f"{path}: columns match forbidden (post-baseline) patterns: {bad}")

    extra = list(cohort_cfg.get("extra_columns", []) or [])   # e.g. scanner: loaded, never a model input
    keep = [id_col, label_col, *strata, *clinical, *extra]
    keep = list(dict.fromkeys(keep))  # dedupe, keep order
    missing = [c for c in keep if c not in header]
    if missing:
        raise KeyError(f"{path}: required columns missing: {missing}")
    dropped = sorted(set(header) - set(keep))
    if dropped:
        log.info("cohort: ignoring %d non-allowlisted columns: %s", len(dropped), dropped)

    df = pd.read_csv(path, usecols=keep, dtype={id_col: str})
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicate patient ids: {df[id_col][df[id_col].duplicated()].tolist()}")
    df = df.set_index(id_col)
    if df[label_col].isna().any():
        raise ValueError(f"{path}: label '{label_col}' has missing values")
    df[label_col] = df[label_col].astype(int)
    for c in categorical:
        df[c] = df[c].astype("category")
    return df
