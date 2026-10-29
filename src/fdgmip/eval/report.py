"""Turn the common prediction table (patient_id, repeat, fold, model, y, p) into comparable metrics.

Metrics are computed on pooled out-of-fold predictions within each repeat, then averaged over repeats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import calibration_slope_intercept
from .decision_curve import decision_curve
from .metrics import auroc, auroc_bootstrap_ci, brier, delong_test


def summarize(preds: pd.DataFrame, n_boot: int, seed: int, dca_thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (metrics long table, decision-curve table)."""
    rows, dca = [], []
    for (model, rep), g in preds.groupby(["model", "repeat"]):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        lo, hi = auroc_bootstrap_ci(y, p, n_boot, seed)
        cal = calibration_slope_intercept(y, p)
        rows += [{"model": model, "repeat": rep, "metric": k, "value": v} for k, v in {
            "auroc": auroc(y, p), "auroc_ci_lo": lo, "auroc_ci_hi": hi, "brier": brier(y, p),
            "cal_slope": cal["slope"], "cal_intercept": cal["intercept"], "n": len(y), "n_events": int(y.sum())}.items()]
        d = decision_curve(y, p, dca_thresholds)
        d.insert(0, "repeat", rep)
        d.insert(0, "model", model)
        dca.append(d)
    return pd.DataFrame(rows), pd.concat(dca, ignore_index=True)


def compare_models(preds: pd.DataFrame, reference: str, candidates: list[str]) -> pd.DataFrame:
    """Paired DeLong test of each candidate vs the reference (e.g. M2 vs M3/M4), per repeat."""
    rows = []
    for rep, g in preds.groupby("repeat"):
        wide = g.pivot(index="patient_id", columns="model", values="p")
        y = g.drop_duplicates("patient_id").set_index("patient_id").loc[wide.index, "y"].to_numpy()
        for c in candidates:
            if c not in wide or reference not in wide:
                continue
            r = delong_test(y, wide[c].to_numpy(), wide[reference].to_numpy())
            rows.append({"repeat": rep, "candidate": c, "reference": reference, **r})
    return pd.DataFrame(rows)
