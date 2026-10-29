"""Decision curve analysis (Vickers & Elkin 2006)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def decision_curve(y, p, thresholds) -> pd.DataFrame:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in np.asarray(thresholds, float):
        if not 0 < t < 1:
            continue
        pred = p >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        w = t / (1 - t)
        rows.append({"threshold": t, "net_benefit": tp / n - fp / n * w,
                     "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return pd.DataFrame(rows)
