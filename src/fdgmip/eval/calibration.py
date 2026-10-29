from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


def calibration_curve_points(y, p, n_bins: int, strategy: str = "quantile") -> pd.DataFrame:
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_fraction": frac})


def calibration_slope_intercept(y, p, eps: float = 1e-6) -> dict[str, float]:
    """Cox recalibration: logit(y) ~ a + b * logit(p). Ideal: a=0, b=1."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(logit, np.asarray(y).astype(int))
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
            "intercept_in_the_large": float(np.log(np.mean(y) / (1 - np.mean(y))) - np.mean(np.log(p / (1 - p))))}
