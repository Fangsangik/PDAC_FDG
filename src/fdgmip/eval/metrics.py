from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def auroc(y, p) -> float:
    return float(roc_auc_score(y, p))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def auroc_bootstrap_ci(y, p, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return tuple(float(v) for v in np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# ---- DeLong (Sun & Xu 2014 fast algorithm) ---------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def _delong(y: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """preds: (k, n). Returns (aucs (k,), covariance (k,k))."""
    order = np.argsort(-y)
    y, preds = y[order], preds[:, order]
    m = int(y.sum())
    n = len(y) - m
    k = preds.shape[0]
    tx = np.array([_midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2) -> dict[str, float]:
    """Paired DeLong test for AUROC(p1) vs AUROC(p2) on the same patients."""
    y = np.asarray(y).astype(int)
    aucs, cov = _delong(y, np.vstack([np.asarray(p1, float), np.asarray(p2, float)]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}
