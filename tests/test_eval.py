import numpy as np
import pandas as pd
from scipy import stats

from fdgmip.eval import (auroc, auroc_bootstrap_ci, calibration_curve_points, calibration_slope_intercept,
                         compare_models, decision_curve, delong_test, summarize)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p_good = 1 / (1 + np.exp(-(2 * y - 1 + rng.normal(0, 1, n))))
    p_noise = rng.random(n)
    return y, p_good, p_noise


def test_auroc_and_ci():
    y, pg, pn = _data()
    assert auroc(y, y) == 1.0
    lo, hi = auroc_bootstrap_ci(y, pg, 200, seed=0)
    assert lo < auroc(y, pg) < hi


def test_delong_identical_and_different():
    y, pg, pn = _data()
    assert delong_test(y, pg, pg)["p_value"] == 1.0
    r = delong_test(y, pg, pn)
    assert r["p_value"] < 0.01 and r["diff"] > 0
    # matches a hand DeLong-free sanity check: AUCs agree with sklearn
    np.testing.assert_allclose(r["auc1"], auroc(y, pg))


def test_calibration_of_well_calibrated_probs():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    c = calibration_slope_intercept(y, p)
    assert abs(c["slope"] - 1) < 0.1 and abs(c["intercept"]) < 0.1
    pts = calibration_curve_points(y, p, n_bins=5)
    assert len(pts) == 5 and (abs(pts.mean_predicted - pts.observed_fraction) < 0.1).all()


def test_decision_curve_limits():
    y, pg, _ = _data()
    d = decision_curve(y, pg, [0.01, 0.5])
    assert abs(d.treat_all.iloc[0] - (y.mean() - (1 - y.mean()) * 0.01 / 0.99)) < 1e-9
    assert (d.treat_none == 0).all()


def test_summarize_and_compare():
    y, pg, pn = _data(n=100)
    pid = [f"p{i}" for i in range(100)]
    preds = pd.concat([
        pd.DataFrame({"patient_id": pid, "repeat": 0, "fold": 0, "model": "m2", "y": y, "p": pn}),
        pd.DataFrame({"patient_id": pid, "repeat": 0, "fold": 0, "model": "m3", "y": y, "p": pg}),
    ])
    metrics, dca = summarize(preds, n_boot=50, seed=0, dca_thresholds=np.arange(0.05, 0.5, 0.05))
    assert set(metrics.model) == {"m2", "m3"} and "auroc" in set(metrics.metric)
    assert set(dca.model) == {"m2", "m3"}
    cmp = compare_models(preds, reference="m2", candidates=["m3"])
    assert cmp.iloc[0]["p_value"] < 0.05 and cmp.iloc[0]["diff"] > 0
