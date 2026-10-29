import numpy as np
import pandas as pd
import pytest

from fdgmip.models import TabularModel, select_columns
from fdgmip.registry import MODELS, import_all

import_all()


def _table(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "pet_suvmax": 4 + 2 * y + rng.normal(0, 1, n),
        "pet_mtv_ml": 10 + 5 * y + rng.normal(0, 3, n),
        "pet_tlg": rng.normal(50, 10, n),
        "clin_age": rng.normal(65, 8, n),
        "clin_sex": pd.Categorical(rng.choice(["M", "F"], n)),
        "clin_resectability": pd.Categorical(rng.choice(["BR", "LA"], n)),
        "clin_log_ca19_9": rng.normal(5, 1, n),
        "scanner": rng.choice(["A", "B"], n),
    }, index=[f"p{i}" for i in range(n)])
    for k in range(16):
        df[f"mip_{k:04d}"] = rng.normal(0, 1, n) + 0.5 * y
    df.loc[df.index[:3], "clin_age"] = np.nan  # missing values must be handled inside the pipeline
    return df, y


PRESETS = {
    "m0": ["pet_suvmax"],
    "m1": ["pet_suvmax", "pet_mtv_ml"],
    "m2": ["pet_suvmax", "pet_mtv_ml", "clin_*"],
    "m3": ["mip_*"],
    "m4": ["mip_*", "clin_*"],
}


@pytest.mark.parametrize("name,cols", PRESETS.items())
def test_every_model_has_same_interface(name, cols):
    X, y = _table()
    m = MODELS.build({"name": "tabular", "columns": cols, "estimator": {"name": "logistic", "C": 1.0, "max_iter": 500},
                      "impute": "median", "scale": "standard"})
    p = m.fit(X.iloc[:60], y[:60]).predict_proba(X.iloc[60:])
    assert p.shape == (20,) and ((0 <= p) & (p <= 1)).all()
    assert len(m.columns_) == len(select_columns(list(X.columns), cols))


def test_wildcard_selection_and_missing_pattern():
    X, _ = _table()
    assert select_columns(list(X.columns), ["mip_*"]) == [c for c in X.columns if c.startswith("mip_")]
    with pytest.raises(KeyError):
        select_columns(list(X.columns), ["nope_*"])


def test_fit_uses_train_fold_only():
    X, y = _table()
    m = TabularModel(["pet_*", "clin_*"], {"name": "logistic", "max_iter": 500}, "median", "standard")
    m.fit(X.iloc[:50], y[:50])
    scaler = m.pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    imputer = m.pipeline.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    mean_before, stat_before = scaler.mean_.copy(), imputer.statistics_.copy()
    num_cols = [c for c in m.columns_ if not str(X[c].dtype) == "category"]
    train = X.iloc[:50][num_cols]
    np.testing.assert_allclose(mean_before, train.fillna(train.median()).mean().to_numpy())  # stats == train-only stats
    p1 = m.predict_proba(X.iloc[50:])
    Xt = X.iloc[50:].copy()
    Xt[num_cols] = Xt[num_cols] * 1000                # wildly different test data
    p2 = m.predict_proba(Xt)
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(imputer.statistics_, stat_before)
    assert not np.allclose(p1, p2)


def test_model_stage_harmonization_is_fit_on_train_only(monkeypatch):
    from fdgmip.registry import HARMONIZERS
    calls = []

    @HARMONIZERS.register("spy")
    class Spy:
        def fit(self, X, batch):
            calls.append(("fit", len(X)))
            return self
        def transform(self, X, batch):
            calls.append(("transform", len(X)))
            return X

    X, y = _table()
    m = TabularModel(["pet_*"], {"name": "logistic"}, "median", "standard",
                     harmonization={"name": "spy", "stage": "model"}, batch_column="scanner")
    m.fit(X.iloc[:50], y[:50]).predict_proba(X.iloc[50:])
    assert calls == [("fit", 50), ("transform", 50), ("transform", 30)]
    HARMONIZERS._items.pop("spy")
