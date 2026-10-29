import numpy as np
import pandas as pd

from fdgmip.eval import load_folds, make_folds, save_folds


def _cohort(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"edf_6mo": rng.integers(0, 2, n), "resectability": rng.choice(["BR", "LA"], n)},
                        index=pd.Index([f"p{i:03d}" for i in range(n)], name="patient_id"))


def test_each_patient_once_per_repeat_and_stratified():
    c = _cohort()
    f = make_folds(c, ["edf_6mo", "resectability"], n_splits=5, n_repeats=2, seed=0)
    for r, g in f.groupby("repeat"):
        assert sorted(g.patient_id) == sorted(c.index)
        assert g.patient_id.is_unique
        joint = c.loc[g.patient_id, "edf_6mo"].astype(str).values + c.loc[g.patient_id, "resectability"].values
        per_fold = pd.crosstab(g.fold.values, joint)
        assert (per_fold.max() - per_fold.min() <= 1).all()   # balanced within +-1 per stratum


def test_folds_file_is_shared_and_views_stay_together(tmp_path):
    c = _cohort()
    f = make_folds(c, ["edf_6mo"], 4, 1, seed=1)
    save_folds(f, tmp_path / "folds.csv")
    a, b = load_folds(tmp_path / "folds.csv"), load_folds(tmp_path / "folds.csv")   # "two models" read it
    pd.testing.assert_frame_equal(a, b)
    # view-level table: 12 views per patient -> fold is a function of patient_id only
    views = pd.DataFrame({"patient_id": np.repeat(c.index, 12), "view": np.tile(range(12), len(c))})
    merged = views.merge(a, on="patient_id")
    assert (merged.groupby("patient_id").fold.nunique() == 1).all()
    assert len(merged) == len(views)


def test_load_rejects_duplicate_patient(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"patient_id": ["a", "a"], "repeat": [0, 0], "fold": [0, 1]}).to_csv(p, index=False)
    import pytest
    with pytest.raises(ValueError):
        load_folds(p)
