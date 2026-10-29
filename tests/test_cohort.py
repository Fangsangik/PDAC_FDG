import pandas as pd
import pytest

from fdgmip.domain.cohort import LeakageError, load_cohort

CFG = {
    "id_column": "patient_id",
    "label_column": "edf_6mo",
    "strata_columns": ["edf_6mo", "resectability"],
    "clinical_columns": ["age", "sex", "resectability"],
    "categorical": ["sex", "resectability"],
    "forbidden_patterns": [r"followup", r"post_", r"_m6$", r"response"],
}


def _write(tmp_path, df):
    p = tmp_path / "c.csv"
    df.to_csv(p, index=False)
    return p


def test_allowlist_drops_unlisted_columns(tmp_path):
    p = _write(tmp_path, pd.DataFrame({
        "patient_id": ["a", "b"], "edf_6mo": [0, 1], "age": [60, 70], "sex": ["M", "F"],
        "resectability": ["BR", "LA"], "notes": ["x", "y"]}))
    df = load_cohort(p, CFG)
    assert "notes" not in df.columns
    assert list(df.index) == ["a", "b"]
    assert str(df["sex"].dtype) == "category"


def test_forbidden_column_raises(tmp_path):
    p = _write(tmp_path, pd.DataFrame({
        "patient_id": ["a"], "edf_6mo": [0], "age": [60], "sex": ["M"], "resectability": ["BR"],
        "ca199_m6": [12.0]}))
    with pytest.raises(LeakageError):
        load_cohort(p, CFG)


def test_duplicate_id_raises(tmp_path):
    p = _write(tmp_path, pd.DataFrame({
        "patient_id": ["a", "a"], "edf_6mo": [0, 1], "age": [1, 2], "sex": ["M", "F"], "resectability": ["BR", "BR"]}))
    with pytest.raises(ValueError):
        load_cohort(p, CFG)
