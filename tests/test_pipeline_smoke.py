"""End-to-end on synthetic DICOM: preprocess -> features -> folds -> train/eval. No real data, no download."""
import json
import stat
import sys

import numpy as np
import pandas as pd
import yaml

from fdgmip.config import load_config
from fdgmip.domain.cohort import load_cohort
from fdgmip.eval import load_folds, make_folds, save_folds
from fdgmip.pipeline import build_design_matrix, run_features, run_preprocess, train_eval
from fdgmip.provenance import write_provenance
from conftest import standard_pet_tags, write_dicom_series

N = 12
FAKE_SEG = '''#!{py}
import sys, pathlib, SimpleITK as sitk
a = sys.argv; img = sitk.ReadImage(a[a.index("-i")+1]); o = pathlib.Path(a[a.index("-o")+1]); o.mkdir()
for organ in a[a.index("--roi_subset")+1:]:
    m = sitk.Image(img.GetSize(), sitk.sitkUInt8); m.CopyInformation(img)
    if organ == "pancreas":
        for z in range(1, 5):
            for y in range(4, 12):
                for x in range(4, 12): m[x, y, z] = 1
    else:
        for z in range(0, 2):
            for y in range(0, 4):
                for x in range(0, 4): m[x, y, z] = 1
    sitk.WriteImage(m, str(o / f"{{organ}}.nii.gz"))
'''


def _make_dataset(root):
    rng = np.random.default_rng(0)
    sel, clin = [], []
    for i in range(N):
        pid = f"P{i:03d}"
        for_uid = f"1.2.3.{i}"
        y = i % 2
        ct_uid = write_dicom_series(root / pid / "ct", "CT", 6, spacing_xy=(2.0, 2.0), thickness=3.0,
                                    for_uid=for_uid, description="CE CT")
        pet_uid = write_dicom_series(root / pid / "pt", "PT", 4, rows=8, cols=8, spacing_xy=(4.0, 4.0), thickness=4.5,
                                     origin=(0.0, 0.0, 0.0), for_uid=for_uid, description="PET AC",
                                     pixel_value=int(3000 + 2000 * y + rng.integers(0, 500)), pet_tags=standard_pet_tags())
        sel.append({"patient_id": pid, "ct_series_uid": ct_uid, "pet_series_uid": pet_uid})
        clin.append({"patient_id": pid, "edf_6mo": y, "resectability": ["BR", "LA"][(i // 2) % 2],
                     "age": 60 + i, "sex": "MF"[i % 2], "log_ca19_9": 4.0 + rng.normal(), "ecog": i % 2,
                     "chemo_regimen": ["FOLFIRINOX", "GnP"][i % 2], "scanner": "A"})
    pd.DataFrame(sel).to_csv(root / "selection.csv", index=False)
    pd.DataFrame(clin).to_csv(root / "clinical.csv", index=False)


def test_end_to_end(tmp_path):
    root = tmp_path / "dicom"
    _make_dataset(root)
    exe = tmp_path / "FakeTS"
    exe.write_text(FAKE_SEG.format(py=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    override = {
        "paths": {"dicom_root": str(root), "series_selection": str(root / "selection.csv"),
                  "clinical_table": str(root / "clinical.csv"), "derived_root": str(tmp_path / "derived"),
                  "outputs_root": str(tmp_path / "outputs")},
        "cohort": {"extra_columns": ["scanner"]},
        "preprocess": {"spacing_mm": [4.0, 4.0, 4.0]},
        "segmentation": {"segmenter": {"name": "totalsegmentator", "task": "total", "executable": str(exe)},
                         "organs": ["pancreas", "liver"]},
        "crop": {"strategy": {"name": "pancreas_centroid", "size_mm": [40, 40, 32], "square_xy": True}},
        "projection": {"projector": {"name": "ma_mip", "rotation_axis": "z", "projection_axis": "y",
                                     "start_deg": 0, "n_angles": 3, "span_deg": 180},
                       "channels": [{"type": "image"}, {"type": "mask", "organ": "pancreas"}]},
        "features": {"backbone": {"name": "torchvision", "arch": "resnet18", "weights": None, "input_size": 32,
                                  "channel_adapter": "first3", "intensity_max": 10.0}},
        "folds": {"n_splits": 2, "n_repeats": 1, "file": str(tmp_path / "folds.csv")},
        "models": {"m0": {"columns": ["pet_suvmax"], "estimator": {"name": "logistic"}, "impute": "median", "scale": "standard"},
                   "m2": {"columns": ["pet_suvmax", "clin_*"], "estimator": {"name": "logistic"}, "impute": "median", "scale": "standard"},
                   "m3": {"columns": ["mip_*"], "estimator": {"name": "logistic", "C": 0.1}, "impute": "median", "scale": "standard"}},
        "eval": {"bootstrap_n": 20, "compare": {"reference": "m2", "candidates": ["m3"]}},
    }
    # first3 needs C>=3 -> add a third channel
    override["projection"]["channels"].append({"type": "image", "restrict_to": "pancreas"})
    (tmp_path / "override.yaml").write_text(yaml.safe_dump(override))
    cfg = load_config("configs/default.yaml", tmp_path / "override.yaml")

    # 1. preprocess (twice: second run must hit the cache and give identical manifest)
    m1 = run_preprocess(cfg)
    assert m1.excluded.isna().all(), m1.excluded.dropna().tolist()
    views = np.load(m1.mip_path[0])
    assert views.shape == (3, 3, 8, 10)   # (n_views, C, z, x): 32mm/4mm, 40mm/4mm; square_xy keeps y == x
    assert (m1.pet_suvmax > 0).all()
    m2 = run_preprocess(cfg)
    pd.testing.assert_frame_equal(m1, m2)
    manifest_path = next((tmp_path / "derived" / "manifests").glob("manifest_*.csv"))

    # 2. features
    feat_path = run_features(cfg, manifest_path)
    feats = pd.read_parquet(feat_path)
    assert len(feats) == N and sum(c.startswith("mip_") for c in feats.columns) == 512

    # 3. folds (one file, shared) + 4. train/eval
    cohort = load_cohort(cfg.get("paths.clinical_table"), cfg.section("cohort"))
    save_folds(make_folds(cohort, cfg.get("cohort.strata_columns"), 2, 1, 0), cfg.get("folds.file"))
    X, y = build_design_matrix(cfg, feats, cohort)
    assert "scanner" in X.columns and "clin_age" in X.columns and "edf_6mo" not in X.columns
    out = tmp_path / "outputs" / "run"
    write_provenance(out, cfg)
    preds = train_eval(cfg, X, y, load_folds(cfg.get("folds.file")), out)
    assert set(preds.model) == {"m0", "m1", "m2", "m3", "m4"}   # default models deep-merged with overrides
    assert preds.groupby("model").patient_id.nunique().eq(N).all()     # every patient predicted out-of-fold once
    for f in ("preds.csv", "metrics.csv", "decision_curve.csv", "comparison.csv", "config.yaml", "provenance.json"):
        assert (out / f).exists()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["seed"] == 0 and "torch" in prov["packages"]
    # each patient's test fold in preds equals the fold file (no leakage across the shared file)
    merged = preds.merge(load_folds(cfg.get("folds.file")), on=["patient_id", "repeat"], suffixes=("", "_file"))
    assert (merged.fold == merged.fold_file).all()
