# fdgmip backbone — design (approved 2026-10-28)

Goal: swappable-parts skeleton for predicting 6-month early distant failure (EDF) in BR/LA PDAC from
baseline FDG PET/CT via multi-angle MIP (MA-MIP) + frozen CNN features, compared against SUV/clinical
logistic baselines (M0–M4). Not a finished study: every experimental knob is YAML.

## Decisions taken
- Package `fdgmip` under `src/` (src-layout, `pip install -e ".[dev]"`); Python 3.12 venv.
- Registry + config, not full DDD. From DDD only: explicit domain objects (`Volume`, `Series`, `Study`,
  `Patient`) and IO/logic separation (`io/` adapters vs `preprocess/`…`models/`).
- `Volume` always carries spacing/origin/direction/FrameOfReferenceUID. SimpleITK conventions:
  array (z,y,x), spacing/origin (x,y,z).
- One model class (`models/tabular.py`) for M0–M4: column glob patterns → impute/scale/one-hot →
  [harmonizer] → estimator (registry). Model presets are the `models:` mapping in the YAML, so one run
  yields all models + DeLong comparisons from a single `preds.csv`.
- MIP span default 180° (config); 360° to be decided after first results.
- TotalSegmentator: optional extra dependency + subprocess adapter only (`--roi_subset`, per-organ files).
- SUV: standard DICOM tag path only; vendor branches are `TODO` in `preprocess/suv.py` (scanner unknown).
- git not initialised yet; provenance records `git_commit: null` until it is.
- PANORAMA segmentation validation: deferred (not in this skeleton).

## Data flow / cache (file system as state machine)
```
data/derived/<patient_id>/
  seg/<hash(segmentation, orientation, ct_uid)>/<organ>.nii.gz
  suv/<hash(orientation, spacing, scan_time_tag, pet_uid)>/suv.nii.gz
  mip/<hash(preprocess, segmentation, crop, organ_masking, projection, uids)>/views.npy  (n_views, C, H, W)
data/derived/manifests/manifest_<hash>.csv      patient_id, excluded, mip_path, pet_*
data/derived/features/<hash>/features.parquet   patient_id, pet_*, mip_*
data/derived/folds/folds.csv                    patient_id, repeat, fold   (generated once, shared)
outputs/<run_id>/{config.yaml, provenance.json, preds.csv, metrics.csv, decision_curve.csv, comparison.csv}
```

## Leakage guards (enforced in code + tests)
- `domain/cohort.py`: allowlist columns only; headers matching `cohort.forbidden_patterns` raise `LeakageError`.
- All learned steps live in the sklearn Pipeline / Harmonizer and are fit per fold on train ids only
  (`test_fit_uses_train_fold_only`, `test_model_stage_harmonization_is_fit_on_train_only`).
- Folds are patient-level, stratified on label × resectability, written once and read by every model
  (`test_folds_file_is_shared_and_views_stay_together`, smoke test checks preds fold == fold file).
- `transfer_mask` refuses CT→PET transfer without an identical FrameOfReferenceUID.
- Segmentation masks only reach crop/masking/channels; labels only come from the cohort table.

## Interfaces (Protocols) and registries
| registry     | contract |
|--------------|----------|
| SEGMENTERS   | `segment(ct, organs) -> {organ: mask Volume on CT grid}` |
| CROPS        | `box(vol, masks) -> BoundingBox` (pancreas_centroid, vertebral_level, whole_torso) |
| PROJECTORS   | `project(vol, center, is_mask) -> (n_views, H, W)` (ma_mip, two_view) |
| BACKBONES    | `extract((n_views, C, H, W)) -> (n_views, D)` (torchvision frozen, foundation_stub) |
| AGGREGATORS  | `aggregate((n_views, D)) -> (D',)` (mean, max, mean_max, concat) |
| HARMONIZERS  | `fit(X, batch)`, `transform(X, batch)` (none; combat/scn stubs) |
| MODELS       | `fit(X, y)`, `predict_proba(X) -> (n,)` (tabular) |

## Open TODOs needing a human decision (grep `TODO`)
- inclusion thresholds, resample spacing, crop extent, SUV clip (`intensity_max`), MTV threshold
- VOI definition for SUVmax/MTV/TLG (organ mask vs lesion VOI)
- vendor SUV tag paths; ComBat wrapper; foundation-model backbone; 180° vs 360°
