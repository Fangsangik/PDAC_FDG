# fdgmip

Config-driven backbone for baseline FDG PET/CT multi-angle MIP models (see `docs/DESIGN.md`).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"      # add ".[seg]" for TotalSegmentator
.venv/bin/python -m pytest -q                                     # synthetic data only

cp configs/local.example.yaml configs/local.yaml                  # git-ignored; real paths go here
python scripts/list_series.py <dicom_root>/<patient_id>           # human picks CT/PET series -> series_selection.csv
python scripts/run_preprocess.py -c configs/default.yaml -c configs/local.yaml
python scripts/run_features.py   -c configs/default.yaml -c configs/local.yaml --manifest data/derived/manifests/manifest_<hash>.csv
python scripts/make_folds.py     -c configs/default.yaml -c configs/local.yaml   # once
python scripts/run_train_eval.py -c configs/default.yaml -c configs/local.yaml --features data/derived/features/<hash>/features.parquet
```

Experiments = another YAML passed as a later `-c` (deep-merged over `default.yaml`). Nothing tunable is in code.
