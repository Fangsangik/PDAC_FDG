"""DICOM -> segmentation -> SUV -> MA-MIP for every patient in paths.series_selection."""
from pathlib import Path

from fdgmip.cli import parse
from fdgmip.pipeline import run_preprocess
from fdgmip.provenance import write_provenance

if __name__ == "__main__":
    _, cfg = parse(__doc__)
    write_provenance(Path(cfg.get("paths.derived_root")) / "manifests", cfg)
    run_preprocess(cfg)
