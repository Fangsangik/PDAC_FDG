"""features + cohort + folds -> out-of-fold predictions, metrics, calibration, DCA, DeLong comparisons."""
import datetime as dt
from pathlib import Path

import pandas as pd

from fdgmip.cli import parse
from fdgmip.domain.cohort import load_cohort
from fdgmip.eval import load_folds
from fdgmip.pipeline import build_design_matrix, train_eval
from fdgmip.provenance import write_provenance

if __name__ == "__main__":
    args, cfg = parse(__doc__, add=lambda ap: (ap.add_argument("--features", required=True),
                                               ap.add_argument("--run-id", default=None)))
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg.get("paths.outputs_root")) / run_id
    write_provenance(out, cfg)
    cohort = load_cohort(cfg.get("paths.clinical_table"), cfg.section("cohort"))
    X, y = build_design_matrix(cfg, pd.read_parquet(args.features), cohort)
    train_eval(cfg, X, y, load_folds(cfg.get("folds.file")), out)
    print(f"results -> {out}")
