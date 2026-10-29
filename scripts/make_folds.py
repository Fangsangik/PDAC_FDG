"""Create the patient-level fold file ONCE. Every model reads this same file."""
from pathlib import Path

from fdgmip.cli import parse
from fdgmip.domain.cohort import load_cohort
from fdgmip.eval import make_folds, save_folds

if __name__ == "__main__":
    _, cfg = parse(__doc__)
    cohort = load_cohort(cfg.get("paths.clinical_table"), cfg.section("cohort"))
    f = cfg.section("folds")
    folds = make_folds(cohort, cfg.get("cohort.strata_columns"), f["n_splits"], f["n_repeats"], cfg.get("seed"))
    out = Path(f["file"])
    if out.exists():
        raise SystemExit(f"{out} exists; delete it explicitly if you really want new folds")
    save_folds(folds, out)
    print(f"wrote {out} ({folds.patient_id.nunique()} patients, {f['n_splits']} folds x {f['n_repeats']} repeats)")
