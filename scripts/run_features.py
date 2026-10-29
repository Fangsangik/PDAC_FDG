"""MIP views -> frozen backbone features -> features.parquet"""
from pathlib import Path

from fdgmip.cli import parse
from fdgmip.pipeline import run_features
from fdgmip.provenance import write_provenance

if __name__ == "__main__":
    args, cfg = parse(__doc__, add=lambda ap: ap.add_argument("--manifest", required=True))
    out = run_features(cfg, args.manifest)
    write_provenance(out.parent, cfg)
