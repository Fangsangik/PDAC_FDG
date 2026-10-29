"""Record everything needed to rerun: full config, seed, git commit, package versions."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from fdgmip.config import Config


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None  # not a git repo (yet)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def write_provenance(out_dir: str | Path, cfg: Config) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(out / "config.yaml")
    info = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": cfg.get("seed"),
        "git_commit": git_commit(),
        "config_sources": list(cfg.sources),
        "config_hash": cfg.hash(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in md.distributions()},
    }
    (out / "provenance.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    return out
