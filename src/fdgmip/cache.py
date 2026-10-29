"""File-system cache: derived_root/<patient_id>/<stage>/<key>/..., key = hash of the relevant config
sections + input identity (series UIDs). Changing a relevant config value changes the key."""
from __future__ import annotations

from pathlib import Path


class Cache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def dir(self, patient_id: str, stage: str, key: str) -> Path:
        d = self.root / patient_id / stage / key
        d.mkdir(parents=True, exist_ok=True)
        return d
