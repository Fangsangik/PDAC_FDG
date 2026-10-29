"""YAML config loading. No defaults live in code: every value must come from a YAML file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(KeyError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, data: dict[str, Any], sources: tuple[str, ...] = ()) -> None:
        self.data = data
        self.sources = sources

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not ...:
                    return default
                raise ConfigError(f"missing config key '{dotted}' (sources={list(self.sources)})")
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        sec = self.get(name)
        if not isinstance(sec, dict):
            raise ConfigError(f"config section '{name}' is not a mapping")
        return sec

    def hash(self, sections: list[str] | None = None, extra: Any = None) -> str:
        """Deterministic hash of selected sections (+ extra), for cache keys."""
        payload = {s: self.get(s) for s in sections} if sections else self.data
        blob = json.dumps({"cfg": payload, "extra": extra}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.data, sort_keys=False))


def load_config(*paths: str | Path) -> Config:
    """Later files override earlier ones (deep merge)."""
    data: dict[str, Any] = {}
    for p in paths:
        loaded = yaml.safe_load(Path(p).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        data = _deep_merge(data, loaded)
    return Config(data, tuple(str(p) for p in paths))
