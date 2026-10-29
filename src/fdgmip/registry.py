"""Minimal name -> class registry. Every swappable part registers itself here."""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = cls
            return cls
        return deco

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {sorted(self._items)}") from None

    def build(self, spec: dict[str, Any], **extra: Any) -> T:
        """spec = {"name": <registered name>, **constructor kwargs}."""
        spec = dict(spec)
        name = spec.pop("name")
        return self.get(name)(**spec, **extra)

    def names(self) -> list[str]:
        return sorted(self._items)


SEGMENTERS: Registry = Registry("segmenter")
CROPS: Registry = Registry("crop strategy")
PROJECTORS: Registry = Registry("projector")
BACKBONES: Registry = Registry("backbone")
AGGREGATORS: Registry = Registry("aggregator")
HARMONIZERS: Registry = Registry("harmonizer")
ESTIMATORS: Registry = Registry("estimator")
MODELS: Registry = Registry("model")


def import_all() -> None:
    """Import every module that registers parts, so registries are populated."""
    import importlib
    for mod in (
        "fdgmip.segmentation.totalseg",
        "fdgmip.preprocess.crop",
        "fdgmip.projection.mip",
        "fdgmip.features.backbone",
        "fdgmip.features.aggregate",
        "fdgmip.harmonization.none",
        "fdgmip.harmonization.combat",
        "fdgmip.harmonization.scn",
        "fdgmip.models.tabular",
    ):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:  # pragma: no cover - only during staged build-out
            if e.name and e.name.startswith("fdgmip"):
                continue
            raise
