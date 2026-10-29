"""Volume: a 3D array that always carries its physical geometry.

Conventions (SimpleITK):
  array index order  : (z, y, x)
  spacing / origin   : physical order (x, y, z), millimetres
  direction          : 3x3 row-major, flattened to 9 floats
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import SimpleITK as sitk

Vec3 = tuple[float, float, float]
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Volume:
    array: np.ndarray
    spacing: Vec3
    origin: Vec3
    direction: tuple[float, ...] = IDENTITY
    frame_of_reference_uid: str | None = None
    modality: str = "UNKNOWN"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"Volume.array must be 3D (z,y,x); got shape {self.array.shape}")
        if len(self.spacing) != 3 or len(self.origin) != 3 or len(self.direction) != 9:
            raise ValueError("spacing/origin need 3 values, direction needs 9")
        object.__setattr__(self, "spacing", tuple(float(v) for v in self.spacing))
        object.__setattr__(self, "origin", tuple(float(v) for v in self.origin))
        object.__setattr__(self, "direction", tuple(float(v) for v in self.direction))

    # ---- geometry -------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=float).reshape(3, 3)

    def index_to_physical(self, index_zyx: Any) -> np.ndarray:
        """Continuous (z,y,x) index -> physical (x,y,z) point."""
        idx = np.asarray(index_zyx, dtype=float)[..., ::-1]  # -> xyz
        return np.asarray(self.origin) + (self.direction_matrix @ (idx * np.asarray(self.spacing)).T).T

    def physical_to_index(self, point_xyz: Any) -> np.ndarray:
        """Physical (x,y,z) point -> continuous (z,y,x) index."""
        p = np.asarray(point_xyz, dtype=float) - np.asarray(self.origin)
        idx_xyz = (np.linalg.inv(self.direction_matrix) @ p.T).T / np.asarray(self.spacing)
        return idx_xyz[..., ::-1]

    def same_grid(self, other: "Volume", atol: float = 1e-3) -> bool:
        return (
            self.shape == other.shape
            and np.allclose(self.spacing, other.spacing, atol=atol)
            and np.allclose(self.origin, other.origin, atol=atol)
            and np.allclose(self.direction, other.direction, atol=atol)
        )

    # ---- conversions ----------------------------------------------------
    def to_sitk(self) -> sitk.Image:
        img = sitk.GetImageFromArray(self.array)
        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction)
        return img

    @classmethod
    def from_sitk(cls, img: sitk.Image, **kwargs: Any) -> "Volume":
        return cls(
            array=sitk.GetArrayFromImage(img),
            spacing=tuple(img.GetSpacing()),
            origin=tuple(img.GetOrigin()),
            direction=tuple(img.GetDirection()),
            **kwargs,
        )

    def replace(self, **changes: Any) -> "Volume":
        return dataclasses.replace(self, **changes)

    def with_array(self, array: np.ndarray) -> "Volume":
        """Same geometry, new voxel values. Shape must match."""
        if array.shape != self.array.shape:
            raise ValueError(f"shape mismatch {array.shape} vs {self.array.shape}; geometry would be wrong")
        return self.replace(array=array)


@dataclass(frozen=True)
class BoundingBox:
    """Half-open index box in (z, y, x)."""
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(h <= l for l, h in zip(self.lo, self.hi)):
            raise ValueError(f"empty box lo={self.lo} hi={self.hi}")

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(l, h) for l, h in zip(self.lo, self.hi))  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(h - l for l, h in zip(self.lo, self.hi))  # type: ignore[return-value]

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((l + h - 1) / 2 for l, h in zip(self.lo, self.hi))  # type: ignore[return-value]

    def clip(self, shape: tuple[int, int, int]) -> "BoundingBox":
        return BoundingBox(
            tuple(max(0, l) for l in self.lo),  # type: ignore[arg-type]
            tuple(min(s, h) for s, h in zip(shape, self.hi)),  # type: ignore[arg-type]
        )
