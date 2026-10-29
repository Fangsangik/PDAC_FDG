"""Crop strategies (registry) + the crop operation itself. All boxes are in the target (PET) grid."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from fdgmip.domain import BoundingBox, Volume
from fdgmip.registry import CROPS


class CropStrategy(Protocol):
    def box(self, vol: Volume, masks: dict[str, Volume]) -> BoundingBox: ...


def mask_centroid_index(mask: Volume) -> np.ndarray:
    idx = np.argwhere(mask.array > 0)
    if len(idx) == 0:
        raise ValueError("empty mask; cannot compute centroid")
    return idx.mean(axis=0)  # (z,y,x) continuous index


def _box_around(center_zyx: np.ndarray, size_mm: tuple[float, float, float], spacing_xyz, square_xy: bool) -> BoundingBox:
    sz_vox = np.asarray(size_mm, dtype=float)[::-1] / np.asarray(spacing_xyz, dtype=float)[::-1]  # (z,y,x)
    if square_xy:
        sz_vox[1] = sz_vox[2] = max(sz_vox[1], sz_vox[2])
    half = np.ceil(sz_vox / 2).astype(int)
    c = np.round(center_zyx).astype(int)
    return BoundingBox(tuple(int(v) for v in c - half), tuple(int(v) for v in c + half))


@CROPS.register("pancreas_centroid")
class PancreasCentroidCrop:
    """Fixed physical-size box centred on the pancreas mask centroid.

    square_xy=True makes the in-plane box square so a rotated MIP around z never clips corners.
    """
    def __init__(self, size_mm: tuple[float, float, float], square_xy: bool, organ: str = "pancreas"):
        self.size_mm, self.square_xy, self.organ = tuple(size_mm), square_xy, organ

    def box(self, vol: Volume, masks: dict[str, Volume]) -> BoundingBox:
        m = masks[self.organ]
        if not m.same_grid(vol):
            raise ValueError(f"{self.organ} mask not on target grid")
        return _box_around(mask_centroid_index(m), self.size_mm, vol.spacing, self.square_xy)


@CROPS.register("vertebral_level")
class VertebralLevelCrop:
    """z-range from two vertebra masks (e.g. vertebrae_T10 .. vertebrae_L3), xy box of xy_size_mm centred on pancreas."""
    def __init__(self, top_organ: str, bottom_organ: str, xy_size_mm: float, square_xy: bool,
                 center_organ: str = "pancreas"):
        self.top, self.bottom, self.xy_size_mm, self.square_xy, self.center_organ = \
            top_organ, bottom_organ, xy_size_mm, square_xy, center_organ

    def box(self, vol: Volume, masks: dict[str, Volume]) -> BoundingBox:
        zs = np.concatenate([np.argwhere(masks[o].array > 0)[:, 0] for o in (self.top, self.bottom)])
        c = mask_centroid_index(masks[self.center_organ])
        b = _box_around(c, (self.xy_size_mm, self.xy_size_mm, 1.0), vol.spacing, self.square_xy)
        return BoundingBox((int(zs.min()), b.lo[1], b.lo[2]), (int(zs.max()) + 1, b.hi[1], b.hi[2]))


@CROPS.register("whole_torso")
class WholeTorsoCrop:
    def __init__(self, square_xy: bool):
        self.square_xy = square_xy

    def box(self, vol: Volume, masks: dict[str, Volume]) -> BoundingBox:
        z, y, x = vol.shape
        if self.square_xy:
            s = max(y, x)
            cy, cx = (y - 1) / 2, (x - 1) / 2
            return BoundingBox((0, int(round(cy - s / 2)), int(round(cx - s / 2))),
                               (z, int(round(cy - s / 2)) + s, int(round(cx - s / 2)) + s))
        return BoundingBox((0, 0, 0), (z, y, x))


def crop(vol: Volume, box: BoundingBox, fill_value: float = 0.0) -> Volume:
    """Extract `box`; regions outside the volume are padded with fill_value. Origin is updated."""
    clipped = box.clip(vol.shape)
    sub = vol.array[clipped.slices]
    pad = [(c_lo - b_lo, b_hi - c_hi) for b_lo, b_hi, c_lo, c_hi in zip(box.lo, box.hi, clipped.lo, clipped.hi)]
    if any(p != (0, 0) for p in pad):
        sub = np.pad(sub, pad, mode="constant", constant_values=fill_value)
    origin = tuple(float(v) for v in vol.index_to_physical(box.lo))
    return vol.replace(array=sub, origin=origin)
