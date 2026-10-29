"""Multi-angle MIP projectors (registry)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from fdgmip.domain import Volume
from fdgmip.preprocess.crop import mask_centroid_index
from fdgmip.projection.rotate import AXIS_INDEX, rotate_about
from fdgmip.registry import PROJECTORS


class Projector(Protocol):
    def angles(self) -> list[float]: ...
    def project(self, vol: Volume, center_zyx, is_mask: bool) -> np.ndarray: ...   # (n_views, H, W)


@PROJECTORS.register("ma_mip")
class MultiAngleMIP:
    """Rotate about `rotation_axis`, take the max along `projection_axis` at each angle.

    angles = start_deg + k * span_deg / n_angles, k = 0..n_angles-1  (span 180 or 360 is a config choice:
    180 covers all projection lines but front/back views are mirror images, not identical, for a CNN).
    """
    def __init__(self, rotation_axis: str, projection_axis: str, start_deg: float, n_angles: int, span_deg: float):
        if rotation_axis == projection_axis:
            raise ValueError("rotation_axis and projection_axis must differ")
        if n_angles < 1:
            raise ValueError("n_angles must be >= 1")
        self.rotation_axis, self.projection_axis = rotation_axis, projection_axis
        self.start_deg, self.n_angles, self.span_deg = float(start_deg), int(n_angles), float(span_deg)

    def angles(self) -> list[float]:
        return [self.start_deg + k * self.span_deg / self.n_angles for k in range(self.n_angles)]

    def project(self, vol: Volume, center_zyx, is_mask: bool) -> np.ndarray:
        if len(set(np.round(vol.spacing, 6))) != 1:
            raise ValueError(f"MA-MIP requires isotropic voxels; got spacing {vol.spacing}")
        order = 0 if is_mask else 1
        proj_ax = AXIS_INDEX[self.projection_axis]
        views = [rotate_about(vol.array, a, self.rotation_axis, center_zyx, order=order).max(axis=proj_ax)
                 for a in self.angles()]
        return np.stack(views).astype(np.uint8 if is_mask else np.float32)


@PROJECTORS.register("two_view")
class TwoViewMIP(MultiAngleMIP):
    """Classic coronal + sagittal pair."""
    def __init__(self, rotation_axis: str, projection_axis: str):
        super().__init__(rotation_axis, projection_axis, start_deg=0.0, n_angles=2, span_deg=180.0)


def rotation_center(vol: Volume, masks: dict[str, Volume], center: str, organ: str = "pancreas") -> np.ndarray:
    if center == "crop_center":
        return (np.asarray(vol.shape, dtype=float) - 1) / 2
    if center == "mask_centroid":
        m = masks[organ]
        if not m.same_grid(vol):
            raise ValueError(f"{organ} mask is not on the crop grid")
        return mask_centroid_index(m)
    raise ValueError(f"unknown rotation center '{center}' (crop_center | mask_centroid)")
