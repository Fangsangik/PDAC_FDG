from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk

from fdgmip.domain import Volume

_INTERP = {"linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor, "bspline": sitk.sitkBSpline}


def resample(vol: Volume, spacing: tuple[float, float, float], interpolation: str, fill_value: float = 0.0) -> Volume:
    """Resample onto a grid with the given (x,y,z) spacing, same origin/direction, covering the same extent.

    Images: 'linear'. Masks: 'nearest' (anything else silently fakes label values).
    """
    if interpolation not in _INTERP:
        raise ValueError(f"interpolation must be one of {list(_INTERP)}")
    if vol.modality == "MASK" and interpolation != "nearest":
        raise ValueError("masks must be resampled with interpolation='nearest'")
    img = vol.to_sitk()
    old_size = np.asarray(img.GetSize(), dtype=float)
    new_size = [int(math.ceil(s * o / n)) for s, o, n in zip(old_size, img.GetSpacing(), spacing)]
    out = sitk.Resample(img, new_size, sitk.Transform(), _INTERP[interpolation],
                        img.GetOrigin(), tuple(float(s) for s in spacing), img.GetDirection(),
                        fill_value, img.GetPixelID())
    return Volume.from_sitk(out, frame_of_reference_uid=vol.frame_of_reference_uid,
                            modality=vol.modality, meta=vol.meta)
