"""In-plane rotation of a 3D array about an arbitrary centre. Internal helper: operates on the
isotropically resampled crop, where voxel == physical unit, so geometry is not carried here."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

AXIS_INDEX = {"z": 0, "y": 1, "x": 2}


def rotate_about(arr: np.ndarray, angle_deg: float, axis: str, center_zyx, order: int, cval: float = 0.0) -> np.ndarray:
    """Rotate `arr` by angle_deg around the line parallel to `axis` passing through center_zyx.

    order=1 for images, order=0 for masks. Output has the input shape (corners outside are cval).
    """
    ax = AXIS_INDEX[axis]
    plane = [i for i in range(3) if i != ax]
    t = np.deg2rad(angle_deg)
    rot = np.eye(3)
    rot[np.ix_(plane, plane)] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    c = np.asarray(center_zyx, dtype=float)
    # affine_transform maps output coords o -> input coords M@o + offset
    return ndimage.affine_transform(arr, rot, offset=c - rot @ c, order=order, cval=cval, prefilter=order > 1)
