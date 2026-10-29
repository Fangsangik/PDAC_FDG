from __future__ import annotations

import numpy as np

from fdgmip.domain import Volume


def mask_out_organs(vol: Volume, masks: dict[str, Volume], organs: list[str], fill_value: float) -> Volume:
    """Replace voxels inside the listed organ masks with fill_value (e.g. to suppress renal/gastric uptake)."""
    arr = vol.array.copy()
    for o in organs:
        m = masks[o]
        if not m.same_grid(vol):
            raise ValueError(f"mask '{o}' is not on the volume grid")
        arr[m.array > 0] = fill_value
    return vol.replace(array=arr)
