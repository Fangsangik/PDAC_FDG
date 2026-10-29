"""Conventional PET metrics inside a VOI: SUVmax, MTV, TLG.

The VOI is whatever mask is passed in (currently the TotalSegmentator pancreas mask).
TODO(clinical): decide whether the VOI should be the organ mask or a lesion VOI; organ masks include
normal parenchyma and may miss extra-pancreatic tumour extension.
"""
from __future__ import annotations

import numpy as np

from fdgmip.domain import Volume


def mtv_threshold_value(suv_in_voi: np.ndarray, spec: dict, liver_mean: float | None = None) -> float:
    """spec = {type: suv_abs, value: 2.5} | {type: pct_max, value: 0.4} | {type: liver_relative, ...}"""
    t = spec["type"]
    if t == "suv_abs":
        return float(spec["value"])
    if t == "pct_max":
        return float(spec["value"]) * float(suv_in_voi.max())
    if t == "liver_relative":
        # TODO: PERCIST-style 1.5 * liver mean + 2 SD needs liver SD; add once liver stats are plumbed.
        raise NotImplementedError("liver_relative MTV threshold not implemented")
    raise ValueError(f"unknown MTV threshold type '{t}'")


def pet_stats(suv: Volume, voi: Volume, mtv_threshold: dict) -> dict[str, float]:
    if not suv.same_grid(voi):
        raise ValueError("VOI mask is not on the SUV grid")
    inside = voi.array > 0
    if not inside.any():
        raise ValueError("empty VOI")
    vals = suv.array[inside]
    voxel_ml = float(np.prod(suv.spacing)) / 1000.0
    thr = mtv_threshold_value(vals, mtv_threshold)
    above = vals >= thr
    mtv = float(above.sum()) * voxel_ml
    return {
        "suvmax": float(vals.max()),
        "suvmean_voi": float(vals.mean()),
        "mtv_ml": mtv,
        "tlg": mtv * (float(vals[above].mean()) if above.any() else 0.0),
        "mtv_threshold_suv": float(thr),
    }
