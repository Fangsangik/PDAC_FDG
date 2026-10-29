"""SUV recomputed from raw DICOM tags. Never trusts a vendor-precomputed SUV.

SUV_bw = C(t_scan) / ( D0 * 2^(-(t_scan - t_inj)/T_half) / weight_g )
  C(t_scan): voxel activity concentration [Bq/ml] as stored (already decay-corrected to scan start
             when DecayCorrection == 'START', the common case)
  D0       : injected dose [Bq]
Reference: QIBA FDG-PET SUV profile / Kinahan & Fletcher 2010.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from fdgmip.domain import Volume

log = logging.getLogger(__name__)


class SUVError(ValueError):
    pass


@dataclass(frozen=True)
class SUVParams:
    dose_bq: float
    injection_time: dt.datetime
    scan_time: dt.datetime
    half_life_s: float
    weight_kg: float
    units: str
    decay_correction: str


def _parse_dt(date: str, time: str) -> dt.datetime:
    time = str(time).split(".")[0].ljust(6, "0")
    return dt.datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S")


def _require(header: Any, name: str) -> Any:
    v = getattr(header, name, None)
    if v in (None, ""):
        raise SUVError(f"DICOM tag '{name}' missing or empty; cannot recompute SUV")
    return v


def extract_params_standard(header: Any, scan_time_tag: str = "Series") -> SUVParams:
    """Standard (DICOM PS3.3 C.8.9.2) tag path. scan_time_tag: 'Series' or 'Acquisition'."""
    seq = _require(header, "RadiopharmaceuticalInformationSequence")
    rp = seq[0]
    date = str(_require(header, f"{scan_time_tag}Date"))
    scan_time = _parse_dt(date, _require(header, f"{scan_time_tag}Time"))
    inj_raw = getattr(rp, "RadiopharmaceuticalStartDateTime", None)
    if inj_raw:
        injection = _parse_dt(str(inj_raw)[:8], str(inj_raw)[8:])
    else:
        injection = _parse_dt(date, _require(rp, "RadiopharmaceuticalStartTime"))
    if scan_time < injection:  # midnight crossing between injection and scan
        log.warning("scan time %s before injection %s; assuming next day", scan_time, injection)
        scan_time += dt.timedelta(days=1)
    return SUVParams(
        dose_bq=float(_require(rp, "RadionuclideTotalDose")),
        injection_time=injection,
        scan_time=scan_time,
        half_life_s=float(_require(rp, "RadionuclideHalfLife")),
        weight_kg=float(_require(header, "PatientWeight")),
        units=str(_require(header, "Units")),
        decay_correction=str(getattr(header, "DecayCorrection", "") or ""),
    )


def extract_params(header: Any, scan_time_tag: str = "Series") -> SUVParams:
    """Dispatch on vendor. Only the standard tag path is implemented.

    TODO(vendor): add branches once the institutional scanner is known, e.g.
      - Philips: private SUVScaleFactor (7053,1000) when Units != BQML
      - GE: private decay-correction conventions
    """
    manufacturer = str(getattr(header, "Manufacturer", "") or "")
    try:
        return extract_params_standard(header, scan_time_tag)
    except SUVError as e:
        raise SUVError(f"{e} (Manufacturer='{manufacturer}'; vendor-specific path not implemented)") from e


def compute_suv(pet: Volume, params: SUVParams) -> Volume:
    """Body-weight SUV. Requires Bq/ml voxels and DecayCorrection in {START, ADMIN}."""
    if params.units.upper() != "BQML":
        raise SUVError(f"Units='{params.units}'; only BQML supported (vendor scaling TODO)")
    dc = params.decay_correction.upper()
    if dc == "START":
        elapsed_s = (params.scan_time - params.injection_time).total_seconds()
        decay = 2.0 ** (-elapsed_s / params.half_life_s)
    elif dc == "ADMIN":
        decay = 1.0
    else:
        raise SUVError(f"DecayCorrection='{params.decay_correction}' not supported (need START or ADMIN)")
    if params.weight_kg <= 0 or params.dose_bq <= 0:
        raise SUVError("non-positive weight or dose")
    factor = params.weight_kg * 1000.0 / (params.dose_bq * decay)
    suv = pet.array.astype(np.float32) * np.float32(factor)
    meta = {**pet.meta, "suv_factor": float(factor), "suv_params": params}
    return pet.replace(array=suv, meta=meta)


def liver_normalize(suv: Volume, liver_mask: Volume) -> Volume:
    """Divide by mean SUV inside the liver mask (mask must be on the SUV grid)."""
    if not suv.same_grid(liver_mask):
        raise ValueError("liver mask is not on the SUV grid; run transfer_mask first")
    m = liver_mask.array > 0
    if not m.any():
        raise ValueError("empty liver mask")
    ref = float(suv.array[m].mean())
    if ref <= 0:
        raise ValueError(f"non-positive liver mean SUV {ref}")
    return suv.replace(array=suv.array / np.float32(ref), meta={**suv.meta, "liver_mean_suv": ref})
