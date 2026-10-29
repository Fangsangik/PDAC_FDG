from __future__ import annotations

import logging

from fdgmip.domain import Series

log = logging.getLogger(__name__)


def check_inclusion(series: Series, criteria: dict) -> list[str]:
    """Return a list of violated criteria (empty = included). Violations are logged, not raised."""
    v: list[str] = []
    mx = criteria.get("max_slice_thickness_mm")
    if mx is not None:
        if series.slice_thickness is None:
            v.append("slice_thickness missing")
        elif series.slice_thickness > mx:
            v.append(f"slice_thickness {series.slice_thickness} > {mx}")
    mn = criteria.get("min_n_files")
    if mn is not None and series.n_files < mn:
        v.append(f"n_files {series.n_files} < {mn}")
    for item in v:
        log.warning("series %s (%s) excluded: %s", series.uid, series.description, item)
    return v
