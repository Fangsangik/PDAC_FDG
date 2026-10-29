"""DICOM adapters: series discovery/inventory and series -> Volume reading."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pydicom
import SimpleITK as sitk

from fdgmip.domain import Series, Volume

log = logging.getLogger(__name__)


def _tag(ds: pydicom.Dataset, name: str, default=None):
    v = getattr(ds, name, None)
    return default if v in (None, "") else v


def list_series(root: str | Path) -> list[Series]:
    """Scan a directory tree and group DICOM files by SeriesInstanceUID."""
    groups: dict[str, list[tuple[str, pydicom.Dataset]]] = defaultdict(list)
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        uid = _tag(ds, "SeriesInstanceUID")
        if uid is None:
            continue
        groups[uid].append((str(p), ds))

    out: list[Series] = []
    for uid, items in groups.items():
        ds = items[0][1]
        st = _tag(ds, "SliceThickness")
        out.append(Series(
            uid=uid,
            modality=str(_tag(ds, "Modality", "")),
            description=str(_tag(ds, "SeriesDescription", "")),
            frame_of_reference_uid=_tag(ds, "FrameOfReferenceUID"),
            slice_thickness=float(st) if st is not None else None,
            n_files=len(items),
            paths=[p for p, _ in items],
        ))
    out.sort(key=lambda s: (s.modality, s.description, s.uid))
    return out


def series_table(series: list[Series]) -> pd.DataFrame:
    """Human-readable inventory; the human picks the CT series from this, not the code."""
    return pd.DataFrame([{
        "modality": s.modality,
        "description": s.description,
        "series_uid": s.uid,
        "frame_of_reference_uid": s.frame_of_reference_uid,
        "slice_thickness": s.slice_thickness,
        "n_files": s.n_files,
    } for s in series])


def read_series(series: Series) -> Volume:
    """Read one series into a Volume. Rescale slope/intercept is applied by GDCM.

    meta['header'] holds the (pixel-free) pydicom header of the first file so that
    SUV computation can read dose / time / weight tags without re-opening files.
    """
    if not series.paths:
        raise ValueError(f"series {series.uid} has no file paths")
    directory = str(Path(series.paths[0]).parent)
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, series.uid)
    if not names:
        raise ValueError(f"GDCM found no files for series {series.uid} in {directory}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    img = reader.Execute()
    header = pydicom.dcmread(names[0], stop_before_pixels=True)
    return Volume.from_sitk(
        img,
        frame_of_reference_uid=series.frame_of_reference_uid,
        modality=series.modality,
        meta={"header": header, "series_uid": series.uid, "n_files": len(names)},
    )
