from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from fdgmip.domain import Volume


def read_nifti(path: str | Path, modality: str = "UNKNOWN", frame_of_reference_uid: str | None = None) -> Volume:
    return Volume.from_sitk(sitk.ReadImage(str(path)), modality=modality,
                            frame_of_reference_uid=frame_of_reference_uid)


def write_volume(vol: Volume, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(vol.to_sitk(), str(path))
