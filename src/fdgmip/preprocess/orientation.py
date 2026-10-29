import SimpleITK as sitk

from fdgmip.domain import Volume


def reorient(vol: Volume, orientation: str) -> Volume:
    """Permute/flip axes so the direction cosines match `orientation` (e.g. 'LPS', 'RAS').

    Physical coordinates of every voxel are unchanged; only array layout + direction change.
    """
    img = sitk.DICOMOrient(vol.to_sitk(), orientation)
    return Volume.from_sitk(img, frame_of_reference_uid=vol.frame_of_reference_uid,
                            modality=vol.modality, meta=vol.meta)
