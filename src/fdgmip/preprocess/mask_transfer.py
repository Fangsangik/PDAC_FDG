"""Move a mask from one grid (CT) to another (PET) via physical coordinates."""
from __future__ import annotations

import SimpleITK as sitk

from fdgmip.domain import Volume


class FrameOfReferenceMismatch(ValueError):
    pass


def transfer_mask(mask: Volume, target: Volume) -> Volume:
    """Nearest-neighbour resample of `mask` onto `target`'s grid.

    Refuses to proceed when the two volumes do not share a FrameOfReferenceUID: an identity
    transform would then be a silent misregistration.
    """
    if mask.frame_of_reference_uid is None or target.frame_of_reference_uid is None:
        raise FrameOfReferenceMismatch("FrameOfReferenceUID missing on mask or target; cannot assume shared space")
    if mask.frame_of_reference_uid != target.frame_of_reference_uid:
        raise FrameOfReferenceMismatch(
            f"mask FoR {mask.frame_of_reference_uid} != target FoR {target.frame_of_reference_uid}; "
            "explicit registration required (not implemented)")
    out = sitk.Resample(mask.to_sitk(), target.to_sitk(), sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    return Volume.from_sitk(out, frame_of_reference_uid=target.frame_of_reference_uid, modality="MASK")


def transfer_masks(masks: dict[str, Volume], target: Volume) -> dict[str, Volume]:
    return {k: transfer_mask(v, target) for k, v in masks.items()}
