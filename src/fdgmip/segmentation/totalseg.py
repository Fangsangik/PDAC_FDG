"""TotalSegmentator adapter: external CLI call only, nothing re-implemented."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np

from fdgmip.domain import Volume
from fdgmip.io.nifti import read_nifti, write_volume
from fdgmip.preprocess.mask_transfer import transfer_mask
from fdgmip.registry import SEGMENTERS

log = logging.getLogger(__name__)


class Segmenter(Protocol):
    def segment(self, ct: Volume, organs: list[str]) -> dict[str, Volume]: ...


@SEGMENTERS.register("totalsegmentator")
class TotalSegmentatorAdapter:
    """Runs `TotalSegmentator -i ct.nii.gz -o <dir> --task <task> --roi_subset <organs...> [extra_args]`
    and reads back one binary NIfTI per organ. Organ names must be TotalSegmentator class names
    (e.g. pancreas, liver, spleen, kidney_left, stomach, vertebrae_L1).
    """
    def __init__(self, task: str, extra_args: list[str] | None = None, executable: str = "TotalSegmentator"):
        self.task, self.extra_args, self.executable = task, list(extra_args or []), executable

    def segment(self, ct: Volume, organs: list[str]) -> dict[str, Volume]:
        if ct.modality != "CT":
            raise ValueError(f"TotalSegmentator expects a CT volume, got modality={ct.modality}")
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "ct.nii.gz", Path(tmp) / "seg"
            write_volume(ct, inp)
            cmd = [self.executable, "-i", str(inp), "-o", str(out), "--task", self.task,
                   "--roi_subset", *organs, *self.extra_args]
            log.info("running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
            masks: dict[str, Volume] = {}
            for organ in organs:
                f = out / f"{organ}.nii.gz"
                if not f.exists():
                    raise FileNotFoundError(f"TotalSegmentator produced no mask for '{organ}' ({f})")
                m = read_nifti(f, modality="MASK", frame_of_reference_uid=ct.frame_of_reference_uid)
                m = m.replace(array=(m.array > 0).astype(np.uint8))
                if not m.same_grid(ct):
                    log.warning("mask '%s' grid differs from CT; resampling (nearest) onto CT grid", organ)
                    m = transfer_mask(m, ct)
                masks[organ] = m
        return masks
