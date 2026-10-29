"""Assemble the (n_views, C, H, W) tensor from a channel spec.

channel spec entries (from config):
  {type: image}                         plain MIP of the (optionally organ-masked) SUV crop
  {type: image, restrict_to: pancreas}  pancreas-only MIP: voxels outside the organ zeroed before projecting
  {type: mask, organ: pancreas}         projected 3D mask, binary (2D GT / extra input channel)
"""
from __future__ import annotations

import numpy as np

from fdgmip.domain import Volume
from fdgmip.projection.mip import Projector


def compose_views(vol: Volume, masks: dict[str, Volume], projector: Projector, center_zyx, channels: list[dict]) -> np.ndarray:
    if not channels:
        raise ValueError("projection.channels is empty")
    out = []
    for ch in channels:
        kind = ch["type"]
        if kind == "image":
            src = vol
            organ = ch.get("restrict_to")
            if organ:
                m = masks[organ]
                if not m.same_grid(vol):
                    raise ValueError(f"mask '{organ}' not on the crop grid")
                src = vol.replace(array=np.where(m.array > 0, vol.array, 0).astype(vol.array.dtype))
            out.append(projector.project(src, center_zyx, is_mask=False).astype(np.float32))
        elif kind == "mask":
            m = masks[ch["organ"]]
            if not m.same_grid(vol):
                raise ValueError(f"mask '{ch['organ']}' not on the crop grid")
            out.append(projector.project(m, center_zyx, is_mask=True).astype(np.float32))
        else:
            raise ValueError(f"unknown channel type '{kind}'")
    return np.stack(out, axis=1)  # (n_views, C, H, W)
