import os
import stat
import sys

import numpy as np
import pytest

from fdgmip.registry import SEGMENTERS
from fdgmip.segmentation import TotalSegmentatorAdapter

FAKE = '''#!{py}
import sys, pathlib, SimpleITK as sitk
a = sys.argv; i = a[a.index("-i")+1]; o = pathlib.Path(a[a.index("-o")+1]); o.mkdir()
organs = a[a.index("--roi_subset")+1:]
img = sitk.ReadImage(i)
for k, organ in enumerate(organs):
    m = sitk.Image(img.GetSize(), sitk.sitkUInt8); m.CopyInformation(img)
    m[0, 0, k] = 1
    sitk.WriteImage(m, str(o / f"{{organ}}.nii.gz"))
'''


@pytest.fixture
def fake_exe(tmp_path):
    p = tmp_path / "FakeTotalSegmentator"
    p.write_text(FAKE.format(py=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def test_adapter_round_trips_masks_on_ct_grid(ct, fake_exe):
    seg = SEGMENTERS.build({"name": "totalsegmentator", "task": "total", "executable": fake_exe})
    masks = seg.segment(ct, ["pancreas", "liver"])
    assert set(masks) == {"pancreas", "liver"}
    for k, (organ, m) in enumerate(masks.items()):
        assert m.same_grid(ct) and m.modality == "MASK" and m.frame_of_reference_uid == ct.frame_of_reference_uid
        assert m.array.sum() == 1 and m.array[k, 0, 0] == 1


def test_adapter_refuses_non_ct(pet, fake_exe):
    with pytest.raises(ValueError):
        TotalSegmentatorAdapter(task="total", executable=fake_exe).segment(pet, ["pancreas"])
