import numpy as np
import pytest

from fdgmip.projection import compose_views, rotate_about
from fdgmip.projection.mip import rotation_center
from fdgmip.registry import PROJECTORS
from conftest import make_volume


def _point_volume(n=21, d=6):
    v = make_volume(shape=(n, n, n), spacing=(1.0, 1.0, 1.0), modality="PT")
    arr = np.zeros(v.shape, dtype=np.float32)
    c = n // 2
    arr[c, c, c + d] = 10.0  # hot voxel displaced along +x
    return v.replace(array=arr), c, d


def test_rotation_moves_point_as_expected():
    v, c, d = _point_volume()
    r90 = rotate_about(v.array, 90, "z", (c, c, c), order=0)
    z, y, x = np.unravel_index(r90.argmax(), r90.shape)
    assert (z, x) == (c, c) and abs(y - c) == d       # moved onto the y axis
    r180 = rotate_about(v.array, 180, "z", (c, c, c), order=0)
    assert np.unravel_index(r180.argmax(), r180.shape) == (c, c, c - d)


def test_ma_mip_shapes_angles_and_angle_zero():
    v, c, d = _point_volume()
    proj = PROJECTORS.build({"name": "ma_mip", "rotation_axis": "z", "projection_axis": "y",
                             "start_deg": 0, "n_angles": 4, "span_deg": 180})
    assert proj.angles() == [0.0, 45.0, 90.0, 135.0]
    views = proj.project(v, (c, c, c), is_mask=False)
    assert views.shape == (4, 21, 21) and views.dtype == np.float32
    np.testing.assert_array_equal(views[0], v.array.max(axis=1))      # angle 0 == plain coronal MIP
    assert np.unravel_index(views[2].argmax(), views[2].shape) == (c, c)  # 90 deg: point now on the ray


def test_ma_mip_rejects_anisotropic():
    v = make_volume(shape=(5, 5, 5), spacing=(1.0, 1.0, 2.0), modality="PT")
    proj = PROJECTORS.build({"name": "ma_mip", "rotation_axis": "z", "projection_axis": "y",
                             "start_deg": 0, "n_angles": 2, "span_deg": 180})
    with pytest.raises(ValueError, match="isotropic"):
        proj.project(v, (2, 2, 2), is_mask=False)


def test_compose_views_channels_and_mask_projection():
    v, c, d = _point_volume()
    mask = np.zeros(v.shape, dtype=np.uint8)
    mask[c - 1:c + 2, c - 1:c + 2, c - 1:c + 2] = 1          # cube at centre, excludes the hot voxel
    masks = {"pancreas": v.replace(array=mask, modality="MASK")}
    proj = PROJECTORS.build({"name": "two_view", "rotation_axis": "z", "projection_axis": "y"})
    center = rotation_center(v, masks, "mask_centroid")
    np.testing.assert_allclose(center, (c, c, c))
    out = compose_views(v, masks, proj, center, [
        {"type": "image"}, {"type": "image", "restrict_to": "pancreas"}, {"type": "mask", "organ": "pancreas"}])
    assert out.shape == (2, 3, 21, 21)
    assert out[0, 0].max() == 10.0            # plain MIP sees the hot voxel
    assert out[0, 1].max() == 0.0             # pancreas-only MIP does not
    assert set(np.unique(out[:, 2])) <= {0.0, 1.0} and out[0, 2].sum() == 9   # projected 3x3x3 cube -> 3x3
