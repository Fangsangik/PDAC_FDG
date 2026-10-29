import numpy as np
import SimpleITK as sitk
import pytest

from fdgmip.domain import BoundingBox, Volume
from conftest import make_volume


def test_sitk_roundtrip_preserves_geometry(pet):
    back = Volume.from_sitk(pet.to_sitk(), modality="PT")
    assert back.same_grid(pet)
    np.testing.assert_array_equal(back.array, pet.array)


def test_index_physical_matches_sitk(pet):
    img = pet.to_sitk()
    idx_zyx = (3, 5, 7)
    expected = img.TransformIndexToPhysicalPoint((7, 5, 3))  # sitk wants (x,y,z)
    np.testing.assert_allclose(pet.index_to_physical(idx_zyx), expected)
    np.testing.assert_allclose(pet.physical_to_index(expected), idx_zyx, atol=1e-9)


def test_index_physical_with_direction():
    flip = (-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0)  # LPS-style flip of x,y
    v = make_volume(shape=(4, 4, 4), spacing=(2.0, 2.0, 2.0), origin=(10.0, 20.0, 30.0), direction=flip)
    img = v.to_sitk()
    np.testing.assert_allclose(v.index_to_physical((1, 2, 3)), img.TransformIndexToPhysicalPoint((3, 2, 1)))


def test_rejects_non_3d():
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 4)), (1, 1, 1), (0, 0, 0))


def test_with_array_rejects_shape_change(pet):
    with pytest.raises(ValueError):
        pet.with_array(np.zeros((1, 1, 1)))


def test_bounding_box():
    b = BoundingBox((0, 2, 4), (3, 6, 10))
    assert b.shape == (3, 4, 6)
    assert b.center == (1.0, 3.5, 6.5)
    assert b.clip((2, 100, 100)).hi == (2, 6, 10)
    with pytest.raises(ValueError):
        BoundingBox((1, 1, 1), (1, 2, 2))
