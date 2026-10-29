import datetime as dt

import numpy as np
import pydicom
import pytest

from fdgmip.domain import BoundingBox
from fdgmip.preprocess import (FrameOfReferenceMismatch, SUVError, compute_suv, crop, liver_normalize,
                               mask_out_organs, reorient, resample, transfer_mask)
from fdgmip.preprocess.crop import mask_centroid_index
from fdgmip.preprocess.suv import extract_params
from fdgmip.registry import CROPS
from conftest import make_volume, sphere_mask, standard_pet_tags

CENTER = (3.0, -4.0, 2.0)


def _phys_centroid(mask):
    return mask.index_to_physical(mask_centroid_index(mask))


# ---- resample / orientation ---------------------------------------------
def test_resample_preserves_physical_position(pancreas_ct):
    out = resample(pancreas_ct, (2.0, 2.0, 2.0), "nearest")
    np.testing.assert_allclose(out.spacing, (2.0, 2.0, 2.0))
    assert set(np.unique(out.array)) <= {0, 1}
    np.testing.assert_allclose(_phys_centroid(out), CENTER, atol=1.0)  # within half a voxel


def test_resample_mask_requires_nearest(pancreas_ct):
    with pytest.raises(ValueError):
        resample(pancreas_ct, (2.0, 2.0, 2.0), "linear")


def test_reorient_keeps_physical_coordinates(pancreas_ct):
    flipped = pancreas_ct.replace(array=pancreas_ct.array[:, ::-1, ::-1].copy(),
                                  origin=(31.0, 31.0, -30.0), direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))
    ref = _phys_centroid(pancreas_ct)
    np.testing.assert_allclose(_phys_centroid(flipped), ref, atol=1e-6)
    out = reorient(flipped, "LPS")
    np.testing.assert_allclose(out.direction, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    np.testing.assert_allclose(_phys_centroid(out), ref, atol=1e-6)
    np.testing.assert_array_equal(out.array, pancreas_ct.array)


# ---- mask transfer ------------------------------------------------------
def test_mask_transfer_matches_direct_rasterisation(pancreas_ct, pet):
    moved = transfer_mask(pancreas_ct, pet)
    direct = sphere_mask(pet, CENTER, 8.0)
    assert moved.same_grid(pet)
    inter = np.logical_and(moved.array, direct.array).sum()
    dice = 2 * inter / (moved.array.sum() + direct.array.sum())
    assert dice > 0.8
    np.testing.assert_allclose(_phys_centroid(moved), CENTER, atol=max(pet.spacing))


def test_mask_transfer_rejects_frame_of_reference_mismatch(pancreas_ct, pet):
    other = pet.replace(frame_of_reference_uid="1.2.3.other")
    with pytest.raises(FrameOfReferenceMismatch):
        transfer_mask(pancreas_ct, other)
    with pytest.raises(FrameOfReferenceMismatch):
        transfer_mask(pancreas_ct, pet.replace(frame_of_reference_uid=None))


# ---- SUV ----------------------------------------------------------------
def _header(**kw):
    ds = pydicom.Dataset()
    for k, v in standard_pet_tags(**kw).items():
        setattr(ds, k, v)
    return ds


def test_suv_matches_hand_calculation():
    dose, weight, half, bqml = 3.7e8, 70.0, 6586.2, 5000.0
    pet = make_volume(shape=(2, 2, 2), modality="PT", fill=bqml)
    p = extract_params(_header(dose_bq=dose, weight_kg=weight, half_life_s=half,
                               inj_time="100000", acq_time="110000"))
    suv = compute_suv(pet, p)
    decay = 2 ** (-3600 / half)
    expected = bqml * weight * 1000 / (dose * decay)
    np.testing.assert_allclose(suv.array, expected, rtol=1e-5)
    assert p.scan_time - p.injection_time == dt.timedelta(hours=1)


def test_suv_missing_tag_is_clear_error():
    ds = _header()
    del ds.PatientWeight
    with pytest.raises(SUVError, match="PatientWeight"):
        extract_params(ds)


def test_suv_rejects_unknown_units():
    pet = make_volume(shape=(2, 2, 2), modality="PT", fill=1.0)
    ds = _header()
    ds.Units = "CNTS"
    with pytest.raises(SUVError, match="Units"):
        compute_suv(pet, extract_params(ds))


def test_liver_normalize():
    suv = make_volume(shape=(2, 2, 2), modality="PT", fill=4.0)
    liver = suv.replace(array=np.ones(suv.shape, dtype=np.uint8), modality="MASK")
    np.testing.assert_allclose(liver_normalize(suv, liver).array, 1.0)


# ---- crop / masking -----------------------------------------------------
def test_pancreas_centroid_crop_is_square_and_centred(pancreas_ct, pet):
    masks = {"pancreas": transfer_mask(pancreas_ct, pet)}
    strat = CROPS.build({"name": "pancreas_centroid", "size_mm": [30, 18, 24], "square_xy": True})
    box = strat.box(pet, masks)
    assert box.shape[1] == box.shape[2]
    out = crop(pet, box)
    assert out.shape == box.shape
    np.testing.assert_allclose(out.index_to_physical(box.center - np.asarray(box.lo)),
                               _phys_centroid(masks["pancreas"]), atol=max(pet.spacing))


def test_crop_pads_outside_volume(pet):
    box = BoundingBox((-2, 0, 0), (3, 5, 5))
    out = crop(pet, box, fill_value=-1.0)
    assert out.shape == (5, 5, 5)
    assert (out.array[:2] == -1.0).all()
    np.testing.assert_allclose(out.index_to_physical((2, 0, 0)), pet.index_to_physical((0, 0, 0)))


def test_whole_torso_and_vertebral_crops(pet):
    assert CROPS.build({"name": "whole_torso", "square_xy": False}).box(pet, {}).shape == pet.shape
    sq = CROPS.build({"name": "whole_torso", "square_xy": True}).box(pet, {})
    assert sq.shape[1] == sq.shape[2]
    z = np.zeros(pet.shape, dtype=np.uint8)
    top, bot = z.copy(), z.copy()
    top[2, 5, 5] = 1
    bot[9, 5, 5] = 1
    masks = {"vertebrae_T10": pet.replace(array=top, modality="MASK"),
             "vertebrae_L3": pet.replace(array=bot, modality="MASK"),
             "pancreas": sphere_mask(pet, CENTER, 8.0)}
    b = CROPS.build({"name": "vertebral_level", "top_organ": "vertebrae_T10", "bottom_organ": "vertebrae_L3",
                     "xy_size_mm": 30, "square_xy": True}).box(pet, masks)
    assert (b.lo[0], b.hi[0]) == (2, 10)


def test_mask_out_organs(pet):
    m = sphere_mask(pet, CENTER, 8.0)
    out = mask_out_organs(pet, {"kidney": m}, ["kidney"], fill_value=0.0)
    assert (out.array[m.array > 0] == 0).all()
    assert (out.array[m.array == 0] == pet.array[m.array == 0]).all()
