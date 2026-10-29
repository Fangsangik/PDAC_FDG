"""Synthetic data only. No patient data is ever used in tests."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from fdgmip.domain import Volume

FOR_UID = "1.2.826.0.1.3680043.9999.1"


def make_volume(shape=(20, 32, 32), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                modality="CT", fill=0.0, dtype=np.float32, **kw) -> Volume:
    return Volume(np.full(shape, fill, dtype=dtype), spacing, origin,
                  frame_of_reference_uid=kw.pop("for_uid", FOR_UID), modality=modality, **kw)


def sphere_mask(vol: Volume, center_xyz, radius_mm) -> Volume:
    """Binary sphere defined in PHYSICAL coordinates, rasterised on vol's grid."""
    zz, yy, xx = np.indices(vol.shape)
    idx = np.stack([zz, yy, xx], -1).reshape(-1, 3)
    phys = vol.index_to_physical(idx)
    d = np.linalg.norm(phys - np.asarray(center_xyz), axis=1).reshape(vol.shape)
    return vol.replace(array=(d <= radius_mm).astype(np.uint8), modality="MASK", meta={})


@pytest.fixture
def ct() -> Volume:
    return make_volume(shape=(40, 64, 64), spacing=(1.0, 1.0, 1.5), origin=(-32.0, -32.0, -30.0), modality="CT")


@pytest.fixture
def pet() -> Volume:
    # Different spacing AND origin from CT; same frame of reference.
    v = make_volume(shape=(16, 24, 24), spacing=(3.0, 3.0, 4.0), origin=(-34.5, -31.0, -29.0), modality="PT")
    rng = np.random.default_rng(0)
    return v.replace(array=rng.random(v.shape, dtype=np.float32))


@pytest.fixture
def pancreas_ct(ct) -> Volume:
    return sphere_mask(ct, center_xyz=(3.0, -4.0, 2.0), radius_mm=8.0)


# ---- synthetic DICOM series ---------------------------------------------
def write_dicom_series(out_dir: Path, modality: str, n_slices: int, rows=16, cols=16,
                       spacing_xy=(2.0, 2.0), thickness=3.0, origin=(0.0, 0.0, 0.0),
                       pixel_value=100, rescale=(1.0, 0.0), for_uid=FOR_UID,
                       description="synthetic", pet_tags: dict | None = None) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    for i in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if modality == "CT" else "1.2.840.10008.5.1.4.1.1.128"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(out_dir / f"{modality}_{i:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = "SYN000"
        ds.PatientName = "SYN^THETIC"
        ds.Modality = modality
        ds.SeriesDescription = description
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + i * thickness]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = list(spacing_xy)
        ds.SliceThickness = thickness
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope, ds.RescaleIntercept = rescale
        ds.PixelData = np.full((rows, cols), pixel_value, dtype=np.uint16).tobytes()
        if pet_tags:
            for k, v in pet_tags.items():
                setattr(ds, k, v)
        path = str(out_dir / f"{modality}_{i:03d}.dcm")
        try:
            ds.save_as(path, enforce_file_format=True)   # pydicom >= 3
        except TypeError:
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(path)                             # pydicom 2.x
    return series_uid


def standard_pet_tags(dose_bq=3.7e8, inj_time="100000", acq_time="110000", weight_kg=70.0,
                      half_life_s=6586.2, date="20240101") -> dict:
    rp = Dataset()
    rp.RadionuclideTotalDose = dose_bq
    rp.RadiopharmaceuticalStartTime = inj_time
    rp.RadionuclideHalfLife = half_life_s
    return {
        "Units": "BQML",
        "DecayCorrection": "START",
        "PatientWeight": weight_kg,
        "SeriesDate": date,
        "SeriesTime": acq_time,
        "AcquisitionDate": date,
        "AcquisitionTime": acq_time,
        "RadiopharmaceuticalInformationSequence": [rp],
    }
