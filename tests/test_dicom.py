import numpy as np

from fdgmip.io import list_series, read_series, series_table
from conftest import write_dicom_series


def test_list_and_read_series(tmp_path):
    ct_uid = write_dicom_series(tmp_path / "ct", "CT", 5, spacing_xy=(0.8, 0.8), thickness=2.5,
                                origin=(-10.0, -20.0, 5.0), description="CE CT portal")
    write_dicom_series(tmp_path / "pt", "PT", 3, spacing_xy=(4.0, 4.0), thickness=3.0, description="PET AC")
    series = list_series(tmp_path)
    tab = series_table(series)
    assert set(tab["modality"]) == {"CT", "PT"}
    assert tab.loc[tab.modality == "CT", "n_files"].item() == 5
    assert tab.loc[tab.modality == "CT", "slice_thickness"].item() == 2.5

    ct = read_series(next(s for s in series if s.uid == ct_uid))
    assert ct.shape == (5, 16, 16)
    np.testing.assert_allclose(ct.spacing, (0.8, 0.8, 2.5))
    np.testing.assert_allclose(ct.origin, (-10.0, -20.0, 5.0))
    assert ct.modality == "CT" and ct.frame_of_reference_uid is not None
    assert ct.meta["header"].Modality == "CT"
