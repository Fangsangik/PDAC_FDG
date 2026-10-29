import numpy as np
import pandas as pd
import pytest

from fdgmip.features import pet_stats
from fdgmip.registry import AGGREGATORS, BACKBONES, HARMONIZERS, import_all
from conftest import make_volume

import_all()


@pytest.fixture(scope="module")
def backbone():
    # weights=None: random init, no download; contract (shape, frozen) is what is tested
    return BACKBONES.build({"name": "torchvision", "arch": "resnet18", "weights": None, "input_size": 64,
                            "channel_adapter": "repeat", "intensity_max": 10.0})


def test_backbone_shape_frozen_and_deterministic(backbone):
    views = np.random.default_rng(0).random((5, 1, 40, 40), dtype=np.float32) * 10
    f = backbone.extract(views)
    assert f.shape == (5, backbone.feature_dim) and backbone.feature_dim == 512
    np.testing.assert_allclose(backbone.extract(views), f)
    assert all(not p.requires_grad for p in backbone.model.parameters())
    assert not backbone.model.training


def test_backbone_channel_adapter_errors(backbone):
    with pytest.raises(ValueError):
        backbone.extract(np.zeros((2, 3, 8, 8), dtype=np.float32))


def test_foundation_stub():
    with pytest.raises(NotImplementedError):
        BACKBONES.build({"name": "foundation_stub"}).extract(np.zeros((1, 1, 8, 8), np.float32))


def test_aggregators():
    f = np.array([[1.0, 5.0], [3.0, 1.0]])
    np.testing.assert_array_equal(AGGREGATORS.build({"name": "mean"}).aggregate(f), [2.0, 3.0])
    np.testing.assert_array_equal(AGGREGATORS.build({"name": "max"}).aggregate(f), [3.0, 5.0])
    assert AGGREGATORS.build({"name": "mean_max"}).aggregate(f).shape == (4,)
    assert AGGREGATORS.build({"name": "concat"}).aggregate(f).shape == (4,)


def test_pet_stats_hand_calculation():
    suv = make_volume(shape=(2, 2, 2), spacing=(10.0, 10.0, 10.0), modality="PT")  # 1 ml voxels
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32).reshape(2, 2, 2)
    suv = suv.replace(array=arr)
    voi = suv.replace(array=np.ones(suv.shape, np.uint8), modality="MASK")
    s = pet_stats(suv, voi, {"type": "suv_abs", "value": 5.0})
    assert s["suvmax"] == 8 and s["mtv_ml"] == 4 and s["tlg"] == 4 * 6.5
    s2 = pet_stats(suv, voi, {"type": "pct_max", "value": 0.5})
    assert s2["mtv_threshold_suv"] == 4.0 and s2["mtv_ml"] == 5
    with pytest.raises(NotImplementedError):
        pet_stats(suv, voi, {"type": "liver_relative"})


def test_harmonizers():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    b = pd.Series(["s1", "s2"])
    assert HARMONIZERS.build({"name": "none"}).fit(X, b).transform(X, b) is X
    for n in ("combat", "scn"):
        with pytest.raises(NotImplementedError):
            HARMONIZERS.build({"name": n}).fit(X, b)
