import pytest

from fdgmip.config import ConfigError, load_config


def test_merge_and_hash(tmp_path):
    (tmp_path / "a.yaml").write_text("x: {a: 1, b: 2}\ny: 3\n")
    (tmp_path / "b.yaml").write_text("x: {b: 9}\n")
    cfg = load_config(tmp_path / "a.yaml", tmp_path / "b.yaml")
    assert cfg.get("x.a") == 1 and cfg.get("x.b") == 9 and cfg.get("y") == 3
    h1 = cfg.hash(["x"])
    cfg.data["y"] = 100
    assert cfg.hash(["x"]) == h1          # unrelated section does not change the key
    assert cfg.hash(["x", "y"]) != h1
    with pytest.raises(ConfigError):
        cfg.get("x.missing")
