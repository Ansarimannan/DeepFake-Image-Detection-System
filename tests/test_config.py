"""config.yaml is a contract, so it is validated rather than trusted."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from deepfake.config import ConfigError, load_config
from tests.conftest import BASE_CONFIG


def _write(tmp_path: Path, mutate) -> Path:
    data = copy.deepcopy(BASE_CONFIG)
    mutate(data)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_and_exposes_basics(cfg):
    assert cfg.seed == 7
    assert cfg.image_size == (32, 32)
    assert cfg.input_shape == (32, 32, 3)
    assert cfg.run_name == "pytest"


def test_class_order_is_fake_then_real(cfg):
    """fake=0, real=1. The sigmoid output is therefore P(real), never P(fake)."""
    assert cfg.class_names == ["fake", "real"]
    assert cfg.class_names.index("fake") == 0
    assert cfg.class_names.index("real") == 1


def test_paths_are_resolved_against_the_project_root(cfg, tmp_path):
    assert Path(cfg.require("data", "root")).is_absolute()
    assert Path(cfg.require("data", "split", "manifest")).is_absolute()
    assert str(cfg.require("data", "root")).startswith(str(tmp_path))


def test_split_fractions_must_sum_to_one(tmp_path):
    path = _write(tmp_path, lambda d: d["data"]["split"].update({"train": 0.9}))
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


def test_callback_mode_must_be_explicit(tmp_path):
    """Guards against Keras' most dangerous silent default.

    Keras' mode='auto' resolves 'val_auc_roc' to MINIMISE because the name
    contains 'auc' rather than 'acc'. Omitting the mode is therefore a config
    error here, not a default.
    """
    path = _write(tmp_path, lambda d: d["training"]["checkpoint"].pop("mode"))
    with pytest.raises(ConfigError, match="explicitly 'min' or 'max'"):
        load_config(path)


def test_unknown_backbone_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["model"].update({"backbone": "ResNet999"}))
    with pytest.raises(ConfigError, match="not supported"):
        load_config(path)


def test_unknown_threshold_strategy_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["evaluation"].update({"threshold_strategy": "vibes"}))
    with pytest.raises(ConfigError, match="not supported"):
        load_config(path)


def test_unknown_frequency_branch_is_rejected(tmp_path):
    path = _write(tmp_path, lambda d: d["model"].update({"frequency_branch": "fourier"}))
    with pytest.raises(ConfigError, match="not supported"):
        load_config(path)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_save_used_writes_a_copy(cfg):
    out = cfg.save_used()
    assert out.exists()
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["project"]["run_name"] == "pytest"
