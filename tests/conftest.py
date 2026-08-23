"""Shared test fixtures.

Tests never touch the real dataset or a trained model. They build a tiny
synthetic dataset on disk and a tiny model, so the whole suite runs in seconds
and can be executed on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfake.config import Config, load_config  # noqa: E402


BASE_CONFIG = {
    "project": {"name": "test", "run_name": "pytest", "seed": 7},
    "data": {
        "root": "data",
        "class_dirs": {"fake": "fake", "real": "real"},
        "extensions": [".png"],
        "split": {"train": 0.6, "val": 0.2, "test": 0.2,
                  "manifest": "artifacts/splits/manifest.csv"},
    },
    "image": {"size": [32, 32], "channels": 3},
    "pipeline": {"batch_size": 4, "cache": True, "shuffle_buffer": 16, "prefetch": True},
    "augmentation": {"enabled": True, "random_flip_horizontal": True,
                     "random_rotation": 0.0, "random_zoom": 0.0,
                     "random_translation": 0.0, "random_contrast": 0.0},
    "model": {"backbone": "MobileNetV2", "weights": None, "head_units": [], "dropout": 0.2},
    "training": {
        "stage1": {"enabled": True, "epochs": 1, "learning_rate": 0.001},
        "stage2": {"enabled": False, "epochs": 1, "learning_rate": 0.0001,
                   "unfreeze_last_n": 4},
        "lr_schedule": "none",
        "reduce_on_plateau": {"factor": 0.5, "patience": 1, "min_lr": 1e-7},
        "early_stopping": {"monitor": "val_auc_roc", "mode": "max", "patience": 2,
                           "restore_best_weights": True},
        "checkpoint": {"monitor": "val_auc_roc", "mode": "max"},
        "class_weight_threshold": 1.5,
    },
    "evaluation": {"threshold_strategy": "f1", "threshold_value": 0.5,
                   "bootstrap_samples": 20, "confidence_level": 0.95,
                   "abstain_margin": 0.1},
    "gradcam": {"samples": 2, "alpha": 0.4},
}


def _write_png(path: Path, rng: np.random.Generator) -> None:
    """Write a small random PNG without needing PIL."""
    import tensorflow as tf

    array = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    path.write_bytes(tf.io.encode_png(tf.convert_to_tensor(array)).numpy())


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A synthetic dataset: 12 fake images, 14 real images."""
    rng = np.random.default_rng(0)
    for class_name, count in (("fake", 12), ("real", 14)):
        directory = tmp_path / "data" / class_name
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            # Fake filenames mimic the CIPLAB naming so the metadata parser is exercised.
            name = f"easy_{i:03d}_1010.png" if class_name == "fake" else f"real_{i:05d}.png"
            _write_png(directory / name, rng)
    return tmp_path


@pytest.fixture
def cfg(dataset_root: Path, tmp_path: Path, monkeypatch) -> Config:
    """A Config whose project root is the temporary directory."""
    import deepfake.config as config_mod

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    return load_config(path)
