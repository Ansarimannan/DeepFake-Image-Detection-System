"""Small shared helpers: logging, seeding and the label-map contract."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")
    # TensorFlow's C++ logging is noisy and says nothing useful here.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seeds(seed: int) -> None:
    """Seed every generator we control.

    This makes the split, the initialisation and the shuffling reproducible. It
    does NOT make training bit-exact: multi-threaded tf.data and non-deterministic
    CPU/GPU kernels still introduce run-to-run variation. Call
    tf.config.experimental.enable_op_determinism() if you need bit-exactness and
    can afford the throughput cost.
    """
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def label_map_payload(cfg, preprocessing: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """The deployment contract that travels with the weights.

    A weights file on its own is not a deployable artifact. What ships is the
    weights plus the label ordering plus the preprocessing plus the operating
    threshold. All four live here.
    """
    class_names = cfg.class_names
    return {
        "class_names": class_names,
        "class_indices": {name: i for i, name in enumerate(class_names)},
        "positive_class": class_names[1],
        "output": "logit",
        "probability": "sigmoid(logit) = P(positive_class) = P(real)",
        "decision_rule": (
            f"predicted_index = int(probability >= threshold); "
            f"predicted_class = class_names[predicted_index]"
        ),
        "threshold": float(threshold),
        "image_size": list(cfg.image_size),
        "channels": int(cfg.require("image", "channels")),
        "color_order": "RGB",
        "preprocessing": preprocessing,
        "backbone": cfg.require("model", "backbone"),
    }
