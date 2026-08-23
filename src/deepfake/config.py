"""Configuration loading and validation.

Every tunable value in this project lives in config.yaml. This module loads it,
validates it, resolves all relative paths against the project root, and exposes
it as a plain nested dict wrapped in a small accessor.

Design rule: no other module may hardcode a hyperparameter or a path.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml

# src/deepfake/config.py -> src/deepfake -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Keys whose values are paths and must be resolved against the project root.
_PATH_KEYS = (
    ("data", "root"),
    ("data", "split", "manifest"),
)


class ConfigError(ValueError):
    """Raised when config.yaml is missing a key or holds an impossible value."""


class Config:
    """Read-only-ish accessor over the parsed config dict."""

    def __init__(self, data: Dict[str, Any], source: Path):
        self._data = data
        self.source = source

    # -- access ------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: cfg.get('training', 'stage1', 'epochs')."""
        node: Any = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, *keys: str) -> Any:
        sentinel = object()
        value = self.get(*keys, default=sentinel)
        if value is sentinel:
            raise ConfigError(f"config.yaml is missing required key: {'.'.join(keys)}")
        return value

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- derived paths -----------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def run_name(self) -> str:
        return str(self.require("project", "run_name"))

    @property
    def run_dir(self) -> Path:
        """artifacts/<run_name>/ - everything a run produces goes here."""
        path = PROJECT_ROOT / "artifacts" / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def seed(self) -> int:
        return int(self.require("project", "seed"))

    @property
    def image_size(self) -> tuple:
        size = self.require("image", "size")
        return (int(size[0]), int(size[1]))

    @property
    def input_shape(self) -> tuple:
        h, w = self.image_size
        return (h, w, int(self.require("image", "channels")))

    @property
    def class_names(self) -> list:
        """Canonical class order. Index IS the training label.

        fake -> 0, real -> 1. A sigmoid output is therefore P(real), never
        P(fake). This single fact is the source of the label-polarity bug that
        existed in the original notebook, so it is fixed here in one place and
        asserted by tests/test_predict.py.
        """
        dirs = self.require("data", "class_dirs")
        if set(dirs) != {"fake", "real"}:
            raise ConfigError("data.class_dirs must define exactly 'fake' and 'real'")
        return ["fake", "real"]

    def class_dir(self, class_name: str) -> Path:
        root = Path(self.require("data", "root"))
        return root / str(self.require("data", "class_dirs")[class_name])

    # -- persistence -------------------------------------------------------
    def save_used(self) -> Path:
        """Copy the resolved config next to the run's artifacts for traceability."""
        out = self.run_dir / "config.used.yaml"
        payload = self.as_dict()
        payload = _stringify_paths(payload)
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return out

    def __repr__(self) -> str:
        return f"Config(run_name={self.run_name!r}, source={self.source})"


def _stringify_paths(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _stringify_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_stringify_paths(v) for v in node]
    if isinstance(node, Path):
        return str(node)
    return node


def _resolve_paths(data: Dict[str, Any]) -> None:
    for keys in _PATH_KEYS:
        node = data
        for key in keys[:-1]:
            node = node.get(key, {})
        leaf = keys[-1]
        if leaf in node and node[leaf] is not None:
            raw = Path(str(node[leaf]))
            node[leaf] = raw if raw.is_absolute() else (PROJECT_ROOT / raw)


def _validate(data: Dict[str, Any]) -> None:
    def need(*keys: str) -> Any:
        node: Any = data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"config.yaml is missing required key: {'.'.join(keys)}")
            node = node[key]
        return node

    for keys in (
        ("project", "run_name"), ("project", "seed"),
        ("data", "root"), ("data", "class_dirs"), ("data", "split"),
        ("image", "size"), ("image", "channels"),
        ("pipeline", "batch_size"),
        ("model", "backbone"), ("model", "weights"), ("model", "dropout"),
        ("training", "stage1"), ("training", "stage2"),
        ("training", "early_stopping"), ("training", "checkpoint"),
        ("evaluation", "threshold_strategy"),
    ):
        need(*keys)

    split = need("data", "split")
    total = float(split["train"]) + float(split["val"]) + float(split["test"])
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"data.split fractions must sum to 1.0, got {total}")
    for name in ("train", "val", "test"):
        if not 0.0 < float(split[name]) < 1.0:
            raise ConfigError(f"data.split.{name} must be strictly between 0 and 1")

    size = need("image", "size")
    if len(size) != 2 or min(size) < 32:
        raise ConfigError("image.size must be [height, width] with both >= 32")

    backbone = str(need("model", "backbone"))
    if backbone not in {"MobileNetV2", "EfficientNetB0"}:
        raise ConfigError(f"model.backbone {backbone!r} is not supported")

    # The direction of a monitored metric is never left to Keras' mode='auto'.
    # 'val_auc_roc' does not contain 'acc', so auto mode would MINIMISE it - the
    # exact bug that made the original project's tuned model score below chance.
    for section in ("early_stopping", "checkpoint"):
        mode = str(need("training", section).get("mode", "")).lower()
        if mode not in {"min", "max"}:
            raise ConfigError(
                f"training.{section}.mode must be explicitly 'min' or 'max', never omitted"
            )

    frequency = str(data.get("model", {}).get("frequency_branch", "none") or "none")
    if frequency not in {"none", "srm"}:
        raise ConfigError(f"model.frequency_branch {frequency!r} is not supported (none | srm)")

    strategy = str(need("evaluation", "threshold_strategy"))
    if strategy not in {"f1", "youden", "fixed"}:
        raise ConfigError(f"evaluation.threshold_strategy {strategy!r} is not supported")

    lr_schedule = str(data.get("training", {}).get("lr_schedule", "none"))
    if lr_schedule not in {"reduce_on_plateau", "cosine", "none"}:
        raise ConfigError(f"training.lr_schedule {lr_schedule!r} is not supported")


def load_config(path: str | Path | None = None) -> Config:
    """Load, validate and resolve config.yaml."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"config file {cfg_path} did not parse to a mapping")
    _validate(data)
    _resolve_paths(data)
    return Config(data, cfg_path)


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Small shared helper so every module writes JSON the same way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
