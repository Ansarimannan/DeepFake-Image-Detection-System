"""Dataset discovery, splitting and the tf.data input pipeline.

Two responsibilities, kept strictly apart:

1. `build_manifest` walks the raw dataset once and writes a CSV listing every
   image with its class, label and split assignment. The split is stratified
   and seeded, and it is written to disk so that training, evaluation and any
   later analysis all see byte-identical splits. The test split is sealed: it
   is only ever read by `evaluate`.

2. `make_dataset` turns one split of that manifest into a tf.data pipeline.

What this module deliberately does NOT do: normalise pixels or augment them.
Both live inside the Keras model (see model.py) so that the training pipeline
and the serving path cannot drift apart, and so that `cache()` here can never
freeze the random augmentations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- manifest
def discover_images(cfg: Config) -> pd.DataFrame:
    """List every image under the configured class directories."""
    extensions = {str(e).lower() for e in cfg.require("data", "extensions")}
    rows: List[Dict[str, object]] = []

    for label, class_name in enumerate(cfg.class_names):  # fake=0, real=1
        directory = cfg.class_dir(class_name)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"class directory for {class_name!r} not found: {directory}\n"
                f"Check data.root and data.class_dirs in {cfg.source}"
            )
        found = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        if not found:
            raise FileNotFoundError(f"no images with {sorted(extensions)} in {directory}")
        for path in found:
            rows.append({
                "path": str(path),
                "filename": path.name,
                "class_name": class_name,
                "label": label,
                "difficulty": _difficulty_from_name(path.name),
                "region_mask": _region_mask_from_name(path.name),
            })

    frame = pd.DataFrame(rows)
    logger.info("discovered %d images: %s", len(frame),
                frame["class_name"].value_counts().to_dict())
    return frame


def _difficulty_from_name(name: str) -> str:
    """CIPLAB fake filenames start with easy_/mid_/hard_. Real images have none."""
    head = name.split("_", 1)[0].lower()
    return head if head in {"easy", "mid", "hard"} else "none"


def _region_mask_from_name(name: str) -> str:
    """CIPLAB fake filenames end in a 4-bit mask: left eye, right eye, nose, mouth.

    e.g. easy_100_1111.jpg -> '1111'. Kept in the manifest because it enables
    difficulty-stratified reporting and, later, auxiliary region supervision.
    """
    stem = Path(name).stem
    tail = stem.rsplit("_", 1)[-1]
    return tail if len(tail) == 4 and set(tail) <= {"0", "1"} else ""


def build_manifest(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Create (or reuse) the stratified train/val/test manifest."""
    manifest_path = Path(cfg.require("data", "split", "manifest"))
    if manifest_path.exists() and not force:
        logger.info("reusing existing manifest: %s", manifest_path)
        return pd.read_csv(manifest_path)

    frame = discover_images(cfg)
    fractions = {s: float(cfg.require("data", "split", s)) for s in SPLITS}
    rng = np.random.default_rng(cfg.seed)

    assignments = np.empty(len(frame), dtype=object)
    # Stratify: split each class independently so the class ratio is preserved
    # in all three splits rather than only in expectation.
    for label in sorted(frame["label"].unique()):
        idx = frame.index[frame["label"] == label].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * fractions["train"]))
        n_val = int(round(n * fractions["val"]))
        n_train = min(n_train, n - 2)               # guarantee val and test are non-empty
        n_val = min(n_val, n - n_train - 1)
        assignments[idx[:n_train]] = "train"
        assignments[idx[n_train:n_train + n_val]] = "val"
        assignments[idx[n_train + n_val:]] = "test"

    frame["split"] = assignments
    _assert_disjoint(frame)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    logger.info("wrote manifest %s", manifest_path)
    logger.info("split sizes: %s", frame["split"].value_counts().to_dict())
    return frame


def _assert_disjoint(frame: pd.DataFrame) -> None:
    """A file may appear in exactly one split. Cheap, and catches a fatal class of bug."""
    counts = frame.groupby("path")["split"].nunique()
    leaked = counts[counts > 1]
    if len(leaked):
        raise RuntimeError(f"{len(leaked)} files appear in more than one split")
    if frame["split"].isna().any():
        raise RuntimeError("some rows were never assigned to a split")


def load_manifest(cfg: Config) -> pd.DataFrame:
    manifest_path = Path(cfg.require("data", "split", "manifest"))
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}\nRun:  python -m deepfake.cli split"
        )
    return pd.read_csv(manifest_path)


def split_frame(cfg: Config, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    frame = load_manifest(cfg)
    subset = frame[frame["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise RuntimeError(f"split {split!r} is empty in the manifest")
    return subset


# ---------------------------------------------------------------- pipeline
def _decode(path: tf.Tensor, label: tf.Tensor, size: Tuple[int, int], channels: int):
    """Read a file, decode it, resize it. Pixels stay in [0, 255] float32.

    Normalisation is a model layer, not a pipeline step - see model.py.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=channels, expand_animations=False)
    image = tf.image.resize(image, size, method="bilinear")
    image = tf.cast(image, tf.float32)
    image.set_shape([size[0], size[1], channels])
    return image, tf.cast(label, tf.float32)


def make_dataset(cfg: Config, split: str, shuffle: bool | None = None,
                 frame: pd.DataFrame | None = None) -> tf.data.Dataset:
    """Build the tf.data pipeline for one split.

    Operation order is deliberate:
        slices -> map(decode+resize) -> cache -> [shuffle] -> batch -> prefetch

    cache() sits after the expensive deterministic work and before shuffling,
    so decoding happens once while the epoch order still varies. Because
    augmentation is a model layer it is applied after this pipeline, per batch,
    per epoch, and is therefore never frozen by the cache.
    """
    frame = split_frame(cfg, split) if frame is None else frame
    if shuffle is None:
        shuffle = (split == "train")   # validation and test are never shuffled

    size = cfg.image_size
    channels = int(cfg.require("image", "channels"))
    batch_size = int(cfg.require("pipeline", "batch_size"))

    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _decode(p, y, size, channels),
                num_parallel_calls=tf.data.AUTOTUNE)

    if bool(cfg.require("pipeline", "cache")):
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(int(cfg.require("pipeline", "shuffle_buffer")),
                        seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if bool(cfg.require("pipeline", "prefetch")):
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def class_weights(cfg: Config, frame: pd.DataFrame) -> Dict[int, float] | None:
    """Balanced class weights, but only when the imbalance actually warrants them."""
    counts = frame["label"].value_counts().to_dict()
    if len(counts) < 2:
        raise RuntimeError("training split contains only one class")
    ratio = max(counts.values()) / min(counts.values())
    threshold = float(cfg.require("training", "class_weight_threshold"))
    if ratio <= threshold:
        logger.info("imbalance ratio %.2fx <= %.2fx, no class weights applied",
                    ratio, threshold)
        return None
    total = sum(counts.values())
    weights = {int(k): total / (len(counts) * v) for k, v in counts.items()}
    logger.info("imbalance ratio %.2fx, class weights %s", ratio, weights)
    return weights


def dataset_summary(cfg: Config) -> Dict[str, object]:
    """Counts per split and per class, for the run report."""
    frame = load_manifest(cfg)
    table = (frame.groupby(["split", "class_name"]).size()
             .unstack(fill_value=0).reindex(list(SPLITS)))
    summary = {
        "total": int(len(frame)),
        "per_split": {s: int(frame[frame["split"] == s].shape[0]) for s in SPLITS},
        "per_split_per_class": {
            s: {c: int(table.loc[s, c]) for c in table.columns} for s in table.index
        },
    }
    for s in SPLITS:
        counts = summary["per_split_per_class"][s]
        n = sum(counts.values())
        summary.setdefault("majority_baseline", {})[s] = (
            round(max(counts.values()) / n, 4) if n else 0.0
        )
    return summary
