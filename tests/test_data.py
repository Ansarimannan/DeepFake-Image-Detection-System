"""The split and the input pipeline are the two places where silent, fatal
data bugs live. Both are asserted here rather than eyeballed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepfake import data as data_mod


def test_manifest_covers_every_image_exactly_once(cfg):
    frame = data_mod.build_manifest(cfg)
    assert len(frame) == 26                       # 12 fake + 14 real
    assert frame["path"].nunique() == len(frame)
    assert set(frame["split"]) == {"train", "val", "test"}


def test_splits_are_disjoint(cfg):
    frame = data_mod.build_manifest(cfg)
    groups = {s: set(frame[frame["split"] == s]["path"]) for s in data_mod.SPLITS}
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])


def test_split_is_stratified(cfg):
    """Both classes must appear in every split, in roughly the source proportion."""
    frame = data_mod.build_manifest(cfg)
    for split in data_mod.SPLITS:
        labels = set(frame[frame["split"] == split]["label"])
        assert labels == {0, 1}, f"{split} lost a class"


def test_split_is_reproducible(cfg):
    first = data_mod.build_manifest(cfg, force=True).sort_values("path")
    second = data_mod.build_manifest(cfg, force=True).sort_values("path")
    assert list(first["split"]) == list(second["split"])


def test_labels_follow_the_class_order(cfg):
    frame = data_mod.build_manifest(cfg)
    assert set(frame[frame["class_name"] == "fake"]["label"]) == {0}
    assert set(frame[frame["class_name"] == "real"]["label"]) == {1}


def test_filename_metadata_is_parsed(cfg):
    frame = data_mod.build_manifest(cfg)
    fakes = frame[frame["class_name"] == "fake"]
    assert set(fakes["difficulty"]) == {"easy"}
    assert set(fakes["region_mask"]) == {"1010"}
    assert set(frame[frame["class_name"] == "real"]["difficulty"]) == {"none"}


def test_pipeline_yields_raw_pixels_of_the_right_shape(cfg):
    data_mod.build_manifest(cfg)
    batch_images, batch_labels = next(iter(data_mod.make_dataset(cfg, "train")))
    assert batch_images.shape[1:] == (32, 32, 3)
    assert batch_images.dtype.name == "float32"
    # Normalisation is a model layer, so the pipeline must NOT rescale.
    assert float(batch_images.numpy().max()) > 1.5
    assert set(np.unique(batch_labels.numpy())) <= {0.0, 1.0}


def test_validation_and_test_are_not_shuffled(cfg):
    """Evaluation pairs predictions with manifest rows by position, so order
    must be stable across iterations."""
    data_mod.build_manifest(cfg)
    frame = data_mod.split_frame(cfg, "test")
    ds = data_mod.make_dataset(cfg, "test", frame=frame)
    first = np.concatenate([y.numpy() for _, y in ds])
    second = np.concatenate([y.numpy() for _, y in ds])
    assert np.array_equal(first, second)
    assert np.array_equal(first, frame["label"].to_numpy().astype("float32"))


def test_class_weights_are_skipped_when_balanced(cfg):
    frame = pd.DataFrame({"label": [0] * 100 + [1] * 110})
    assert data_mod.class_weights(cfg, frame) is None


def test_class_weights_are_applied_when_imbalanced(cfg):
    frame = pd.DataFrame({"label": [0] * 100 + [1] * 900})
    weights = data_mod.class_weights(cfg, frame)
    assert weights is not None
    assert weights[0] > weights[1]                # the rare class gets the larger weight


def test_summary_reports_the_majority_baseline(cfg):
    data_mod.build_manifest(cfg)
    summary = data_mod.dataset_summary(cfg)
    assert summary["total"] == 26
    for split in data_mod.SPLITS:
        assert 0.0 < summary["majority_baseline"][split] < 1.0
