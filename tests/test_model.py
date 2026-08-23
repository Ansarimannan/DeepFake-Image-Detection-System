"""The model carries its own preprocessing and augmentation. Both are asserted
here, because a mismatch between training and serving is invisible at runtime."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from deepfake import model as model_mod


def test_mobilenet_preprocessing_maps_to_minus_one_to_one(cfg):
    """MobileNetV2 ImageNet weights expect [-1, 1], not [0, 1].

    Feeding [0, 1] (Rescaling(1./255)) was the original project's silent
    accuracy killer, so the exact mapping is pinned by this test.
    """
    spec = model_mod.preprocessing_spec(cfg)
    assert spec["range"] == "[-1, 1]"
    assert np.isclose(0.0 * spec["scale"] + spec["offset"], -1.0)
    assert np.isclose(255.0 * spec["scale"] + spec["offset"], 1.0)


def test_preprocess_layer_is_inside_the_model(cfg):
    model = model_mod.build_model(cfg)
    layer = model.get_layer("preprocess")
    black = layer(tf.zeros((1, 32, 32, 3))).numpy()
    white = layer(tf.fill((1, 32, 32, 3), 255.0)).numpy()
    assert np.allclose(black, -1.0)
    assert np.allclose(white, 1.0)


def test_output_is_a_single_linear_logit(cfg):
    """A linear output keeps the loss numerically stable and lets Grad-CAM
    differentiate the logit instead of a saturated sigmoid."""
    model = model_mod.build_model(cfg)
    assert model.output_shape == (None, 1)
    logit_layer = model.get_layer("logit")
    assert logit_layer.activation.__name__ == "linear"

    # Force the head to emit +5. A sigmoid output could never leave [0, 1], so
    # seeing 5 proves nothing squashes the output.
    weights, bias = logit_layer.get_weights()
    logit_layer.set_weights([np.zeros_like(weights), np.full_like(bias, 5.0)])
    logits = model.predict(np.zeros((2, 32, 32, 3), dtype="float32"), verbose=0)
    assert logits.shape == (2, 1)
    assert np.allclose(logits, 5.0), f"output was squashed: {logits.ravel()}"


def test_augmentation_is_training_only(cfg):
    model = model_mod.build_model(cfg)
    batch = np.random.default_rng(0).uniform(0, 255, (4, 32, 32, 3)).astype("float32")
    a = model(batch, training=False).numpy()
    b = model(batch, training=False).numpy()
    assert np.allclose(a, b), "inference must be deterministic"


def test_backbone_starts_frozen(cfg):
    model = model_mod.build_model(cfg)
    backbone = model_mod.get_backbone(model)
    assert backbone.trainable is False
    counts = model_mod.count_parameters(model)
    assert counts["trainable"] < counts["non_trainable"]


def test_unfreeze_keeps_batchnorm_frozen(cfg):
    """Fine-tuning BatchNorm on a small dataset destroys the ImageNet moving
    statistics, so it must stay frozen even when its neighbours are unfrozen."""
    model = model_mod.build_model(cfg)
    report = model_mod.unfreeze_top(model, 30)
    backbone = model_mod.get_backbone(model)
    tail = backbone.layers[-30:]
    bn_layers = [l for l in tail if isinstance(l, layers.BatchNormalization)]
    assert bn_layers, "expected BatchNorm layers in the unfrozen tail"
    assert all(l.trainable is False for l in bn_layers)
    assert report["unfrozen"] > 0
    assert report["frozen_batchnorm"] == len(bn_layers)


def test_metrics_are_threshold_free_and_logit_aware(cfg):
    """AUC must know it is receiving logits, and accuracy must threshold at 0,
    not at 0.5, because 0 is the logit of probability 0.5."""
    metrics = {m.name: m for m in model_mod.build_metrics()}
    assert set(metrics) == {"accuracy", "auc_roc", "auc_pr"}
    assert float(metrics["accuracy"].get_config()["threshold"]) == 0.0
    assert metrics["auc_roc"].get_config()["from_logits"] is True
    assert metrics["auc_pr"].get_config()["curve"].upper() == "PR"


def test_srm_kernels_are_high_pass_and_frozen(cfg):
    """An SRM residual filter must sum to zero: it suppresses image content and
    leaves the noise residual. A non-zero sum would leak brightness through."""
    layer = model_mod.build_srm_layer(3)
    kernel = layer.get_weights()[0]                 # (5, 5, channels, 3)
    assert layer.trainable is False
    assert kernel.shape == (5, 5, 3, 3)
    for k in range(kernel.shape[-1]):
        for c in range(kernel.shape[2]):
            assert abs(float(kernel[:, :, c, k].sum())) < 1e-6

    # A flat image has no residual, so the interior response must be ~0. The
    # outer 2-pixel ring is excluded: 'same' padding pads with zeros, which is
    # itself an edge the high-pass filter correctly reacts to.
    flat = tf.fill((1, 32, 32, 3), 128.0)
    interior = layer(flat)[:, 2:-2, 2:-2, :]
    assert float(tf.reduce_max(tf.abs(interior))) < 1e-3


def test_frequency_branch_is_off_by_default(cfg):
    model = model_mod.build_model(cfg)
    assert "srm" not in [l.name for l in model.layers]


def test_frequency_branch_fuses_two_streams(cfg):
    cfg._data["model"]["frequency_branch"] = "srm"
    model = model_mod.build_model(cfg)
    names = [l.name for l in model.layers]
    assert "srm" in names and "srm_gap" in names and "fusion" in names
    assert model.output_shape == (None, 1)
    # The fused head sees the backbone features plus the residual features.
    assert model.get_layer("fusion").output_shape[-1] > model.get_layer("gap").output_shape[-1]


def test_compile_sets_a_from_logits_loss(cfg):
    model = model_mod.build_model(cfg)
    model_mod.compile_model(model, 1e-4)
    assert model.loss.get_config()["from_logits"] is True
