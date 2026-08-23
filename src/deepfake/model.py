"""Model construction, compilation and the freeze/unfreeze logic.

Three decisions in here are load-bearing. They are documented in DECISIONS.md.

1. Preprocessing is a LAYER, not a pipeline step.
   MobileNetV2's ImageNet weights expect inputs in [-1, 1], produced by
   x/127.5 - 1. Feeding [0, 1] from Rescaling(1./255) would shift every
   activation in the network off-distribution. Putting the correct rescaling
   inside the model means training and serving cannot disagree: the saved
   model accepts raw [0, 255] pixels and normalises them itself.

2. Augmentation is a LAYER, not a pipeline step.
   Calling .cache() after an augmentation map freezes the first epoch's random
   transforms and replays them forever. As model layers, the transforms run
   per batch, per epoch, and are automatically disabled at inference.

3. The output layer is LINEAR, not sigmoid.
   Training uses from_logits=True, which is numerically stable, and Grad-CAM
   differentiates the logit rather than a saturated sigmoid whose gradients
   vanish. Inference applies the sigmoid explicitly in predict.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import Config

logger = logging.getLogger(__name__)

# How each supported backbone wants its inputs, given raw [0, 255] pixels.
# scale/offset are the arguments to keras.layers.Rescaling.
PREPROCESSING: Dict[str, Dict[str, object]] = {
    # x / 127.5 - 1  ->  [-1, 1]
    "MobileNetV2": {"scale": 1.0 / 127.5, "offset": -1.0, "range": "[-1, 1]"},
    # EfficientNet carries its own Normalization layer, so pass pixels through.
    "EfficientNetB0": {"scale": 1.0, "offset": 0.0, "range": "[0, 255]"},
}


def preprocessing_spec(cfg: Config) -> Dict[str, object]:
    """The preprocessing contract, recorded in label_map.json and asserted by tests."""
    backbone = str(cfg.require("model", "backbone"))
    spec = dict(PREPROCESSING[backbone])
    spec["backbone"] = backbone
    spec["input_pixels"] = "[0, 255] float32, RGB, resized with bilinear interpolation"
    spec["applied_by"] = "the model itself (keras.layers.Rescaling)"
    return spec


# Three classic SRM (spatial rich model) high-pass residual kernels, the same
# ones used by the two-stream manipulation-detection literature. They suppress
# image content and leave the noise residual, which is where splice boundaries,
# resampling traces and compression inconsistencies live. They are FIXED: making
# them trainable lets the network drift back toward content features, which is
# the thing this branch exists to avoid.
SRM_KERNELS = [
    # 5x5 second-order (KV) filter
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    # 3x3 Laplacian-like, embedded in 5x5
    [[0, 0, 0, 0, 0],
     [0, -1, 2, -1, 0],
     [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0],
     [0, 0, 0, 0, 0]],
    # first-order horizontal difference
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
]
SRM_NORMALISERS = [12.0, 4.0, 2.0]


def build_srm_layer(channels: int) -> layers.DepthwiseConv2D:
    """A frozen depthwise convolution holding the SRM kernels.

    depth_multiplier=3 means every input channel is filtered by all three
    kernels, so an RGB image becomes a 9-channel noise residual.
    """
    layer = layers.DepthwiseConv2D(
        kernel_size=5, padding="same", use_bias=False,
        depth_multiplier=len(SRM_KERNELS), trainable=False, name="srm",
    )
    layer.build((None, None, None, channels))
    kernel = np.zeros((5, 5, channels, len(SRM_KERNELS)), dtype="float32")
    for k, (values, norm) in enumerate(zip(SRM_KERNELS, SRM_NORMALISERS)):
        weights = np.asarray(values, dtype="float32") / norm
        for c in range(channels):
            kernel[:, :, c, k] = weights
    layer.set_weights([kernel])
    return layer


def build_frequency_branch(cfg: Config, inputs) -> "keras.KerasTensor | None":
    """Optional second stream over the SRM noise residual.

    The RGB stream carries ImageNet semantics; this one carries local
    high-frequency statistics. They answer different questions, so their pooled
    features are concatenated before the head.
    """
    mode = str(cfg.get("model", "frequency_branch", default="none") or "none").lower()
    if mode == "none":
        return None
    if mode != "srm":
        raise ValueError(f"model.frequency_branch {mode!r} is not supported (none | srm)")

    channels = int(cfg.require("image", "channels"))
    width = int(cfg.get("model", "frequency_width", default=32))

    x = build_srm_layer(channels)(inputs)
    for i, multiplier in enumerate((1, 2, 4)):
        x = layers.Conv2D(width * multiplier, 3, strides=2, padding="same",
                          use_bias=False, name=f"srm_conv_{i}")(x)
        x = layers.BatchNormalization(name=f"srm_bn_{i}")(x)
        x = layers.ReLU(name=f"srm_relu_{i}")(x)
    return layers.GlobalAveragePooling2D(name="srm_gap")(x)


def build_backbone(cfg: Config) -> keras.Model:
    name = str(cfg.require("model", "backbone"))
    weights = cfg.require("model", "weights")
    factory = {
        "MobileNetV2": keras.applications.MobileNetV2,
        "EfficientNetB0": keras.applications.EfficientNetB0,
    }[name]
    backbone = factory(include_top=False, weights=weights, input_shape=cfg.input_shape)
    backbone._name = name.lower()
    return backbone


def build_augmentation(cfg: Config) -> keras.Sequential | None:
    """Conservative, forensics-aware augmentation.

    No vertical flip (faces are never upside down at inference) and only mild
    photometric jitter, because contrast and brightness noise attack exactly the
    high-frequency statistics that betray a splice.
    """
    if not bool(cfg.get("augmentation", "enabled", default=False)):
        return None
    seed = cfg.seed
    steps: List[layers.Layer] = []
    if cfg.get("augmentation", "random_flip_horizontal", default=False):
        steps.append(layers.RandomFlip("horizontal", seed=seed))
    rotation = float(cfg.get("augmentation", "random_rotation", default=0.0))
    if rotation > 0:
        steps.append(layers.RandomRotation(rotation, seed=seed))
    zoom = float(cfg.get("augmentation", "random_zoom", default=0.0))
    if zoom > 0:
        steps.append(layers.RandomZoom(zoom, seed=seed))
    translation = float(cfg.get("augmentation", "random_translation", default=0.0))
    if translation > 0:
        steps.append(layers.RandomTranslation(translation, translation, seed=seed))
    contrast = float(cfg.get("augmentation", "random_contrast", default=0.0))
    if contrast > 0:
        steps.append(layers.RandomContrast(contrast, seed=seed))
    if not steps:
        return None
    return keras.Sequential(steps, name="augmentation")


def build_model(cfg: Config) -> keras.Model:
    """Assemble input -> augment -> normalise -> backbone -> head -> logit."""
    backbone = build_backbone(cfg)
    backbone.trainable = False                      # stage 1: frozen

    spec = PREPROCESSING[str(cfg.require("model", "backbone"))]
    augmentation = build_augmentation(cfg)

    inputs = keras.Input(shape=cfg.input_shape, name="image")
    augmented = inputs
    if augmentation is not None:
        augmented = augmentation(inputs)            # training-only, automatically
    x = layers.Rescaling(float(spec["scale"]), offset=float(spec["offset"]),
                         name="preprocess")(augmented)
    # training=False keeps the backbone's BatchNormalization layers in inference
    # mode for the whole life of the model, including stage 2. Without this the
    # moving statistics estimated on ImageNet get overwritten by a 1.4k-image
    # dataset the moment anything is unfrozen.
    features = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(features)

    # Optional second stream over the SRM noise residual, taken from the same
    # augmented image but BEFORE the ImageNet rescaling, since the residual is
    # a statement about pixel statistics rather than about backbone inputs.
    frequency = build_frequency_branch(cfg, augmented)
    if frequency is not None:
        x = layers.Concatenate(name="fusion")([x, frequency])

    dropout = float(cfg.require("model", "dropout"))
    if dropout > 0:
        x = layers.Dropout(dropout, name="head_dropout")(x)
    for i, units in enumerate(cfg.get("model", "head_units", default=[]) or []):
        x = layers.Dense(int(units), activation="relu", name=f"head_dense_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"head_dropout_{i}")(x)

    # LINEAR output. This is a logit, not a probability.
    outputs = layers.Dense(1, activation=None, name="logit")(x)

    return keras.Model(inputs, outputs, name="deepfake_classifier")


def get_backbone(model: keras.Model) -> keras.Model:
    """Find the nested backbone inside the outer model.

    Looked up by type rather than stored as an attribute, so it also works on a
    model that was loaded back from disk.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential):
            return layer
    raise RuntimeError("no nested backbone model found inside this model")


def build_metrics() -> list:
    """Threshold-free metrics first: they are what this task should be judged on.

    Note threshold=0.0 on accuracy: the model emits logits, and logit 0
    corresponds to probability 0.5.
    """
    return [
        keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
        keras.metrics.AUC(name="auc_roc", from_logits=True),
        keras.metrics.AUC(name="auc_pr", curve="PR", from_logits=True),
    ]


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=build_metrics(),
    )
    return model


def unfreeze_top(model: keras.Model, n_layers: int) -> Dict[str, int]:
    """Unfreeze the last n backbone layers, keeping every BatchNorm frozen.

    Returns a small report so the caller can log exactly what changed.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    total = len(backbone.layers)
    cut = max(total - int(n_layers), 0)
    unfrozen, frozen_bn = 0, 0
    for i, layer in enumerate(backbone.layers):
        if i < cut:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            # Fine-tuning BatchNorm on a small dataset destroys the ImageNet
            # moving statistics. Keep them frozen.
            layer.trainable = False
            frozen_bn += 1
        else:
            layer.trainable = True
            unfrozen += 1
    report = {
        "backbone_layers": total,
        "unfrozen": unfrozen,
        "frozen_batchnorm": frozen_bn,
        "frozen": total - unfrozen,
    }
    logger.info("stage 2 unfreeze: %s", report)
    return report


def count_parameters(model: keras.Model) -> Dict[str, int]:
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    non_trainable = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
