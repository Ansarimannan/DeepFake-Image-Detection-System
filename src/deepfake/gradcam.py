"""Grad-CAM attention maps.

Two things here are non-obvious.

1. Finding the feature map. The textbook recipe is
   model.get_layer('out_relu'), which fails when the backbone is added as a
   single nested Functional layer: its internal layers are not members of the
   outer model and the graph cannot be traced across the boundary. Instead we
   take the tensor feeding GlobalAveragePooling2D. That tensor IS the
   backbone's final feature map and it lives in the outer graph, so this
   approach is both simpler and backbone-agnostic (no layer name is ever
   mentioned).

2. Differentiating the right thing. Differentiating the sigmoid output
   produces noise once the sigmoid saturates, because its gradient vanishes.
   This model emits a logit, so the gradient is taken with respect to the
   logit and stays informative at any confidence.

A caveat that belongs next to every heatmap produced here: attribution
explains what a model looked at, not whether it is right. Read the accuracy
first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import data as data_mod
from .config import Config, load_config
from .predict import Predictor
from .utils import set_seeds, setup_logging

logger = logging.getLogger(__name__)


def feature_tensor(model: keras.Model) -> tf.Tensor:
    """The spatial feature map entering the backbone's GlobalAveragePooling2D.

    Named "gap" by build_model. The name is checked first because a model with
    the SRM frequency branch has a second pooling layer ("srm_gap"), and the
    attention map we want is the backbone's.
    """
    try:
        return model.get_layer("gap").input
    except ValueError:
        pass
    for layer in model.layers:
        if isinstance(layer, layers.GlobalAveragePooling2D):
            return layer.input
    raise RuntimeError("no GlobalAveragePooling2D layer found; cannot locate features")


def heatmap(model: keras.Model, image: np.ndarray) -> np.ndarray:
    """Grad-CAM for one image, shape (H, W) normalised to [0, 1]."""
    grad_model = keras.Model(model.inputs, [feature_tensor(model), model.output])
    batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        features, logit = grad_model(batch, training=False)
        tape.watch(features)
        score = logit[:, 0]                       # the logit, not the sigmoid

    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))          # one weight per channel
    cam = tf.reduce_sum(features[0] * weights, axis=-1)      # weighted channel sum
    cam = tf.maximum(cam, 0)                                 # positive evidence only
    peak = tf.reduce_max(cam)
    if peak > 0:
        cam = cam / peak
    return cam.numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a jet-coloured heatmap over the original image, both in [0, 1]."""
    import matplotlib.cm as cm

    resized = tf.image.resize(cam[..., None], image.shape[:2]).numpy()[..., 0]
    coloured = cm.get_cmap("jet")(np.uint8(255 * resized))[..., :3]
    base = image / 255.0
    return np.clip(coloured * alpha + base * (1 - alpha) + base * alpha, 0, 1)


def run(cfg: Config | None = None, split: str = "test") -> Path:
    """Render a grid of original / heatmap / overlay for a few test images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    set_seeds(cfg.seed)
    predictor = Predictor.from_run(cfg)
    n = int(cfg.require("gradcam", "samples"))
    alpha = float(cfg.require("gradcam", "alpha"))

    frame = data_mod.split_frame(cfg, split)
    # A balanced sample: half fake, half real, so the figure is not all one class.
    per_class = max(n // 2, 1)
    picks = pd.concat([
        group.sample(min(per_class, len(group)), random_state=cfg.seed)
        for _, group in frame.groupby("label")
    ]).reset_index(drop=True)

    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)

    for row, (_, record) in enumerate(picks.iterrows()):
        image = predictor.load_image(record["path"])
        result = predictor.predict_one(record["path"])
        cam = heatmap(predictor.model, image)

        axes[row, 0].imshow(image.astype("uint8"))
        axes[row, 0].set_title(f"true: {record['class_name']}", fontsize=10)
        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title(
            f"pred: {result['predicted_class']}  P(real)={result['probability_real']:.3f}",
            fontsize=10)
        axes[row, 2].imshow(overlay(image, cam, alpha))
        axes[row, 2].set_title("overlay", fontsize=10)
        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle("Grad-CAM on the test split (attribution explains attention, not correctness)",
                 fontsize=12)
    fig.tight_layout()
    out = cfg.run_dir / "gradcam.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def main() -> None:
    setup_logging()
    run()


if __name__ == "__main__":
    main()
