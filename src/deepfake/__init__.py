"""Deepfake image detection: real vs fake face classification.

Layout:
    config.py    load and validate config.yaml, resolve paths
    data.py      dataset discovery, stratified split manifest, tf.data pipeline
    model.py     backbone + head, preprocessing and augmentation as layers
    train.py     two-stage transfer learning with explicit callback directions
    evaluate.py  sealed-test metrics, bootstrap intervals, calibration
    predict.py   inference against the saved deployment contract
    gradcam.py   attention maps taken from the logit
    cli.py       one entry point for all of the above
"""

__version__ = "2.0.0"

__all__ = ["config", "data", "model", "train", "evaluate", "predict", "gradcam", "cli"]
