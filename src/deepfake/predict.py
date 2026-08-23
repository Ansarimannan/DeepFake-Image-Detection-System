"""Single-image and batch inference.

The label-polarity bug from the original notebook lived here. With
class_names = ['fake', 'real'] the label index of 'real' is 1, so the sigmoid
output is P(real) and a value at or above the threshold means REAL. The old
code returned {'class': 'real', 'label': 'Fake'} for the same image.

The fix is to never write a class name as a literal. The predicted name is
always class_names[int(probability >= threshold)], and tests/test_predict.py
asserts that the returned class and label agree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import Config, load_config
from .utils import read_json, setup_logging

logger = logging.getLogger(__name__)


class Predictor:
    """A loaded model plus its deployment contract.

    The contract (label_map.json) carries the class order, the image size, the
    preprocessing description and the operating threshold. Loading them together
    is what stops training and serving from drifting apart.
    """

    def __init__(self, model: keras.Model, label_map: Dict[str, object]):
        self.model = model
        self.label_map = label_map
        self.class_names: List[str] = list(label_map["class_names"])
        self.image_size = tuple(int(v) for v in label_map["image_size"])
        self.channels = int(label_map.get("channels", 3))
        self.threshold = float(label_map.get("threshold", 0.5))
        if label_map.get("output") != "logit":
            raise ValueError(
                "label_map.json says the model does not output logits; "
                "this Predictor applies the sigmoid itself"
            )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_run(cls, cfg: Config | None = None) -> "Predictor":
        cfg = cfg or load_config()
        run_dir = cfg.run_dir
        model_path = run_dir / "model.keras"
        contract_path = run_dir / "label_map.json"
        for path in (model_path, contract_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path.name} in {run_dir}\n"
                    f"Run:  python -m deepfake.cli train"
                )
        return cls(keras.models.load_model(model_path), read_json(contract_path))

    # -- preprocessing -----------------------------------------------------
    def load_image(self, path: str | Path) -> np.ndarray:
        """Decode and resize only. Normalisation is a layer inside the model.

        This is exactly what data.make_dataset does, so the training path and
        this path cannot diverge.
        """
        raw = tf.io.read_file(str(path))
        image = tf.io.decode_image(raw, channels=self.channels, expand_animations=False)
        image = tf.image.resize(image, self.image_size, method="bilinear")
        return tf.cast(image, tf.float32).numpy()

    # -- inference ---------------------------------------------------------
    def predict_batch(self, paths: Iterable[str | Path],
                      threshold: float | None = None,
                      abstain_margin: float = 0.0) -> List[Dict[str, object]]:
        paths = [Path(p) for p in paths]
        if not paths:
            return []
        threshold = self.threshold if threshold is None else float(threshold)
        batch = np.stack([self.load_image(p) for p in paths])
        logits = self.model.predict(batch, verbose=0).reshape(-1)
        probabilities = tf.sigmoid(logits).numpy()

        results = []
        for path, logit, probability in zip(paths, logits, probabilities):
            index = int(probability >= threshold)
            name = self.class_names[index]        # never a literal string
            uncertain = abs(float(probability) - threshold) < abstain_margin
            results.append({
                "path": str(path),
                "logit": round(float(logit), 4),
                "probability_real": round(float(probability), 4),
                "probability_fake": round(float(1.0 - probability), 4),
                "predicted_index": index,
                "predicted_class": name,
                "label": name,                    # identical by construction
                "threshold": threshold,
                "uncertain": bool(uncertain),
                "decision": "abstain" if uncertain else name,
            })
        return results

    def predict_one(self, path: str | Path, **kwargs) -> Dict[str, object]:
        return self.predict_batch([path], **kwargs)[0]


def predict_paths(paths: Iterable[str | Path], cfg: Config | None = None,
                  abstain: bool = True) -> List[Dict[str, object]]:
    cfg = cfg or load_config()
    predictor = Predictor.from_run(cfg)
    margin = float(cfg.get("evaluation", "abstain_margin", default=0.0) or 0.0) if abstain else 0.0
    return predictor.predict_batch(paths, abstain_margin=margin)


def main(argv: List[str] | None = None) -> None:
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser(description="Classify images as real or fake.")
    parser.add_argument("paths", nargs="+", help="image files to classify")
    args = parser.parse_args(argv)
    for result in predict_paths(args.paths):
        print(f"{Path(result['path']).name:40s} {result['decision']:9s} "
              f"P(real)={result['probability_real']:.3f}")


if __name__ == "__main__":
    main()
