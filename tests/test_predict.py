"""Label polarity is the single most dangerous bug class in a binary
classifier: everything runs, nothing errors, and every answer is backwards.

The original notebook shipped exactly that: with class_names ['fake', 'real'],
a sigmoid output above 0.5 means REAL, but the code returned
{'class': 'real', 'label': 'Fake'}. These tests pin the correct behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from deepfake.predict import Predictor
from deepfake.utils import label_map_payload
from deepfake import model as model_mod


class _ConstantLogit(keras.Model):
    """A stand-in model that returns a fixed logit, so the polarity logic can be
    tested without training anything."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = float(logit)

    def call(self, inputs, training=False):
        batch = tf.shape(inputs)[0]
        return tf.fill((batch, 1), self.logit)


@pytest.fixture
def contract(cfg):
    return label_map_payload(cfg, model_mod.preprocessing_spec(cfg), threshold=0.5)


def _image_path(cfg):
    from deepfake import data as data_mod
    data_mod.build_manifest(cfg)
    return data_mod.split_frame(cfg, "test")["path"].iloc[0]


def test_high_probability_means_real(cfg, contract):
    """logit +4 -> P(real) ~ 0.982 -> the answer must be 'real'."""
    predictor = Predictor(_ConstantLogit(4.0), contract)
    result = predictor.predict_one(_image_path(cfg))
    assert result["probability_real"] > 0.9
    assert result["predicted_index"] == 1
    assert result["predicted_class"] == "real"


def test_low_probability_means_fake(cfg, contract):
    """logit -4 -> P(real) ~ 0.018 -> the answer must be 'fake'."""
    predictor = Predictor(_ConstantLogit(-4.0), contract)
    result = predictor.predict_one(_image_path(cfg))
    assert result["probability_real"] < 0.1
    assert result["predicted_index"] == 0
    assert result["predicted_class"] == "fake"


@pytest.mark.parametrize("logit", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_class_and_label_never_disagree(cfg, contract, logit):
    """The exact contradiction the old predict_image() produced."""
    predictor = Predictor(_ConstantLogit(logit), contract)
    result = predictor.predict_one(_image_path(cfg))
    assert result["predicted_class"] == result["label"]
    assert result["predicted_class"] == contract["class_names"][result["predicted_index"]]


def test_probabilities_are_complementary(cfg, contract):
    predictor = Predictor(_ConstantLogit(0.7), contract)
    result = predictor.predict_one(_image_path(cfg))
    assert np.isclose(result["probability_real"] + result["probability_fake"], 1.0, atol=1e-3)


def test_threshold_is_honoured(cfg, contract):
    predictor = Predictor(_ConstantLogit(0.5), contract)     # P(real) ~ 0.62
    assert predictor.predict_one(_image_path(cfg), threshold=0.5)["predicted_class"] == "real"
    assert predictor.predict_one(_image_path(cfg), threshold=0.8)["predicted_class"] == "fake"


def test_abstain_band_flags_uncertain_predictions(cfg, contract):
    predictor = Predictor(_ConstantLogit(0.0), contract)      # P(real) = 0.5 exactly
    result = predictor.predict_one(_image_path(cfg), abstain_margin=0.1)
    assert result["uncertain"] is True
    assert result["decision"] == "abstain"


def test_loaded_image_is_raw_pixels_at_the_contract_size(cfg, contract):
    predictor = Predictor(_ConstantLogit(0.0), contract)
    image = predictor.load_image(_image_path(cfg))
    assert image.shape == (32, 32, 3)
    assert image.max() > 1.5, "predict must not pre-normalise; the model does that"


def test_predictor_rejects_a_probability_model(cfg, contract):
    """If someone swaps in a sigmoid model, fail loudly instead of double-applying."""
    broken = dict(contract, output="probability")
    with pytest.raises(ValueError, match="logits"):
        Predictor(_ConstantLogit(0.0), broken)
