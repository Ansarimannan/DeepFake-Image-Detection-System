"""Evaluation must be sound by construction: baselines reported, thresholds
chosen on validation only, and every metric a measured classification statistic."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from deepfake import evaluate as eval_mod


def _perfect(n=200):
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    p = np.where(y == 1, 0.9, 0.1)
    return y, p


def test_metrics_on_a_perfect_classifier(cfg):
    y, p = _perfect()
    m = eval_mod.compute_metrics(y, p, 0.5, ["fake", "real"], cfg, "test")
    assert m["accuracy"] == 1.0
    assert m["auc_roc"] == 1.0
    assert m["per_class"]["fake"]["recall"] == 1.0
    assert m["confusion_matrix"]["fp_fake_called_real"] == 0


def test_metrics_on_a_coin_flip(cfg):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = rng.uniform(0, 1, 400)
    m = eval_mod.compute_metrics(y, p, 0.5, ["fake", "real"], cfg, "test")
    assert 0.35 < m["auc_roc"] < 0.65, "a random scorer must land near 0.5 AUC"
    assert abs(m["lift_over_baseline"]) < 0.2


def test_majority_baseline_is_always_reported(cfg):
    y = np.array([1] * 90 + [0] * 10)
    p = np.full(100, 0.9)
    m = eval_mod.compute_metrics(y, p, 0.5, ["fake", "real"], cfg, "test")
    assert m["majority_baseline"] == 0.9
    assert m["accuracy"] == 0.9
    assert m["lift_over_baseline"] == 0.0, "predicting the majority class adds nothing"


def test_confidence_interval_brackets_the_point_estimate(cfg):
    y, p = _perfect(120)
    p = p + np.random.default_rng(1).normal(0, 0.15, len(p))
    p = np.clip(p, 0, 1)
    m = eval_mod.compute_metrics(y, p, 0.5, ["fake", "real"], cfg, "test")
    low, high = m["confidence_interval"]["accuracy"]
    assert low <= m["accuracy"] <= high


def test_calibration_error_is_small_for_calibrated_scores(cfg):
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)     # perfectly calibrated by construction
    m = eval_mod.compute_metrics(y, p, 0.5, ["fake", "real"], cfg, "test")
    assert m["calibration"]["expected_calibration_error"] < 0.05


def test_threshold_is_chosen_from_the_data_not_hardcoded(cfg):
    y = np.array([0] * 100 + [1] * 100)
    p = np.concatenate([np.full(100, 0.05), np.full(100, 0.30)])
    info = eval_mod.choose_threshold(y, p, cfg)
    assert 0.05 < info["threshold"] <= 0.30, info
    assert info["strategy"] == "f1"


def test_abstain_band_is_reported(cfg):
    y, p = _perfect(100)
    p = np.full(100, 0.52)                             # everything sits on the fence
    m = eval_mod.compute_metrics(y, p, 0.5, ["fake", "real"], cfg, "test")
    assert m["abstain"]["abstained"] == 100


def test_no_random_or_regression_metric_calls():
    """Every reported metric must be a measured classification statistic:
    no module may call a random-number generator in place of a metric, and
    R-squared (a regression statistic) is banned for this binary label.

    The check parses the AST rather than grepping the text, so mentions in
    docstrings do not trip it and a rename cannot hide a violation.
    """
    import ast

    def call_name(node: ast.Call) -> str:
        parts = []
        func = node.func
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))

    banned = {"np.random.uniform", "numpy.random.uniform", "random.uniform",
              "r2_score", "sklearn.metrics.r2_score"}
    for path in Path(eval_mod.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = call_name(node)
                assert name not in banned, f"{path.name} calls {name}()"
