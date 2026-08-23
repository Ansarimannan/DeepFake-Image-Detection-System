"""Sealed-test evaluation.

Rules this module enforces:

* The operating threshold is chosen on the VALIDATION split and then applied
  unchanged to the TEST split. Choosing a threshold on the data you report is
  how a weak model is made to look strong.
* Every headline number is reported next to the majority-class baseline and
  with a bootstrap confidence interval. On a few hundred images the interval is
  wide enough to change how a result should be described.
* No R-squared. It is a regression statistic for a continuous target and it is
  not interpretable for a Bernoulli label; the appropriate probabilistic scores
  are Brier score, log loss and a calibration curve, all computed below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from . import data as data_mod
from .config import Config, load_config
from .utils import label_map_payload, read_json, set_seeds, setup_logging, write_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- inference
def predict_split(model: keras.Model, cfg: Config, split: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (y_true, probability, frame) for one split, order preserved.

    The split is not shuffled (see data.make_dataset), so the rows of `frame`
    line up with the returned arrays. That alignment is asserted below rather
    than assumed.

    With evaluation.tta_hflip enabled, each image is also scored mirrored and the
    two probabilities are averaged. A horizontal flip is a genuine symmetry of a
    face, and it is already in the training augmentation, so averaging over it
    reduces prediction variance without changing what the model represents.
    """
    frame = data_mod.split_frame(cfg, split)
    ds = data_mod.make_dataset(cfg, split, shuffle=False, frame=frame)
    proba = tf.sigmoid(model.predict(ds, verbose=0).reshape(-1)).numpy()

    if bool(cfg.get("evaluation", "tta_hflip", default=False)):
        flipped = ds.map(lambda x, y: (tf.image.flip_left_right(x), y),
                         num_parallel_calls=tf.data.AUTOTUNE)
        mirrored = tf.sigmoid(model.predict(flipped, verbose=0).reshape(-1)).numpy()
        proba = (proba + mirrored) / 2.0
        logger.info("%s: averaged over horizontal-flip TTA", split)

    y_true = frame["label"].to_numpy().astype(int)
    if len(proba) != len(y_true):
        raise RuntimeError(
            f"prediction/label length mismatch on {split}: {len(proba)} vs {len(y_true)}"
        )
    return y_true, proba, frame


# ---------------------------------------------------------------- threshold
def choose_threshold(y_true: np.ndarray, proba: np.ndarray, cfg: Config) -> Dict[str, float]:
    """Pick the operating point on the validation split.

    'f1'     maximises F1 for the FAKE class, the class you actually want to
             catch, which is label 0, so it is scored on the flipped problem.
    'youden' maximises tpr - fpr, a cost-neutral choice.
    'fixed'  uses evaluation.threshold_value unchanged.
    """
    strategy = str(cfg.require("evaluation", "threshold_strategy"))
    if strategy == "fixed":
        return {"threshold": float(cfg.require("evaluation", "threshold_value")),
                "strategy": strategy}

    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        best = int(np.argmax(tpr - fpr))
        return {"threshold": float(thresholds[best]), "strategy": strategy,
                "youden_j": float((tpr - fpr)[best])}

    # strategy == "f1", scored on the fake class (label 0, low probability)
    fake_true = 1 - y_true
    fake_score = 1.0 - proba
    precision, recall, thresholds = precision_recall_curve(fake_true, fake_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    fake_threshold = float(thresholds[best]) if len(thresholds) else 0.5
    return {
        "threshold": float(1.0 - fake_threshold),   # back to P(real) space
        "strategy": strategy,
        "fake_f1_at_threshold": float(f1[best]) if len(thresholds) else 0.0,
    }


# ---------------------------------------------------------------- metrics
def _bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                  n: int, level: float, seed: int) -> Dict[str, list]:
    """Percentile bootstrap intervals for accuracy and ROC AUC."""
    rng = np.random.default_rng(seed)
    size = len(y_true)
    accuracies, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        yt, pp = y_true[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        accuracies.append(float(np.mean((pp >= threshold).astype(int) == yt)))
        aucs.append(float(roc_auc_score(yt, pp)))
    alpha = (1.0 - level) / 2.0
    def interval(values):
        if not values:
            return [float("nan"), float("nan")]
        return [round(float(np.quantile(values, alpha)), 4),
                round(float(np.quantile(values, 1 - alpha)), 4)]
    return {"accuracy": interval(accuracies), "auc_roc": interval(aucs)}


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                    class_names: list, cfg: Config, split: str) -> Dict[str, object]:
    y_pred = (proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()          # positive class = real (label 1)

    counts = np.bincount(y_true, minlength=2)
    majority = float(counts.max() / counts.sum())

    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)

    metrics: Dict[str, object] = {
        "split": split,
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "majority_baseline": round(majority, 4),
        "accuracy": round(float(np.mean(y_pred == y_true)), 4),
        "balanced_accuracy": round(float(
            0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))), 4),
        "auc_roc": round(float(roc_auc_score(y_true, proba)), 4),
        "auc_pr_real": round(float(average_precision_score(y_true, proba)), 4),
        "auc_pr_fake": round(float(average_precision_score(1 - y_true, 1 - proba)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7))), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "confusion_matrix": {
            "labels": class_names,
            "matrix": matrix.tolist(),
            "tn_fake_correct": int(tn), "fp_fake_called_real": int(fp),
            "fn_real_called_fake": int(fn), "tp_real_correct": int(tp),
        },
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            } for name in class_names
        },
    }
    metrics["lift_over_baseline"] = round(metrics["accuracy"] - majority, 4)
    metrics["confidence_interval"] = {
        "level": float(cfg.require("evaluation", "confidence_level")),
        **_bootstrap_ci(
            y_true, proba, threshold,
            int(cfg.require("evaluation", "bootstrap_samples")),
            float(cfg.require("evaluation", "confidence_level")),
            cfg.seed,
        ),
    }
    metrics["calibration"] = _calibration(y_true, proba)

    margin = float(cfg.get("evaluation", "abstain_margin", default=0.0) or 0.0)
    if margin > 0:
        metrics["abstain"] = _abstain_stats(y_true, proba, threshold, margin)
    return metrics


def _calibration(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> Dict[str, object]:
    """Reliability curve plus expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        confidence = float(proba[mask].mean())
        accuracy = float(y_true[mask].mean())
        weight = float(mask.sum()) / len(y_true)
        ece += weight * abs(accuracy - confidence)
        rows.append({"bin": [round(float(edges[b]), 2), round(float(edges[b + 1]), 2)],
                     "n": int(mask.sum()),
                     "mean_predicted": round(confidence, 4),
                     "observed_rate_real": round(accuracy, 4)})
    return {"expected_calibration_error": round(float(ece), 4), "bins": rows}


def _abstain_stats(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                   margin: float) -> Dict[str, object]:
    """How the model performs if uncertain cases are sent to a human instead."""
    uncertain = np.abs(proba - threshold) < margin
    decided = ~uncertain
    accuracy = (float(np.mean((proba[decided] >= threshold).astype(int) == y_true[decided]))
                if decided.any() else float("nan"))
    return {
        "margin": float(margin),
        "abstained": int(uncertain.sum()),
        "abstained_fraction": round(float(uncertain.mean()), 4),
        "accuracy_on_decided": round(accuracy, 4) if decided.any() else None,
    }


def stratified_by_difficulty(frame: pd.DataFrame, y_true: np.ndarray,
                             proba: np.ndarray, threshold: float) -> Dict[str, object]:
    """Accuracy per CIPLAB difficulty tier (easy / mid / hard fakes).

    The dataset encodes this in the filename and the original project ignored it.
    A detector that only catches the 'easy' tier is a very different product
    from one that catches 'hard', and a single accuracy number hides that.
    """
    y_pred = (proba >= threshold).astype(int)
    out: Dict[str, object] = {}
    for tier, group in frame.assign(_correct=(y_pred == y_true)).groupby("difficulty"):
        out[str(tier)] = {
            "n": int(len(group)),
            "accuracy": round(float(group["_correct"].mean()), 4),
        }
    return out


# ---------------------------------------------------------------- figures
def _save_figures(run_dir: Path, y_true: np.ndarray, proba: np.ndarray,
                  metrics: Dict[str, object], class_names: list) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0, 0].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    axes[0, 0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    axes[0, 1].plot(recall, precision, lw=2,
                    label=f"AP = {metrics['auc_pr_real']:.3f}")
    axes[0, 1].set(xlabel="recall (real)", ylabel="precision (real)",
                   title="Precision-Recall")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    matrix = np.array(metrics["confusion_matrix"]["matrix"])
    axes[1, 0].imshow(matrix, cmap="Blues")
    axes[1, 0].set(xticks=[0, 1], yticks=[0, 1],
                   xticklabels=class_names, yticklabels=class_names,
                   xlabel="predicted", ylabel="actual",
                   title=f"Confusion matrix (acc {metrics['accuracy']:.3f})")
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, int(matrix[i, j]), ha="center", va="center",
                            color="black")

    bins = metrics["calibration"]["bins"]
    if bins:
        axes[1, 1].plot([b["mean_predicted"] for b in bins],
                        [b["observed_rate_real"] for b in bins], "o-", lw=2)
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    axes[1, 1].set(xlabel="mean predicted P(real)", ylabel="observed rate of real",
                   title=f"Calibration (ECE {metrics['calibration']['expected_calibration_error']:.3f})")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Test-split evaluation, n = {metrics['n']}, "
                 f"majority baseline {metrics['majority_baseline']:.3f}")
    fig.tight_layout()
    out = run_dir / "evaluation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- entry
def evaluate(cfg: Config | None = None) -> Dict[str, object]:
    cfg = cfg or load_config()
    set_seeds(cfg.seed)
    run_dir = cfg.run_dir
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"no trained model at {model_path}\nRun:  python -m deepfake.cli train"
        )
    model = keras.models.load_model(model_path)
    class_names = cfg.class_names

    # 1. Validation split: choose the operating threshold here, and only here.
    y_val, p_val, _ = predict_split(model, cfg, "val")
    threshold_info = choose_threshold(y_val, p_val, cfg)
    threshold = float(threshold_info["threshold"])
    logger.info("threshold chosen on validation: %.4f (%s)",
                threshold, threshold_info["strategy"])
    val_metrics = compute_metrics(y_val, p_val, threshold, class_names, cfg, "val")

    # 2. Test split: sealed until now, evaluated once, threshold applied as-is.
    y_test, p_test, test_frame = predict_split(model, cfg, "test")
    test_metrics = compute_metrics(y_test, p_test, threshold, class_names, cfg, "test")
    test_metrics["by_difficulty"] = stratified_by_difficulty(
        test_frame, y_test, p_test, threshold)

    figure = _save_figures(run_dir, y_test, p_test, test_metrics, class_names)

    # 3. The threshold is part of the deployment contract, so update it.
    label_map = read_json(run_dir / "label_map.json") if (run_dir / "label_map.json").exists() else None
    from . import model as model_mod
    payload = label_map_payload(cfg, model_mod.preprocessing_spec(cfg), threshold)
    if label_map:
        payload = {**label_map, **payload}
    write_json(run_dir / "label_map.json", payload)

    results = {
        "run_name": cfg.run_name,
        "threshold_selection": threshold_info,
        "validation": val_metrics,
        "test": test_metrics,
        "figure": str(figure),
    }
    write_json(run_dir / "metrics.json", results)

    # Worst mistakes, for the failure gallery.
    mistakes = test_frame.assign(
        probability_real=p_test,
        predicted=(p_test >= threshold).astype(int),
    )
    mistakes = mistakes[mistakes["predicted"] != mistakes["label"]]
    mistakes["margin"] = (mistakes["probability_real"] - threshold).abs()
    mistakes.sort_values("margin", ascending=False).head(40).to_csv(
        run_dir / "worst_mistakes.csv", index=False)

    _log_summary(test_metrics)
    return results


def _log_summary(m: Dict[str, object]) -> None:
    ci = m["confidence_interval"]
    logger.info("TEST  n=%d  accuracy=%.4f %s  baseline=%.4f  lift=%+.4f",
                m["n"], m["accuracy"], ci["accuracy"], m["majority_baseline"],
                m["lift_over_baseline"])
    logger.info("TEST  auc_roc=%.4f %s  auc_pr_fake=%.4f  brier=%.4f  ece=%.4f",
                m["auc_roc"], ci["auc_roc"], m["auc_pr_fake"], m["brier"],
                m["calibration"]["expected_calibration_error"])
    for name, stats in m["per_class"].items():
        logger.info("TEST  %-5s precision=%.3f recall=%.3f f1=%.3f n=%d",
                    name, stats["precision"], stats["recall"], stats["f1"],
                    stats["support"])


def main() -> None:
    setup_logging()
    evaluate()


if __name__ == "__main__":
    main()
