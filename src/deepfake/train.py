"""Two-stage transfer-learning training.

Stage 1  backbone frozen, head only, learning rate 1e-4.
Stage 2  top N backbone layers unfrozen (BatchNorm stays frozen), 1e-5.

Everything that decides an outcome is explicit:

* Callback directions are stated (`mode: max`), never inferred. Keras'
  mode='auto' resolves 'val_auc_roc' to MINIMISE, because the name contains
  'auc' and not 'acc'. Left to that default, training would checkpoint and
  restore the worst epoch instead of the best.

* Exactly one learning-rate controller is active. Combining a
  LearningRateScheduler (which sets the rate each epoch) with
  ReduceLROnPlateau (which multiplies it) means the scheduler silently
  overwrites every reduction.

* The test split is never touched here. It is opened once, by evaluate.py.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from tensorflow import keras

from . import data as data_mod
from . import model as model_mod
from .config import Config, load_config
from .utils import label_map_payload, set_seeds, setup_logging, write_json

logger = logging.getLogger(__name__)


def _callbacks(cfg: Config, checkpoint_path: Path, stage: str) -> List[keras.callbacks.Callback]:
    es = cfg.require("training", "early_stopping")
    ck = cfg.require("training", "checkpoint")

    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor=str(es["monitor"]),
            mode=str(es["mode"]),                       # explicit, always
            patience=int(es["patience"]),
            restore_best_weights=bool(es["restore_best_weights"]),
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=str(ck["monitor"]),
            mode=str(ck["mode"]),                       # explicit, always
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(checkpoint_path.parent / f"history_{stage}.csv"), append=False
        ),
    ]

    schedule = str(cfg.get("training", "lr_schedule", default="none"))
    if schedule == "reduce_on_plateau":
        rop = cfg.require("training", "reduce_on_plateau")
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=float(rop["factor"]),
            patience=int(rop["patience"]),
            min_lr=float(rop["min_lr"]),
            verbose=1,
        ))
    elif schedule == "cosine":
        total = int(cfg.require("training", "stage1", "epochs"))
        base_lr = float(cfg.require("training", "stage1", "learning_rate"))
        callbacks.append(keras.callbacks.LearningRateScheduler(
            lambda epoch: base_lr * 0.5 * (1 + math.cos(math.pi * epoch / max(total, 1))),
            verbose=0,
        ))
    # schedule == "none": the optimizer's fixed rate is used.
    return callbacks


def _history_to_dict(history: keras.callbacks.History) -> Dict[str, List[float]]:
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def _best(history: keras.callbacks.History, monitor: str) -> float:
    """Best validation value of the monitored metric in this stage.

    All monitors used here (auc_roc, auc_pr, accuracy) are maximised; loss is
    minimised. The direction is decided by the metric, never guessed from
    its name at callback level.
    """
    key = monitor if monitor.startswith("val_") else f"val_{monitor}"
    values = history.history.get(key)
    if not values:
        raise KeyError(f"monitor {key!r} not found in history keys {list(history.history)}")
    return float(min(values) if "loss" in key else max(values))


def _select_stage(report: Dict[str, object], stage_paths: Dict[str, Path],
                  monitor: str) -> tuple:
    """Pick the stage whose checkpoint genuinely scored best on the monitor."""
    higher_is_better = "loss" not in monitor
    candidates = [
        (name, float(info["best_val_monitor"]))
        for name, info in report["stages"].items()
        if "best_val_monitor" in info and stage_paths[name].exists()
    ]
    if not candidates:
        raise RuntimeError("no stage produced a checkpoint; nothing to save")
    return max(candidates, key=lambda kv: kv[1]) if higher_is_better \
        else min(candidates, key=lambda kv: kv[1])


def train(cfg: Config | None = None) -> Dict[str, object]:
    """Run the full training procedure and return a summary dict."""
    cfg = cfg or load_config()
    set_seeds(cfg.seed)
    run_dir = cfg.run_dir
    cfg.save_used()

    # ---- data ---------------------------------------------------------
    data_mod.build_manifest(cfg)
    train_frame = data_mod.split_frame(cfg, "train")
    val_frame = data_mod.split_frame(cfg, "val")
    train_ds = data_mod.make_dataset(cfg, "train", frame=train_frame)
    val_ds = data_mod.make_dataset(cfg, "val", frame=val_frame)
    weights = data_mod.class_weights(cfg, train_frame)

    summary = data_mod.dataset_summary(cfg)
    logger.info("data: %s", summary["per_split"])
    logger.info("majority-class baseline per split: %s", summary["majority_baseline"])

    # ---- model --------------------------------------------------------
    model = model_mod.build_model(cfg)
    model_mod.compile_model(model, float(cfg.require("training", "stage1", "learning_rate")))
    params_stage1 = model_mod.count_parameters(model)
    logger.info("stage 1 parameters: %s", params_stage1)

    # One checkpoint file per stage. A single shared file would be wrong: the
    # stage-2 callback starts with a fresh "best so far", so a stage-2 epoch
    # worse than stage 1's best would silently overwrite it. The two files are
    # compared at the end and the genuine winner becomes model.keras.
    final_path = run_dir / "model.keras"
    stage_paths = {
        "stage1": run_dir / "model_stage1.keras",
        "stage2": run_dir / "model_stage2.keras",
    }
    monitor = str(cfg.require("training", "checkpoint", "monitor"))

    report: Dict[str, object] = {
        "run_name": cfg.run_name,
        "data": summary,
        "class_weights": weights,
        "parameters": {"stage1": params_stage1},
        "stages": {},
    }

    started = time.time()

    # ---- stage 1: frozen backbone -------------------------------------
    if bool(cfg.require("training", "stage1", "enabled")):
        epochs1 = int(cfg.require("training", "stage1", "epochs"))
        logger.info("stage 1: training the head for up to %d epochs", epochs1)
        history1 = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs1,
            callbacks=_callbacks(cfg, stage_paths["stage1"], "stage1"),
            class_weight=weights, verbose=2,
        )
        report["stages"]["stage1"] = {
            "epochs_run": len(history1.history["loss"]),
            "best_val_monitor": _best(history1, monitor),
            "history": _history_to_dict(history1),
        }
        epochs_done = len(history1.history["loss"])
    else:
        epochs_done = 0

    # ---- stage 2: fine-tune the top of the backbone --------------------
    if bool(cfg.require("training", "stage2", "enabled")):
        unfreeze_report = model_mod.unfreeze_top(
            model, int(cfg.require("training", "stage2", "unfreeze_last_n"))
        )
        # Recompiling is mandatory: trainability changes only take effect when
        # the training function is rebuilt.
        model_mod.compile_model(
            model, float(cfg.require("training", "stage2", "learning_rate"))
        )
        params_stage2 = model_mod.count_parameters(model)
        logger.info("stage 2 parameters: %s", params_stage2)

        epochs2 = int(cfg.require("training", "stage2", "epochs"))
        logger.info("stage 2: fine-tuning for up to %d epochs", epochs2)
        history2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs_done + epochs2,
            initial_epoch=epochs_done,           # keeps the epoch axis continuous
            callbacks=_callbacks(cfg, stage_paths["stage2"], "stage2"),
            class_weight=weights, verbose=2,
        )
        report["parameters"]["stage2"] = params_stage2
        report["stages"]["stage2"] = {
            "unfreeze": unfreeze_report,
            "epochs_run": len(history2.history["loss"]),
            "best_val_monitor": _best(history2, monitor),
            "history": _history_to_dict(history2),
        }

    report["train_seconds"] = round(time.time() - started, 1)

    # ---- select the winning stage and persist it -----------------------
    winner, winner_score = _select_stage(report, stage_paths, monitor)
    report["selected"] = {"stage": winner, "monitor": monitor, "value": winner_score}
    logger.info("selected %s (%s = %.4f) as the final model", winner, monitor, winner_score)

    best_model = keras.models.load_model(stage_paths[winner])
    best_model.save(final_path)
    logger.info("saved model: %s", final_path)

    write_json(
        run_dir / "label_map.json",
        label_map_payload(cfg, model_mod.preprocessing_spec(cfg), threshold=0.5),
    )
    write_json(run_dir / "train_report.json", report)
    logger.info("training complete in %.1fs", report["train_seconds"])
    return report


def main() -> None:
    setup_logging()
    train()


if __name__ == "__main__":
    main()
