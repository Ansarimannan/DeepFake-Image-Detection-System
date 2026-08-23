"""Single entry point for every operation.

    python -m deepfake.cli split      build/refresh the train/val/test manifest
    python -m deepfake.cli info       show the dataset and split summary
    python -m deepfake.cli train      two-stage training, writes artifacts/<run>/
    python -m deepfake.cli evaluate   sealed-test evaluation, writes metrics.json
    python -m deepfake.cli gradcam    attention figure for a few test images
    python -m deepfake.cli predict    classify one or more image files
    python -m deepfake.cli all        split -> train -> evaluate -> gradcam
    python -m deepfake.cli smoke      2-minute end-to-end check on a tiny subset

Every command accepts --config to point at a different YAML file, which is how
you run an ablation without editing anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow "python src/deepfake/cli.py" as well as "python -m deepfake.cli".
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "deepfake"

from .config import load_config                       # noqa: E402
from .utils import setup_logging                      # noqa: E402

logger = logging.getLogger(__name__)


def _cmd_split(args) -> int:
    from . import data as data_mod
    cfg = load_config(args.config)
    frame = data_mod.build_manifest(cfg, force=args.force)
    print(frame.groupby(["split", "class_name"]).size().unstack(fill_value=0))
    print(f"\nmanifest: {cfg.require('data', 'split', 'manifest')}")
    return 0


def _cmd_info(args) -> int:
    from . import data as data_mod
    cfg = load_config(args.config)
    data_mod.build_manifest(cfg)
    print(json.dumps(data_mod.dataset_summary(cfg), indent=2))
    return 0


def _cmd_train(args) -> int:
    from .train import train
    train(load_config(args.config))
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluate import evaluate
    results = evaluate(load_config(args.config))
    test = results["test"]
    print(json.dumps({
        "accuracy": test["accuracy"],
        "accuracy_95ci": test["confidence_interval"]["accuracy"],
        "majority_baseline": test["majority_baseline"],
        "lift_over_baseline": test["lift_over_baseline"],
        "auc_roc": test["auc_roc"],
        "auc_roc_95ci": test["confidence_interval"]["auc_roc"],
        "brier": test["brier"],
        "threshold": test["threshold"],
    }, indent=2))
    return 0


def _cmd_gradcam(args) -> int:
    from .gradcam import run
    print(run(load_config(args.config)))
    return 0


def _cmd_predict(args) -> int:
    from .predict import predict_paths
    for result in predict_paths(args.paths, load_config(args.config)):
        print(f"{Path(result['path']).name:40s} {result['decision']:9s} "
              f"P(real)={result['probability_real']:.3f} "
              f"P(fake)={result['probability_fake']:.3f}")
    return 0


def _cmd_all(args) -> int:
    for step in (_cmd_split, _cmd_train, _cmd_evaluate, _cmd_gradcam):
        code = step(args)
        if code:
            return code
    return 0


def _cmd_smoke(args) -> int:
    """Fast end-to-end sanity check: does every piece still connect?

    Runs one tiny epoch on a handful of images. It proves the pipeline, the
    model, the save/load round trip and the prediction contract all work. It
    proves nothing about accuracy, and says so.
    """
    from . import data as data_mod, model as model_mod
    from .predict import Predictor
    from .utils import label_map_payload, set_seeds, write_json

    cfg = load_config(args.config)
    set_seeds(cfg.seed)
    data_mod.build_manifest(cfg)

    train_frame = data_mod.split_frame(cfg, "train").groupby("label").head(16)
    val_frame = data_mod.split_frame(cfg, "val").groupby("label").head(8)
    train_ds = data_mod.make_dataset(cfg, "train", frame=train_frame)
    val_ds = data_mod.make_dataset(cfg, "val", shuffle=False, frame=val_frame)

    model = model_mod.build_model(cfg)
    model_mod.compile_model(model, 1e-4)
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)

    run_dir = cfg.run_dir
    model_path = run_dir / "smoke_model.keras"
    model.save(model_path)

    import tensorflow.keras as keras
    reloaded = keras.models.load_model(model_path)
    contract = label_map_payload(cfg, model_mod.preprocessing_spec(cfg), 0.5)
    predictor = Predictor(reloaded, contract)
    sample = val_frame["path"].head(4).tolist()
    for result in predictor.predict_batch(sample):
        assert result["predicted_class"] == result["label"], "label polarity broken"
        print(f"  {Path(result['path']).name:35s} -> {result['predicted_class']:5s} "
              f"P(real)={result['probability_real']:.3f}")

    write_json(run_dir / "smoke.json", {"status": "ok", "model": str(model_path)})
    print("\nSMOKE OK: pipeline, model, save/load and prediction contract all work.")
    print("This says nothing about accuracy. Run 'train' then 'evaluate' for that.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # --config and -v are accepted on BOTH sides of the subcommand, because
    # `cli train --config x.yaml` is what everyone types first and argparse
    # would otherwise reject it with an unhelpful "unrecognized arguments".
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help="path to a config YAML (default: ./config.yaml)")
    common.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(
        prog="deepfake", parents=[common],
        description="Real vs fake face detection: data, training, evaluation, inference.",
    )
    # No set_defaults here on purpose. argparse re-parses a subcommand in its own
    # namespace and copies the result over the outer one, so a default declared
    # on either side would clobber a value given on the other. Both --config
    # actions use SUPPRESS, and parse_args() fills the gaps afterwards.
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, func):
        child = sub.add_parser(name, help=help_text, parents=[common])
        child.set_defaults(func=func)
        return child

    add("split", "build the train/val/test manifest", _cmd_split).add_argument(
        "--force", action="store_true", help="rebuild even if it exists")
    add("info", "dataset and split summary", _cmd_info)
    add("train", "two-stage training", _cmd_train)
    add("evaluate", "sealed-test evaluation", _cmd_evaluate)
    add("gradcam", "attention figure", _cmd_gradcam)
    add("all", "split, train, evaluate, gradcam", _cmd_all)
    add("smoke", "fast end-to-end sanity check", _cmd_smoke)
    add("predict", "classify image files", _cmd_predict).add_argument("paths", nargs="+")

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and normalise, so every command sees the same attribute set."""
    args = build_parser().parse_args(argv)
    for name, default in (("config", None), ("verbose", False), ("force", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
