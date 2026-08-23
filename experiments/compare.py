"""Compare every finished run in artifacts/ on the sealed test split.

Reads only artifacts/<run>/metrics.json, so it cannot report a number that was
not actually measured.

    python experiments/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    ("run", 16), ("acc", 7), ("95% CI", 16), ("base", 7), ("lift", 8),
    ("auc_roc", 8), ("auc_pr_fake", 12), ("brier", 7), ("ece", 7), ("thr", 7),
]


SKIPPED = []


def rows():
    for metrics_path in sorted((ROOT / "artifacts").glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        # The standalone notebook writes its own flatter metrics.json into
        # artifacts/notebook/. Only CLI runs have the {"validation", "test"}
        # shape this table needs, so anything else is reported and skipped
        # rather than crashing the comparison.
        if "test" not in data:
            SKIPPED.append(metrics_path.parent.name)
            continue
        test = data["test"]
        ci = test["confidence_interval"]["accuracy"]
        yield {
            "run": metrics_path.parent.name,
            "acc": f"{test['accuracy']:.4f}",
            "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            "base": f"{test['majority_baseline']:.4f}",
            "lift": f"{test['lift_over_baseline']:+.4f}",
            "auc_roc": f"{test['auc_roc']:.4f}",
            "auc_pr_fake": f"{test['auc_pr_fake']:.4f}",
            "brier": f"{test['brier']:.4f}",
            "ece": f"{test['calibration']['expected_calibration_error']:.4f}",
            "thr": f"{test['threshold']:.3f}",
            "_auc": test["auc_roc"],
        }


def main() -> int:
    collected = sorted(rows(), key=lambda r: r["_auc"], reverse=True)
    if not collected:
        print("no runs found; train and evaluate something first")
        return 1

    header = "  ".join(name.ljust(width) for name, width in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in collected:
        print("  ".join(str(row[name]).ljust(width) for name, width in COLUMNS))
    print("\nSorted by ROC AUC, which is threshold-free. Accuracy depends on the "
          "operating point and moves without the model changing (see DECISIONS.md D-020).")
    print("Overlapping confidence intervals mean the difference is not established.")
    if SKIPPED:
        print(f"\nskipped (not a CLI run): {', '.join(sorted(set(SKIPPED)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
