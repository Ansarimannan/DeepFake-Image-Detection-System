"""Derive an experiment config from the base config by dotted overrides.

An ablation should differ from the baseline in exactly one thing, and it should
be obvious from the file which thing that is. Hand-copying config.yaml makes both
of those hard, so the variants are generated instead:

    python experiments/derive.py res384 image.size=[384,384]
    python experiments/derive.py unfreeze_all training.stage2.unfreeze_last_n=154

Writes experiments/<name>.yaml with project.run_name set to <name>, and records
the overrides in a comment header so the file explains itself.

Then:
    python -m deepfake.cli train    --config experiments/res384.yaml
    python -m deepfake.cli evaluate --config experiments/res384.yaml
    python experiments/compare.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_dotted(node: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"{dotted}: {key} is not a section in the base config")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"{dotted}: {keys[-1]} does not exist in the base config")
    node[keys[-1]] = value


def parse_value(raw: str):
    """YAML-ish literal parsing: 384, [384,384], true, 1e-5, plain strings."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return yaml.safe_load(raw)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    name, overrides = argv[0], argv[1:]
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    applied = []
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"override must be key.path=value, got {override!r}")
        dotted, raw = override.split("=", 1)
        value = parse_value(raw)
        set_dotted(config, dotted, value)
        applied.append(f"{dotted} = {value!r}")

    config["project"]["run_name"] = name

    out = ROOT / "experiments" / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join([
        f"# Experiment: {name}",
        "# Derived from config.yaml by experiments/derive.py. Do not hand-edit;",
        "# regenerate instead, so the diff against the baseline stays exact.",
        "#",
        "# Overrides applied:",
        *[f"#   {line}" for line in applied],
        "",
    ])
    out.write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {out}")
    for line in applied:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
