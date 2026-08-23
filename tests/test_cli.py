"""The CLI is the only entry point, so its argument handling is worth pinning."""

from __future__ import annotations

import pytest

from deepfake.cli import build_parser, parse_args


@pytest.mark.parametrize("argv", [
    ["--config", "experiments/x.yaml", "train"],   # before the subcommand
    ["train", "--config", "experiments/x.yaml"],   # after it
])
def test_config_is_accepted_on_either_side_of_the_subcommand(argv):
    """`cli train --config x.yaml` is what everyone types first. It used to fail
    with 'unrecognized arguments', which silently wasted two training runs."""
    args = parse_args(argv)
    assert args.config == "experiments/x.yaml"
    assert args.command == "train"


def test_config_defaults_to_none_when_not_given():
    args = parse_args(["evaluate"])
    assert args.config is None
    assert args.verbose is False


def test_every_command_is_reachable():
    parser = build_parser()
    for command in ("split", "info", "train", "evaluate", "gradcam", "all", "smoke"):
        assert parse_args([command]).command == command
    assert parse_args(["predict", "a.jpg", "b.jpg"]).paths == ["a.jpg", "b.jpg"]


def test_a_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
