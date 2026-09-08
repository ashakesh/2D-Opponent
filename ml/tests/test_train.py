"""CLI defaults for the PPO trainer."""

from fightml.train import parse_args


def test_tensorboard_off_by_default():
    args = parse_args([])
    assert args.tb == ""


def test_tensorboard_path_when_requested():
    args = parse_args(["--tb", "runs/tb"])
    assert args.tb == "runs/tb"
