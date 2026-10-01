"""CLI defaults for the mix4 trainer."""

from fightml.train import parse_args


def test_tensorboard_off_by_default():
    args = parse_args([])
    assert args.tb == ""


def test_tensorboard_path_when_requested():
    args = parse_args(["--tb", "runs/tb"])
    assert args.tb == "runs/tb"


def test_mix4_defaults():
    args = parse_args([])
    assert args.p2 == "mix"
    assert args.timesteps == 30_000_000
    assert args.save == "runs/ppo_mix4"
    assert args.eval_after is True
    assert args.pool == "runs/pool4"
    assert args.ckpt_every == 1_000_000
    assert args.opponent_pool_frac == 0.5
    assert args.timeout_lead == 2000
    assert args.timeout_win == 0.25
    assert args.timeout_draw == -0.25
    assert args.ent_coef == 0.02
    assert args.dummy_only is False


def test_dummy_only_flag():
    args = parse_args(["--dummy-only"])
    assert args.dummy_only is True


def test_no_eval_after():
    args = parse_args(["--no-eval-after"])
    assert args.eval_after is False
