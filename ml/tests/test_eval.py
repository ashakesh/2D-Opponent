"""Headless qualitative-eval helpers (no pygame)."""

from __future__ import annotations

import pytest

from fightsim.api import Simulator
from fightml.env import Phase1Action
from fightml.eval import (
    evaluate,
    format_summary,
    parse_args,
    pick_style,
    run_episode,
)


def test_sticky_policy_predicts_once_per_frameskip():
    calls = {"n": 0}

    def predict(_obs) -> int:
        calls["n"] += 1
        return int(Phase1Action.NEUTRAL)

    sim = Simulator()
    result = run_episode(sim, predict, "idle", frameskip=4, seed=0, max_frames=8)
    assert result.frames == 8
    assert calls["n"] == 2
    assert result.round_state == "fighting"
    assert result.p2_style == "idle"


def test_evaluate_vs_block_reports_counts():
    def predict(_obs) -> int:
        return int(Phase1Action.NEUTRAL)

    stats = evaluate(predict, "block", episodes=2, frameskip=4, seed=1, max_frames=6)
    assert stats["n"] == 2
    assert stats["incomplete"] == 2
    assert stats["p1_win"] + stats["p2_win"] + stats["draw"] + stats["incomplete"] == 2
    line = format_summary(stats)
    assert "p2=block" in line
    assert "n=2" in line


def test_pick_style_mix_is_a_dummy():
    class _Rng:
        def choice(self, seq):
            return seq[0]

    assert pick_style("mix", _Rng()) == "aggressive"
    assert pick_style("mash", _Rng()) == "mash"
    with pytest.raises(ValueError):
        pick_style("human", _Rng())


def test_eval_cli_defaults_not_headless():
    args = parse_args(["runs/ppo_p1"])
    assert args.p2 == "block"
    assert args.headless is False
    assert args.episodes == 0
    args = parse_args(["ckpt.zip", "--p2", "human", "--headless"])
    assert args.p2 == "human"
    assert args.headless is True
