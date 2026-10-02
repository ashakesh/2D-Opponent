"""Headless qualitative-eval helpers (no pygame). Agent is P2."""

from __future__ import annotations

import pytest

from fightsim.api import Simulator
from fightsim.dummy import PERSONALITY_STYLES
from fightml.env import Phase1Action
from fightml.eval import (
    evaluate,
    evaluate_per_style,
    format_per_style,
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
    assert result.p1_style == "idle"


def test_evaluate_vs_block_reports_counts():
    def predict(_obs) -> int:
        return int(Phase1Action.NEUTRAL)

    stats = evaluate(predict, "block", episodes=2, frameskip=4, seed=1, max_frames=6)
    assert stats["n"] == 2
    assert stats["incomplete"] == 2
    assert stats["p1_win"] + stats["agent_win"] + stats["draw"] + stats["incomplete"] == 2
    line = format_summary(stats)
    assert "p1=block" in line
    assert "agent_win=" in line
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
    assert args.p1 == "block"
    assert args.headless is False
    assert args.episodes == 0
    assert args.per_style is False
    args = parse_args(["ckpt.zip", "--p1", "human", "--headless"])
    assert args.p1 == "human"
    assert args.headless is True
    boxed = parse_args(["runs/ppo_p1", "--box", "no", "--native"])
    assert boxed.box == "no"
    assert boxed.native is True
    assert boxed.fullscreen is False
    with pytest.raises(SystemExit):
        parse_args(["runs/ppo_p1", "--native", "--fullscreen"])


def test_run_episode_both_orientations():
    def predict(_obs) -> int:
        return int(Phase1Action.NEUTRAL)

    sim = Simulator()
    left = run_episode(sim, predict, "idle", 4, seed=0, max_frames=2, p1_on_left=True)
    right = run_episode(sim, predict, "idle", 4, seed=0, max_frames=2, p1_on_left=False)
    assert left.p1_on_left is True
    assert right.p1_on_left is False
    assert sim.get_state().p1.x > sim.get_state().p2.x


def test_evaluate_per_style_has_six_personalities():
    def predict(_obs) -> int:
        return int(Phase1Action.NEUTRAL)

    table = evaluate_per_style(predict, episodes_per_style=2, frameskip=4, seed=0, max_frames=4)
    for style in PERSONALITY_STYLES:
        assert style in table
        assert table[style]["n"] == 2
        sides = {ep.p1_on_left for ep in table[style]["episodes"]}
        assert sides == {True, False}
    assert table["total"]["n"] == 2 * len(PERSONALITY_STYLES)
    text = format_per_style(table)
    assert "p1=zoner" in text
    assert "p1=mix" in text
