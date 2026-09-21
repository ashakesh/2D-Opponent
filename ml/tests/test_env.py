"""Headless contract tests for the Phase-1 Gymnasium env."""

from __future__ import annotations

import numpy as np

from fightsim.config import HEALTH_MAX, SUBPIXEL
from fightml.env import FightEnv, Phase1Action, action_to_raw, terminal_bonus
from fightsim.state import RoundState
from fightml.obs import OBS_SIZE, flatten_state
from fightml.spawn import GAP_MAX, GAP_MIN


def _close(env: FightEnv, gap_px: int = 28) -> None:
    st = env.sim.get_state()
    mid = 400 * SUBPIXEL
    st.p1.x = mid - (gap_px * SUBPIXEL) // 2
    st.p2.x = mid + (gap_px * SUBPIXEL) // 2
    st.p1.facing = 1
    st.p2.facing = -1
    st.p1.y = 0
    st.p2.y = 0


def test_reset_obs_shape_and_dtype():
    env = FightEnv(dummy_style="idle", dummy_only=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert flatten_state(env.sim.get_state()).shape == obs.shape
    assert info["round_state"] == "fighting"


def test_step_advances_frame_by_frameskip():
    env = FightEnv(dummy_style="idle", frameskip=4, dummy_only=True)
    env.reset(seed=0)
    start = env.sim.get_state().frame
    env.step(int(Phase1Action.NEUTRAL))
    assert env.sim.get_state().frame == start + 4
    assert env.sim.get_state().round_state.value == "fighting"


def test_episode_ends_on_ko():
    env = FightEnv(dummy_style="idle", frameskip=4, dummy_only=True)
    env.reset(seed=0)
    _close(env, 28)
    env.sim.get_state().p2.health = 200
    terminated = False
    for _ in range(80):
        _, _, terminated, truncated, info = env.step(int(Phase1Action.THROW))
        assert truncated is False
        if terminated:
            break
    assert terminated
    assert info["round_state"] == "p1_win"


def test_reward_non_negative_when_only_p2_takes_damage():
    env = FightEnv(dummy_style="idle", frameskip=4, damage_coef=0.1, dummy_only=True)
    env.reset(seed=0)
    _close(env, 36)
    obs, reward, terminated, truncated, info = env.step(int(Phase1Action.LIGHT))
    assert not terminated
    assert reward >= 0.0
    assert info["p1_health"] == HEALTH_MAX
    if info["p2_health"] < HEALTH_MAX:
        assert reward > 0.0


def test_command_special_flags_only_on_specials():
    facing = 1
    for act in Phase1Action:
        raw = action_to_raw(int(act), facing)
        if act is Phase1Action.SPECIAL_1:
            assert raw.command_special_1 is True
            assert raw.command_special_2 is False
            assert raw.command_special_3 is False
        elif act is Phase1Action.SPECIAL_2:
            assert raw.command_special_1 is False
            assert raw.command_special_2 is True
            assert raw.command_special_3 is False
        elif act is Phase1Action.SPECIAL_3:
            assert raw.command_special_3 is True
            assert raw.command_special_1 is False
            assert raw.command_special_2 is False
        else:
            assert raw.command_special_1 is False
            assert raw.command_special_2 is False
            assert raw.command_special_3 is False
    crouch = action_to_raw(int(Phase1Action.CROUCH_LIGHT), facing)
    assert crouch.down is True and crouch.a is True
    block = action_to_raw(int(Phase1Action.CROUCH_BLOCK), facing)
    assert block.down is True and block.left is True and block.right is False


def test_canonical_spawn_keeps_p1_on_left():
    env = FightEnv(dummy_style="idle", random_spawn=False, dummy_only=True)
    env.reset(seed=0)
    st = env.sim.get_state()
    assert st.p1.x < st.p2.x
    assert st.p1.facing > 0
    assert st.p2.facing < 0


def test_random_spawn_uses_both_sides_and_legal_gap():
    env = FightEnv(dummy_style="idle", random_spawn=True, dummy_only=True)
    left = 0
    right = 0
    for seed in range(48):
        env.reset(seed=seed)
        st = env.sim.get_state()
        gap = abs(st.p1.x - st.p2.x)
        assert GAP_MIN <= gap <= GAP_MAX
        assert st.p1.y == 0 and st.p2.y == 0
        if st.p1.x < st.p2.x:
            left += 1
            assert st.p1.facing > 0
        else:
            right += 1
            assert st.p1.facing < 0
    assert left > 0 and right > 0


def test_terminal_bonus_timeout_lead():
    assert terminal_bonus(RoundState.P1_WIN, 10000, 0, timed_out=False) == 1.0
    assert terminal_bonus(RoundState.P2_WIN, 0, 10000, timed_out=False) == -1.0
    assert terminal_bonus(RoundState.P1_WIN, 10000, 10000, timed_out=True) == 0.0
    assert terminal_bonus(RoundState.P1_WIN, 10000, 9600, timed_out=True) == 0.0
    assert terminal_bonus(RoundState.P1_WIN, 10000, 8000, timed_out=True) == 0.5
    assert terminal_bonus(RoundState.P2_WIN, 7000, 10000, timed_out=True) == -1.0


def test_timeout_full_health_is_not_a_win():
    env = FightEnv(
        dummy_style="idle",
        frameskip=4,
        dummy_only=True,
        random_spawn=False,
        damage_coef=0.0,
        whiff_penalty=0.0,
    )
    env.reset(seed=0)
    env.sim.get_state().timer = 3
    terminated = False
    reward = 0.0
    info = {}
    for _ in range(8):
        _, reward, terminated, _, info = env.step(int(Phase1Action.NEUTRAL))
        if terminated:
            break
    assert terminated
    assert info["round_state"] in ("draw", "p1_win", "p2_win")
    assert reward == 0.0


def test_whiff_at_range_is_penalized():
    env = FightEnv(
        dummy_style="idle",
        frameskip=4,
        dummy_only=True,
        random_spawn=False,
        damage_coef=0.0,
        whiff_penalty=0.02,
    )
    env.reset(seed=0)
    st = env.sim.get_state()
    st.p1.x = 120 * SUBPIXEL
    st.p2.x = 680 * SUBPIXEL
    total = 0.0
    for _ in range(8):
        _, reward, terminated, _, _ = env.step(int(Phase1Action.LIGHT))
        total += reward
        if terminated:
            break
    assert total < 0.0


def test_dummy_only_info_kind():
    env = FightEnv(dummy_style="idle", dummy_only=True, random_spawn=False)
    _, info = env.reset(seed=0)
    assert info["p2_kind"].startswith("dummy:")


def test_pool_reset_reports_pool_kind():
    from pathlib import Path

    import pytest

    runs = Path(__file__).resolve().parents[1] / "runs"
    pool = runs / "pool"
    mix2 = runs / "ppo_mix2.zip"
    if not mix2.is_file() and not (pool.is_dir() and any(pool.glob("*.zip"))):
        pytest.skip("no frozen pool zip on disk")
    env = FightEnv(
        dummy_style="idle",
        dummy_only=False,
        random_spawn=False,
        pool_dir=str(pool),
        pool_frac=1.0,
    )
    _, info = env.reset(seed=0)
    assert info["p2_kind"].startswith("pool:")
