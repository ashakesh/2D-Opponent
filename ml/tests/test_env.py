"""Headless contract tests for the Phase-1 Gymnasium env."""

from __future__ import annotations

import numpy as np

from fightsim.config import HEALTH_MAX, SUBPIXEL
from fightml.env import FightEnv, Phase1Action, action_to_raw
from fightml.obs import OBS_SIZE, flatten_state


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
    env = FightEnv(dummy_style="idle")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert flatten_state(env.sim.get_state()).shape == obs.shape
    assert info["round_state"] == "fighting"


def test_step_advances_frame_by_frameskip():
    env = FightEnv(dummy_style="idle", frameskip=4)
    env.reset(seed=0)
    start = env.sim.get_state().frame
    env.step(int(Phase1Action.NEUTRAL))
    assert env.sim.get_state().frame == start + 4
    assert env.sim.get_state().round_state.value == "fighting"


def test_episode_ends_on_ko():
    env = FightEnv(dummy_style="idle", frameskip=4)
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
    env = FightEnv(dummy_style="idle", frameskip=4, damage_coef=0.1)
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
