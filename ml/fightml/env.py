"""Gymnasium env wrapping fightsim.Simulator.

Frameskip, dummy P2, semantic actions, and rewards live here — never in the sim.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fightsim.api import Simulator
from fightsim.config import HEALTH_MAX
from fightsim.dummy import dummy_input
from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.state import RoundState
from fightml.obs import OBS_SIZE, flatten_state

DUMMY_STYLES: tuple[str, ...] = ("idle", "block", "mash", "crouch_block")
DEFAULT_FRAMESKIP = 4
DEFAULT_DAMAGE_COEF = 0.1


class Phase1Action(IntEnum):
    NEUTRAL = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    BLOCK = 5
    LIGHT = 6
    HEAVY = 7
    THROW = 8
    SPECIAL_1 = 9
    SPECIAL_2 = 10
    CROUCH_LIGHT = 11
    CROUCH_BLOCK = 12


N_ACTIONS = len(Phase1Action)


def action_to_raw(action: int, facing: int) -> RawInput:
    """Map a discrete Phase-1 action to one frame of RawInput."""
    act = Phase1Action(int(action))
    if act is Phase1Action.CROUCH_LIGHT:
        return RawInput(down=True, a=True)
    if act is Phase1Action.CROUCH_BLOCK:
        block = semantic_to_raw(SemanticAction.BLOCK, facing)
        return RawInput(left=block.left, right=block.right, down=True)
    return semantic_to_raw(SemanticAction(act.name.lower()), facing)


def _resolve_styles(p2: str) -> tuple[str, ...]:
    if p2 == "mix":
        return DUMMY_STYLES
    if p2 in DUMMY_STYLES:
        return (p2,)
    raise ValueError(f"unknown dummy style: {p2}")


class FightEnv(gym.Env):
    """Single-agent P1 vs a fightsim dummy. One policy step = frameskip sim frames."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dummy_style: str = "block",
        frameskip: int = DEFAULT_FRAMESKIP,
        damage_coef: float = DEFAULT_DAMAGE_COEF,
    ) -> None:
        super().__init__()
        self.sim = Simulator()
        self.frameskip = int(frameskip)
        self.damage_coef = float(damage_coef)
        self._styles = _resolve_styles(dummy_style)
        self._p2_style = self._styles[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._p2_style = str(self.np_random.choice(self._styles))
        state = self.sim.reset(int(seed))
        return flatten_state(state), self._info(state, frames=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self.sim.get_state()
        p1_raw = action_to_raw(int(action), state.p1.facing)
        h1_before = state.p1.health
        h2_before = state.p2.health

        frames = 0
        for _ in range(self.frameskip):
            state = self.sim.get_state()
            if state.round_state != RoundState.FIGHTING:
                break
            p2_raw = dummy_input(state, 2, self._p2_style)
            state = self.sim.step(p1_raw, p2_raw)
            frames += 1

        dealt = h2_before - state.p2.health
        taken = h1_before - state.p1.health
        reward = self.damage_coef * (dealt - taken) / HEALTH_MAX
        terminated = state.round_state != RoundState.FIGHTING
        if terminated:
            if state.round_state is RoundState.P1_WIN:
                reward += 1.0
            elif state.round_state is RoundState.P2_WIN:
                reward -= 1.0
        return flatten_state(state), float(reward), terminated, False, self._info(state, frames)

    def _info(self, state, frames: int) -> dict[str, Any]:
        return {
            "frames": frames,
            "round_state": state.round_state.value,
            "p1_health": state.p1.health,
            "p2_health": state.p2.health,
            "p2_style": self._p2_style,
        }


def make_fight_env(
    dummy_style: str = "block",
    frameskip: int = DEFAULT_FRAMESKIP,
    damage_coef: float = DEFAULT_DAMAGE_COEF,
) -> Callable[[], FightEnv]:
    """Thunk for DummyVecEnv / make_vec_env (avoids late-binding lambda capture)."""

    def _thunk() -> FightEnv:
        return FightEnv(
            dummy_style=dummy_style, frameskip=frameskip, damage_coef=damage_coef
        )

    return _thunk
