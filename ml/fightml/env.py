"""Gymnasium env wrapping fightsim.Simulator.

Frameskip, P2 opponent, semantic actions, and rewards live here — never in the sim.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fightsim.api import Simulator
from fightsim.config import HEALTH_MAX, ROUND_TIMER, SUBPIXEL
from fightsim.dummy import DUMMY_STYLES, PERSONALITY_STYLES
from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.state import RoundState
from fightml.obs import OBS_SIZE, flatten_state
from fightml.opponent import DummyOpponent, PolicyOpponent, collect_pool_paths
from fightml.spawn import randomize_spawn

DEFAULT_FRAMESKIP = 4
DEFAULT_DAMAGE_COEF = 0.1
DEFAULT_TIMEOUT_LEAD = 2000
DEFAULT_TIMEOUT_WIN = 0.25
DEFAULT_TIMEOUT_DRAW = -0.25
DEFAULT_TIME_COST = 0.25
DEFAULT_WHIFF_PENALTY = 0.02
DEFAULT_WHIFF_RANGE_PX = 100
DEFAULT_POOL_FRAC = 0.5


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
    SPECIAL_3 = 13


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
        return PERSONALITY_STYLES
    if p2 in DUMMY_STYLES:
        return (p2,)
    raise ValueError(f"unknown dummy style: {p2}")


def _timed_out(events) -> bool:
    return any(e.kind == "round_end" and e.data.get("timeout") for e in events)


def terminal_bonus(
    round_state: RoundState,
    p1_health: int,
    p2_health: int,
    timed_out: bool,
    timeout_lead: int = DEFAULT_TIMEOUT_LEAD,
    timeout_win: float = DEFAULT_TIMEOUT_WIN,
    timeout_draw: float = DEFAULT_TIMEOUT_DRAW,
) -> float:
    """KO uses ±1. Timeout win is weaker than a KO; indecisive clock-outs are negative."""
    if not timed_out:
        if round_state is RoundState.P1_WIN:
            return 1.0
        if round_state is RoundState.P2_WIN:
            return -1.0
        return 0.0
    lead = int(p1_health) - int(p2_health)
    if lead >= int(timeout_lead):
        return float(timeout_win)
    if lead <= -int(timeout_lead):
        return -1.0
    return float(timeout_draw)


class FightEnv(gym.Env):
    """Single-agent P1 vs dummy or frozen-pool P2. One policy step = frameskip sim frames."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dummy_style: str = "mix",
        frameskip: int = DEFAULT_FRAMESKIP,
        damage_coef: float = DEFAULT_DAMAGE_COEF,
        random_spawn: bool = True,
        pool_dir: str = "",
        pool_frac: float = DEFAULT_POOL_FRAC,
        dummy_only: bool = False,
        timeout_lead: int = DEFAULT_TIMEOUT_LEAD,
        timeout_win: float = DEFAULT_TIMEOUT_WIN,
        timeout_draw: float = DEFAULT_TIMEOUT_DRAW,
        time_cost: float = DEFAULT_TIME_COST,
        whiff_penalty: float = DEFAULT_WHIFF_PENALTY,
        whiff_range_px: float = DEFAULT_WHIFF_RANGE_PX,
    ) -> None:
        super().__init__()
        self.sim = Simulator()
        self.frameskip = int(frameskip)
        self.damage_coef = float(damage_coef)
        self.random_spawn = bool(random_spawn)
        self.pool_dir = str(pool_dir)
        self.pool_frac = float(pool_frac)
        self.dummy_only = bool(dummy_only)
        self.timeout_lead = int(timeout_lead)
        self.timeout_win = float(timeout_win)
        self.timeout_draw = float(timeout_draw)
        self.time_cost = float(time_cost)
        self.whiff_penalty = float(whiff_penalty)
        self.whiff_range_px = float(whiff_range_px)
        self._styles = _resolve_styles(dummy_style)
        self._p2_kind = f"dummy:{self._styles[0]}"
        self._opponent: DummyOpponent | PolicyOpponent = DummyOpponent(self._styles[0])
        self._policy_cache: dict[str, PolicyOpponent] = {}
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

    def action_masks(self) -> np.ndarray:
        from fightml.mask import action_mask

        return action_mask(self.sim.get_state(), pid=1)

    def _pick_opponent(self, seed: int) -> None:
        pool = [] if self.dummy_only else collect_pool_paths(self.pool_dir)
        use_pool = bool(pool) and float(self.np_random.random()) < self.pool_frac
        if use_pool:
            path = str(pool[int(self.np_random.integers(0, len(pool)))])
            opp = self._policy_cache.get(path)
            if opp is None:
                opp = PolicyOpponent(path, frameskip=self.frameskip)
                self._policy_cache[path] = opp
            opp.reset(seed)
            self._opponent = opp
            self._p2_kind = f"pool:{Path(path).stem}"
            return
        style = str(self.np_random.choice(self._styles))
        dummy = DummyOpponent(style)
        dummy.reset(seed)
        self._opponent = dummy
        self._p2_kind = f"dummy:{dummy.resolved}"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        state = self.sim.reset(int(seed))
        if self.random_spawn:
            randomize_spawn(state, self.np_random)
        self._pick_opponent(int(seed))
        return flatten_state(state), self._info(state, frames=0)

    def _time_cost_per_step(self) -> float:
        policy_steps = ROUND_TIMER / max(self.frameskip, 1)
        if policy_steps <= 0:
            return 0.0
        return self.time_cost / policy_steps

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self.sim.get_state()
        p1_raw = action_to_raw(int(action), state.p1.facing)
        h1_before = state.p1.health
        h2_before = state.p2.health
        gap_px = abs(state.p1.x - state.p2.x) / SUBPIXEL

        frames = 0
        whiffs = 0
        timed_out = False
        for _ in range(self.frameskip):
            state = self.sim.get_state()
            if state.round_state != RoundState.FIGHTING:
                break
            p2_raw = self._opponent.raw(state, 2)
            state = self.sim.step(p1_raw, p2_raw)
            frames += 1
            events = self.sim.get_events()
            whiffs += sum(1 for e in events if e.kind == "whiff" and e.attacker == 1)
            if _timed_out(events):
                timed_out = True

        dealt = h2_before - state.p2.health
        taken = h1_before - state.p1.health
        reward = self.damage_coef * (dealt - taken) / HEALTH_MAX
        if gap_px < self.whiff_range_px:
            reward -= self.whiff_penalty * whiffs
        reward -= self._time_cost_per_step()
        terminated = state.round_state != RoundState.FIGHTING
        if terminated:
            if not timed_out:
                timed_out = _timed_out(state.last_events)
            reward += terminal_bonus(
                state.round_state,
                state.p1.health,
                state.p2.health,
                timed_out,
                self.timeout_lead,
                self.timeout_win,
                self.timeout_draw,
            )
        return flatten_state(state), float(reward), terminated, False, self._info(state, frames)

    def _info(self, state, frames: int) -> dict[str, Any]:
        dummy_style = (
            self._opponent.resolved if isinstance(self._opponent, DummyOpponent) else ""
        )
        return {
            "frames": frames,
            "round_state": state.round_state.value,
            "p1_health": state.p1.health,
            "p2_health": state.p2.health,
            "p2_style": dummy_style,
            "p2_kind": self._p2_kind,
        }


def make_fight_env(
    dummy_style: str = "mix",
    frameskip: int = DEFAULT_FRAMESKIP,
    damage_coef: float = DEFAULT_DAMAGE_COEF,
    random_spawn: bool = True,
    pool_dir: str = "",
    pool_frac: float = DEFAULT_POOL_FRAC,
    dummy_only: bool = False,
    timeout_lead: int = DEFAULT_TIMEOUT_LEAD,
    timeout_win: float = DEFAULT_TIMEOUT_WIN,
    timeout_draw: float = DEFAULT_TIMEOUT_DRAW,
    time_cost: float = DEFAULT_TIME_COST,
    whiff_penalty: float = DEFAULT_WHIFF_PENALTY,
    whiff_range_px: float = DEFAULT_WHIFF_RANGE_PX,
) -> Callable[[], Any]:
    """Thunk for DummyVecEnv. Wraps ActionMasker when sb3-contrib is installed."""

    def _thunk():
        env = FightEnv(
            dummy_style=dummy_style,
            frameskip=frameskip,
            damage_coef=damage_coef,
            random_spawn=random_spawn,
            pool_dir=pool_dir,
            pool_frac=pool_frac,
            dummy_only=dummy_only,
            timeout_lead=timeout_lead,
            timeout_win=timeout_win,
            timeout_draw=timeout_draw,
            time_cost=time_cost,
            whiff_penalty=whiff_penalty,
            whiff_range_px=whiff_range_px,
        )
        try:
            from sb3_contrib.common.wrappers import ActionMasker

            return ActionMasker(env, lambda e: e.action_masks())
        except ImportError:
            return env

    return _thunk
