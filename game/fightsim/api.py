"""Public Simulator: reset / step / get_state / save_state / load_state / render."""

from __future__ import annotations

from typing import Any, Optional

from fightsim.events import CombatEvent
from fightsim.input import RawInput
from fightsim.sim import step_state_mutate
from fightsim.state import GameState, dumps_state, initial_state, loads_state
from fightsim.rng import RngState, rand_u32, seed_rng


class Simulator:
    """Headless by default. render() is optional and must not affect state."""

    def __init__(self) -> None:
        self._state: GameState = initial_state(0)
        self._events: list[CombatEvent] = []
        self._seed: int = 0

    def reset(self, seed: int) -> GameState:
        self._seed = int(seed)
        self._state = initial_state(self._seed)
        self._events = []
        return self._state

    def step(self, p1: RawInput, p2: RawInput) -> GameState:
        """
        Takes a step in the game, mutating its game state.
        """
        self._state, self._events = step_state_mutate(self._state, p1, p2)
        return self._state

    def get_state(self) -> GameState:
        """
        Warning:
            This state will be modified when Simulator.step is called. To make a copy of the state, use save_state and load_state. 
        Todo:
            This function is currently being used *dangerously* in the rest of the codebase. That, is the private member._state is being modified outside of this class. Unfortunately this is necessary for dummy bots (in fightsim/dummy.py) to use the RNGenerator within the game state.
            Will need to migrate that codebase into using the Simulator.rand_u32.

            Once this is no longer being used dangerously. This function should make a copy of self._state and return it. So that it isn't being used dangerously again.
        """
        return self._state

    def rand_u32(self) -> int:
        num, self._state.rng = rand_u32(self._state.rng)
        return num
    def save_state(self) -> bytes:
        return dumps_state(self._state)

    def load_state(self, snapshot: bytes) -> None:
        self._state = loads_state(snapshot)
        self._events = list(self._state.last_events)

    def get_events(self) -> list[CombatEvent]:
        return list(self._events)

    def render(self, surface: Optional[Any] = None) -> None:
        try:
            from fightsim.render import render_state

            render_state(self._state, surface)
        except ImportError:
            return
