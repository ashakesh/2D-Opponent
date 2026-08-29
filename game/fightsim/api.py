"""Public Simulator: reset / step / get_state / save_state / load_state / render."""

from __future__ import annotations

from typing import Any, Optional

from fightsim.events import CombatEvent
from fightsim.input import RawInput
from fightsim.sim import step_state
from fightsim.state import GameState, dumps_state, initial_state, loads_state


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
        self._state, self._events = step_state(self._state, p1, p2)
        return self._state

    def get_state(self) -> GameState:
        return self._state

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
