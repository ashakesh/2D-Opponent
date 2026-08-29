"""Replay tape: version, config, seed, initial snapshot, per-frame inputs/state/events.

Rewards are intentionally omitted — those belong in the ML wrapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fightsim.config import CHARACTER_CONFIG, SIMULATOR_VERSION
from fightsim.events import CombatEvent
from fightsim.input import RawInput
from fightsim.state import GameState, dumps_state


def _input_dict(inp: RawInput) -> dict[str, bool]:
    return inp.to_dict()


@dataclass
class Replay:
    simulator_version: str
    character_config: str
    seed: int
    initial_state: dict[str, Any]
    frames: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulator_version": self.simulator_version,
            "character_config": self.character_config,
            "seed": self.seed,
            "initial_state": self.initial_state,
            "frames": self.frames,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Replay:
        return Replay(
            simulator_version=str(d["simulator_version"]),
            character_config=str(d["character_config"]),
            seed=int(d["seed"]),
            initial_state=dict(d["initial_state"]),
            frames=list(d["frames"]),
        )

    @staticmethod
    def from_bytes(blob: bytes) -> Replay:
        return Replay.from_dict(json.loads(blob.decode("utf-8")))


class ReplayRecorder:
    def __init__(self, seed: int, initial: GameState) -> None:
        self.replay = Replay(
            simulator_version=SIMULATOR_VERSION,
            character_config=CHARACTER_CONFIG,
            seed=seed,
            initial_state=initial.to_dict(),
        )

    def record(
        self,
        p1: RawInput,
        p2: RawInput,
        state: GameState,
        events: list[CombatEvent],
    ) -> None:
        self.replay.frames.append(
            {
                "p1": _input_dict(p1),
                "p2": _input_dict(p2),
                "state": state.to_dict(),
                "events": [e.to_dict() for e in events],
            }
        )


def playback_matches(replay: Replay) -> bool:
    """Re-run recorded inputs from the initial snapshot; compare serialized states."""
    from fightsim.api import Simulator

    sim = Simulator()
    sim.load_state(dumps_state(GameState.from_dict(replay.initial_state)))
    for frame in replay.frames:
        p1 = RawInput.from_dict(frame["p1"])
        p2 = RawInput.from_dict(frame["p2"])
        sim.step(p1, p2)
        got = json.dumps(sim.get_state().to_dict(), sort_keys=True, separators=(",", ":"))
        want = json.dumps(frame["state"], sort_keys=True, separators=(",", ":"))
        if got != want:
            return False
    return True
