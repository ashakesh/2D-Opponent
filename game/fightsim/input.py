"""Raw controller input, facing-relative motions, semantic adapter.

The simulator core only consumes RawInput. SemanticAction is a helper for
Phase-1 ML and scripted dummies. command_special_* skips motion detection; raw
policies must not set those flags if the experiment is 'discover specials'.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fightsim.config import DASH_WINDOW, INPUT_BUFFER, MOTION_WINDOW


@dataclass(frozen=True)
class RawInput:
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    a: bool = False
    b: bool = False
    c: bool = False
    d: bool = False
    # Debug / semantic ML only — not part of a real controller.
    command_special_1: bool = False
    command_special_2: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "left": self.left,
            "right": self.right,
            "up": self.up,
            "down": self.down,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "command_special_1": self.command_special_1,
            "command_special_2": self.command_special_2,
        }

    @staticmethod
    def from_dict(d: dict[str, bool]) -> RawInput:
        return RawInput(
            left=bool(d.get("left", False)),
            right=bool(d.get("right", False)),
            up=bool(d.get("up", False)),
            down=bool(d.get("down", False)),
            a=bool(d.get("a", False)),
            b=bool(d.get("b", False)),
            c=bool(d.get("c", False)),
            d=bool(d.get("d", False)),
            command_special_1=bool(d.get("command_special_1", False)),
            command_special_2=bool(d.get("command_special_2", False)),
        )

    def horiz(self) -> int:
        dx = 0
        if self.right:
            dx += 1
        if self.left:
            dx -= 1
        return dx

    def vert(self) -> int:
        dy = 0
        if self.up:
            dy += 1
        if self.down:
            dy -= 1
        return dy


class SemanticAction(str, Enum):
    NEUTRAL = "neutral"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    BLOCK = "block"
    LIGHT = "light"
    HEAVY = "heavy"
    THROW = "throw"
    SPECIAL_1 = "special_1"
    SPECIAL_2 = "special_2"


def semantic_to_raw(action: SemanticAction, facing: int) -> RawInput:
    """Map one semantic verb to a single frame of raw input.

    SPECIAL_1 / SPECIAL_2 set command flags so they complete in one step().
    BLOCK holds back relative to current facing.
    """
    if action is SemanticAction.NEUTRAL:
        return RawInput()
    if action is SemanticAction.LEFT:
        return RawInput(left=True)
    if action is SemanticAction.RIGHT:
        return RawInput(right=True)
    if action is SemanticAction.UP:
        return RawInput(up=True)
    if action is SemanticAction.DOWN:
        return RawInput(down=True)
    if action is SemanticAction.BLOCK:
        if facing >= 0:
            return RawInput(left=True)
        return RawInput(right=True)
    if action is SemanticAction.LIGHT:
        return RawInput(a=True)
    if action is SemanticAction.HEAVY:
        return RawInput(b=True)
    if action is SemanticAction.THROW:
        return RawInput(c=True)
    if action is SemanticAction.SPECIAL_1:
        return RawInput(command_special_1=True)
    if action is SemanticAction.SPECIAL_2:
        return RawInput(command_special_2=True)
    return RawInput()


def stick_numpad(inp: RawInput, facing: int) -> int:
    """Numpad direction relative to facing: 6=forward, 4=back, 2=down, 8=up."""
    face = 1 if facing >= 0 else -1
    dx = inp.horiz()
    dy = inp.vert()
    fwd = dx * face
    return 5 + fwd + 3 * dy


def empty_buffer() -> list[int]:
    return [5] * INPUT_BUFFER


def push_buffer(buf: list[int], numpad: int) -> list[int]:
    out = buf[-INPUT_BUFFER + 1 :] + [int(numpad)]
    if len(out) < INPUT_BUFFER:
        out = [5] * (INPUT_BUFFER - len(out)) + out
    return out[-INPUT_BUFFER:]


def _subsequence(buf: list[int], seq: tuple[int, ...], window: int) -> bool:
    recent = buf[-window:]
    i = 0
    for d in recent:
        if d == seq[i]:
            i += 1
            if i == len(seq):
                return True
    return False


def motion_236(buf: list[int]) -> bool:
    """Quarter-circle forward."""
    return _subsequence(buf, (2, 3, 6), MOTION_WINDOW)


def motion_623(buf: list[int]) -> bool:
    """DP / shoryu motion."""
    return _subsequence(buf, (6, 2, 3), MOTION_WINDOW)


def tap_forward(buf: list[int]) -> bool:
    """66 dash: forward tap, then forward again (rising into 6)."""
    if not buf or buf[-1] != 6:
        return False
    if len(buf) >= 2 and buf[-2] == 6:
        return False
    window = buf[-DASH_WINDOW:-1]
    return 6 in window


def tap_back(buf: list[int]) -> bool:
    if not buf or buf[-1] != 4:
        return False
    if len(buf) >= 2 and buf[-2] == 4:
        return False
    return 4 in buf[-DASH_WINDOW:-1]


def holding_back(inp: RawInput, facing: int) -> bool:
    face = 1 if facing >= 0 else -1
    return inp.horiz() * face < 0


def holding_forward(inp: RawInput, facing: int) -> bool:
    face = 1 if facing >= 0 else -1
    return inp.horiz() * face > 0
