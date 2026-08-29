"""Rule-based dummy for playtesting. Not an ML opponent pool."""

from __future__ import annotations

from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.state import GameState


def dummy_input(state: GameState, pid: int, style: str) -> RawInput:
    fighter = state.p1 if pid == 1 else state.p2
    if style in ("idle", "dummy"):
        return RawInput()
    if style == "block":
        return semantic_to_raw(SemanticAction.BLOCK, fighter.facing)
    if style == "mash":
        return RawInput(a=(state.frame % 8) < 2)
    if style == "crouch_block":
        raw = semantic_to_raw(SemanticAction.BLOCK, fighter.facing)
        return RawInput(
            left=raw.left,
            right=raw.right,
            down=True,
        )
    raise ValueError(f"unknown dummy style: {style}")
