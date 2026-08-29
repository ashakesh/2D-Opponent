"""Backend-agnostic simulator contract.

Any engine (fightsim today, another game later) can be wrapped for training
if it matches this protocol. This module must not import fightsim.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class FightingSimulator(Protocol):
    """Frame-by-frame two-player combat backend."""

    def reset(self, seed: int) -> Any:
        """Start a new round. Deterministic for a given seed."""

    def step(self, p1: Any, p2: Any) -> Any:
        """Advance exactly one simulation frame. Both players always act."""

    def get_state(self) -> Any:
        """Current explicit game state (serializable)."""

    def save_state(self) -> bytes:
        """Canonical snapshot for rollback / replay."""

    def load_state(self, snapshot: bytes) -> None:
        """Restore a snapshot from save_state()."""

    def render(self, surface: Optional[Any] = None) -> None:
        """Optional debug present. Must not be required for stepping."""
