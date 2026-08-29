"""Typed combat events recorded each frame for replays and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CombatEvent:
    kind: str
    attacker: int = 0
    defender: int = 0
    move_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "attacker": self.attacker,
            "defender": self.defender,
            "move_id": self.move_id,
            "data": dict(self.data),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CombatEvent:
        return CombatEvent(
            kind=str(d["kind"]),
            attacker=int(d.get("attacker", 0)),
            defender=int(d.get("defender", 0)),
            move_id=str(d.get("move_id", "")),
            data=dict(d.get("data") or {}),
        )
