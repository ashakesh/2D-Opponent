"""Clip table and pose resolve. Render-only — never imported from step()."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fightsim.config import CHARACTER_CONFIG, KNOCKDOWN_FRAMES, WAKEUP_INVULN
from fightsim.moves import MOVES
from fightsim.physics import airborne
from fightsim.state import FSM, FighterState, Projectile

ATTACK_MOVES = (
    "stand_light",
    "crouch_light",
    "stand_heavy",
    "jump_light",
    "throw",
    "fireball",
    "dp",
    "pulse",
)

PHASES = ("startup", "active", "recovery")


def _files(stem: str, n: int) -> list[str]:
    return [f"{stem}_{i:02d}.png" for i in range(n)]


CLIPS: dict[str, dict[str, object]] = {
    "idle": {"files": _files("idle", 4), "loop": True},
    "walk": {"files": _files("walk", 4), "loop": True},
    "crouch": {"files": _files("crouch", 1), "loop": False},
    "jump_rise": {"files": _files("jump_rise", 1), "loop": False},
    "jump_fall": {"files": _files("jump_fall", 1), "loop": False},
    "land": {"files": _files("land", 1), "loop": False},
    "dash_fwd": {"files": _files("dash_fwd", 1), "loop": False},
    "dash_back": {"files": _files("dash_back", 1), "loop": False},
    "hitstun": {"files": _files("hitstun", 1), "loop": False},
    "hitstun_air": {"files": _files("hitstun_air", 1), "loop": False},
    "blockstun": {"files": _files("blockstun", 1), "loop": False},
    "jump_light_land_recovery": {"files": _files("jump_light_land_recovery", 1), "loop": False},
    "knockdown": {"files": _files("knockdown", 2), "loop": False},
    "wakeup": {"files": _files("wakeup", 1), "loop": False},
    "dead": {"files": _files("dead", 1), "loop": False},
    "fireball_proj": {"files": _files("fireball_proj", 2), "loop": True},
}

for _move in ATTACK_MOVES:
    for _phase in PHASES:
        _name = f"{_move}_{_phase}"
        CLIPS[_name] = {"files": _files(_name, 1), "loop": False}


def all_clip_filenames() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for spec in CLIPS.values():
        for name in spec["files"]:  # type: ignore[union-attr]
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def fighter_asset_root(character: str = CHARACTER_CONFIG) -> Path:
    return Path(__file__).resolve().parent / "assets" / "fighters" / character


def load_rig(character: str = CHARACTER_CONFIG) -> dict:
    return json.loads((fighter_asset_root(character) / "rig.json").read_text())


def pivot_topleft(
    feet_x: int, feet_y: int, pivot_x: int, pivot_y: int, width: int, facing: int
) -> tuple[int, int]:
    """Screen top-left of a sprite whose feet sit at (feet_x, feet_y)."""
    px = pivot_x if facing >= 0 else width - pivot_x
    return feet_x - px, feet_y - pivot_y


def resolve_sprite_path(filename: str, character: str = CHARACTER_CONFIG) -> Path | None:
    path = fighter_asset_root(character) / "art" / filename
    return path if path.is_file() else None


@dataclass(frozen=True)
class PoseRef:
    clip: str
    filename: str
    facing: int


def _pose(clip: str, index: int, facing: int) -> PoseRef:
    files: list[str] = CLIPS[clip]["files"]  # type: ignore[assignment]
    if not files:
        raise KeyError(clip)
    return PoseRef(clip, files[index % len(files)], facing)


def _attack_clip(f: FighterState) -> str:
    move = MOVES.get(f.move_id)
    if move is None:
        return "idle"
    if f.state_frame < move.startup:
        phase = "startup"
    elif move.is_active(f.state_frame):
        phase = "active"
    else:
        phase = "recovery"
    name = f"{f.move_id}_{phase}"
    if f.move_id == "jump_light" and phase == "recovery" and not airborne(f):
        return "jump_light_land_recovery"
    if name not in CLIPS:
        return "idle"
    return name


def resolve_clip(f: FighterState, frame: int = 0) -> PoseRef:
    facing = 1 if f.facing >= 0 else -1
    if f.fsm == FSM.DEAD:
        return _pose("dead", 0, facing)
    if f.fsm == FSM.ATTACK and f.move_id:
        return _pose(_attack_clip(f), 0, facing)
    if f.fsm == FSM.HITSTUN:
        clip = "hitstun_air" if airborne(f) else "hitstun"
        return _pose(clip, 0, facing)
    if f.fsm == FSM.BLOCKSTUN:
        return _pose("blockstun", 0, facing)
    if f.fsm == FSM.KNOCKDOWN:
        if f.state_frame >= KNOCKDOWN_FRAMES - WAKEUP_INVULN:
            return _pose("wakeup", 0, facing)
        idx = 0 if f.state_frame * 2 < KNOCKDOWN_FRAMES else 1
        return _pose("knockdown", idx, facing)
    if f.fsm == FSM.DASH:
        clip = "dash_back" if f.move_id == "dash_back" else "dash_fwd"
        return _pose(clip, 0, facing)
    if f.landing_lock > 0 and not airborne(f):
        return _pose("land", 0, facing)
    if f.fsm == FSM.JUMP:
        clip = "jump_rise" if f.vy > 0 else "jump_fall"
        return _pose(clip, 0, facing)
    if f.fsm == FSM.CROUCH:
        return _pose("crouch", 0, facing)
    if f.fsm == FSM.WALK:
        return _pose("walk", frame // 4, facing)
    return _pose("idle", frame // 10, facing)


def resolve_projectile_clip(p: Projectile, frame: int = 0) -> PoseRef:
    facing = 1 if p.vx >= 0 else -1
    return _pose("fireball_proj", frame // 4, facing)
