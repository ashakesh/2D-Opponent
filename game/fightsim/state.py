"""Explicit, serializable simulation state. No hidden fields."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fightsim.config import (
    CHARACTER_CONFIG,
    CROUCH_PUSH_H,
    CROUCH_PUSH_W,
    HEALTH_MAX,
    INPUT_BUFFER,
    METER_START,
    ROUND_TIMER,
    SIMULATOR_VERSION,
    STAND_PUSH_H,
    STAND_PUSH_W,
    START_P1_X,
    START_P2_X,
)
from fightsim.events import CombatEvent
from fightsim.rng import RngState, seed_rng


class FSM(str, Enum):
    IDLE = "idle"
    WALK = "walk"
    CROUCH = "crouch"
    JUMP = "jump"
    DASH = "dash"
    ATTACK = "attack"
    HITSTUN = "hitstun"
    BLOCKSTUN = "blockstun"
    KNOCKDOWN = "knockdown"
    DEAD = "dead"


class RoundState(str, Enum):
    FIGHTING = "fighting"
    P1_WIN = "p1_win"
    P2_WIN = "p2_win"
    DRAW = "draw"


class Height(str, Enum):
    HIGH = "high"
    LOW = "low"
    OVERHEAD = "overhead"
    THROW = "throw"


@dataclass
class Projectile:
    owner: int
    x: int
    y: int
    vx: int
    vy: int
    w: int
    h: int
    damage: int
    hitstun: int
    blockstun: int
    hitstop: int
    knockback_x: int
    knockback_y: int
    lifetime: int
    height: str = Height.HIGH.value

    def to_dict(self) -> dict[str, int | str]:
        return {
            "owner": self.owner,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "w": self.w,
            "h": self.h,
            "damage": self.damage,
            "hitstun": self.hitstun,
            "blockstun": self.blockstun,
            "hitstop": self.hitstop,
            "knockback_x": self.knockback_x,
            "knockback_y": self.knockback_y,
            "lifetime": self.lifetime,
            "height": self.height,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Projectile:
        return Projectile(
            owner=int(d["owner"]),
            x=int(d["x"]),
            y=int(d["y"]),
            vx=int(d["vx"]),
            vy=int(d["vy"]),
            w=int(d["w"]),
            h=int(d["h"]),
            damage=int(d["damage"]),
            hitstun=int(d["hitstun"]),
            blockstun=int(d["blockstun"]),
            hitstop=int(d["hitstop"]),
            knockback_x=int(d["knockback_x"]),
            knockback_y=int(d["knockback_y"]),
            lifetime=int(d["lifetime"]),
            height=str(d.get("height", Height.HIGH.value)),
        )


@dataclass
class FighterState:
    x: int
    y: int
    vx: int
    vy: int
    facing: int
    health: int
    meter: int
    fsm: FSM
    state_frame: int
    move_id: str
    hitstun: int
    blockstun: int
    invuln: int
    move_has_hit: bool
    move_spawned: bool
    can_special_cancel: bool
    input_buffer: list[int]
    prev_a: bool
    prev_b: bool
    prev_c: bool
    push_w: int
    push_h: int
    combo_received: int
    knockdown_pending: bool
    bounce_count: int
    landing_lock: int
    jump_lock: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "facing": self.facing,
            "health": self.health,
            "meter": self.meter,
            "fsm": self.fsm.value,
            "state_frame": self.state_frame,
            "move_id": self.move_id,
            "hitstun": self.hitstun,
            "blockstun": self.blockstun,
            "invuln": self.invuln,
            "move_has_hit": self.move_has_hit,
            "move_spawned": self.move_spawned,
            "can_special_cancel": self.can_special_cancel,
            "input_buffer": list(self.input_buffer),
            "prev_a": self.prev_a,
            "prev_b": self.prev_b,
            "prev_c": self.prev_c,
            "push_w": self.push_w,
            "push_h": self.push_h,
            "combo_received": self.combo_received,
            "knockdown_pending": self.knockdown_pending,
            "bounce_count": self.bounce_count,
            "landing_lock": self.landing_lock,
            "jump_lock": self.jump_lock,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FighterState:
        buf = [int(x) for x in d["input_buffer"]]
        if len(buf) < INPUT_BUFFER:
            buf = [5] * (INPUT_BUFFER - len(buf)) + buf
        return FighterState(
            x=int(d["x"]),
            y=int(d["y"]),
            vx=int(d["vx"]),
            vy=int(d["vy"]),
            facing=int(d["facing"]),
            health=int(d["health"]),
            meter=int(d["meter"]),
            fsm=FSM(d["fsm"]),
            state_frame=int(d["state_frame"]),
            move_id=str(d["move_id"]),
            hitstun=int(d["hitstun"]),
            blockstun=int(d["blockstun"]),
            invuln=int(d["invuln"]),
            move_has_hit=bool(d["move_has_hit"]),
            move_spawned=bool(d["move_spawned"]),
            can_special_cancel=bool(d["can_special_cancel"]),
            input_buffer=buf[-INPUT_BUFFER:],
            prev_a=bool(d["prev_a"]),
            prev_b=bool(d["prev_b"]),
            prev_c=bool(d["prev_c"]),
            push_w=int(d["push_w"]),
            push_h=int(d["push_h"]),
            combo_received=int(d["combo_received"]),
            knockdown_pending=bool(d["knockdown_pending"]),
            bounce_count=int(d.get("bounce_count", 0)),
            landing_lock=int(d.get("landing_lock", 0)),
            jump_lock=int(d.get("jump_lock", 0)),
        )


@dataclass
class GameState:
    frame: int
    round_state: RoundState
    timer: int
    rng: RngState
    p1: FighterState
    p2: FighterState
    projectiles: list[Projectile]
    hitstop: int
    version: str = SIMULATOR_VERSION
    config_id: str = CHARACTER_CONFIG
    last_events: list[CombatEvent] = field(default_factory=list)

    def fighter(self, pid: int) -> FighterState:
        return self.p1 if pid == 1 else self.p2

    def opponent(self, pid: int) -> FighterState:
        return self.p2 if pid == 1 else self.p1

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "round_state": self.round_state.value,
            "timer": self.timer,
            "rng": {"state": self.rng.state, "inc": self.rng.inc},
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict(),
            "projectiles": [p.to_dict() for p in self.projectiles],
            "hitstop": self.hitstop,
            "version": self.version,
            "config_id": self.config_id,
            "last_events": [e.to_dict() for e in self.last_events],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GameState:
        rng_d = d["rng"]
        return GameState(
            frame=int(d["frame"]),
            round_state=RoundState(d["round_state"]),
            timer=int(d["timer"]),
            rng=RngState(int(rng_d["state"]), int(rng_d["inc"])),
            p1=FighterState.from_dict(d["p1"]),
            p2=FighterState.from_dict(d["p2"]),
            projectiles=[Projectile.from_dict(p) for p in d["projectiles"]],
            hitstop=int(d["hitstop"]),
            version=str(d.get("version", SIMULATOR_VERSION)),
            config_id=str(d.get("config_id", CHARACTER_CONFIG)),
            last_events=[CombatEvent.from_dict(e) for e in d.get("last_events") or []],
        )


def _blank_fighter(x: int, facing: int) -> FighterState:
    return FighterState(
        x=x,
        y=0,
        vx=0,
        vy=0,
        facing=facing,
        health=HEALTH_MAX,
        meter=METER_START,
        fsm=FSM.IDLE,
        state_frame=0,
        move_id="",
        hitstun=0,
        blockstun=0,
        invuln=0,
        move_has_hit=False,
        move_spawned=False,
        can_special_cancel=False,
        input_buffer=[5] * INPUT_BUFFER,
        prev_a=False,
        prev_b=False,
        prev_c=False,
        push_w=STAND_PUSH_W,
        push_h=STAND_PUSH_H,
        combo_received=0,
        knockdown_pending=False,
        bounce_count=0,
        landing_lock=0,
        jump_lock=0,
    )


def initial_state(seed: int) -> GameState:
    return GameState(
        frame=0,
        round_state=RoundState.FIGHTING,
        timer=ROUND_TIMER,
        rng=seed_rng(seed),
        p1=_blank_fighter(START_P1_X, 1),
        p2=_blank_fighter(START_P2_X, -1),
        projectiles=[],
        hitstop=0,
        version=SIMULATOR_VERSION,
        config_id=CHARACTER_CONFIG,
        last_events=[],
    )


def dumps_state(state: GameState) -> bytes:
    payload = state.to_dict()
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def loads_state(blob: bytes) -> GameState:
    data = json.loads(blob.decode("utf-8"))
    return GameState.from_dict(data)


def crouch_dims() -> tuple[int, int]:
    return CROUCH_PUSH_W, CROUCH_PUSH_H


def stand_dims() -> tuple[int, int]:
    return STAND_PUSH_W, STAND_PUSH_H
