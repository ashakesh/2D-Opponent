"""Frame-data tables. Engine code should not special-case individual moves."""

from __future__ import annotations

from dataclasses import dataclass

from fightsim.config import (
    HITSTOP_DEFAULT,
    PULSE_COST,
    SPECIAL_COST,
    SUBPIXEL,
    THROW_RANGE,
    px,
)
from fightsim.state import Height


@dataclass(frozen=True)
class HitboxDef:
    offset_x: int
    offset_y: int
    w: int
    h: int


@dataclass(frozen=True)
class MoveDef:
    move_id: str
    startup: int
    active: int
    recovery: int
    damage: int
    hitstun: int
    blockstun: int
    hitstop: int
    knockback_x: int
    knockback_y: int
    hitbox: HitboxDef | None
    height: str
    meter_cost: int = 0
    meter_gain: int = 80
    spawns_projectile: bool = False
    projectile_frame: int = 0
    invuln_frames: int = 0
    knockdown: bool = False
    is_throw: bool = False
    throw_range: int = 0
    air_ok: bool = False
    ground_ok: bool = True
    cancelable: bool = True
    recovery_on_hit: int | None = None

    @property
    def total(self) -> int:
        return self.startup + self.active + self.recovery

    def duration(self, hit: bool) -> int:
        rec = self.recovery_on_hit if hit and self.recovery_on_hit is not None else self.recovery
        return self.startup + self.active + rec

    def is_active(self, frame: int) -> bool:
        return self.startup <= frame < self.startup + self.active


STAND_LIGHT = MoveDef(
    move_id="stand_light",
    startup=3,
    active=2,
    recovery=8,
    damage=400,
    hitstun=8,
    blockstun=6,
    hitstop=HITSTOP_DEFAULT,
    knockback_x=2 * SUBPIXEL,
    knockback_y=0,
    hitbox=HitboxDef(px(36), px(52), px(42), px(24)),
    height=Height.HIGH.value,
    meter_gain=50,
    cancelable=True,
)

CROUCH_LIGHT = MoveDef(
    move_id="crouch_light",
    startup=5,
    active=3,
    recovery=10,
    damage=450,
    hitstun=8,
    blockstun=6,
    hitstop=HITSTOP_DEFAULT,
    knockback_x=2 * SUBPIXEL,
    knockback_y=0,
    hitbox=HitboxDef(px(52), px(16), px(48), px(18)),
    height=Height.LOW.value,
    meter_gain=50,
    cancelable=True,
)

STAND_HEAVY = MoveDef(
    move_id="stand_heavy",
    startup=8,
    active=3,
    recovery=16,
    damage=800,
    hitstun=18,
    blockstun=10,
    hitstop=5,
    knockback_x=5 * SUBPIXEL,
    knockback_y=0,
    hitbox=HitboxDef(px(44), px(50), px(50), px(28)),
    height=Height.HIGH.value,
    meter_gain=80,
    cancelable=True,
)

JUMP_LIGHT = MoveDef(
    move_id="jump_light",
    startup=4,
    active=10,
    recovery=6,
    damage=550,
    hitstun=14,
    blockstun=8,
    hitstop=HITSTOP_DEFAULT,
    knockback_x=14 * SUBPIXEL,
    knockback_y=14 * SUBPIXEL,
    hitbox=HitboxDef(px(48), px(8), px(50), px(32)),
    height=Height.OVERHEAD.value,
    meter_gain=60,
    air_ok=True,
    ground_ok=False,
    cancelable=False,
    knockdown=True,
)

THROW = MoveDef(
    move_id="throw",
    startup=4,
    active=2,
    recovery=22,
    recovery_on_hit=8,
    damage=1200,
    hitstun=22,
    blockstun=0,
    hitstop=6,
    knockback_x=16 * SUBPIXEL,
    knockback_y=14 * SUBPIXEL,
    hitbox=None,
    height=Height.THROW.value,
    meter_gain=100,
    knockdown=True,
    is_throw=True,
    throw_range=THROW_RANGE,
    cancelable=False,
)

FIREBALL = MoveDef(
    move_id="fireball",
    startup=12,
    active=2,
    recovery=18,
    damage=0,
    hitstun=0,
    blockstun=0,
    hitstop=0,
    knockback_x=0,
    knockback_y=0,
    hitbox=None,
    height=Height.HIGH.value,
    meter_cost=SPECIAL_COST,
    meter_gain=80,
    spawns_projectile=True,
    projectile_frame=12,
    cancelable=False,
)

DP = MoveDef(
    move_id="dp",
    startup=5,
    active=6,
    recovery=22,
    recovery_on_hit=10,
    damage=1500,
    hitstun=24,
    blockstun=8,
    hitstop=6,
    knockback_x=10 * SUBPIXEL,
    knockback_y=18 * SUBPIXEL,
    hitbox=HitboxDef(px(28), px(70), px(36), px(70)),
    height=Height.HIGH.value,
    meter_cost=SPECIAL_COST,
    meter_gain=120,
    invuln_frames=4,
    knockdown=True,
    cancelable=False,
)

PULSE = MoveDef(
    move_id="pulse",
    startup=4,
    active=4,
    recovery=16,
    recovery_on_hit=10,
    damage=50,
    hitstun=12,
    blockstun=8,
    hitstop=3,
    knockback_x=14 * SUBPIXEL,
    knockback_y=0,
    hitbox=HitboxDef(px(10), px(40), px(96), px(80)),
    height=Height.HIGH.value,
    meter_cost=PULSE_COST,
    meter_gain=40,
    knockdown=False,
    cancelable=False,
)

MOVES: dict[str, MoveDef] = {
    m.move_id: m
    for m in (
        STAND_LIGHT,
        CROUCH_LIGHT,
        STAND_HEAVY,
        JUMP_LIGHT,
        THROW,
        FIREBALL,
        DP,
        PULSE,
    )
}

NORMALS = frozenset({"stand_light", "crouch_light", "stand_heavy", "jump_light"})
LIGHTS = frozenset({"stand_light", "crouch_light"})
SPECIALS = frozenset({"fireball", "dp", "pulse"})

PROJ_W = px(20)
PROJ_H = px(16)
PROJ_SPEED = 6 * SUBPIXEL
PROJ_LIFETIME = 70
PROJ_DAMAGE = 600
PROJ_HITSTUN = 8
PROJ_BLOCKSTUN = 8
PROJ_HITSTOP = 3
PROJ_KNOCKBACK = 3 * SUBPIXEL
PROJ_Y = px(52)
