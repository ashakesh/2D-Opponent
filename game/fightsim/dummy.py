"""Rule-based dummies for playtesting. Not an ML opponent pool.

Random choices consume GameState.rng (PCG32) so a seed + style is reproducible.
"""

from __future__ import annotations

from fightsim.config import PULSE_COST, STAGE_WIDTH, SUBPIXEL, THROW_RANGE
from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.physics import airborne
from fightsim.rng import RngState, rand_u32, seed_rng
from fightsim.state import FSM, GameState

# Pulse hitbox reaches ~106px; throw 52px; stand heavy ~94px.
_HEAVY_RANGE = 80 * SUBPIXEL
_PULSE_RANGE = 100 * SUBPIXEL
_IDEAL_ZONER_GAP = 200 * SUBPIXEL
_IDEAL_BAND = 28 * SUBPIXEL
_CORNER_MARGIN = 72 * SUBPIXEL
# Fireball is 6px/f; jump clears the ball in a few frames. Jump only when a
# dodge would actually beat the hit, not when the projectile is still mid-screen.
_FIREBALL_DODGE_RANGE = 120 * SUBPIXEL

KEEP_STYLES: tuple[str, ...] = ("idle", "block", "mash", "crouch_block")
PERSONALITY_STYLES: tuple[str, ...] = (
    "aggressive",
    "defensive",
    "rushdown",
    "mobile",
    "zoner",
    "grappler",
)
DUMMY_STYLES: tuple[str, ...] = KEEP_STYLES + PERSONALITY_STYLES


def pick_mix_style(rng: RngState) -> tuple[str, RngState]:
    val, rng = rand_u32(rng)
    return PERSONALITY_STYLES[val % len(PERSONALITY_STYLES)], rng


def mix_rng_from_seed(seed: int) -> RngState:
    return seed_rng(int(seed) ^ 0xD1CE5)


def _roll(state: GameState, n: int = 100) -> int:
    val, state.rng = rand_u32(state.rng)
    return val % n


def _me_other(state: GameState, pid: int):
    if pid == 1:
        return state.p1, state.p2
    return state.p2, state.p1


def _gap(me, other) -> int:
    return abs(me.x - other.x)


def _toward(me) -> RawInput:
    if me.facing >= 0:
        return RawInput(right=True)
    return RawInput(left=True)


def _away(me) -> RawInput:
    if me.facing >= 0:
        return RawInput(left=True)
    return RawInput(right=True)


def _crouch_block(me) -> RawInput:
    block = semantic_to_raw(SemanticAction.BLOCK, me.facing)
    return RawInput(left=block.left, right=block.right, down=True)


def _special(cmd: int) -> RawInput:
    if cmd == 1:
        return RawInput(command_special_1=True)
    if cmd == 2:
        return RawInput(command_special_2=True)
    return RawInput(command_special_3=True)


def dummy_input(state: GameState, pid: int, style: str) -> RawInput:
    me, other = _me_other(state, pid)
    if style in ("idle", "dummy"):
        return RawInput()
    if style == "block":
        return semantic_to_raw(SemanticAction.BLOCK, me.facing)
    if style == "mash":
        return RawInput(a=(state.frame % 8) < 2)
    if style == "crouch_block":
        return _crouch_block(me)
    if style == "aggressive":
        return _aggressive(state, me, other)
    if style == "defensive":
        return _defensive(state, me, other)
    if style == "rushdown":
        return _rushdown(state, me, other)
    if style == "mobile":
        return _mobile(state, me, other)
    if style == "zoner":
        return _zoner(state, me, other)
    if style == "grappler":
        return _grappler(state, me, other)
    raise ValueError(f"unknown dummy style: {style}")


def _aggressive(state: GameState, me, other) -> RawInput:
    gap = _gap(me, other)
    if airborne(other) and me.meter >= 250 and _roll(state, 100) < 25:
        return _special(2)
    if gap > 180 * SUBPIXEL and me.meter >= 250 and me.fsm == FSM.IDLE and _roll(state, 100) < 12:
        return _special(1)
    if gap < 50 * SUBPIXEL and _roll(state, 100) < 20:
        return RawInput(b=True)
    if _roll(state, 100) < 18:
        return RawInput(a=True, down=_roll(state, 100) < 40)
    walk = _toward(me)
    if _roll(state, 100) < 8:
        return RawInput(right=walk.right, left=walk.left, a=True)
    return walk


def _incoming_projectile(state: GameState, me) -> bool:
    my_id = 1 if me is state.p1 else 2
    for proj in state.projectiles:
        if proj.owner == my_id:
            continue
        if abs(me.x - proj.x) > _FIREBALL_DODGE_RANGE:
            continue
        if proj.vx > 0 and me.x > proj.x:
            return True
        if proj.vx < 0 and me.x < proj.x:
            return True
    return False


def _dodge_fireball(state: GameState, me) -> RawInput | None:
    if airborne(me):
        return None
    if _incoming_projectile(state, me):
        return RawInput(up=True)
    return None


def _near_corner(me) -> bool:
    return me.x < _CORNER_MARGIN or me.x > STAGE_WIDTH - _CORNER_MARGIN


def _guard_reactions(state: GameState, me, other) -> RawInput | None:
    """Shared defensive/zoner answers. Jump close fireballs; throw/heavy/pulse only in range."""
    dodged = _dodge_fireball(state, me)
    if dodged is not None:
        return dodged
    gap = _gap(me, other)
    if gap <= THROW_RANGE + 8 * SUBPIXEL:
        if _roll(state, 100) < 55:
            return RawInput(c=True)
        return RawInput(b=True)
    if gap <= _HEAVY_RANGE:
        return RawInput(b=True)
    if gap <= _PULSE_RANGE and me.meter >= PULSE_COST:
        return _special(3)
    return None


def _defensive(state: GameState, me, other) -> RawInput:
    reacted = _guard_reactions(state, me, other)
    if reacted is not None:
        return reacted
    # Hold ground: idle or crouch-block. Never walk back by default.
    if _roll(state, 100) < 55:
        return _crouch_block(me)
    return RawInput()


def _rushdown(state: GameState, me, other) -> RawInput:
    gap = _gap(me, other)
    if gap < 70 * SUBPIXEL and me.meter >= 250 and _roll(state, 100) < 10:
        return _special(2)
    if gap > 90 * SUBPIXEL:
        t = _toward(me)
        if _roll(state, 100) < 30:
            # Attempt 66: hold forward; actual dash needs a tap. Alternate frames help a little.
            if state.frame % 14 in (0, 2):
                return t
            if state.frame % 14 == 1:
                return RawInput()
        return t
    r = _roll(state, 100)
    if r < 28:
        return RawInput(a=True)
    if r < 48:
        return RawInput(a=True, down=True)
    if r < 62:
        return RawInput(b=True)
    return _toward(me)


def _mobile(state: GameState, me, other) -> RawInput:
    gap = _gap(me, other)
    if airborne(me):
        if _roll(state, 100) < 55:
            return RawInput(a=True)
        return RawInput()
    if gap < 70 * SUBPIXEL and me.meter >= 250 and _roll(state, 100) < 12:
        return _special(2)
    if gap > 200 * SUBPIXEL and me.meter >= 250 and _roll(state, 100) < 18:
        return _special(1)
    r = _roll(state, 100)
    if r < 14:
        return RawInput(up=True, right=_toward(me).right, left=_toward(me).left)
    if r < 24:
        return _away(me)
    if r < 36:
        return _toward(me)
    if r < 50:
        return RawInput(a=True)
    if r < 58:
        return RawInput(up=True)
    return RawInput() if r < 70 else _toward(me)


def _zoner(state: GameState, me, other) -> RawInput:
    reacted = _guard_reactions(state, me, other)
    if reacted is not None:
        return reacted
    gap = _gap(me, other)
    # Pinned in the corner: cannot hold the mid-screen gap. Crouch-block.
    if _near_corner(me) and gap < _IDEAL_ZONER_GAP:
        return _crouch_block(me)
    if gap > _IDEAL_ZONER_GAP + _IDEAL_BAND:
        return _toward(me)
    if gap < _IDEAL_ZONER_GAP - _IDEAL_BAND:
        return _away(me)
    return RawInput()


def _grappler(state: GameState, me, other) -> RawInput:
    dodged = _dodge_fireball(state, me)
    if dodged is not None:
        return dodged
    gap = _gap(me, other)
    if gap <= THROW_RANGE + 4 * SUBPIXEL and _roll(state, 100) < 45:
        return RawInput(c=True)
    if gap < 60 * SUBPIXEL and me.meter >= 150 and _roll(state, 100) < 25:
        return _special(3)
    if gap < 80 * SUBPIXEL and _roll(state, 100) < 18:
        return RawInput(b=True)
    return _toward(me)
