"""Gravity, integration, walls, pushbox separation, knockdown bounce."""

from __future__ import annotations

from fightsim.collision import overlap_x, pushbox
from fightsim.config import (
    AIR_PUSH_H,
    AIR_PUSH_W,
    BOUNCE_HITSTUN,
    BOUNCE_MIN_VY,
    BOUNCE_VX_DEN,
    BOUNCE_VX_NUM,
    BOUNCE_VY_DEN,
    BOUNCE_VY_NUM,
    CROUCH_PUSH_H,
    CROUCH_PUSH_W,
    GRAVITY_DOWN,
    GRAVITY_UP,
    HITSTUN_GRAVITY_DOWN,
    JUMP_COOLDOWN,
    KNOCKDOWN_PUSH_H,
    KNOCKDOWN_PUSH_W,
    LANDING_RECOVERY,
    MAX_BOUNCES,
    STAGE_WIDTH,
    STAND_PUSH_H,
    STAND_PUSH_W,
    SUBPIXEL,
)
from fightsim.state import FSM, FighterState, GameState


def airborne(f: FighterState) -> bool:
    return f.y > 0


def apply_gravity_and_integrate(f: FighterState) -> None:
    if f.y > 0 or f.vy > 0:
        if f.vy > 0:
            g = GRAVITY_UP
        elif f.fsm in (FSM.HITSTUN, FSM.DEAD):
            g = HITSTUN_GRAVITY_DOWN
        else:
            g = GRAVITY_DOWN
        f.vy -= g
        f.y += f.vy
    else:
        f.y = 0
        if f.vy < 0:
            f.vy = 0
    if f.y <= 0:
        f.y = 0
        if f.vy < 0 and f.fsm not in (FSM.HITSTUN, FSM.DEAD):
            f.vy = 0
    f.x += f.vx


def _launched_bounce(f: FighterState) -> bool:
    return f.fsm in (FSM.HITSTUN, FSM.DEAD) and f.bounce_count < MAX_BOUNCES


def clamp_to_stage(f: FighterState) -> str | None:
    """Keep in bounds. One wall bounce while launched (shares bounce_count with floor)."""
    hw = f.push_w // 2
    bounced = None
    if f.x < hw:
        f.x = hw
        if _launched_bounce(f) and f.vx < 0:
            f.vx = -f.vx * BOUNCE_VX_NUM // BOUNCE_VX_DEN
            f.bounce_count += 1
            bounced = "wall"
        elif f.vx < 0:
            f.vx = 0
    max_x = STAGE_WIDTH - hw
    if f.x > max_x:
        f.x = max_x
        if _launched_bounce(f) and f.vx > 0:
            f.vx = -f.vx * BOUNCE_VX_NUM // BOUNCE_VX_DEN
            f.bounce_count += 1
            bounced = "wall"
        elif f.vx > 0:
            f.vx = 0
    return bounced


def update_body_size(f: FighterState) -> None:
    if airborne(f):
        f.push_w, f.push_h = AIR_PUSH_W, AIR_PUSH_H
    elif f.fsm == FSM.KNOCKDOWN:
        f.push_w, f.push_h = KNOCKDOWN_PUSH_W, KNOCKDOWN_PUSH_H
    elif f.fsm == FSM.CROUCH or f.move_id == "crouch_light":
        f.push_w, f.push_h = CROUCH_PUSH_W, CROUCH_PUSH_H
    else:
        f.push_w, f.push_h = STAND_PUSH_W, STAND_PUSH_H


def resolve_push(state: GameState) -> None:
    a, b = state.p1, state.p2
    pa, pb = pushbox(a), pushbox(b)
    ov = overlap_x(pa, pb)
    if ov <= 0:
        return
    left, right = (a, b) if a.x <= b.x else (b, a)
    half = ov // 2
    rest = ov - half
    left.x -= half
    right.x += rest
    clamp_to_stage(a)
    clamp_to_stage(b)
    pa, pb = pushbox(a), pushbox(b)
    ov = overlap_x(pa, pb)
    if ov > 0:
        if a.x <= b.x:
            if a.x <= a.push_w // 2:
                b.x += ov
            elif b.x >= STAGE_WIDTH - b.push_w // 2:
                a.x -= ov
            else:
                a.x -= ov // 2
                b.x += ov - ov // 2
        else:
            if b.x <= b.push_w // 2:
                a.x += ov
            elif a.x >= STAGE_WIDTH - a.push_w // 2:
                b.x -= ov
            else:
                b.x -= ov // 2
                a.x += ov - ov // 2
        clamp_to_stage(a)
        clamp_to_stage(b)


def land_if_needed(f: FighterState) -> str | None:
    """Returns 'bounce' if a floor bounce happened this frame."""
    if f.y > 0:
        return None
    incoming_vy = f.vy
    # Still launching off the ground this frame — do not treat as a landing.
    if incoming_vy > 0:
        return None
    f.y = 0
    if f.fsm == FSM.JUMP:
        f.fsm = FSM.IDLE
        f.state_frame = 0
        f.vx = 0
        f.vy = 0
        f.move_id = ""
        f.jump_lock = max(f.jump_lock, JUMP_COOLDOWN)
        return None
    if f.fsm == FSM.ATTACK and f.move_id == "jump_light":
        f.y = 0
        f.vx = 0
        f.vy = 0
        f.jump_lock = max(f.jump_lock, JUMP_COOLDOWN)
        if f.move_has_hit:
            f.fsm = FSM.IDLE
            f.state_frame = 0
            f.move_id = ""
            f.can_special_cancel = False
            f.move_has_hit = False
            f.landing_lock = LANDING_RECOVERY
        return None
    if f.fsm == FSM.DEAD:
        if f.bounce_count < MAX_BOUNCES and incoming_vy <= -BOUNCE_MIN_VY:
            f.bounce_count += 1
            f.y = SUBPIXEL
            f.vy = -incoming_vy * BOUNCE_VY_NUM // BOUNCE_VY_DEN
            f.vx = f.vx * BOUNCE_VX_NUM // BOUNCE_VX_DEN
            return "bounce"
        f.vx = 0
        f.vy = 0
        return None
    if f.fsm == FSM.HITSTUN and f.knockdown_pending:
        if f.bounce_count < MAX_BOUNCES and incoming_vy <= -BOUNCE_MIN_VY:
            f.bounce_count += 1
            f.y = SUBPIXEL
            f.vy = -incoming_vy * BOUNCE_VY_NUM // BOUNCE_VY_DEN
            f.vx = f.vx * BOUNCE_VX_NUM // BOUNCE_VX_DEN
            f.hitstun = max(f.hitstun, BOUNCE_HITSTUN)
            return "bounce"
        f.fsm = FSM.KNOCKDOWN
        f.state_frame = 0
        f.vx = 0
        f.vy = 0
        f.hitstun = 0
        return None
    if f.fsm == FSM.HITSTUN and f.hitstun <= 0:
        f.fsm = FSM.IDLE
        f.vx = 0
        f.vy = 0
    elif incoming_vy < 0 and f.fsm not in (FSM.HITSTUN, FSM.DEAD):
        f.vy = 0
    return None


def settle_living_if_grounded(f: FighterState) -> None:
    """Post-round: cancel attacks/stun once a living fighter is on the floor."""
    if f.health <= 0 or f.fsm == FSM.DEAD:
        return
    if airborne(f) or f.vy > 0:
        return
    f.y = 0
    f.vx = 0
    f.vy = 0
    f.fsm = FSM.IDLE
    f.move_id = ""
    f.state_frame = 0
    f.hitstun = 0
    f.blockstun = 0
    f.can_special_cancel = False
    f.knockdown_pending = False
