"""Gravity, integration, walls, pushbox separation."""

from __future__ import annotations

from fightsim.collision import overlap_x, pushbox
from fightsim.config import (
    AIR_PUSH_H,
    AIR_PUSH_W,
    CROUCH_PUSH_H,
    CROUCH_PUSH_W,
    GRAVITY,
    STAGE_WIDTH,
    STAND_PUSH_H,
    STAND_PUSH_W,
)
from fightsim.state import FSM, FighterState, GameState


def airborne(f: FighterState) -> bool:
    return f.y > 0


def apply_gravity_and_integrate(f: FighterState) -> None:
    if f.y > 0 or f.vy > 0:
        f.vy -= GRAVITY
        f.y += f.vy
    else:
        f.y = 0
        if f.vy < 0:
            f.vy = 0
    if f.y <= 0:
        f.y = 0
        if f.vy < 0:
            f.vy = 0
    f.x += f.vx


def clamp_to_stage(f: FighterState) -> None:
    hw = f.push_w // 2
    if f.x < hw:
        f.x = hw
        if f.vx < 0:
            f.vx = 0
    max_x = STAGE_WIDTH - hw
    if f.x > max_x:
        f.x = max_x
        if f.vx > 0:
            f.vx = 0


def update_body_size(f: FighterState) -> None:
    if airborne(f):
        f.push_w, f.push_h = AIR_PUSH_W, AIR_PUSH_H
    elif f.fsm == FSM.CROUCH or f.move_id == "crouch_light":
        f.push_w, f.push_h = CROUCH_PUSH_W, CROUCH_PUSH_H
    elif f.fsm == FSM.KNOCKDOWN:
        f.push_w, f.push_h = CROUCH_PUSH_W, CROUCH_PUSH_H
    else:
        f.push_w, f.push_h = STAND_PUSH_W, STAND_PUSH_H


def resolve_push(state: GameState) -> None:
    a, b = state.p1, state.p2
    pa, pb = pushbox(a), pushbox(b)
    ov = overlap_x(pa, pb)
    if ov <= 0:
        return
    # Split overlap; the left fighter moves left.
    left, right = (a, b) if a.x <= b.x else (b, a)
    half = ov // 2
    rest = ov - half
    left.x -= half
    right.x += rest
    clamp_to_stage(a)
    clamp_to_stage(b)
    # If still overlapping because of a wall, shove the inner fighter.
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


def land_if_needed(f: FighterState) -> None:
    if f.y > 0:
        return
    f.y = 0
    if f.vy < 0:
        f.vy = 0
    if f.fsm == FSM.JUMP:
        f.fsm = FSM.IDLE
        f.state_frame = 0
        f.vx = 0
        f.move_id = ""
    elif f.fsm == FSM.ATTACK and f.move_id == "jump_light":
        f.fsm = FSM.IDLE
        f.state_frame = 0
        f.vx = 0
        f.move_id = ""
        f.can_special_cancel = False
        f.move_has_hit = False
    elif f.fsm == FSM.HITSTUN and f.knockdown_pending:
        f.fsm = FSM.KNOCKDOWN
        f.state_frame = 0
        f.vx = 0
        f.hitstun = 0
    elif f.fsm == FSM.HITSTUN and f.hitstun <= 0:
        f.fsm = FSM.IDLE
        f.vx = 0
