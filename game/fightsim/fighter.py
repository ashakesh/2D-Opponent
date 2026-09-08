"""Per-frame fighter updates. New moves stay on state_frame 0 until end-of-frame."""

from __future__ import annotations

from fightsim.config import (
    DASH_BACK_SPEED,
    DASH_FRAMES,
    DASH_SPEED,
    JUMP_VX,
    JUMP_VY,
    KNOCKDOWN_FRAMES,
    SPECIAL_COST,
    SUBPIXEL,
    WALK_SPEED,
    WAKEUP_INVULN,
)
from fightsim.events import CombatEvent
from fightsim.input import (
    RawInput,
    holding_back,
    motion_236,
    motion_623,
    push_buffer,
    stick_numpad,
    tap_back,
    tap_forward,
)
from fightsim.moves import (
    DP,
    FIREBALL,
    LIGHTS,
    MOVES,
    PROJ_BLOCKSTUN,
    PROJ_DAMAGE,
    PROJ_H,
    PROJ_HITSTOP,
    PROJ_HITSTUN,
    PROJ_KNOCKBACK,
    PROJ_LIFETIME,
    PROJ_SPEED,
    PROJ_W,
    PROJ_Y,
    SPECIALS,
)
from fightsim.physics import airborne
from fightsim.state import FSM, FighterState, Projectile


def buffer_input(f: FighterState, inp: RawInput) -> None:
    num = stick_numpad(inp, f.facing)
    f.input_buffer = push_buffer(f.input_buffer, num)


def store_prev_buttons(f: FighterState, inp: RawInput) -> None:
    f.prev_a = inp.a
    f.prev_b = inp.b
    f.prev_c = inp.c


def _a_press(f: FighterState, inp: RawInput) -> bool:
    return inp.a and not f.prev_a


def _b_press(f: FighterState, inp: RawInput) -> bool:
    return inp.b and not f.prev_b


def _c_press(f: FighterState, inp: RawInput) -> bool:
    return inp.c and not f.prev_c


def is_actionable(f: FighterState) -> bool:
    if f.health <= 0 or f.fsm == FSM.DEAD:
        return False
    if f.landing_lock > 0:
        return False
    if f.fsm in (FSM.HITSTUN, FSM.BLOCKSTUN, FSM.KNOCKDOWN, FSM.DEAD):
        return False
    if f.fsm == FSM.ATTACK:
        return False
    if f.fsm == FSM.DASH:
        return False
    return True


def _start_move(
    f: FighterState, move_id: str, events: list[CombatEvent], pid: int, cancel: bool
) -> None:
    move = MOVES[move_id]
    f.fsm = FSM.ATTACK
    f.move_id = move_id
    f.state_frame = 0
    f.move_has_hit = False
    f.move_spawned = False
    f.can_special_cancel = False
    f.meter -= move.meter_cost
    if f.meter < 0:
        f.meter = 0
    if move.invuln_frames > 0:
        f.invuln = max(f.invuln, move.invuln_frames)
    if not airborne(f):
        f.vx = 0
    events.append(
        CombatEvent(
            kind=(
                "special_cancel"
                if cancel and move_id in SPECIALS
                else ("gatling" if cancel else "attack_started")
            ),
            attacker=pid,
            defender=0,
            move_id=move_id,
        )
    )


def try_special(f: FighterState, inp: RawInput, events: list[CombatEvent], pid: int) -> bool:
    cancel = f.fsm == FSM.ATTACK and f.can_special_cancel
    if not (is_actionable(f) or cancel):
        return False
    if airborne(f):
        return False
    want_dp = inp.command_special_2 or (motion_623(f.input_buffer) and _a_press(f, inp))
    want_fb = inp.command_special_1 or (motion_236(f.input_buffer) and _a_press(f, inp))
    if want_dp and f.meter >= SPECIAL_COST:
        _start_move(f, DP.move_id, events, pid, cancel)
        return True
    if want_fb and f.meter >= SPECIAL_COST:
        _start_move(f, FIREBALL.move_id, events, pid, cancel)
        return True
    return False


def try_normals_and_move(
    f: FighterState, inp: RawInput, events: list[CombatEvent], pid: int
) -> None:
    if try_special(f, inp, events, pid):
        return
    if (
        f.fsm == FSM.ATTACK
        and f.can_special_cancel
        and f.move_id in LIGHTS
        and _b_press(f, inp)
        and not airborne(f)
    ):
        _start_move(f, "stand_heavy", events, pid, cancel=True)
        return
    if not is_actionable(f):
        return

    air = airborne(f)

    if not air and _c_press(f, inp):
        _start_move(f, "throw", events, pid, False)
        return
    if air and _a_press(f, inp):
        _start_move(f, "jump_light", events, pid, False)
        return
    if not air and _b_press(f, inp):
        _start_move(f, "stand_heavy", events, pid, False)
        return
    if not air and _a_press(f, inp):
        if inp.down:
            _start_move(f, "crouch_light", events, pid, False)
        else:
            _start_move(f, "stand_light", events, pid, False)
        return

    if air:
        f.fsm = FSM.JUMP
        return

    if inp.up:
        f.fsm = FSM.JUMP
        f.state_frame = 0
        f.move_id = ""
        f.vy = JUMP_VY
        hx = inp.horiz()
        if hx > 0:
            f.vx = JUMP_VX
        elif hx < 0:
            f.vx = -JUMP_VX
        else:
            f.vx = 0
        return

    if tap_forward(f.input_buffer) and not inp.down and not inp.up:
        f.fsm = FSM.DASH
        f.state_frame = 0
        f.move_id = "dash_fwd"
        f.vx = DASH_SPEED * (1 if f.facing >= 0 else -1)
        return
    if tap_back(f.input_buffer) and not inp.down and not inp.up:
        f.fsm = FSM.DASH
        f.state_frame = 0
        f.move_id = "dash_back"
        back = -1 if f.facing >= 0 else 1
        f.vx = DASH_BACK_SPEED * back
        return

    if inp.down:
        f.fsm = FSM.CROUCH
        f.vx = 0
        return

    if inp.horiz() != 0:
        f.fsm = FSM.WALK
        f.vx = WALK_SPEED * (1 if inp.horiz() > 0 else -1)
        return

    f.fsm = FSM.IDLE
    f.vx = 0
    f.move_id = ""


def _end_combo(f: FighterState, events: list[CombatEvent], pid: int) -> None:
    if f.combo_received > 0:
        events.append(
            CombatEvent(
                kind="combo_ended",
                attacker=0,
                defender=pid,
                data={"length": f.combo_received},
            )
        )
    f.combo_received = 0
    f.bounce_count = 0


def advance_stun(f: FighterState, events: list[CombatEvent], pid: int) -> None:
    if f.landing_lock > 0:
        f.landing_lock -= 1
    if f.invuln > 0:
        f.invuln -= 1

    if f.fsm == FSM.DEAD:
        f.vx = 0
        return

    if f.fsm == FSM.HITSTUN:
        if f.hitstun > 0:
            f.hitstun -= 1
        if f.hitstun <= 0 and not airborne(f):
            if f.knockdown_pending:
                f.fsm = FSM.KNOCKDOWN
                f.state_frame = 0
                f.vx = 0
            else:
                f.fsm = FSM.IDLE
                f.vx = 0
                f.knockdown_pending = False
                _end_combo(f, events, pid)
        return

    if f.fsm == FSM.BLOCKSTUN:
        if f.blockstun > 0:
            f.blockstun -= 1
        if f.blockstun <= 0:
            f.fsm = FSM.IDLE
            f.vx = 0
        return

    if f.fsm == FSM.KNOCKDOWN:
        f.vx = 0
        f.vy = 0
        f.y = 0
        f.state_frame += 1
        if f.state_frame >= KNOCKDOWN_FRAMES:
            f.fsm = FSM.IDLE
            f.state_frame = 0
            f.knockdown_pending = False
            f.invuln = max(f.invuln, WAKEUP_INVULN)
            _end_combo(f, events, pid)


def end_of_frame_move_tick(f: FighterState, events: list[CombatEvent], pid: int) -> None:
    if f.fsm == FSM.DASH:
        f.state_frame += 1
        if f.move_id == "dash_fwd":
            f.vx = DASH_SPEED * (1 if f.facing >= 0 else -1)
        elif f.move_id == "dash_back":
            back = -1 if f.facing >= 0 else 1
            f.vx = DASH_BACK_SPEED * back
        if f.state_frame >= DASH_FRAMES:
            f.fsm = FSM.IDLE
            f.state_frame = 0
            f.move_id = ""
            f.vx = 0
        return

    if f.fsm == FSM.ATTACK and f.move_id:
        move = MOVES[f.move_id]
        f.state_frame += 1
        if f.state_frame >= move.total:
            if not f.move_has_hit and f.move_id not in SPECIALS | {"fireball"}:
                events.append(
                    CombatEvent(kind="whiff", attacker=pid, defender=0, move_id=f.move_id)
                )
            f.fsm = FSM.IDLE if not airborne(f) else FSM.JUMP
            f.state_frame = 0
            f.move_id = ""
            f.can_special_cancel = False
            if not airborne(f):
                f.vx = 0


def maybe_die(f: FighterState) -> None:
    if f.health <= 0 and f.fsm != FSM.DEAD:
        f.health = 0
        f.fsm = FSM.DEAD
        f.vx = 0
        f.move_id = ""


def update_facing(f: FighterState, other: FighterState) -> None:
    if airborne(f):
        return
    if f.fsm in (FSM.ATTACK, FSM.HITSTUN, FSM.BLOCKSTUN, FSM.DASH, FSM.KNOCKDOWN, FSM.DEAD):
        return
    if other.x > f.x:
        f.facing = 1
    elif other.x < f.x:
        f.facing = -1


def block_flags(f: FighterState, inp: RawInput) -> tuple[bool, bool]:
    back = holding_back(inp, f.facing)
    crouch = bool(inp.down) or f.fsm == FSM.CROUCH
    return back, crouch


def spawn_fireball_if_needed(
    f: FighterState, pid: int, events: list[CombatEvent]
) -> Projectile | None:
    if f.fsm != FSM.ATTACK or f.move_id != FIREBALL.move_id:
        return None
    if f.move_spawned:
        return None
    if f.state_frame != FIREBALL.projectile_frame:
        return None
    f.move_spawned = True
    proj = Projectile(
        owner=pid,
        x=f.x + f.facing * (40 * SUBPIXEL),
        y=PROJ_Y,
        vx=PROJ_SPEED * f.facing,
        vy=0,
        w=PROJ_W,
        h=PROJ_H,
        damage=PROJ_DAMAGE,
        hitstun=PROJ_HITSTUN,
        blockstun=PROJ_BLOCKSTUN,
        hitstop=PROJ_HITSTOP,
        knockback_x=PROJ_KNOCKBACK,
        knockback_y=0,
        lifetime=PROJ_LIFETIME,
    )
    events.append(
        CombatEvent(kind="projectile_spawned", attacker=pid, defender=0, move_id="fireball")
    )
    return proj
