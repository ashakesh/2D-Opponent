"""Pure step: (GameState, RawInput, RawInput) → (GameState, events)."""

from __future__ import annotations

import copy

from fightsim.combat import detect_and_apply
from fightsim.events import CombatEvent
from fightsim.fighter import (
    advance_stun,
    block_flags,
    buffer_input,
    end_of_frame_move_tick,
    maybe_die,
    spawn_fireball_if_needed,
    store_prev_buttons,
    try_normals_and_move,
    update_facing,
)
from fightsim.input import RawInput
from fightsim.physics import (
    apply_gravity_and_integrate,
    clamp_to_stage,
    land_if_needed,
    resolve_push,
    settle_living_if_grounded,
    update_body_size,
)
from fightsim.state import FSM, GameState, RoundState


def step_state(state: GameState, p1_in: RawInput, p2_in: RawInput) -> tuple[GameState, list[CombatEvent]]:
    s = copy.deepcopy(state)
    events: list[CombatEvent] = []
    s.frame += 1

    if s.round_state != RoundState.FIGHTING:
        _step_post_round(s, events)
        s.last_events = events
        return s, events

    buffer_input(s.p1, p1_in)
    buffer_input(s.p2, p2_in)

    if s.hitstop > 0:
        s.hitstop -= 1
        # Keep direction buffer; do not consume button edges so cancels buffer through hitstop.
        s.last_events = events
        return s, events

    advance_stun(s.p1, events, 1)
    advance_stun(s.p2, events, 2)

    try_normals_and_move(s.p1, p1_in, events, 1)
    try_normals_and_move(s.p2, p2_in, events, 2)

    _run_physics(s, events)

    b1, c1 = block_flags(s.p1, p1_in)
    b2, c2 = block_flags(s.p2, p2_in)
    detect_and_apply(s, b1, c1, b2, c2, events)

    for pid, f in ((1, s.p1), (2, s.p2)):
        proj = spawn_fireball_if_needed(f, pid, events)
        if proj is not None:
            s.projectiles.append(proj)

    for pid, f in ((1, s.p1), (2, s.p2)):
        kind = land_if_needed(f)
        if kind == "bounce":
            events.append(CombatEvent(kind="bounce", defender=pid, data={"surface": "floor"}))
    update_body_size(s.p1)
    update_body_size(s.p2)
    update_facing(s.p1, s.p2)
    update_facing(s.p2, s.p1)

    maybe_die(s.p1)
    maybe_die(s.p2)
    _resolve_round(s, events)

    end_of_frame_move_tick(s.p1, events, 1)
    end_of_frame_move_tick(s.p2, events, 2)

    if s.timer > 0:
        s.timer -= 1
        if s.timer == 0 and s.round_state == RoundState.FIGHTING:
            _timeout(s, events)

    store_prev_buttons(s.p1, p1_in)
    store_prev_buttons(s.p2, p2_in)
    s.last_events = events
    return s, events


def _dec_locks(f) -> None:
    if f.landing_lock > 0:
        f.landing_lock -= 1
    if f.jump_lock > 0:
        f.jump_lock -= 1
    if f.invuln > 0:
        f.invuln -= 1


def _run_physics(s: GameState, events: list[CombatEvent]) -> None:
    apply_gravity_and_integrate(s.p1)
    apply_gravity_and_integrate(s.p2)
    for pid, f in ((1, s.p1), (2, s.p2)):
        kind = land_if_needed(f)
        if kind == "bounce":
            events.append(CombatEvent(kind="bounce", defender=pid, data={"surface": "floor"}))
    update_body_size(s.p1)
    update_body_size(s.p2)
    for pid, f in ((1, s.p1), (2, s.p2)):
        kind = clamp_to_stage(f)
        if kind == "wall":
            events.append(CombatEvent(kind="bounce", defender=pid, data={"surface": "wall"}))
    resolve_push(s)


def _step_post_round(s: GameState, events: list[CombatEvent]) -> None:
    if s.hitstop > 0:
        s.hitstop -= 1
    _dec_locks(s.p1)
    _dec_locks(s.p2)
    _run_physics(s, events)
    for pid, f in ((1, s.p1), (2, s.p2)):
        kind = land_if_needed(f)
        if kind == "bounce":
            events.append(CombatEvent(kind="bounce", defender=pid, data={"surface": "floor"}))
    update_body_size(s.p1)
    update_body_size(s.p2)
    settle_living_if_grounded(s.p1)
    settle_living_if_grounded(s.p2)
    end_of_frame_move_tick(s.p1, events, 1)
    end_of_frame_move_tick(s.p2, events, 2)


def _resolve_round(s: GameState, events: list[CombatEvent]) -> None:
    if s.round_state != RoundState.FIGHTING:
        return
    d1 = s.p1.fsm == FSM.DEAD or s.p1.health <= 0
    d2 = s.p2.fsm == FSM.DEAD or s.p2.health <= 0
    if not d1 and not d2:
        return
    if d1 and d2:
        s.round_state = RoundState.DRAW
    elif d1:
        s.round_state = RoundState.P2_WIN
    else:
        s.round_state = RoundState.P1_WIN
    events.append(
        CombatEvent(
            kind="round_end",
            data={"result": s.round_state.value},
        )
    )


def _timeout(s: GameState, events: list[CombatEvent]) -> None:
    if s.p1.health > s.p2.health:
        s.round_state = RoundState.P1_WIN
    elif s.p2.health > s.p1.health:
        s.round_state = RoundState.P2_WIN
    else:
        s.round_state = RoundState.DRAW
    events.append(
        CombatEvent(kind="round_end", data={"result": s.round_state.value, "timeout": True})
    )
