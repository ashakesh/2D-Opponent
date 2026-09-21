"""Hit / block / throw / projectile resolution."""

from __future__ import annotations

from fightsim.collision import (
    aabb_overlap,
    any_overlap,
    hitbox_from_local,
    hurtboxes,
    projectile_box,
)
from fightsim.config import (
    AIR_BLOWBACK_X,
    AIR_BLOWBACK_Y,
    CHIP_DEN,
    CHIP_NUM,
    METER_GAIN_BLOCK_ATTACKER,
    METER_GAIN_BLOCK_DEFENDER,
    METER_MAX,
    SUBPIXEL,
)
from fightsim.events import CombatEvent
from fightsim.moves import (
    MOVES,
    NORMALS,
    PROJ_BLOCKSTUN,
    PROJ_DAMAGE,
    PROJ_HITSTUN,
    MoveDef,
)
from fightsim.physics import airborne
from fightsim.state import FSM, FighterState, GameState, Height, Projectile


def _clamp_meter(f: FighterState) -> None:
    if f.meter < 0:
        f.meter = 0
    if f.meter > METER_MAX:
        f.meter = METER_MAX


def _blocking(f: FighterState, holding_back: bool, crouching: bool, height: str) -> bool:
    if not holding_back:
        return False
    if airborne(f):
        return False
    if f.fsm in (FSM.ATTACK, FSM.DASH, FSM.HITSTUN, FSM.KNOCKDOWN, FSM.DEAD):
        return False
    if f.fsm == FSM.BLOCKSTUN:
        pass
    elif f.fsm not in (FSM.IDLE, FSM.WALK, FSM.CROUCH, FSM.BLOCKSTUN):
        return False
    if height == Height.THROW.value:
        return False
    if height == Height.LOW.value:
        return crouching or f.fsm == FSM.CROUCH
    if height == Height.OVERHEAD.value:
        return (not crouching) and f.fsm != FSM.CROUCH
    return True


def attack_box(f: FighterState, *, spent_ok: bool = False) -> tuple[int, int, int, int] | None:
    if f.fsm != FSM.ATTACK or not f.move_id:
        return None
    move = MOVES.get(f.move_id)
    if move is None or move.hitbox is None:
        return None
    if not move.is_active(f.state_frame):
        return None
    if f.move_has_hit and not spent_ok:
        return None
    hb = move.hitbox
    return hitbox_from_local(f, hb.offset_x, hb.offset_y, hb.w, hb.h)


def current_hitbox(f: FighterState) -> tuple[int, int, int, int] | None:
    return attack_box(f, spent_ok=False)


def attack_boxes(f: FighterState, *, spent_ok: bool = False) -> list[tuple[int, int, int, int]]:
    box = attack_box(f, spent_ok=spent_ok)
    return [] if box is None else [box]


def throw_connects(attacker: FighterState, defender: FighterState, move: MoveDef) -> bool:
    if not move.is_throw:
        return False
    if not move.is_active(attacker.state_frame):
        return False
    if attacker.move_has_hit:
        return False
    if airborne(attacker) or airborne(defender):
        return False
    if defender.invuln > 0:
        return False
    if defender.fsm in (FSM.DEAD, FSM.KNOCKDOWN):
        return False
    dist = abs(attacker.x - defender.x)
    return dist <= move.throw_range


def apply_hit(
    state: GameState,
    attacker: FighterState,
    defender: FighterState,
    attacker_id: int,
    defender_id: int,
    move: MoveDef,
    blocked: bool,
    is_throw: bool,
    events: list[CombatEvent],
    damage_override: int | None = None,
    hitstun_override: int | None = None,
    blockstun_override: int | None = None,
    hitstop_override: int | None = None,
    knockback_x_override: int | None = None,
    knockback_y_override: int | None = None,
    height: str | None = None,
) -> None:
    dmg = move.damage if damage_override is None else damage_override
    hs = move.hitstun if hitstun_override is None else hitstun_override
    bs = move.blockstun if blockstun_override is None else blockstun_override
    stop = move.hitstop if hitstop_override is None else hitstop_override
    kbx = move.knockback_x if knockback_x_override is None else knockback_x_override
    kby = move.knockback_y if knockback_y_override is None else knockback_y_override
    hgt = move.height if height is None else height
    is_pulse = move.move_id == "pulse"

    attacker.move_has_hit = True
    if move.move_id in NORMALS:
        attacker.can_special_cancel = True

    face = attacker.facing
    was_air = airborne(defender)

    if blocked and not is_throw:
        defender.fsm = FSM.BLOCKSTUN
        defender.blockstun = bs
        defender.hitstun = 0
        defender.move_id = ""
        defender.can_special_cancel = False
        if defender.combo_received > 0:
            events.append(
                CombatEvent(
                    kind="combo_ended",
                    attacker=attacker_id,
                    defender=defender_id,
                    data={"length": defender.combo_received},
                )
            )
        defender.combo_received = 0
        defender.vx = kbx * face // 2 if kbx else 2 * SUBPIXEL * face
        defender.vy = 0
        chip = 0
        if move.move_id == "fireball_proj" and dmg > 0:
            chip = dmg * CHIP_NUM // CHIP_DEN
            defender.health -= chip
            if defender.health < 0:
                defender.health = 0
        attacker.meter += METER_GAIN_BLOCK_ATTACKER
        defender.meter += METER_GAIN_BLOCK_DEFENDER
        events.append(
            CombatEvent(
                kind="block",
                attacker=attacker_id,
                defender=defender_id,
                move_id=move.move_id,
                data={"height": hgt, "chip": chip},
            )
        )
    else:
        dmg = dmg if not is_throw or dmg else move.damage
        defender.health -= dmg
        if defender.health < 0:
            defender.health = 0
        if defender.fsm == FSM.HITSTUN:
            defender.combo_received += 1
        else:
            defender.combo_received = 1
            defender.bounce_count = 0
        defender.fsm = FSM.HITSTUN
        defender.hitstun = hs if not (is_throw or move.knockdown) else max(hs, 12)
        defender.blockstun = 0
        defender.move_id = ""
        defender.can_special_cancel = False
        defender.landing_lock = 0
        launch = (not is_pulse) and (
            is_throw or move.knockdown or was_air or move.move_id == "jump_light"
        )
        defender.knockdown_pending = bool(launch)
        if launch and (was_air or move.move_id == "jump_light"):
            kbx = max(kbx, AIR_BLOWBACK_X)
            if was_air:
                kby = max(kby, AIR_BLOWBACK_Y)
        if is_pulse:
            kby = 0
        defender.vx = kbx * face
        defender.vy = kby
        attacker.meter += move.meter_gain
        kind = "throw" if is_throw else "hit"
        events.append(
            CombatEvent(
                kind=kind,
                attacker=attacker_id,
                defender=defender_id,
                move_id=move.move_id,
                data={
                    "damage": dmg,
                    "combo": defender.combo_received,
                    "height": hgt,
                },
            )
        )
        if defender.knockdown_pending:
            events.append(
                CombatEvent(
                    kind="knockdown",
                    attacker=attacker_id,
                    defender=defender_id,
                    move_id=move.move_id,
                )
            )
    _clamp_meter(attacker)
    _clamp_meter(defender)
    if stop > state.hitstop:
        state.hitstop = stop


def detect_and_apply(
    state: GameState, block1: bool, crouch1: bool, block2: bool, crouch2: bool, events: list[CombatEvent]
) -> None:
    p1, p2 = state.p1, state.p2
    m1 = MOVES.get(p1.move_id) if p1.move_id else None
    m2 = MOVES.get(p2.move_id) if p2.move_id else None

    p1_throw = bool(m1 and throw_connects(p1, p2, m1))
    p2_throw = bool(m2 and throw_connects(p2, p1, m2))
    if p1_throw and p2_throw:
        p1.move_has_hit = True
        p2.move_has_hit = True
        events.append(CombatEvent(kind="throw_clash", attacker=0, defender=0, move_id="throw"))
    elif p1_throw and m1:
        apply_hit(state, p1, p2, 1, 2, m1, blocked=False, is_throw=True, events=events)
    elif p2_throw and m2:
        apply_hit(state, p2, p1, 2, 1, m2, blocked=False, is_throw=True, events=events)

    p1_hit = attack_boxes(p1)
    p2_hurt = hurtboxes(p2)
    if p1_hit and m1 and p2.invuln <= 0 and p2.fsm != FSM.DEAD:
        if any_overlap(p1_hit, p2_hurt):
            blocked = _blocking(p2, block2, crouch2, m1.height)
            apply_hit(state, p1, p2, 1, 2, m1, blocked=blocked, is_throw=False, events=events)

    p2_hit = attack_boxes(p2)
    p1_hurt = hurtboxes(p1)
    if p2_hit and m2 and p1.invuln <= 0 and p1.fsm != FSM.DEAD:
        if any_overlap(p2_hit, p1_hurt):
            blocked = _blocking(p1, block1, crouch1, m2.height)
            apply_hit(state, p2, p1, 2, 1, m2, blocked=blocked, is_throw=False, events=events)

    _resolve_projectiles(state, block1, crouch1, block2, crouch2, events)


def _pulse_erases(fighter: FighterState, pid: int, proj: Projectile, events: list[CombatEvent]) -> bool:
    if fighter.move_id != "pulse":
        return False
    if proj.owner == pid:
        return False
    boxes = attack_boxes(fighter, spent_ok=True)
    if not boxes:
        return False
    if not any_overlap(boxes, [projectile_box(proj)]):
        return False
    events.append(
        CombatEvent(
            kind="projectile_erased",
            attacker=pid,
            defender=proj.owner,
            move_id="pulse",
        )
    )
    return True


def _resolve_projectiles(
    state: GameState,
    block1: bool,
    crouch1: bool,
    block2: bool,
    crouch2: bool,
    events: list[CombatEvent],
) -> None:
    alive: list[Projectile] = []
    for i, p in enumerate(state.projectiles):
        p.x += p.vx
        p.y += p.vy
        p.lifetime -= 1
        if p.lifetime <= 0:
            continue
        if _pulse_erases(state.p1, 1, p, events) or _pulse_erases(state.p2, 2, p, events):
            continue
        pb = projectile_box(p)
        skip = False
        for j, q in enumerate(state.projectiles):
            if j <= i or q.owner == p.owner:
                continue
            if q.lifetime <= 0:
                continue
            if aabb_overlap(pb, projectile_box(q)):
                q.lifetime = 0
                skip = True
                events.append(CombatEvent(kind="projectile_clash"))
                break
        if skip:
            continue
        defender = state.p2 if p.owner == 1 else state.p1
        defender_id = 2 if p.owner == 1 else 1
        attacker = state.p1 if p.owner == 1 else state.p2
        if defender.fsm == FSM.DEAD or defender.invuln > 0:
            alive.append(p)
            continue
        if any_overlap([pb], hurtboxes(defender)):
            holding = block2 if p.owner == 1 else block1
            crouch = crouch2 if p.owner == 1 else crouch1
            blocked = _blocking(defender, holding, crouch, p.height)
            fake = MoveDef(
                move_id="fireball_proj",
                startup=0,
                active=1,
                recovery=0,
                damage=p.damage,
                hitstun=p.hitstun,
                blockstun=p.blockstun,
                hitstop=p.hitstop,
                knockback_x=p.knockback_x,
                knockback_y=p.knockback_y,
                hitbox=None,
                height=p.height,
                meter_gain=80,
                cancelable=False,
            )
            apply_hit(
                state,
                attacker,
                defender,
                p.owner,
                defender_id,
                fake,
                blocked=blocked,
                is_throw=False,
                events=events,
                damage_override=p.damage if p.damage else PROJ_DAMAGE,
                hitstun_override=p.hitstun or PROJ_HITSTUN,
                blockstun_override=p.blockstun or PROJ_BLOCKSTUN,
            )
            events.append(
                CombatEvent(
                    kind="projectile_hit",
                    attacker=p.owner,
                    defender=defender_id,
                    move_id="fireball",
                    data={"blocked": blocked},
                )
            )
            continue
        alive.append(p)
    state.projectiles = alive
