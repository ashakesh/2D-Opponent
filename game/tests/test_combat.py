from fightsim.api import Simulator
from fightsim.config import (
    CHIP_DEN,
    CHIP_NUM,
    CROUCH_PUSH_W,
    HEALTH_MAX,
    KNOCKDOWN_PUSH_W,
    METER_START,
    SUBPIXEL,
)
from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.moves import CROUCH_LIGHT, DP, PULSE, PROJ_DAMAGE, STAND_HEAVY, STAND_LIGHT, THROW
from fightsim.state import FSM, RoundState


def _close(sim: Simulator, gap_px: int = 36) -> None:
    st = sim.get_state()
    mid = 400 * SUBPIXEL
    st.p1.x = mid - (gap_px * SUBPIXEL) // 2
    st.p2.x = mid + (gap_px * SUBPIXEL) // 2
    st.p1.facing = 1
    st.p2.facing = -1
    st.p1.y = 0
    st.p2.y = 0


def _wait(sim: Simulator, n: int, p1: RawInput | None = None, p2: RawInput | None = None) -> None:
    idle = RawInput()
    a = p1 if p1 is not None else idle
    b = p2 if p2 is not None else idle
    for _ in range(n):
        sim.step(a, b)


def test_light_deals_exact_damage():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 36)
    sim.step(RawInput(a=True), RawInput())
    _wait(sim, 20)
    st = sim.get_state()
    assert st.p2.health == HEALTH_MAX - STAND_LIGHT.damage
    assert st.p1.health == HEALTH_MAX


def test_holding_back_blocks_crouch_heavy():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 36)
    block = semantic_to_raw(SemanticAction.BLOCK, -1)
    p2 = RawInput(left=block.left, right=block.right, down=True)
    saw_blockstun = False
    sim.step(RawInput(b=True), p2)
    for _ in range(24):
        st = sim.get_state()
        if st.p2.fsm == FSM.BLOCKSTUN:
            saw_blockstun = True
        sim.step(RawInput(), p2)
    st = sim.get_state()
    assert st.p2.health == HEALTH_MAX
    assert st.p2.fsm == FSM.CROUCH
    assert saw_blockstun is False


def test_throw_beats_standing_block():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 28)
    block = semantic_to_raw(SemanticAction.BLOCK, -1)
    p2 = RawInput(left=block.left, right=block.right, down=True)
    sim.step(RawInput(c=True), p2)
    _wait(sim, 12, p2=p2)
    st = sim.get_state()
    assert st.p2.health == HEALTH_MAX - THROW.damage


def test_command_fireball_spawns_projectile():
    sim = Simulator()
    sim.reset(0)
    sim.step(RawInput(command_special_1=True), RawInput())
    found = False
    for _ in range(28):
        sim.step(RawInput(), RawInput())
        if sim.get_state().projectiles:
            found = True
            break
    assert found
    proj = sim.get_state().projectiles[0]
    assert proj.owner == 1
    assert proj.vx > 0


def test_fireball_can_hit():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 200 * SUBPIXEL
    st.p2.x = 420 * SUBPIXEL
    st.p1.facing = 1
    st.p2.facing = -1
    sim.step(RawInput(command_special_1=True), RawInput())
    hit = False
    for _ in range(80):
        sim.step(RawInput(), RawInput())
        if sim.get_state().p2.health < HEALTH_MAX:
            hit = True
            break
    assert hit
    assert sim.get_state().p2.health == HEALTH_MAX - PROJ_DAMAGE


def test_point_blank_dp_beats_fireball():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 40)
    st = sim.get_state()
    st.p1.meter = 1000
    st.p2.meter = 1000
    sim.step(RawInput(command_special_1=True), RawInput(command_special_2=True))
    _wait(sim, 30)
    st = sim.get_state()
    assert st.p1.health == HEALTH_MAX - DP.damage
    assert st.p2.health == HEALTH_MAX


def test_throw_launches_backward():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 28)
    x0 = sim.get_state().p2.x
    sim.step(RawInput(c=True), RawInput())
    _wait(sim, 25)
    st = sim.get_state()
    assert st.p2.health == HEALTH_MAX - THROW.damage
    assert st.p2.x > x0 + 20 * SUBPIXEL


def test_crouch_kick_outranges_standing_light():
    gap = 88
    sim = Simulator()
    sim.reset(0)
    _close(sim, gap)
    sim.step(RawInput(a=True), RawInput())
    _wait(sim, 16)
    assert sim.get_state().p2.health == HEALTH_MAX

    sim = Simulator()
    sim.reset(0)
    _close(sim, gap)
    sim.step(RawInput(a=True, down=True), RawInput())
    _wait(sim, 20)
    assert sim.get_state().p2.health == HEALTH_MAX - CROUCH_LIGHT.damage


def test_dp_deals_more_than_heavy():
    assert DP.damage > 800
    assert DP.damage > STAND_LIGHT.damage * 3


def test_air_hit_knockdown():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 380 * SUBPIXEL
    st.p2.x = 410 * SUBPIXEL
    st.p1.facing = 1
    st.p2.facing = -1
    st.p2.y = 50 * SUBPIXEL
    st.p2.fsm = FSM.JUMP
    sim.step(RawInput(command_special_2=True), RawInput())
    knocked = False
    for _ in range(30):
        sim.step(RawInput(), RawInput())
        p2 = sim.get_state().p2
        if p2.knockdown_pending or p2.fsm == FSM.KNOCKDOWN:
            knocked = True
            break
    assert knocked
    assert sim.get_state().p2.health < HEALTH_MAX


def test_knockdown_hurtbox_wider_than_crouch():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 28)
    sim.step(RawInput(c=True), RawInput())
    wide = False
    for _ in range(80):
        sim.step(RawInput(), RawInput())
        p2 = sim.get_state().p2
        if p2.fsm == FSM.KNOCKDOWN:
            assert p2.push_w == KNOCKDOWN_PUSH_W
            assert p2.push_w > CROUCH_PUSH_W
            wide = True
            break
    assert wide


def test_throw_can_bounce():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 28)
    sim.step(RawInput(c=True), RawInput())
    saw_bounce = False
    hittable = False
    for _ in range(50):
        sim.step(RawInput(), RawInput())
        st = sim.get_state()
        if any(e.kind == "bounce" for e in st.last_events):
            saw_bounce = True
        if st.p2.fsm == FSM.HITSTUN and st.p2.y > 0 and st.p2.bounce_count >= 1:
            hittable = True
    assert saw_bounce
    assert hittable


def test_light_into_heavy_gatling():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 36)
    sim.step(RawInput(a=True), RawInput())
    gattled = False
    for _ in range(20):
        st = sim.get_state()
        if st.p1.can_special_cancel and st.p1.move_id in ("stand_light",):
            sim.step(RawInput(b=True), RawInput())
            if sim.get_state().p1.move_id == "stand_heavy":
                gattled = True
                break
        else:
            sim.step(RawInput(), RawInput())
    assert gattled


def test_throw_on_hit_recovers_while_victim_still_launched():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 28)
    sim.step(RawInput(c=True), RawInput())
    recovered_while_launched = False
    for _ in range(50):
        sim.step(RawInput(), RawInput())
        st = sim.get_state()
        p1_free = st.p1.fsm in (FSM.IDLE, FSM.WALK, FSM.CROUCH) and not st.p1.move_id
        p2_up = st.p2.fsm == FSM.HITSTUN or st.p2.y > 0
        if p1_free and p2_up:
            recovered_while_launched = True
            break
    assert recovered_while_launched


def test_pulse_pushes_with_little_damage():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 40)
    x0 = sim.get_state().p2.x
    sim.get_state().p1.meter = 1000
    sim.step(RawInput(command_special_3=True), RawInput())
    _wait(sim, 16)
    st = sim.get_state()
    assert st.p2.health == HEALTH_MAX - PULSE.damage
    assert st.p2.x > x0 + 8 * SUBPIXEL
    assert st.p2.y < 20 * SUBPIXEL


def test_pulse_erases_incoming_fireball():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 280 * SUBPIXEL
    st.p2.x = 520 * SUBPIXEL
    st.p1.facing = 1
    st.p2.facing = -1
    st.p1.meter = 1000
    st.p2.meter = 1000
    sim.step(RawInput(), RawInput(command_special_1=True))
    erased = False
    for _ in range(90):
        st = sim.get_state()
        if st.projectiles and abs(st.projectiles[0].x - st.p1.x) < 80 * SUBPIXEL:
            if st.p1.move_id != "pulse":
                sim.step(RawInput(command_special_3=True), RawInput())
            else:
                sim.step(RawInput(), RawInput())
        else:
            sim.step(RawInput(), RawInput())
        st = sim.get_state()
        if any(e.kind == "projectile_erased" for e in st.last_events):
            erased = True
            break
        if st.p1.health < HEALTH_MAX:
            break
    assert erased
    assert sim.get_state().p1.health == HEALTH_MAX
    assert not sim.get_state().projectiles


def test_blocked_fireball_chips_blocked_jab_does_not():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 32)
    block = semantic_to_raw(SemanticAction.BLOCK, -1)
    p2 = RawInput(left=block.left, right=block.right, down=True)
    sim.step(RawInput(a=True), p2)
    _wait(sim, 18, p2=p2)
    assert sim.get_state().p2.health == HEALTH_MAX

    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 200 * SUBPIXEL
    st.p2.x = 500 * SUBPIXEL
    st.p1.facing = 1
    st.p2.facing = -1
    st.p1.meter = 1000
    block = semantic_to_raw(SemanticAction.BLOCK, -1)
    p2 = RawInput(left=block.left, right=block.right, down=True)
    sim.step(RawInput(command_special_1=True), p2)
    hit = False
    for _ in range(90):
        sim.step(RawInput(), p2)
        if sim.get_state().p2.health < HEALTH_MAX:
            hit = True
            break
    assert hit
    chip = PROJ_DAMAGE * CHIP_NUM // CHIP_DEN
    assert sim.get_state().p2.health == HEALTH_MAX - chip


def test_defender_gains_more_meter_on_block():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 32)
    sim.get_state().p1.meter = METER_START
    sim.get_state().p2.meter = METER_START
    block = semantic_to_raw(SemanticAction.BLOCK, -1)
    p2 = RawInput(left=block.left, right=block.right, down=True)
    sim.step(RawInput(b=True), p2)
    _wait(sim, 24, p2=p2)
    st = sim.get_state()
    assert st.p2.meter > st.p1.meter
    assert st.p2.meter > METER_START
    assert st.p1.meter == METER_START


def test_stand_light_misses_crouch_heavy_still_hits():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 36)
    sim.step(RawInput(a=True), RawInput(down=True))
    _wait(sim, 20, p2=RawInput(down=True))
    assert sim.get_state().p2.health == HEALTH_MAX

    sim = Simulator()
    sim.reset(0)
    _close(sim, 36)
    sim.step(RawInput(b=True), RawInput(down=True))
    _wait(sim, 24, p2=RawInput(down=True))
    assert sim.get_state().p2.health == HEALTH_MAX - STAND_HEAVY.damage


def test_fireball_hits_crouch():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 200 * SUBPIXEL
    st.p2.x = 420 * SUBPIXEL
    st.p1.facing = 1
    st.p2.facing = -1
    st.p1.meter = 1000
    sim.step(RawInput(command_special_1=True), RawInput(down=True))
    hit = False
    for _ in range(90):
        sim.step(RawInput(), RawInput(down=True))
        if sim.get_state().p2.health < HEALTH_MAX:
            hit = True
            break
    assert hit


def test_blocked_heavy_pushes_attacker_farther_than_light():
    from fightsim.combat import block_push_speeds
    from fightsim.moves import STAND_HEAVY

    light_atk, _ = block_push_speeds(STAND_LIGHT, STAND_LIGHT.damage)
    heavy_atk, _ = block_push_speeds(STAND_HEAVY, STAND_HEAVY.damage)
    pulse_atk, pulse_dfn = block_push_speeds(PULSE, PULSE.damage)
    assert heavy_atk > light_atk
    assert heavy_atk < pulse_dfn
    assert light_atk < pulse_atk

    def attacker_slide(button: RawInput, wait: int) -> int:
        sim = Simulator()
        sim.reset(0)
        _close(sim, 36)
        if button.command_special_3:
            sim.get_state().p1.meter = 1000
        x0 = sim.get_state().p1.x
        block = semantic_to_raw(SemanticAction.BLOCK, -1)
        p2 = RawInput(left=block.left, right=block.right, down=True)
        sim.step(button, p2)
        _wait(sim, wait, p2=p2)
        return x0 - sim.get_state().p1.x

    light_slide = attacker_slide(RawInput(a=True), 16)
    heavy_slide = attacker_slide(RawInput(b=True), 20)
    pulse_slide = attacker_slide(RawInput(command_special_3=True), 16)
    # Standing light misses crouch, so no attacker push.
    assert light_slide < heavy_slide
    assert heavy_slide < pulse_slide


def test_hop_allows_jab_but_not_instant_rejump():
    sim = Simulator()
    sim.reset(0)
    sim.step(RawInput(up=True), RawInput())
    for _ in range(80):
        sim.step(RawInput(), RawInput())
        st = sim.get_state()
        if st.p1.y == 0 and st.p1.fsm != FSM.JUMP:
            break
    st = sim.get_state()
    assert st.p1.y == 0
    assert st.p1.jump_lock > 0
    sim.step(RawInput(up=True), RawInput())
    assert sim.get_state().p1.y == 0
    assert sim.get_state().p1.fsm != FSM.JUMP
    sim.step(RawInput(a=True), RawInput())
    assert sim.get_state().p1.move_id == "stand_light"


def test_whiff_jump_light_recovers_on_ground():
    from fightsim.animation import resolve_clip
    from fightsim.moves import JUMP_LIGHT

    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 200 * SUBPIXEL
    st.p2.x = 700 * SUBPIXEL
    st.p1.y = 6 * SUBPIXEL
    st.p1.vy = -8 * SUBPIXEL
    st.p1.fsm = FSM.ATTACK
    st.p1.move_id = "jump_light"
    st.p1.state_frame = JUMP_LIGHT.startup + JUMP_LIGHT.active
    st.p1.move_has_hit = False
    sim.step(RawInput(), RawInput())
    st = sim.get_state()
    assert st.p1.y == 0
    assert st.p1.fsm == FSM.ATTACK
    assert st.p1.move_id == "jump_light"
    assert resolve_clip(st.p1).clip == "jump_light_land_recovery"


def test_ko_keeps_launch_and_settles_dead_on_ground():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 28)
    sim.get_state().p2.health = 100
    sim.step(RawInput(c=True), RawInput())
    ended = 0
    launched = False
    for _ in range(80):
        st = sim.get_state()
        ended += sum(1 for e in st.last_events if e.kind == "round_end")
        if st.p2.fsm == FSM.DEAD and (st.p2.y > 0 or st.p2.vx != 0 or st.p2.vy != 0):
            launched = True
        sim.step(RawInput(a=True), RawInput(a=True))
    st = sim.get_state()
    assert st.round_state == RoundState.P1_WIN
    assert launched
    assert st.p2.fsm == FSM.DEAD
    assert st.p2.y == 0
    assert st.p2.vx == 0
    assert st.p2.vy == 0
    assert st.p1.fsm == FSM.IDLE
    assert st.p1.y == 0
    assert ended == 1


def test_grounded_winner_cancels_to_idle_after_round():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 36)
    sim.get_state().p2.health = 1
    sim.step(RawInput(a=True), RawInput())
    for _ in range(20):
        st = sim.get_state()
        if st.round_state != RoundState.FIGHTING:
            break
        sim.step(RawInput(), RawInput())
    st = sim.get_state()
    assert st.round_state == RoundState.P1_WIN
    assert st.p2.fsm == FSM.DEAD
    sim.step(RawInput(a=True), RawInput(b=True))
    st = sim.get_state()
    assert st.p1.fsm == FSM.IDLE
    assert st.p1.move_id == ""
    assert st.p2.fsm == FSM.DEAD


def test_airborne_winner_lands_idle_after_ko():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 40)
    st = sim.get_state()
    st.p2.health = 1
    st.p1.y = 24 * SUBPIXEL
    st.p1.vy = 4 * SUBPIXEL
    st.p1.fsm = FSM.JUMP
    sim.step(RawInput(a=True), RawInput())
    for _ in range(90):
        sim.step(RawInput(up=True, a=True), RawInput())
        st = sim.get_state()
        if st.round_state != RoundState.FIGHTING and st.p1.y == 0 and st.p2.y == 0:
            break
    st = sim.get_state()
    assert st.round_state == RoundState.P1_WIN
    assert st.p1.fsm == FSM.IDLE
    assert st.p1.y == 0
    assert st.p2.fsm == FSM.DEAD
    assert st.p2.y == 0

