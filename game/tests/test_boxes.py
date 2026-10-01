from fightsim.api import Simulator
from fightsim.collision import any_overlap, character_box, hitbox_from_local, hurtboxes, projectile_box
from fightsim.combat import attack_boxes
from fightsim.config import CHAR_BOX_H, CHAR_BOX_W, CROUCH_PUSH_H, SUBPIXEL
from fightsim.input import RawInput
from fightsim.moves import PROJ_H, PROJ_W, PROJ_Y, STAND_HEAVY, STAND_LIGHT
from fightsim.physics import update_body_size
from fightsim.state import FSM, Projectile, initial_state


def test_character_box_size_is_fixed_stand_rect():
    s = initial_state(0)
    box = character_box(s.p1)
    assert box[2] - box[0] == CHAR_BOX_W
    assert box[3] - box[1] == CHAR_BOX_H
    assert box[1] == s.p1.y
    assert (box[0] + box[2]) // 2 == s.p1.x


def test_character_box_ignores_crouch_air_knockdown_push():
    s = initial_state(0)
    f = s.p1
    stand = character_box(f)
    f.fsm = FSM.CROUCH
    update_body_size(f)
    assert character_box(f) == stand
    f.y = 40 * SUBPIXEL
    update_body_size(f)
    air = character_box(f)
    assert air[2] - air[0] == CHAR_BOX_W
    assert air[3] - air[1] == CHAR_BOX_H
    f.y = 0
    f.fsm = FSM.KNOCKDOWN
    update_body_size(f)
    kd = character_box(f)
    assert kd[2] - kd[0] == CHAR_BOX_W
    assert kd[3] - kd[1] == CHAR_BOX_H
    assert f.push_h != CHAR_BOX_H


def test_standing_hurtboxes_include_head_above_character_box():
    s = initial_state(0)
    boxes = hurtboxes(s.p1)
    assert len(boxes) == 2
    body, head = boxes
    assert head[3] > CHAR_BOX_H
    assert head[1] >= CHAR_BOX_H
    assert head[3] > body[3]


def test_crouch_hurtboxes_include_lower_head():
    s = initial_state(0)
    f = s.p1
    stand_head_top = hurtboxes(f)[1][3]
    f.fsm = FSM.CROUCH
    update_body_size(f)
    boxes = hurtboxes(f)
    assert len(boxes) == 2
    body, head = boxes
    assert head[1] >= CROUCH_PUSH_H
    assert head[1] <= CROUCH_PUSH_H + SUBPIXEL
    assert head[3] > CROUCH_PUSH_H
    assert head[3] < stand_head_top


def test_knockdown_hurtbox_is_body_only():
    s = initial_state(0)
    f = s.p1
    f.fsm = FSM.KNOCKDOWN
    update_body_size(f)
    assert len(hurtboxes(f)) == 1


def test_idle_attack_boxes_empty():
    s = initial_state(0)
    assert attack_boxes(s.p1) == []


def test_active_light_has_one_attack_box():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 400 * SUBPIXEL
    st.p2.x = 700 * SUBPIXEL
    sim.step(RawInput(a=True), RawInput())
    found = False
    for _ in range(8):
        boxes = attack_boxes(sim.get_state().p1)
        if boxes:
            found = True
            assert len(boxes) == 1
            break
        sim.step(RawInput(), RawInput())
    assert found


def test_any_overlap_true_false():
    a = [(0, 0, 10, 10)]
    b = [(5, 5, 15, 15)]
    c = [(20, 20, 30, 30)]
    assert any_overlap(a, b)
    assert not any_overlap(a, c)
    assert not any_overlap([], b)


def test_stand_light_box_misses_crouch_hurt_heavy_and_fireball_hit():
    s = initial_state(0)
    p1, p2 = s.p1, s.p2
    p1.x = 400 * SUBPIXEL
    p2.x = 440 * SUBPIXEL
    p1.facing = 1
    p2.facing = -1
    p2.fsm = FSM.CROUCH
    update_body_size(p2)
    crouch = hurtboxes(p2)
    hb = STAND_LIGHT.hitbox
    light = hitbox_from_local(p1, hb.offset_x, hb.offset_y, hb.w, hb.h)
    assert not any_overlap([light], crouch)
    hh = STAND_HEAVY.hitbox
    heavy = hitbox_from_local(p1, hh.offset_x, hh.offset_y, hh.w, hh.h)
    assert any_overlap([heavy], crouch)
    proj = Projectile(
        owner=1,
        x=p2.x,
        y=PROJ_Y,
        vx=0,
        vy=0,
        w=PROJ_W,
        h=PROJ_H,
        damage=0,
        hitstun=0,
        blockstun=0,
        hitstop=0,
        knockback_x=0,
        knockback_y=0,
        lifetime=10,
    )
    assert any_overlap([projectile_box(proj)], crouch)
