from fightsim.api import Simulator
from fightsim.collision import any_overlap, character_box, hurtboxes
from fightsim.combat import attack_boxes
from fightsim.config import CHAR_BOX_H, CHAR_BOX_W, SUBPIXEL
from fightsim.input import RawInput
from fightsim.physics import update_body_size
from fightsim.state import FSM, initial_state


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


def test_hurtboxes_is_one_rect_today():
    s = initial_state(0)
    boxes = hurtboxes(s.p1)
    assert len(boxes) == 1


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
