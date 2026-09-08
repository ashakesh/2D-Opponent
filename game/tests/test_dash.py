from fightsim.api import Simulator
from fightsim.input import RawInput, tap_back, tap_forward
from fightsim.state import FSM


def test_strict_dash_requires_release_inside_window():
    # 6, 5, 6 within 7 frames → dash
    buf = [5] * 25 + [6, 5, 6]
    assert tap_forward(buf)
    buf = [5] * 25 + [4, 5, 4]
    assert tap_back(buf)


def test_walk_then_delayed_tap_is_not_a_dash():
    # Many forwards, then a long neutral, then one forward — old 12f window would dash.
    buf = [6] * 20 + [5] * 10 + [6]
    assert not tap_forward(buf)


def test_held_forward_is_not_a_dash():
    buf = [5] * 10 + [6] * 8
    assert not tap_forward(buf)


def test_sim_walk_pause_tap_does_not_dash():
    sim = Simulator()
    sim.reset(0)
    for _ in range(20):
        sim.step(RawInput(right=True), RawInput())
    for _ in range(10):
        sim.step(RawInput(), RawInput())
    sim.step(RawInput(right=True), RawInput())
    assert sim.get_state().p1.fsm != FSM.DASH


def test_sim_quick_double_tap_dashes():
    sim = Simulator()
    sim.reset(0)
    sim.step(RawInput(right=True), RawInput())
    sim.step(RawInput(), RawInput())
    sim.step(RawInput(right=True), RawInput())
    assert sim.get_state().p1.fsm == FSM.DASH
