from fightsim.api import Simulator
from fightsim.input import RawInput


def test_save_load_roundtrip_after_reset():
    sim = Simulator()
    sim.reset(3)
    blob = sim.save_state()
    other = Simulator()
    other.load_state(blob)
    assert other.save_state() == blob


def test_save_mid_sequence_continues_identically():
    tape = []
    x = 99
    for _ in range(80):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        p1 = RawInput(right=bool(x & 1), a=bool(x & 2), down=bool(x & 4))
        p2 = RawInput(left=bool(x & 8), a=bool(x & 16))
        tape.append((p1, p2))

    full = Simulator()
    full.reset(11)
    for p1, p2 in tape:
        full.step(p1, p2)

    prefix = Simulator()
    prefix.reset(11)
    for p1, p2 in tape[:30]:
        prefix.step(p1, p2)
    snap = prefix.save_state()

    resumed = Simulator()
    resumed.load_state(snap)
    for p1, p2 in tape[30:]:
        resumed.step(p1, p2)

    assert resumed.save_state() == full.save_state()
