from fightsim.api import Simulator
from fightsim.dummy import dummy_input
from fightsim.input import RawInput, SemanticAction, semantic_to_raw


def test_dummy_idle_is_neutral():
    sim = Simulator()
    sim.reset(0)
    inp = dummy_input(sim.get_state(), 2, "idle")
    assert inp == RawInput()


def test_dummy_block_holds_back():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    inp = dummy_input(st, 2, "block")
    expected = semantic_to_raw(SemanticAction.BLOCK, st.p2.facing)
    assert inp == expected
    assert inp.right is True  # P2 faces left at start


def test_mash_dummy_runs_without_error():
    sim = Simulator()
    sim.reset(0)
    for _ in range(40):
        p2 = dummy_input(sim.get_state(), 2, "mash")
        sim.step(RawInput(), p2)
    assert sim.get_state().frame == 40
