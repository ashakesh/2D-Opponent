from fightsim.api import Simulator
from fightsim.config import PULSE_COST, SPECIAL_COST, STAGE_WIDTH, START_P1_X, START_P2_X, SUBPIXEL, THROW_RANGE
from fightsim.dummy import DUMMY_STYLES, dummy_input, mix_rng_from_seed, pick_mix_style
from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.state import Projectile


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


def test_personality_dummies_run():
    for style in DUMMY_STYLES:
        sim = Simulator()
        sim.reset(1)
        for _ in range(30):
            p2 = dummy_input(sim.get_state(), 2, style)
            sim.step(RawInput(), p2)
        assert sim.get_state().frame == 30


def test_dummy_same_seed_matches():
    def run():
        sim = Simulator()
        sim.reset(9)
        blobs = []
        for _ in range(40):
            p2 = dummy_input(sim.get_state(), 2, "aggressive")
            sim.step(RawInput(right=True), p2)
            blobs.append(sim.save_state())
        return blobs

    assert run() == run()


def test_mix_pick_is_seed_deterministic():
    def picks():
        rng = mix_rng_from_seed(3)
        out = []
        for _ in range(8):
            style, rng = pick_mix_style(rng)
            out.append(style)
        return out

    assert picks() == picks()
    assert len(set(picks())) >= 1


def _fireball_at(owner: int, x: int, vx: int) -> Projectile:
    return Projectile(
        owner=owner,
        x=x,
        y=40 * SUBPIXEL,
        vx=vx,
        vy=0,
        w=16 * SUBPIXEL,
        h=16 * SUBPIXEL,
        damage=600,
        hitstun=14,
        blockstun=12,
        hitstop=3,
        knockback_x=4 * SUBPIXEL,
        knockback_y=0,
        lifetime=40,
    )


def test_zoner_holds_ideal_gap_no_fireball_or_dp():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = START_P1_X
    st.p2.x = START_P2_X
    # ~300px gap, well past the 200px ideal — walk in, never 236/623.
    inp = dummy_input(st, 2, "zoner")
    assert inp.left is True
    assert inp.command_special_1 is False
    assert inp.command_special_2 is False


def test_zoner_crouch_blocks_when_cornered():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p2.x = STAGE_WIDTH - 20 * SUBPIXEL
    st.p1.x = st.p2.x - 140 * SUBPIXEL
    st.p2.facing = -1
    st.p1.facing = 1
    inp = dummy_input(st, 2, "zoner")
    assert inp.down is True
    assert inp.command_special_1 is False
    assert inp.command_special_2 is False


def test_defensive_holds_ground_when_far():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.x = 80 * SUBPIXEL
    inp = dummy_input(st, 2, "defensive")
    walking = (inp.left or inp.right) and not inp.down
    assert walking is False
    assert inp.command_special_1 is False
    assert inp.command_special_2 is False


def test_defensive_zoner_grappler_jump_close_fireball():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.projectiles.append(_fireball_at(1, st.p2.x - 80 * SUBPIXEL, 6 * SUBPIXEL))
    for style in ("defensive", "zoner", "grappler"):
        inp = dummy_input(st, 2, style)
        assert inp.up is True, style


def test_defensive_zoner_grappler_ignore_distant_fireball():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.projectiles.append(_fireball_at(1, st.p1.x + 40 * SUBPIXEL, 6 * SUBPIXEL))
    for style in ("defensive", "zoner", "grappler"):
        inp = dummy_input(st, 2, style)
        assert inp.up is False, style


def test_defensive_throw_or_heavy_when_close():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p2.x = st.p1.x + THROW_RANGE
    st.p2.meter = 0
    inp = dummy_input(st, 2, "defensive")
    assert inp.c or inp.b


def test_zoner_pulses_when_too_close():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p2.x = st.p1.x + 90 * SUBPIXEL
    st.p2.meter = PULSE_COST
    inp = dummy_input(st, 2, "zoner")
    assert inp.command_special_3 is True
    assert inp.command_special_1 is False
    assert inp.command_special_2 is False


def test_grappler_jumps_to_air_to_air():
    from fightsim.state import FSM

    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.fsm = FSM.JUMP
    st.p1.y = 40 * SUBPIXEL
    st.p2.x = st.p1.x + 100 * SUBPIXEL
    st.p2.meter = 0
    inp = dummy_input(st, 2, "grappler")
    assert inp.up is True
    assert inp.command_special_2 is False


def test_aggressive_dp_anti_air_in_range_not_jump():
    from fightsim.state import FSM

    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    st.p1.fsm = FSM.JUMP
    st.p1.y = 40 * SUBPIXEL
    st.p2.x = st.p1.x + 80 * SUBPIXEL
    st.p2.meter = SPECIAL_COST
    inp = dummy_input(st, 2, "aggressive")
    assert inp.command_special_2 is True
    assert inp.up is False
