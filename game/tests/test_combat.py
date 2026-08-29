from fightsim.api import Simulator
from fightsim.config import HEALTH_MAX, SUBPIXEL
from fightsim.input import RawInput, SemanticAction, semantic_to_raw
from fightsim.moves import STAND_LIGHT, THROW


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


def test_holding_back_blocks_standing_light():
    sim = Simulator()
    sim.reset(0)
    _close(sim, 32)
    block = semantic_to_raw(SemanticAction.BLOCK, -1)
    # Crouch+back so P2 does not walk out of range.
    p2 = RawInput(left=block.left, right=block.right, down=True)
    sim.step(RawInput(a=True), p2)
    _wait(sim, 18, p2=p2)
    st = sim.get_state()
    assert st.p2.health == HEALTH_MAX
    # Should have been in blockstun at some point; after recovery they may be idle.
    assert st.p2.fsm.value in ("blockstun", "idle", "crouch")


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
    for _ in range(24):
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
