from fightsim.api import Simulator
from fightsim.input import RawInput


def _lcg(n: int, seed: int) -> list[tuple[RawInput, RawInput]]:
    x = seed
    tape: list[tuple[RawInput, RawInput]] = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        p1 = RawInput(
            left=bool(x & 1),
            right=bool(x & 2),
            down=bool(x & 4),
            up=bool(x & 8),
            a=bool(x & 16),
            b=bool(x & 32),
            c=bool(x & 64),
            command_special_1=bool(x & 128),
        )
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        p2 = RawInput(
            left=bool(x & 1),
            right=bool(x & 2),
            down=bool(x & 4),
            up=bool(x & 8),
            a=bool(x & 16),
            b=bool(x & 32),
            c=bool(x & 64),
            command_special_2=bool(x & 128),
        )
        tape.append((p1, p2))
    return tape


def test_same_seed_and_inputs_match():
    tape = _lcg(180, 42)
    a, b = Simulator(), Simulator()
    a.reset(7)
    b.reset(7)
    for p1, p2 in tape:
        a.step(p1, p2)
        b.step(p1, p2)
        assert a.save_state() == b.save_state()


def test_different_seed_diverges():
    a, b = Simulator(), Simulator()
    a.reset(1)
    b.reset(2)
    idle = RawInput()
    for _ in range(10):
        a.step(idle, idle)
        b.step(idle, idle)
    # Positions are identical (RNG unused in MVP movement), but rng field differs.
    assert a.get_state().rng != b.get_state().rng
    assert a.save_state() != b.save_state()
