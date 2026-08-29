import sys


def test_sim_module_does_not_import_pygame():
    before = set(sys.modules)
    import fightsim.sim  # noqa: F401

    added = set(sys.modules) - before
    assert "pygame" not in added
    assert not any(name == "pygame" or name.startswith("pygame.") for name in added)


def test_render_is_noop_without_pygame():
    from fightsim.api import Simulator

    sim = Simulator()
    sim.reset(0)
    sim.render()

