"""Phase-2/3 scaffolding: viewer obs, dummy opponent, action mask."""

from __future__ import annotations

from fightsim.state import FSM, initial_state
from fightml.env import Phase1Action
from fightml.mask import action_mask
from fightml.obs import OBS_SIZE, flatten_state
from fightml.opponent import DummyOpponent, collect_pool_paths


def test_view_pid_swaps_me_without_changing_size():
    state = initial_state(0)
    p1 = flatten_state(state, view_pid=1)
    p2 = flatten_state(state, view_pid=2)
    assert p1.shape == (OBS_SIZE,)
    assert p2.shape == (OBS_SIZE,)
    assert not (p1 == p2).all()
    # me.x is the first feature; P1 starts left of P2.
    assert p1[0] < p2[0]


def test_dummy_opponent_mix_resolves_personality():
    opp = DummyOpponent("mix")
    opp.reset(0)
    assert opp.resolved == "aggressive"
    opp.reset(1)
    assert opp.resolved == "defensive"


def test_action_mask_air_and_meter():
    state = initial_state(0)
    ground = action_mask(state, pid=1)
    assert ground[int(Phase1Action.THROW)]
    assert ground[int(Phase1Action.SPECIAL_1)]
    state.p1.y = 1
    air = action_mask(state, pid=1)
    assert not air[int(Phase1Action.THROW)]
    assert not air[int(Phase1Action.SPECIAL_1)]
    state.p1.y = 0
    state.p1.meter = 0
    poor = action_mask(state, pid=1)
    assert not poor[int(Phase1Action.SPECIAL_1)]
    assert not poor[int(Phase1Action.SPECIAL_3)]
    state.p1.fsm = FSM.HITSTUN
    state.p1.meter = 1000
    stun = action_mask(state, pid=1)
    assert not stun[int(Phase1Action.LIGHT)]


def test_collect_pool_paths_empty_and_dir(tmp_path):
    assert collect_pool_paths("") == []
    z = tmp_path / "ppo_a.zip"
    z.write_bytes(b"x")
    found = collect_pool_paths(str(tmp_path))
    assert str(z) in found


def test_collect_pool_paths_relative_runs_ignores_cwd(tmp_path, monkeypatch):
    import pytest
    from fightml.paths import runs_root

    monkeypatch.chdir(tmp_path)
    pool = runs_root() / "pool"
    if not pool.is_dir() or not any(pool.glob("*.zip")):
        pytest.skip("no frozen pool zip on disk")
    found = collect_pool_paths("runs/pool")
    assert any(str(pool) in p for p in found)
