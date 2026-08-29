from fightsim.api import Simulator
from fightsim.input import RawInput
from fightsim.replay import Replay, ReplayRecorder, playback_matches


def test_replay_roundtrip_bytes_and_playback():
    sim = Simulator()
    seed = 5
    sim.reset(seed)
    rec = ReplayRecorder(seed, sim.get_state())
    x = 3
    for _ in range(60):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        p1 = RawInput(right=bool(x & 1), a=bool(x & 4), command_special_1=bool(x & 8))
        p2 = RawInput(left=bool(x & 2), a=bool(x & 16))
        sim.step(p1, p2)
        rec.record(p1, p2, sim.get_state(), sim.get_events())

    blob = rec.replay.to_bytes()
    loaded = Replay.from_bytes(blob)
    assert loaded.simulator_version == rec.replay.simulator_version
    assert loaded.seed == seed
    assert playback_matches(loaded)
