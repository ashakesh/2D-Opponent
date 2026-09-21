import json

from fightsim.animation import (
    CLIPS,
    all_clip_filenames,
    fighter_asset_root,
    load_rig,
    pivot_topleft,
    resolve_clip,
    resolve_sprite_path,
)
from fightsim.state import FSM, initial_state


def test_resolve_clip_idle_and_light_phases():
    s = initial_state(0)
    pose = resolve_clip(s.p1, 0)
    assert pose.clip == "idle"
    assert pose.filename == "idle_00.png"
    s.p1.fsm = FSM.ATTACK
    s.p1.move_id = "stand_light"
    s.p1.state_frame = 0
    assert resolve_clip(s.p1).clip == "stand_light_startup"
    s.p1.state_frame = 3
    assert resolve_clip(s.p1).clip == "stand_light_active"
    s.p1.state_frame = 5
    assert resolve_clip(s.p1).clip == "stand_light_recovery"


def test_resolve_clip_pulse_and_land():
    s = initial_state(0)
    s.p1.fsm = FSM.ATTACK
    s.p1.move_id = "pulse"
    s.p1.state_frame = 0
    assert resolve_clip(s.p1).clip == "pulse_startup"
    s.p1.state_frame = 4
    assert resolve_clip(s.p1).clip == "pulse_active"
    s.p1.fsm = FSM.IDLE
    s.p1.move_id = ""
    s.p1.landing_lock = 3
    assert resolve_clip(s.p1).clip == "land"


def test_manifest_matches_clip_table():
    root = fighter_asset_root()
    data = json.loads((root / "manifest.json").read_text())
    for name, spec in CLIPS.items():
        assert name in data["clips"]
        assert data["clips"][name]["files"] == spec["files"]


def test_every_clip_has_art():
    root = fighter_asset_root() / "art"
    for filename in all_clip_filenames():
        assert (root / filename).is_file(), filename


def test_rig_canvas_and_pivot():
    canvas = load_rig()["canvas"]
    assert canvas["w"] == 160
    assert canvas["h"] == 128
    assert canvas["pivot_x"] == 56
    assert canvas["pivot_y"] == 120


def test_pivot_topleft_facing():
    assert pivot_topleft(100, 200, 56, 120, 160, 1) == (100 - 56, 200 - 120)
    assert pivot_topleft(100, 200, 56, 120, 160, -1) == (100 - (160 - 56), 200 - 120)


def test_sprite_path_resolves_art():
    root = fighter_asset_root()
    assert resolve_sprite_path("idle_00.png") == root / "art" / "idle_00.png"
    assert resolve_sprite_path("stand_light_startup_00.png") == root / "art" / "stand_light_startup_00.png"
    assert resolve_sprite_path("knockdown_00.png") == root / "art" / "knockdown_00.png"
    assert resolve_sprite_path("no_such_sprite.png") is None
