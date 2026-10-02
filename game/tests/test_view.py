import sys

import pytest

from fightsim.api import Simulator
from fightsim.play import parse_args as parse_play_args
from fightsim.render import lab_hud_line, play_p2_caption, round_result_text, seconds_remaining
from fightsim.state import RoundState
from fightsim.view import letterbox_scale, max_fit_scale, quantized_scale, scaled_size, visible_size


def test_quantized_scale_nearest_quarter():
    assert quantized_scale(800, 400) == 1.0
    assert scaled_size(1.25) == (1000, 500)
    assert quantized_scale(1200, 600) == 1.5
    assert quantized_scale(1600, 800) == 2.0
    assert quantized_scale(2400, 1200) == 3.0
    # 900x450 is halfway between 1.0 and 1.25; ties pick smaller.
    assert quantized_scale(900, 450) == 1.0
    # 1920x1080 is closer to 2.50 (2000x1000) than 2.25 (1800x900).
    assert quantized_scale(1920, 1080) == 2.5
    assert quantized_scale(799, 400) == 1.0


def test_max_fit_scale_letterbox():
    assert max_fit_scale(800, 400) == 1.0
    assert max_fit_scale(1920, 1080) == 2.25
    assert max_fit_scale(1600, 800) == 2.0
    assert max_fit_scale(799, 399) == 1.0


def test_letterbox_scale_never_exceeds_fit():
    assert letterbox_scale(1920, 1080) == 2.25
    assert scaled_size(letterbox_scale(1920, 1080)) == (1800, 900)
    assert letterbox_scale(799, 399) == 0.75
    sw, sh = scaled_size(0.75)
    assert sw <= 799 and sh <= 399
    assert visible_size(1920, 1080, 1440, 900) == (1440, 900)
    assert letterbox_scale(1440, 900) == 1.75
    assert scaled_size(1.75) == (1400, 700)
    for w, h in (
        (1920, 1080),
        (1440, 900),
        (1366, 768),
        (2560, 1440),
        (1280, 800),
        (800, 400),
        (799, 399),
        (700, 350),
    ):
        s = letterbox_scale(w, h)
        sw, sh = scaled_size(s)
        assert sw <= w
        assert sh <= h


def test_view_import_does_not_load_pygame():
    before = set(sys.modules)
    import fightsim.view  # noqa: F401

    added = set(sys.modules) - before
    assert "pygame" not in added
    assert not any(name == "pygame" or name.startswith("pygame.") for name in added)


def test_play_cli_window_flags():
    args = parse_play_args([])
    assert args.native is False
    assert args.fullscreen is False
    assert args.box == "yes"
    native = parse_play_args(["--native", "--box", "no"])
    assert native.native is True
    assert native.box == "no"
    with pytest.raises(SystemExit):
        parse_play_args(["--native", "--fullscreen"])


def test_play_hud_copy():
    sim = Simulator()
    sim.reset(0)
    st = sim.get_state()
    assert "frame " in lab_hud_line(st)
    assert "hitstop" in lab_hud_line(st)
    assert round_result_text(st) is None
    assert seconds_remaining(st) == 99
    assert play_p2_caption("agent") == "Player 2 — Agent"
    assert play_p2_caption("human") == "Player 2 — Human"
    assert play_p2_caption("crouch_block") == "Player 2 — Crouch Block"
    assert play_p2_caption(None) == "Player 2"
    st.round_state = RoundState.P1_WIN
    assert round_result_text(st) == "P1 Wins"
