import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from fightsim.api import Simulator
from fightsim.collision import character_box
from fightsim.config import STAGE_WIDTH_PX, SUBPIXEL
from fightsim.render import GROUND_SCREEN_Y, lab_hud_line, render_state

P1_DEBUG = (80, 160, 255)


def _outline_hits(surface: pygame.Surface, box: tuple[int, int, int, int]) -> int:
    l, b, r, t = box
    x = l // SUBPIXEL
    y = GROUND_SCREEN_Y - (t // SUBPIXEL)
    w = max(1, (r // SUBPIXEL) - (l // SUBPIXEL))
    h = max(1, (GROUND_SCREEN_Y - (b // SUBPIXEL)) - y)
    hits = 0
    for px in range(x, x + w):
        if surface.get_at((px, y))[:3] == P1_DEBUG:
            hits += 1
        if surface.get_at((px, y + h - 1))[:3] == P1_DEBUG:
            hits += 1
    for py in range(y, y + h):
        if surface.get_at((x, py))[:3] == P1_DEBUG:
            hits += 1
        if surface.get_at((x + w - 1, py))[:3] == P1_DEBUG:
            hits += 1
    return hits


def test_box_no_does_not_stroke_character_box():
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((STAGE_WIDTH_PX, 400))
    surface = pygame.Surface((STAGE_WIDTH_PX, 400))
    sim = Simulator()
    sim.reset(0)
    state = sim.get_state()
    render_state(state, surface, box="no")
    assert _outline_hits(surface, character_box(state.p1)) == 0


def test_box_no_omits_lab_hud_line():
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((STAGE_WIDTH_PX, 400))
    surface = pygame.Surface((STAGE_WIDTH_PX, 400))
    sim = Simulator()
    sim.reset(0)
    state = sim.get_state()
    render_state(state, surface, p2_label="zoner", box="no")
    lab = lab_hud_line(state)
    assert "frame " in lab
    # Play HUD must not paint the lab strip; debug mode still does.
    lab_surface = pygame.Surface((STAGE_WIDTH_PX, 400))
    render_state(state, lab_surface, p2_label="zoner", box="yes")
    assert lab_surface.get_at((STAGE_WIDTH_PX // 2, 18))[:3] != (18, 18, 24)
    play_center = surface.get_at((STAGE_WIDTH_PX // 2, 18))[:3]
    lab_center = lab_surface.get_at((STAGE_WIDTH_PX // 2, 18))[:3]
    assert play_center != lab_center
