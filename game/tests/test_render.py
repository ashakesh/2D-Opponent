import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from fightsim.api import Simulator
from fightsim.collision import character_box
from fightsim.config import STAGE_WIDTH_PX, SUBPIXEL
from fightsim.render import GROUND_SCREEN_Y, render_state

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
