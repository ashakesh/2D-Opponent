"""Rectangle debug renderer. Imported only from play() / Simulator.render()."""

from __future__ import annotations

from typing import Any, Optional

from fightsim.collision import projectile_box, pushbox
from fightsim.combat import current_hitbox
from fightsim.config import HEALTH_MAX, METER_MAX, STAGE_WIDTH_PX, SUBPIXEL
from fightsim.state import GameState

GROUND_SCREEN_Y = 340


def _sx(x: int) -> int:
    return x // SUBPIXEL


def _sy(y: int) -> int:
    return GROUND_SCREEN_Y - (y // SUBPIXEL)


def render_state(state: GameState, surface: Optional[Any] = None, p2_label: str | None = None) -> Any:
    import pygame

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    flip = False
    if surface is None:
        if not pygame.display.get_init() or pygame.display.get_surface() is None:
            pygame.display.set_mode((STAGE_WIDTH_PX, 400))
            pygame.display.set_caption("fightsim")
        surface = pygame.display.get_surface()
        flip = True

    surface.fill((18, 18, 24))
    pygame.draw.line(
        surface, (60, 60, 70), (0, GROUND_SCREEN_Y), (STAGE_WIDTH_PX, GROUND_SCREEN_Y), 2
    )

    _draw_fighter(surface, state.p1, (80, 160, 255))
    _draw_fighter(surface, state.p2, (220, 80, 80))

    for p in state.projectiles:
        _rect(surface, projectile_box(p), (240, 220, 80), 0)

    _health(surface, state)
    _hud(surface, state, p2_label)

    if flip:
        pygame.display.flip()
    return surface


def _rect(surface: Any, box: tuple[int, int, int, int], color: tuple[int, int, int], width: int) -> None:
    import pygame

    l, b, r, t = box
    x = _sx(l)
    y = _sy(t)
    w = max(1, _sx(r) - _sx(l))
    h = max(1, _sy(b) - _sy(t))
    pygame.draw.rect(surface, color, (x, y, w, h), width)


def _font(size: int):
    import pygame

    return pygame.font.Font(None, size)


def _draw_fighter(surface: Any, f: Any, color: tuple[int, int, int]) -> None:
    pb = pushbox(f)
    _rect(surface, pb, color, 0)
    hurt = (
        pb[0] - 4 * SUBPIXEL,
        pb[1],
        pb[2] + 4 * SUBPIXEL,
        pb[3] + 4 * SUBPIXEL,
    )
    _rect(surface, hurt, (80, 220, 120), 1)
    box = current_hitbox(f)
    if box is not None:
        _rect(surface, box, (255, 40, 40), 0)

    cx = _sx(f.x)
    cy = _sy(f.y + f.push_h)
    face = ">" if f.facing >= 0 else "<"
    label = f"{face} {f.fsm.value} {f.move_id or ''} f{f.state_frame}"
    surface.blit(_font(16).render(label, True, (220, 220, 220)), (cx - 40, cy - 16))


def _health(surface: Any, state: GameState) -> None:
    import pygame

    def bar(x: int, y: int, w: int, h: int, frac: float, col: tuple[int, int, int]) -> None:
        pygame.draw.rect(surface, (40, 40, 40), (x, y, w, h))
        pygame.draw.rect(surface, col, (x, y, max(0, int(w * frac)), h))
        pygame.draw.rect(surface, (200, 200, 200), (x, y, w, h), 1)

    bar(20, 16, 300, 16, state.p1.health / HEALTH_MAX, (80, 160, 255))
    bar(STAGE_WIDTH_PX - 320, 16, 300, 16, state.p2.health / HEALTH_MAX, (220, 80, 80))
    bar(20, 36, 300, 8, state.p1.meter / METER_MAX, (80, 200, 120))
    bar(STAGE_WIDTH_PX - 320, 36, 300, 8, state.p2.meter / METER_MAX, (80, 200, 120))


def _hud(surface: Any, state: GameState, p2_label: str | None = None) -> None:
    font = _font(18)
    t = f"frame {state.frame}  timer {state.timer}  {state.round_state.value}  hitstop {state.hitstop}"
    surface.blit(font.render(t, True, (240, 240, 240)), (STAGE_WIDTH_PX // 2 - 180, 12))
    if state.last_events:
        kinds = ",".join(e.kind for e in state.last_events)
        surface.blit(font.render(kinds, True, (255, 200, 80)), (20, 360))
    if p2_label:
        tag = f"P2: {p2_label}"
        img = font.render(tag, True, (220, 200, 160))
        surface.blit(img, (STAGE_WIDTH_PX - img.get_width() - 16, 372))
