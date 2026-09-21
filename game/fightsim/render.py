"""Rectangle debug renderer plus optional PNG blit. Imported only from play() / Simulator.render()."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

from fightsim.animation import (
    load_rig,
    pivot_topleft,
    resolve_clip,
    resolve_projectile_clip,
    resolve_sprite_path,
)
from fightsim.collision import hurtboxes, projectile_box, pushbox
from fightsim.combat import attack_boxes
from fightsim.config import HEALTH_MAX, METER_MAX, STAGE_WIDTH_PX, SUBPIXEL
from fightsim.state import GameState

GROUND_SCREEN_Y = 340

_image_cache: dict[str, Any] = {}
_missing_warned: set[str] = set()


def _sx(x: int) -> int:
    return x // SUBPIXEL


def _sy(y: int) -> int:
    return GROUND_SCREEN_Y - (y // SUBPIXEL)


def render_state(
    state: GameState,
    surface: Optional[Any] = None,
    p2_label: str | None = None,
    box: str = "yes",
) -> Any:
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

    debug_only = box == "yes"
    overlay = box == "maybe"

    _draw_fighter_mode(surface, state.p1, (80, 160, 255), state.frame, debug_only, overlay)
    _draw_fighter_mode(surface, state.p2, (220, 80, 80), state.frame, debug_only, overlay)

    for p in state.projectiles:
        if debug_only:
            _rect(surface, projectile_box(p), (240, 220, 80), 0)
        elif not _draw_projectile_sprite(surface, p, state.frame):
            _rect(surface, projectile_box(p), (240, 220, 80), 0)
        elif overlay:
            _rect(surface, projectile_box(p), (240, 220, 80), 1)

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


def _draw_fighter_mode(
    surface: Any,
    f: Any,
    color: tuple[int, int, int],
    frame: int,
    debug_only: bool,
    overlay: bool,
) -> None:
    if debug_only:
        _draw_fighter_debug(surface, f, color)
        return
    if not _draw_fighter_sprite(surface, f, frame):
        _draw_fighter_debug(surface, f, color)
        return
    if overlay:
        _draw_fighter_overlay(surface, f, color)


def _draw_fighter_overlay(surface: Any, f: Any, color: tuple[int, int, int]) -> None:
    _rect(surface, pushbox(f), color, 1)
    for hurt in hurtboxes(f):
        _rect(surface, hurt, (80, 220, 120), 1)
    for hit in attack_boxes(f):
        _rect(surface, hit, (255, 40, 40), 1)


def _draw_fighter_debug(surface: Any, f: Any, color: tuple[int, int, int]) -> None:
    pb = pushbox(f)
    _rect(surface, pb, color, 0)
    for hurt in hurtboxes(f):
        _rect(surface, hurt, (80, 220, 120), 1)
    for box in attack_boxes(f):
        _rect(surface, box, (255, 40, 40), 0)

    cx = _sx(f.x)
    cy = _sy(f.y + f.push_h)
    face = ">" if f.facing >= 0 else "<"
    label = f"{face} {f.fsm.value} {f.move_id or ''} f{f.state_frame}"
    surface.blit(_font(16).render(label, True, (220, 220, 220)), (cx - 40, cy - 16))


def _sprite_path(filename: str) -> Path | None:
    return resolve_sprite_path(filename)


def _load_image(filename: str) -> Any | None:
    import pygame

    path = _sprite_path(filename)
    if path is None:
        if filename not in _missing_warned:
            _missing_warned.add(filename)
            warnings.warn(f"fightsim sprite missing: {filename}", stacklevel=2)
        return None
    key = str(path)
    cached = _image_cache.get(key)
    if cached is not None:
        return cached
    img = pygame.image.load(str(path)).convert_alpha()
    _image_cache[key] = img
    return img


def _draw_fighter_sprite(surface: Any, f: Any, frame: int) -> bool:
    import pygame

    pose = resolve_clip(f, frame)
    img = _load_image(pose.filename)
    if img is None:
        return False
    if pose.facing < 0:
        img = pygame.transform.flip(img, True, False)
    canvas = load_rig()["canvas"]
    x, y = pivot_topleft(
        _sx(f.x),
        _sy(f.y),
        int(canvas["pivot_x"]),
        int(canvas["pivot_y"]),
        img.get_width(),
        pose.facing,
    )
    surface.blit(img, (x, y))
    return True


def _draw_projectile_sprite(surface: Any, p: Any, frame: int) -> bool:
    import pygame

    pose = resolve_projectile_clip(p, frame)
    img = _load_image(pose.filename)
    if img is None:
        return False
    if pose.facing < 0:
        img = pygame.transform.flip(img, True, False)
    pb = projectile_box(p)
    cx = (_sx(pb[0]) + _sx(pb[2])) // 2
    cy = (_sy(pb[1]) + _sy(pb[3])) // 2
    rect = img.get_rect(center=(cx, cy))
    surface.blit(img, rect)
    return True


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
