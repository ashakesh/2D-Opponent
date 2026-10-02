"""Pygame window, scale, and keyboard maps. Not imported from fightsim.__init__.

Training must not import this module. The sim loop stays in each CLI.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any

from fightsim.config import FPS, STAGE_WIDTH_PX
from fightsim.input import RawInput

VIEW_WIDTH = STAGE_WIDTH_PX
VIEW_HEIGHT = 400
VIEW_SIZE = (VIEW_WIDTH, VIEW_HEIGHT)
SCALE_STEP = 0.25
MIN_SCALE = 1.0


def scaled_size(
    scale: float,
    logical_w: int = VIEW_WIDTH,
    logical_h: int = VIEW_HEIGHT,
) -> tuple[int, int]:
    s = float(scale)
    return (int(round(logical_w * s)), int(round(logical_h * s)))


def quantized_scale(
    win_w: int,
    win_h: int,
    logical_w: int = VIEW_WIDTH,
    logical_h: int = VIEW_HEIGHT,
) -> float:
    """Nearest scale >= 1.0 in 0.25 steps. Ties pick the smaller scale."""
    if win_w <= 0 or win_h <= 0 or logical_w <= 0 or logical_h <= 0:
        return MIN_SCALE
    span = max(win_w / logical_w, win_h / logical_h, MIN_SCALE)
    n_hi = max(round(MIN_SCALE / SCALE_STEP), math.ceil(span / SCALE_STEP))
    n_lo = round(MIN_SCALE / SCALE_STEP)
    best = MIN_SCALE
    best_d = float("inf")
    for n in range(n_lo, n_hi + 1):
        s = n * SCALE_STEP
        sw, sh = scaled_size(s, logical_w, logical_h)
        d = (sw - win_w) ** 2 + (sh - win_h) ** 2
        if d < best_d:
            best_d = d
            best = s
    return best


def max_fit_scale(
    win_w: int,
    win_h: int,
    logical_w: int = VIEW_WIDTH,
    logical_h: int = VIEW_HEIGHT,
) -> float:
    """Largest 0.25-step scale >= 1.0 that fits in the window."""
    if win_w <= 0 or win_h <= 0 or logical_w <= 0 or logical_h <= 0:
        return MIN_SCALE
    raw = min(win_w / logical_w, win_h / logical_h)
    n = math.floor(raw / SCALE_STEP + 1e-9)
    return max(MIN_SCALE, n * SCALE_STEP)


def visible_size(
    surf_w: int,
    surf_h: int,
    desk_w: int,
    desk_h: int,
) -> tuple[int, int]:
    """Per-axis min of pygame surface vs desktop (visible-safe)."""
    fw = min(int(surf_w), int(desk_w)) if desk_w > 0 else int(surf_w)
    fh = min(int(surf_h), int(desk_h)) if desk_h > 0 else int(surf_h)
    return (max(1, fw), max(1, fh))


def letterbox_scale(
    fit_w: int,
    fit_h: int,
    logical_w: int = VIEW_WIDTH,
    logical_h: int = VIEW_HEIGHT,
) -> float:
    """Largest 0.25-step scale whose scaled size fits. May be below 1.0."""
    if fit_w <= 0 or fit_h <= 0 or logical_w <= 0 or logical_h <= 0:
        return SCALE_STEP
    raw = min(fit_w / logical_w, fit_h / logical_h)
    n = max(1, math.floor(raw / SCALE_STEP + 1e-9))
    while n >= 1:
        s = n * SCALE_STEP
        sw, sh = scaled_size(s, logical_w, logical_h)
        if sw <= fit_w and sh <= fit_h:
            return s
        n -= 1
    return SCALE_STEP


def desktop_size() -> tuple[int, int]:
    """Monitor pixel size. Requires pygame.init()."""
    import pygame

    getter = getattr(pygame.display, "get_desktop_sizes", None)
    if getter is not None:
        sizes = getter()
        if sizes:
            w, h = int(sizes[0][0]), int(sizes[0][1])
            if w > 0 and h > 0:
                return (w, h)
    info = pygame.display.Info()
    w, h = int(info.current_w), int(info.current_h)
    if w > 0 and h > 0:
        return (w, h)
    return VIEW_SIZE


def add_window_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--native",
        action="store_true",
        help="Lock 800x400 1:1 (not resizable). Lab view.",
    )
    group.add_argument(
        "--fullscreen",
        action="store_true",
        help="Fullscreen: largest 0.25-step scale that fits, letterboxed. F11 toggles when not --native.",
    )


def p1_from_keys(keys: Any) -> RawInput:
    import pygame

    return RawInput(
        left=bool(keys[pygame.K_a]),
        right=bool(keys[pygame.K_d]),
        up=bool(keys[pygame.K_w]),
        down=bool(keys[pygame.K_s]),
        a=bool(keys[pygame.K_j]),
        b=bool(keys[pygame.K_k]),
        c=bool(keys[pygame.K_l]),
        command_special_1=bool(keys[pygame.K_i]),
        command_special_2=bool(keys[pygame.K_o]),
        command_special_3=bool(keys[pygame.K_p]),
    )


def p2_from_keys(keys: Any) -> RawInput:
    import pygame

    return RawInput(
        left=bool(keys[pygame.K_LEFT]),
        right=bool(keys[pygame.K_RIGHT]),
        up=bool(keys[pygame.K_UP]),
        down=bool(keys[pygame.K_DOWN]),
        a=bool(keys[pygame.K_KP1]),
        b=bool(keys[pygame.K_KP2]),
        c=bool(keys[pygame.K_KP3]),
        command_special_1=bool(keys[pygame.K_KP4]),
        command_special_2=bool(keys[pygame.K_KP5]),
        command_special_3=bool(keys[pygame.K_KP6]),
    )


@dataclass
class FrameEvents:
    quit: bool = False
    reset_pressed: bool = False
    keys: Any = None


class Display:
    """Owns the OS window. Callers draw onto `logical` (800x400) then present()."""

    def __init__(
        self,
        *,
        caption: str,
        native: bool = False,
        fullscreen: bool = False,
    ) -> None:
        import pygame

        if native and fullscreen:
            raise ValueError("--native and --fullscreen are mutually exclusive")
        pygame.init()
        pygame.font.init()
        self.native = bool(native)
        self.fullscreen = bool(fullscreen) and not self.native
        self._windowed_size = VIEW_SIZE
        self._desktop_size = desktop_size()
        self.logical = pygame.Surface(VIEW_SIZE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(caption)
        self.screen = self._open()

    def _open(self) -> Any:
        import pygame

        if self.native:
            return pygame.display.set_mode(VIEW_SIZE)
        if self.fullscreen:
            return pygame.display.set_mode(self._desktop_size, pygame.FULLSCREEN)
        return pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)

    def pump(self) -> FrameEvents:
        import pygame

        ev = FrameEvents()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ev.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    ev.quit = True
                elif event.key == pygame.K_r:
                    ev.reset_pressed = True
                elif event.key == pygame.K_F11 and not self.native:
                    if not self.fullscreen:
                        self._windowed_size = self.screen.get_size()
                    self.fullscreen = not self.fullscreen
                    self.screen = self._open()
            elif (
                event.type == pygame.VIDEORESIZE
                and not self.native
                and not self.fullscreen
            ):
                s = quantized_scale(int(event.w), int(event.h))
                self._windowed_size = scaled_size(s)
                self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        ev.keys = pygame.key.get_pressed()
        self.clock.tick(FPS)
        return ev

    def present(self, overlay: str | None = None) -> None:
        import pygame

        if overlay:
            font = pygame.font.Font(None, 20)
            img = font.render(overlay, True, (200, 200, 200))
            self.logical.blit(img, (20, 380))
        if self.native:
            self.screen.blit(self.logical, (0, 0))
            pygame.display.flip()
            return
        win_w, win_h = self.screen.get_size()
        if self.fullscreen:
            sw, sh = self.screen.get_size()
            dw, dh = self._desktop_size
            fit_w, fit_h = visible_size(sw, sh, dw, dh)
            scale = letterbox_scale(fit_w, fit_h)
            scaled_w, scaled_h = scaled_size(scale)
            scaled = pygame.transform.scale(self.logical, (scaled_w, scaled_h))
            ox = (fit_w - scaled_w) // 2
            oy = (fit_h - scaled_h) // 2
            self.screen.fill((0, 0, 0))
            self.screen.blit(scaled, (ox, oy))
            pygame.display.flip()
            return
        scale = quantized_scale(win_w, win_h)
        scaled_w, scaled_h = scaled_size(scale)
        scaled = pygame.transform.scale(self.logical, (scaled_w, scaled_h))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        import pygame

        pygame.quit()


def require_pygame(install_hint: str) -> None:
    try:
        import pygame  # noqa: F401
    except ImportError:
        print(install_hint, file=sys.stderr)
        sys.exit(1)
