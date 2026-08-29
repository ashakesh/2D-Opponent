"""Keyboard play loop. Requires pygame (`pip install -e ".[play]"`)."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from fightsim.api import Simulator
from fightsim.config import FPS, STAGE_WIDTH_PX
from fightsim.dummy import dummy_input
from fightsim.input import RawInput


def _p1_from_keys(keys: Any) -> RawInput:
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
    )


def _p2_from_keys(keys: Any) -> RawInput:
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
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="fightsim debug play")
    parser.add_argument("--p2", default="idle", choices=("idle", "block", "mash", "crouch_block", "human"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import pygame
    except ImportError:
        print("pygame is required to play. Install with: pip install -e \".[play]\"", file=sys.stderr)
        sys.exit(1)

    from fightsim.render import render_state

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((STAGE_WIDTH_PX, 400))
    pygame.display.set_caption("fightsim")
    clock = pygame.time.Clock()

    sim = Simulator()
    sim.reset(args.seed)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                sim.reset(args.seed)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        p1 = _p1_from_keys(keys)
        if args.p2 == "human":
            p2 = _p2_from_keys(keys)
        else:
            p2 = dummy_input(sim.get_state(), 2, args.p2)

        sim.step(p1, p2)
        render_state(sim.get_state(), screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
