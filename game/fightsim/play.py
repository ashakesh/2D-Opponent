"""Keyboard play loop. Requires pygame (`pip install -e ".[play]"`)."""

from __future__ import annotations

import argparse

from fightsim.api import Simulator
from fightsim.dummy import (
    DUMMY_STYLES,
    dummy_input,
    mix_rng_from_seed,
    pick_mix_style,
)
from fightsim.view import (
    Display,
    add_window_args,
    p1_from_keys,
    p2_from_keys,
    require_pygame,
)

_P2_CHOICES = DUMMY_STYLES + ("human", "mix")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fightsim debug play")
    parser.add_argument("--p2", default="idle", choices=_P2_CHOICES)
    parser.add_argument("--box", default="yes", choices=("yes", "no", "maybe"))
    parser.add_argument("--seed", type=int, default=0)
    add_window_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    require_pygame('pygame is required to play. Install with: pip install -e ".[play]"')

    from fightsim.render import render_state

    display = Display(caption="fightsim", native=args.native, fullscreen=args.fullscreen)
    sim = Simulator()
    sim.reset(args.seed)
    mix_rng = mix_rng_from_seed(args.seed)
    p2_style = args.p2
    if args.p2 == "mix":
        p2_style, mix_rng = pick_mix_style(mix_rng)

    running = True
    while running:
        frame = display.pump()
        if frame.quit:
            running = False
            continue
        if frame.reset_pressed:
            sim.reset(args.seed)
            if args.p2 == "mix":
                p2_style, mix_rng = pick_mix_style(mix_rng)

        p1 = p1_from_keys(frame.keys)
        if args.p2 == "human":
            p2 = p2_from_keys(frame.keys)
            label = "human"
        else:
            p2 = dummy_input(sim.get_state(), 2, p2_style)
            label = p2_style

        sim.step(p1, p2)
        render_state(sim.get_state(), display.logical, p2_label=label, box=args.box)
        display.present()

    display.close()


if __name__ == "__main__":
    main()
