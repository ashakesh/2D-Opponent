"""Qualitative eval: watch a checkpoint vs dummies, or fight it as P2.

Training never imports this module. pygame is required only for the window.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable

from fightsim.api import Simulator
from fightsim.config import FPS, STAGE_WIDTH_PX
from fightsim.dummy import dummy_input
from fightsim.input import RawInput
from fightsim.state import RoundState

from fightml.env import DUMMY_STYLES, PERSONALITY_STYLES, action_to_raw
from fightml.obs import flatten_state

PredictFn = Callable[[Any], int]
P2_CHOICES = DUMMY_STYLES + ("mix", "human")
RESET_PAUSE_FRAMES = int(1.5 * FPS)


@dataclass
class EpisodeResult:
    round_state: str
    frames: int
    p1_health: int
    p2_health: int
    p2_style: str


class StickyPolicy:
    """Hold one semantic action for `frameskip` sim frames (same as FightEnv)."""

    def __init__(self, predict: PredictFn, frameskip: int) -> None:
        self.predict = predict
        self.frameskip = max(1, int(frameskip))
        self._raw = RawInput()
        self._left = 0

    def reset(self) -> None:
        self._raw = RawInput()
        self._left = 0

    def raw(self, state) -> RawInput:
        if state.round_state != RoundState.FIGHTING:
            return RawInput()
        if self._left <= 0:
            action = int(self.predict(flatten_state(state)))
            self._raw = action_to_raw(action, state.p1.facing)
            self._left = self.frameskip
        self._left -= 1
        return self._raw


def load_predict(path: str, deterministic: bool = True) -> PredictFn:
    from stable_baselines3 import PPO

    model = PPO.load(path)

    def predict(obs) -> int:
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    return predict


def _p2_from_keys(keys: Any) -> RawInput:
    """Same mapping as fightsim.play P2 (human is always P2; the agent is P1)."""
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


def pick_style(p2: str, rng) -> str:
    if p2 == "human":
        raise ValueError("human is not a scripted dummy style")
    if p2 == "mix":
        return str(rng.choice(PERSONALITY_STYLES))
    if p2 in DUMMY_STYLES:
        return p2
    raise ValueError(f"unknown dummy style: {p2}")


def run_episode(
    sim: Simulator,
    predict: PredictFn,
    p2_style: str,
    frameskip: int,
    seed: int,
    max_frames: int | None = None,
) -> EpisodeResult:
    """Headless agent (P1) vs one dummy style. Does not import pygame."""
    if p2_style not in DUMMY_STYLES:
        raise ValueError(f"run_episode expects a concrete dummy, got {p2_style}")
    sim.reset(int(seed))
    policy = StickyPolicy(predict, frameskip)
    frames = 0
    while sim.get_state().round_state == RoundState.FIGHTING:
        if max_frames is not None and frames >= max_frames:
            break
        state = sim.get_state()
        sim.step(policy.raw(state), dummy_input(state, 2, p2_style))
        frames += 1
    state = sim.get_state()
    return EpisodeResult(
        round_state=state.round_state.value,
        frames=frames,
        p1_health=state.p1.health,
        p2_health=state.p2.health,
        p2_style=p2_style,
    )


def evaluate(
    predict: PredictFn,
    p2: str,
    episodes: int,
    frameskip: int,
    seed: int,
    max_frames: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    sim = Simulator()
    results = [
        run_episode(
            sim,
            predict,
            pick_style(p2, rng),
            frameskip,
            seed=seed + i,
            max_frames=max_frames,
        )
        for i in range(int(episodes))
    ]
    n = len(results)
    return {
        "p2": p2,
        "n": n,
        "p1_win": sum(1 for r in results if r.round_state == "p1_win"),
        "p2_win": sum(1 for r in results if r.round_state == "p2_win"),
        "draw": sum(1 for r in results if r.round_state == "draw"),
        "incomplete": sum(1 for r in results if r.round_state == "fighting"),
        "mean_p1_health": sum(r.p1_health for r in results) / n,
        "mean_p2_health": sum(r.p2_health for r in results) / n,
        "mean_frames": sum(r.frames for r in results) / n,
        "episodes": results,
    }


def format_summary(stats: dict[str, Any]) -> str:
    return (
        f"p2={stats['p2']}  n={stats['n']}  "
        f"p1_win={stats['p1_win']}  p2_win={stats['p2_win']}  "
        f"draw={stats['draw']}  incomplete={stats['incomplete']}  "
        f"p1_hp={stats['mean_p1_health']:.0f}  p2_hp={stats['mean_p2_health']:.0f}  "
        f"frames={stats['mean_frames']:.0f}"
    )


def watch(
    predict: PredictFn,
    p2: str,
    frameskip: int,
    seed: int,
    episodes: int = 0,
) -> list[EpisodeResult]:
    try:
        import pygame
    except ImportError:
        print(
            "pygame is required to watch or play vs the agent. "
            'Install with: pip install -e "../game[play]"',
            file=sys.stderr,
        )
        sys.exit(1)

    from fightsim.render import render_state

    human = p2 == "human"
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((STAGE_WIDTH_PX, 400))
    pygame.display.set_caption(f"fightml eval — P1 agent vs P2 {p2}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    if human:
        print(
            "You are P2 (red). Arrows move, Num1/2/3 light/heavy/throw, "
            "Num4/5 specials. R reset, Esc quit."
        )

    import numpy as np

    rng = np.random.default_rng(seed)
    sim = Simulator()
    policy = StickyPolicy(predict, frameskip)
    round_seed = int(seed)
    style = "human" if human else pick_style(p2, rng)
    sim.reset(round_seed)
    finished: list[EpisodeResult] = []
    over_frames = 0
    recorded = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                round_seed += 1
                style = "human" if human else pick_style(p2, rng)
                sim.reset(round_seed)
                policy.reset()
                over_frames = 0
                recorded = False

        state = sim.get_state()
        keys = pygame.key.get_pressed()
        if state.round_state == RoundState.FIGHTING:
            p2_raw = _p2_from_keys(keys) if human else dummy_input(state, 2, style)
            sim.step(policy.raw(state), p2_raw)
            recorded = False
            over_frames = 0
        else:
            if not recorded:
                st = sim.get_state()
                finished.append(
                    EpisodeResult(
                        round_state=st.round_state.value,
                        frames=st.frame,
                        p1_health=st.p1.health,
                        p2_health=st.p2.health,
                        p2_style=style,
                    )
                )
                recorded = True
                if episodes > 0 and len(finished) >= episodes:
                    running = False
            elif not human:
                over_frames += 1
                if over_frames >= RESET_PAUSE_FRAMES:
                    round_seed += 1
                    style = pick_style(p2, rng)
                    sim.reset(round_seed)
                    policy.reset()
                    over_frames = 0
                    recorded = False

        render_state(sim.get_state(), screen)
        label = f"P1 agent   P2 {style}   skip={frameskip}   R reset"
        screen.blit(font.render(label, True, (200, 200, 200)), (20, 380))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return finished


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch or score a trained P1 policy vs dummies / a human P2."
    )
    parser.add_argument("model", help="SB3 checkpoint from train --save (zip)")
    parser.add_argument(
        "--p2",
        type=str,
        default="block",
        choices=P2_CHOICES,
        help="Scripted dummy, mix, or human (P2 keyboard)",
    )
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Watch: stop after N finished rounds (0 = until quit). Headless: rounds to run.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print win/health stats; no pygame. Cannot use --p2 human.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of greedy actions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        if args.p2 == "human":
            print("--headless cannot use --p2 human", file=sys.stderr)
            sys.exit(2)
        n = args.episodes if args.episodes > 0 else 20
        predict = load_predict(args.model, deterministic=not args.stochastic)
        stats = evaluate(predict, args.p2, n, args.frameskip, args.seed)
        print(format_summary(stats))
        return

    predict = load_predict(args.model, deterministic=not args.stochastic)
    finished = watch(predict, args.p2, args.frameskip, args.seed, episodes=args.episodes)
    if finished:
        n = len(finished)
        stats = {
            "p2": args.p2,
            "n": n,
            "p1_win": sum(1 for r in finished if r.round_state == "p1_win"),
            "p2_win": sum(1 for r in finished if r.round_state == "p2_win"),
            "draw": sum(1 for r in finished if r.round_state == "draw"),
            "incomplete": 0,
            "mean_p1_health": sum(r.p1_health for r in finished) / n,
            "mean_p2_health": sum(r.p2_health for r in finished) / n,
            "mean_frames": sum(r.frames for r in finished) / n,
        }
        print(format_summary(stats))


if __name__ == "__main__":
    main()
