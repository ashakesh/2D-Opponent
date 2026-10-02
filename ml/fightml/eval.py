"""Qualitative eval: the checkpoint is always P2. --p1 selects dummy or human.

Training never imports pygame. watch() is the only pygame path.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable

from fightsim.api import Simulator
from fightsim.config import FPS
from fightsim.dummy import dummy_input
from fightsim.input import RawInput
from fightsim.state import RoundState

from fightml.env import DUMMY_STYLES, PERSONALITY_STYLES, action_to_raw
from fightml.obs import flatten_state
from fightml.paths import resolve_artifact
from fightml.spawn import place_fighters

PredictFn = Callable[[Any], int]
P1_CHOICES = DUMMY_STYLES + ("mix", "human")
RESET_PAUSE_FRAMES = int(1.5 * FPS)


@dataclass
class EpisodeResult:
    round_state: str
    frames: int
    p1_health: int
    p2_health: int
    p1_style: str
    p1_on_left: bool = True


class StickyPolicy:
    """Hold one semantic action for `frameskip` sim frames. Default view is P2 (eval)."""

    def __init__(self, predict: PredictFn, frameskip: int, view_pid: int = 2) -> None:
        self.predict = predict
        self.frameskip = max(1, int(frameskip))
        self.view_pid = int(view_pid)
        self._raw = RawInput()
        self._left = 0

    def reset(self) -> None:
        self._raw = RawInput()
        self._left = 0

    def raw(self, state) -> RawInput:
        if state.round_state != RoundState.FIGHTING:
            return RawInput()
        if self._left <= 0:
            action = int(self.predict(flatten_state(state, view_pid=self.view_pid)))
            fighter = state.p1 if self.view_pid == 1 else state.p2
            self._raw = action_to_raw(action, fighter.facing)
            self._left = self.frameskip
        self._left -= 1
        return self._raw


def _load_model(path: str):
    try:
        from sb3_contrib import MaskablePPO

        try:
            return MaskablePPO.load(path)
        except Exception:
            pass
    except ImportError:
        pass
    from stable_baselines3 import PPO

    return PPO.load(path)


def load_predict(path: str, deterministic: bool = True) -> PredictFn:
    model = _load_model(str(resolve_artifact(path)))

    def predict(obs) -> int:
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    return predict


def pick_style(p1: str, rng) -> str:
    if p1 == "human":
        raise ValueError("human is not a scripted dummy style")
    if p1 == "mix":
        return str(rng.choice(PERSONALITY_STYLES))
    if p1 in DUMMY_STYLES:
        return p1
    raise ValueError(f"unknown dummy style: {p1}")


def _summarize(results: list[EpisodeResult], p1: str) -> dict[str, Any]:
    n = len(results)
    p1_win = sum(1 for r in results if r.round_state == "p1_win") if n else 0
    agent_win = sum(1 for r in results if r.round_state == "p2_win") if n else 0
    return {
        "p1": p1,
        "n": n,
        "p1_win": p1_win,
        "agent_win": agent_win,
        "p2_win": agent_win,
        "draw": sum(1 for r in results if r.round_state == "draw") if n else 0,
        "incomplete": sum(1 for r in results if r.round_state == "fighting") if n else 0,
        "mean_p1_health": (sum(r.p1_health for r in results) / n) if n else 0.0,
        "mean_p2_health": (sum(r.p2_health for r in results) / n) if n else 0.0,
        "mean_frames": (sum(r.frames for r in results) / n) if n else 0.0,
        "episodes": results,
    }


def run_episode(
    sim: Simulator,
    predict: PredictFn,
    p1_style: str,
    frameskip: int,
    seed: int,
    max_frames: int | None = None,
    p1_on_left: bool = True,
) -> EpisodeResult:
    """Headless dummy/human-side P1 vs agent P2. Does not import pygame."""
    if p1_style not in DUMMY_STYLES:
        raise ValueError(f"run_episode expects a concrete dummy, got {p1_style}")
    sim.reset(int(seed))
    place_fighters(sim.get_state(), p1_on_left=p1_on_left)
    policy = StickyPolicy(predict, frameskip, view_pid=2)
    frames = 0
    while sim.get_state().round_state == RoundState.FIGHTING:
        if max_frames is not None and frames >= max_frames:
            break
        state = sim.get_state()
        sim.step(dummy_input(state, 1, p1_style), policy.raw(state))
        frames += 1
    state = sim.get_state()
    return EpisodeResult(
        round_state=state.round_state.value,
        frames=frames,
        p1_health=state.p1.health,
        p2_health=state.p2.health,
        p1_style=p1_style,
        p1_on_left=p1_on_left,
    )


def evaluate(
    predict: PredictFn,
    p1: str,
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
            pick_style(p1, rng),
            frameskip,
            seed=seed + i,
            max_frames=max_frames,
            p1_on_left=(i % 2 == 0),
        )
        for i in range(int(episodes))
    ]
    return _summarize(results, p1)


def evaluate_per_style(
    predict: PredictFn,
    episodes_per_style: int,
    frameskip: int,
    seed: int,
    max_frames: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Score each personality as P1, half the episodes with P1 on the right."""
    sim = Simulator()
    n = max(1, int(episodes_per_style))
    by_style: dict[str, dict[str, Any]] = {}
    all_eps: list[EpisodeResult] = []
    for si, style in enumerate(PERSONALITY_STYLES):
        results: list[EpisodeResult] = []
        for j in range(n):
            results.append(
                run_episode(
                    sim,
                    predict,
                    style,
                    frameskip,
                    seed=seed + si * 100 + j,
                    max_frames=max_frames,
                    p1_on_left=(j % 2 == 0),
                )
            )
        by_style[style] = _summarize(results, style)
        all_eps.extend(results)
    by_style["total"] = _summarize(all_eps, "mix")
    return by_style


def format_summary(stats: dict[str, Any]) -> str:
    return (
        f"p1={stats['p1']}  n={stats['n']}  "
        f"agent_win={stats['agent_win']}  p1_win={stats['p1_win']}  "
        f"draw={stats['draw']}  incomplete={stats['incomplete']}  "
        f"p1_hp={stats['mean_p1_health']:.0f}  p2_hp={stats['mean_p2_health']:.0f}  "
        f"frames={stats['mean_frames']:.0f}"
    )


def format_per_style(by_style: dict[str, dict[str, Any]]) -> str:
    lines = [format_summary(by_style[s]) for s in PERSONALITY_STYLES if s in by_style]
    if "total" in by_style:
        lines.append(format_summary(by_style["total"]))
    return "\n".join(lines)


def watch(
    predict: PredictFn,
    p1: str,
    frameskip: int,
    seed: int,
    episodes: int = 0,
    box: str = "yes",
    native: bool = False,
    fullscreen: bool = False,
) -> list[EpisodeResult]:
    from fightsim.render import render_state
    from fightsim.view import Display, p1_from_keys, require_pygame

    require_pygame(
        "pygame is required to watch or play vs the agent. "
        'Install with: pip install -e "../game[play]"'
    )

    human = p1 == "human"
    display = Display(
        caption=f"fightml eval — P1 {p1} vs P2 agent",
        native=native,
        fullscreen=fullscreen,
    )

    if human:
        print(
            "You are P1 (blue). WASD move, J/K/L light/heavy/throw, "
            "I/O/P specials. R reset, Esc quit. Agent is P2 (red)."
        )

    import numpy as np

    rng = np.random.default_rng(seed)
    sim = Simulator()
    policy = StickyPolicy(predict, frameskip, view_pid=2)
    round_seed = int(seed)
    style = "human" if human else pick_style(p1, rng)
    sim.reset(round_seed)
    finished: list[EpisodeResult] = []
    over_frames = 0
    recorded = False
    running = True

    while running:
        frame = display.pump()
        if frame.quit:
            running = False
            continue
        if frame.reset_pressed:
            round_seed += 1
            style = "human" if human else pick_style(p1, rng)
            sim.reset(round_seed)
            policy.reset()
            over_frames = 0
            recorded = False

        state = sim.get_state()
        keys = frame.keys
        if state.round_state == RoundState.FIGHTING:
            p1_raw = p1_from_keys(keys) if human else dummy_input(state, 1, style)
            sim.step(p1_raw, policy.raw(state))
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
                        p1_style=style,
                    )
                )
                recorded = True
                if episodes > 0 and len(finished) >= episodes:
                    running = False
            elif not human:
                over_frames += 1
                if over_frames >= RESET_PAUSE_FRAMES:
                    round_seed += 1
                    style = pick_style(p1, rng)
                    sim.reset(round_seed)
                    policy.reset()
                    over_frames = 0
                    recorded = False

        render_state(sim.get_state(), display.logical, p2_label="agent", box=box)
        overlay = None
        if box != "no":
            overlay = f"P1 {style}   P2 agent   skip={frameskip}   R reset"
        display.present(overlay)

    display.close()
    return finished


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch or score a trained policy as P2 vs a dummy / human P1."
    )
    parser.add_argument(
        "model",
        help="SB3 checkpoint from train --save (zip). Relative paths resolve under ml/runs.",
    )
    parser.add_argument(
        "--p1",
        type=str,
        default="block",
        choices=P1_CHOICES,
        help="Scripted dummy, mix, or human (P1 keyboard: WASD+JKL)",
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
        help="Print win/health stats; no pygame. Cannot use --p1 human.",
    )
    parser.add_argument(
        "--per-style",
        action="store_true",
        help="Headless: run episodes_per_style vs each personality as P1 (both orientations).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of greedy actions.",
    )
    parser.add_argument("--box", default="yes", choices=("yes", "no", "maybe"))
    from fightsim.view import add_window_args

    add_window_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        if args.p1 == "human":
            print("--headless cannot use --p1 human", file=sys.stderr)
            sys.exit(2)
        predict = load_predict(args.model, deterministic=not args.stochastic)
        if args.per_style:
            n = args.episodes if args.episodes > 0 else 10
            table = evaluate_per_style(predict, n, args.frameskip, args.seed)
            print(format_per_style(table))
            return
        n = args.episodes if args.episodes > 0 else 20
        stats = evaluate(predict, args.p1, n, args.frameskip, args.seed)
        print(format_summary(stats))
        return

    predict = load_predict(args.model, deterministic=not args.stochastic)
    finished = watch(
        predict,
        args.p1,
        args.frameskip,
        args.seed,
        episodes=args.episodes,
        box=args.box,
        native=args.native,
        fullscreen=args.fullscreen,
    )
    if finished:
        print(format_summary(_summarize(finished, args.p1)))


if __name__ == "__main__":
    main()
