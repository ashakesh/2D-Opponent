"""Maskable PPO vs dummy mix ∪ frozen pool. Rewards stay in FightEnv."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from fightml.env import make_fight_env
from fightml.paths import resolve_runs_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Maskable PPO (P1) against dummies and/or a frozen pool."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--p2",
        type=str,
        default="mix",
        help="Dummy personalities when not sampling the pool: mix or a single style",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument(
        "--save",
        type=str,
        default="runs/ppo_mix3",
        help="Checkpoint stem. Relative paths land in ml/runs, independent of cwd.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="runs/pool",
        help="Frozen-checkpoint dir (also P2 pool). Relative paths land in ml/runs.",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=100_000,
        help="Save a pool checkpoint every N timesteps (0 disables).",
    )
    parser.add_argument(
        "--opponent-pool-frac",
        type=float,
        default=0.5,
        help="Probability of sampling a frozen zip as P2 when the pool is non-empty.",
    )
    parser.add_argument(
        "--dummy-only",
        action="store_true",
        help="Never sample the pool; dummy personalities only.",
    )
    parser.add_argument("--timeout-lead", type=int, default=2000)
    parser.add_argument("--whiff-penalty", type=float, default=0.02)
    parser.add_argument(
        "--tb",
        type=str,
        default="",
        help="TensorBoard log dir under ml/runs when relative. Off by default.",
    )
    parser.add_argument(
        "--eval-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, print a short per-style headless table (use --no-eval-after to skip).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=2,
        help="Episodes per personality for --eval-after (both orientations).",
    )
    return parser.parse_args(argv)


def make_vec_env(args: argparse.Namespace) -> VecMonitor:
    thunk = make_fight_env(
        dummy_style=args.p2,
        frameskip=args.frameskip,
        random_spawn=True,
        pool_dir=args.pool,
        pool_frac=args.opponent_pool_frac,
        dummy_only=args.dummy_only,
        timeout_lead=args.timeout_lead,
        whiff_penalty=args.whiff_penalty,
    )
    env = DummyVecEnv([thunk for _ in range(int(args.n_envs))])
    env.seed(int(args.seed))
    return VecMonitor(env)


def train(args: argparse.Namespace) -> Path:
    from sb3_contrib import MaskablePPO

    save_path = resolve_runs_path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    args.pool = str(resolve_runs_path(args.pool))

    tb_log: str | None = None
    if args.tb:
        tb_path = resolve_runs_path(args.tb)
        tb_path.mkdir(parents=True, exist_ok=True)
        tb_log = str(tb_path)

    env = make_vec_env(args)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        tensorboard_log=tb_log,
        verbose=1,
    )
    callbacks = []
    if int(args.ckpt_every) > 0:
        pool = Path(args.pool)
        pool.mkdir(parents=True, exist_ok=True)
        freq = max(int(args.ckpt_every) // max(int(args.n_envs), 1), 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=freq,
                save_path=str(pool),
                name_prefix="ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=callbacks or None,
        progress_bar=False,
    )
    model.save(str(save_path))
    env.close()
    return save_path


def _eval_after(path: Path, args: argparse.Namespace) -> None:
    from fightml.eval import evaluate_per_style, format_per_style, load_predict

    predict = load_predict(str(path), deterministic=True)
    table = evaluate_per_style(
        predict,
        episodes_per_style=int(args.eval_episodes),
        frameskip=int(args.frameskip),
        seed=int(args.seed) + 17,
    )
    print(format_per_style(table))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = train(args)
    print(f"saved {path}")
    if args.eval_after:
        _eval_after(path, args)


if __name__ == "__main__":
    main()
