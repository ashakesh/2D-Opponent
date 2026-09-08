"""PPO vs a fightsim dummy. Frameskip and rewards stay in FightEnv."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from fightml.env import make_fight_env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO (P1) against a fightsim dummy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument(
        "--p2",
        type=str,
        default="block",
        help="Dummy style: idle, block, mash, crouch_block, or mix",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--save", type=str, default="runs/ppo_p1")
    parser.add_argument(
        "--tb",
        type=str,
        default="",
        help="TensorBoard log dir. Off by default; tensorboard is not a required package.",
    )
    return parser.parse_args(argv)


def make_vec_env(p2: str, n_envs: int, frameskip: int, seed: int) -> VecMonitor:
    thunk = make_fight_env(dummy_style=p2, frameskip=frameskip)
    env = DummyVecEnv([thunk for _ in range(n_envs)])
    env.seed(seed)
    return VecMonitor(env)


def train(args: argparse.Namespace) -> Path:
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tb_log: str | None = None
    if args.tb:
        tb_path = Path(args.tb)
        tb_path.mkdir(parents=True, exist_ok=True)
        tb_log = str(tb_path)

    env = make_vec_env(args.p2, args.n_envs, args.frameskip, args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        tensorboard_log=tb_log,
        verbose=1,
    )
    model.learn(total_timesteps=int(args.timesteps), progress_bar=False)
    model.save(str(save_path))
    env.close()
    return save_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = train(args)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
