"""
Small parking-v0 training script for reward and environment experiments.

Run from this folder:
    uv run train_parking_experiment.py --total-timesteps 300000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


ALGORITHMS: Dict[str, type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ppo": PPO,
    "sac": SAC,
}


def build_env_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "duration": args.duration,
        "vehicles_count": args.vehicles_count,
        "parking_spots": args.parking_spots,
        "add_walls": args.add_walls,
    }


def make_vector_env(
    env_config: Dict[str, Any],
    n_envs: int,
    seed: int,
    vec_env_class: type[DummyVecEnv] | type[SubprocVecEnv],
) -> VecEnv:
    vec_env_kwargs = None
    if vec_env_class is SubprocVecEnv:
        vec_env_kwargs = {"start_method": "spawn"}
    return make_vec_env(
        "parking-v0",
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"config": env_config, "render_mode": None},
        vec_env_cls=vec_env_class,
        vec_env_kwargs=vec_env_kwargs,
    )


def make_model(args: argparse.Namespace, env: VecEnv, model_dir: Path) -> BaseAlgorithm:
    algorithm_class = ALGORITHMS[args.algo]
    common_kwargs: Dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "env": env,
        "seed": args.seed,
        "verbose": args.verbose,
        "tensorboard_log": str(model_dir.parent / "tensorboard"),
        "device": args.device,
    }

    return algorithm_class(**common_kwargs)


def run_visual_test(
    model: BaseAlgorithm,
    env_config: Dict[str, Any],
    episodes: int,
    seed: int,
    sleep_seconds: float,
) -> None:
    env = gym.make("parking-v0", render_mode="human", config=env_config)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            obs, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            env.render()
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                env.render()
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)

            print(
                f"test_episode={episode} seed={episode_seed} "
                f"return={total_reward:.3f} steps={steps} "
                f"success={bool(info['is_success'])} "
                f"crashed={bool(info['crashed'])} truncated={truncated}"
            )
    finally:
        env.close()


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def non_negative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def add_bool_pair(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    destination = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize an SB3 agent in highway-env parking-v0."
    )
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="a2c")
    parser.add_argument("--total-timesteps", type=positive_int, default=300_000)
    parser.add_argument("--n-envs", type=positive_int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/parking_experiment"))
    parser.add_argument("--eval-freq", type=positive_int, default=3_000)
    parser.add_argument("--n-eval-episodes", type=positive_int, default=5)
    parser.add_argument("--test-episodes", type=non_negative_int, default=3)
    parser.add_argument("--test-sleep", type=float, default=0.03)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--dummy-vec-env", action="store_true")

    parser.add_argument("--duration", type=positive_int, default=60)
    parser.add_argument("--vehicles-count", type=non_negative_int, default=0)
    parser.add_argument("--parking-spots", type=non_negative_int, default=10)
    add_bool_pair(parser, "add-walls", True, "Add boundary walls to the parking lot.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser()
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_config = build_env_config(args)
    vec_env_class = DummyVecEnv if args.dummy_vec_env or args.n_envs == 1 else SubprocVecEnv
    train_env = make_vector_env(env_config, args.n_envs, args.seed, vec_env_class)
    eval_env = make_vector_env(env_config, 1, args.seed + 100_000, DummyVecEnv)

    best_model_path = model_dir / "best_model.zip"
    last_model_path = model_dir / "model.zip"
    metadata_path = output_dir / "training_metadata.json"

    try:
        model = make_model(args, train_env, model_dir)
        model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

        eval_freq = max(args.eval_freq // args.n_envs, 1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(output_dir),
            eval_freq=eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=args.verbose,
        )

        start_time = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            progress_bar=args.progress_bar,
        )
        elapsed_seconds = time.perf_counter() - start_time
        model.save(last_model_path)

        if not best_model_path.is_file():
            model.save(best_model_path)
            print(
                "No eval checkpoint was produced before training ended; "
                "saved the final model as best_model.zip."
            )

        metadata = {
            "algo": args.algo,
            "total_timesteps": args.total_timesteps,
            "elapsed_seconds": elapsed_seconds,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "eval_freq": args.eval_freq,
            "eval_freq_sb3_calls": eval_freq,
            "n_eval_episodes": args.n_eval_episodes,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "env_config": env_config,
            "vec_normalize": False,
            "normalization_note": (
                "No VecNormalize is used. parking-v0's KinematicsGoal observation "
                "divides observation, achieved_goal, and desired_goal by the configured scales."
            ),
        }
        with metadata_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(metadata, file_pointer, indent=2)

        print(f"Saved best model to {best_model_path}")
        print(f"Saved last model to {last_model_path}")
        print(f"Saved metadata to {metadata_path}")

        if args.test_episodes > 0:
            print(f"Loading best model from {best_model_path} for rendered testing.")
            best_model = ALGORITHMS[args.algo].load(str(best_model_path), device=args.device)
            run_visual_test(
                best_model,
                env_config,
                args.test_episodes,
                args.seed + 200_000,
                args.test_sleep,
            )
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
