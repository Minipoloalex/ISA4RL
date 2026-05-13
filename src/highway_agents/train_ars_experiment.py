"""
Small ARS experiment for continuous highway-env tasks with Box observations.

Run from this folder:
    uv run train_ars_experiment.py --total-timesteps 10000 --progress-bar
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401
from sb3_contrib import ARS
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize


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


def build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    return {"duration": args.duration}


def make_vector_env(env_id: str, env_config: dict[str, Any], seed: int) -> VecEnv:
    return make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"config": env_config, "render_mode": None},
        vec_env_cls=DummyVecEnv,
    )


def maybe_normalize_env(env: VecEnv, training: bool, enabled: bool) -> VecEnv:
    if not enabled:
        return env
    return VecNormalize(env, training=training, norm_obs=True, norm_reward=False)


def validate_spaces(env: VecEnv) -> None:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            f"ARS experiment expects a Box observation space, got {env.observation_space}."
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(f"ARS experiment expects a Box action space, got {env.action_space}.")


def run_visual_test(
    model: ARS,
    env_id: str,
    env_config: dict[str, Any],
    episodes: int,
    seed: int,
    sleep_seconds: float,
    vec_normalize_path: Path | None,
) -> None:
    def make_env() -> gym.Env:
        return Monitor(gym.make(env_id, render_mode="human", config=env_config))

    env = DummyVecEnv([make_env])
    if vec_normalize_path is not None:
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            env.seed(episode_seed)
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            env.render()
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                total_reward += float(rewards[0])
                steps += 1
                done = bool(dones[0])
                info = infos[0]
                env.render()
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)

            crashed = bool(info["crashed"]) if "crashed" in info else None
            truncated = bool(info["TimeLimit.truncated"]) if "TimeLimit.truncated" in info else None
            print(
                f"test_episode={episode} seed={episode_seed} "
                f"return={total_reward:.3f} steps={steps} "
                f"crashed={crashed} truncated={truncated}"
            )
    finally:
        env.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small sb3-contrib ARS agent on a continuous highway-env task."
    )
    parser.add_argument("--env-id", default="racetrack-v0")
    parser.add_argument("--total-timesteps", type=positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results_experiments/ars"))
    parser.add_argument("--eval-freq", type=positive_int, default=2_000)
    parser.add_argument("--n-eval-episodes", type=positive_int, default=3)
    parser.add_argument("--test-episodes", type=non_negative_int, default=1)
    parser.add_argument("--test-sleep", type=float, default=0.03)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    parser.add_argument("--n-delta", type=positive_int, default=8)
    parser.add_argument("--n-top", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--delta-std", type=float, default=0.05)
    parser.add_argument("--n-eval-rollouts-per-delta", type=positive_int, default=1)
    parser.add_argument("--no-normalize-obs", dest="normalize_obs", action="store_false")
    parser.set_defaults(normalize_obs=True)

    parser.add_argument("--duration", type=positive_int, default=60)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser()
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    tensorboard_dir = output_dir / "tensorboard"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_config = build_env_config(args)
    train_env = maybe_normalize_env(
        make_vector_env(args.env_id, env_config, args.seed),
        training=True,
        enabled=args.normalize_obs,
    )
    eval_env = maybe_normalize_env(
        make_vector_env(args.env_id, env_config, args.seed + 100_000),
        training=False,
        enabled=args.normalize_obs,
    )
    validate_spaces(train_env)

    best_model_path = model_dir / "best_model.zip"
    last_model_path = model_dir / "model.zip"
    vec_normalize_path = model_dir / "vec_normalize.pkl"
    metadata_path = output_dir / "training_metadata.json"

    try:
        model = ARS(
            "MlpPolicy",
            train_env,
            n_delta=args.n_delta,
            n_top=args.n_top,
            learning_rate=args.learning_rate,
            delta_std=args.delta_std,
            n_eval_episodes=args.n_eval_rollouts_per_delta,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            verbose=args.verbose,
            device=args.device,
        )
        model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(output_dir),
            eval_freq=args.eval_freq,
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
        if args.normalize_obs:
            train_env.save(str(vec_normalize_path))
        if not best_model_path.is_file():
            model.save(best_model_path)

        metadata = {
            "algo": "ars",
            "env_id": args.env_id,
            "total_timesteps": args.total_timesteps,
            "elapsed_seconds": elapsed_seconds,
            "seed": args.seed,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "vec_normalize_path": str(vec_normalize_path) if args.normalize_obs else None,
            "env_config": env_config,
        }
        with metadata_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(metadata, file_pointer, indent=2)

        print(f"Saved best model to {best_model_path}")
        print(f"Saved last model to {last_model_path}")
        print(f"Saved metadata to {metadata_path}")

        if args.test_episodes > 0:
            best_model = ARS.load(str(best_model_path), device=args.device)
            run_visual_test(
                best_model,
                args.env_id,
                env_config,
                args.test_episodes,
                args.seed + 200_000,
                args.test_sleep,
                vec_normalize_path if args.normalize_obs else None,
            )
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
