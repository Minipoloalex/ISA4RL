"""
Small TQC experiment for continuous highway-env tasks.

Run from this folder:
    uv run train_tqc_experiment.py --total-timesteps 10000 --progress-bar
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


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
    if args.env_id == "parking-v0":
        return {
            "duration": args.duration,
            "vehicles_count": args.vehicles_count,
            "parking_spots": args.parking_spots,
            "add_walls": args.add_walls,
        }
    return {"duration": args.duration}


def make_vector_env(env_id: str, env_config: dict[str, Any], seed: int) -> VecEnv:
    return make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"config": env_config, "render_mode": None},
        vec_env_cls=DummyVecEnv,
    )


def validate_spaces(env: VecEnv, use_her: bool) -> str:
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(f"TQC experiment expects a Box action space, got {env.action_space}.")
    if isinstance(env.observation_space, gym.spaces.Dict):
        return "MultiInputPolicy"
    if use_her:
        raise TypeError(
            f"HER replay expects a dict goal observation space, got {env.observation_space}."
        )
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            f"TQC experiment expects a Box or Dict observation space, got {env.observation_space}."
        )
    return "MlpPolicy"


def make_model(args: argparse.Namespace, train_env: VecEnv, tensorboard_dir: Path) -> TQC:
    policy = validate_spaces(train_env, args.her)
    replay_buffer_class = None
    replay_buffer_kwargs = None
    if args.her:
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = {
            "n_sampled_goal": args.her_n_sampled_goal,
            "goal_selection_strategy": "future",
        }

    return TQC(
        policy,
        train_env,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        top_quantiles_to_drop_per_net=args.top_quantiles_to_drop_per_net,
        policy_kwargs={"net_arch": [256, 256, 256]},
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        verbose=args.verbose,
        device=args.device,
    )


def run_visual_test(
    model: TQC,
    env_id: str,
    env_config: dict[str, Any],
    episodes: int,
    seed: int,
    sleep_seconds: float,
) -> None:
    env = gym.make(env_id, render_mode="human", config=env_config)
    env = Monitor(env)
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

            success = bool(info["is_success"]) if "is_success" in info else None
            crashed = bool(info["crashed"]) if "crashed" in info else None
            print(
                f"test_episode={episode} seed={episode_seed} "
                f"return={total_reward:.3f} steps={steps} "
                f"success={success} crashed={crashed} truncated={truncated}"
            )
    finally:
        env.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small sb3-contrib TQC agent on a continuous highway-env task."
    )
    parser.add_argument("--env-id", default="parking-v0")
    parser.add_argument("--total-timesteps", type=positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results_experiments/tqc"))
    parser.add_argument("--eval-freq", type=positive_int, default=2_000)
    parser.add_argument("--n-eval-episodes", type=positive_int, default=3)
    parser.add_argument("--test-episodes", type=non_negative_int, default=1)
    parser.add_argument("--test-sleep", type=float, default=0.03)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=positive_int, default=100_000)
    parser.add_argument("--learning-starts", type=non_negative_int, default=1_000)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--top-quantiles-to-drop-per-net", type=non_negative_int, default=2)
    parser.add_argument("--no-her", dest="her", action="store_false")
    parser.set_defaults(her=True)
    parser.add_argument("--her-n-sampled-goal", type=positive_int, default=4)

    parser.add_argument("--duration", type=positive_int, default=60)
    parser.add_argument("--vehicles-count", type=non_negative_int, default=0)
    parser.add_argument("--parking-spots", type=non_negative_int, default=10)
    parser.add_argument("--add-walls", dest="add_walls", action="store_true")
    parser.add_argument("--no-add-walls", dest="add_walls", action="store_false")
    parser.set_defaults(add_walls=True)

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
    train_env = make_vector_env(args.env_id, env_config, args.seed)
    eval_env = make_vector_env(args.env_id, env_config, args.seed + 100_000)
    best_model_path = model_dir / "best_model.zip"
    last_model_path = model_dir / "model.zip"
    metadata_path = output_dir / "training_metadata.json"

    try:
        model = make_model(args, train_env, tensorboard_dir)
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
        if not best_model_path.is_file():
            model.save(best_model_path)

        metadata = {
            "algo": "tqc",
            "taxonomy": "model-free, off-policy, actor-critic, value-based distributional critic, maximum-entropy continuous-control RL",
            "env_id": args.env_id,
            "policy": model.policy_class.__name__,
            "total_timesteps": args.total_timesteps,
            "elapsed_seconds": elapsed_seconds,
            "seed": args.seed,
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "env_config": env_config,
            "uses_her_replay_buffer": args.her,
        }
        with metadata_path.open("w", encoding="utf-8") as file_pointer:
            json.dump(metadata, file_pointer, indent=2)

        print(f"Saved best model to {best_model_path}")
        print(f"Saved last model to {last_model_path}")
        print(f"Saved metadata to {metadata_path}")

        if args.test_episodes > 0:
            best_model = TQC.load(str(best_model_path), device=args.device)
            run_visual_test(
                best_model,
                args.env_id,
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
