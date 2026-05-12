"""
Train one highway-env configuration with multiple seeds and compare performance.

Run from this folder:
    uv run train_seed_consistency.py
"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import highway_env  # noqa: F401
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from common.file_utils import BEST_MODEL_FILE, MODEL_FILE, RESULTS_TRAIN_METADATA_PATH, save_json
from methods.evaluate import evaluate
from methods.train import train
from methods.utils.sb3_utils import (
    _parse_policy_kwargs,
    _resolve_schedule_placeholders,
    make_model_helper,
    make_vec_env_helper,
    map_algo_name_to_class,
)


EXPERIMENT_NAME = "exit_a2c_seed_consistency"
RESULTS_ROOT = Path(__file__).resolve().parent / "results_experiments" / EXPERIMENT_NAME
SEEDS = [0, 1, 2, 3, 4]
PROGRESS_BAR = True

CONFIG: dict[str, Any] = {
    "env_config": {
        "env_id": "exit-v0",
        "config": {
            "high_speed_reward": 0.01,
            "exit_position": 500,
            "exit_length": 150,
            "road_length": 1150,
            "duration": 26,
            "lanes_count": 3,
            "vehicles_density": 1.0
        },
        "train_timesteps": 100_000,
        "eval_freq": 1_000,
        "n_eval_episodes": 10,
        "n_test_episodes": 50,
    },
    "obs_config": {
        "type": "ExitObservation",
        "vehicles_count": 15,
        "features": [
            "presence",
            "x",
            "y",
            "vx",
            "vy",
            "cos_h",
            "sin_h",
        ],
        "clip": False,
    },
    "algo_config": {
      "algo": "a2c",
      "n_envs": 8,
      "policy": "MlpPolicy",
      "ent_coef": 0.0
    },
}

def build_env_kwargs() -> dict[str, Any]:
    env_config = deepcopy(CONFIG["env_config"]["config"])
    env_config["observation"] = deepcopy(CONFIG["obs_config"])
    return env_config


def build_policy_params() -> tuple[str, dict[str, Any], int]:
    algo_config = deepcopy(CONFIG["algo_config"])
    algo_name = algo_config.pop("algo")
    n_envs = algo_config.pop("n_envs")

    for key in ["action_space", "env_wrapper", "frame_stack", "normalize", "id"]:
        algo_config.pop(key, None)

    for param, value in algo_config.items():
        algo_config[param] = _resolve_schedule_placeholders(value)

    if "policy_kwargs" in algo_config and algo_config["policy_kwargs"] is not None:
        algo_config["policy_kwargs"] = _parse_policy_kwargs(algo_config["policy_kwargs"])

    return algo_name, algo_config, n_envs


def make_train_env(env_kwargs: dict[str, Any], n_envs: int, output_dir: Path) -> VecEnv:
    train_vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    train_vec_env_kwargs = {"start_method": "spawn"} if train_vec_env_cls is SubprocVecEnv else None
    return make_vec_env_helper(
        CONFIG["env_config"]["env_id"],
        env_kwargs,
        n_envs,
        train_vec_env_cls,
        train_vec_env_kwargs,
        monitor_dir=str(output_dir),
    )


def make_eval_env(env_kwargs: dict[str, Any], output_dir: Path) -> VecEnv:
    return make_vec_env_helper(
        CONFIG["env_config"]["env_id"],
        env_kwargs,
        1,
        DummyVecEnv,
        None,
        monitor_dir=str(output_dir),
    )


def summarize_eval_results(eval_results: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(entry["reward"]) for entry in eval_results]
    lengths = [int(entry["length"]) for entry in eval_results]
    return {
        "reward_mean": mean(rewards),
        "reward_std": stdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "length_mean": mean(lengths),
        "length_std": stdev(lengths) if len(lengths) > 1 else 0.0,
    }


def train_and_evaluate_seed(seed: int) -> dict[str, Any]:
    output_dir = RESULTS_ROOT / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = RESULTS_TRAIN_METADATA_PATH(output_dir)
    if metadata_path.exists():
        raise FileExistsError(f"Seed output folder already contains a trained run: {output_dir}")

    save_json(output_dir / "experiment_config.json", CONFIG)
    env_kwargs = build_env_kwargs()
    algo_name, policy_params, n_envs = build_policy_params()
    algo_cls = map_algo_name_to_class(algo_name)
    eval_freq = CONFIG["env_config"]["eval_freq"] // n_envs

    train_env = None
    eval_env = None
    model = None
    try:
        train_env = make_train_env(env_kwargs, n_envs, output_dir)
        eval_env = make_eval_env(env_kwargs, output_dir)
        model = make_model_helper(
            train_env,
            algo_cls=algo_cls,
            folder_name=str(output_dir),
            model_path=output_dir / "models" / MODEL_FILE,
            policy_params=policy_params,
            device="cpu",
        )
        train(
            env=train_env,
            model=model,
            timesteps=CONFIG["env_config"]["train_timesteps"],
            folder_name=str(output_dir),
            eval_env=eval_env,
            n_eval_episodes=CONFIG["env_config"]["n_eval_episodes"],
            eval_freq=eval_freq,
            seed=seed,
            progress_bar=PROGRESS_BAR,
        )

        train_env.close()
        train_env = None
        eval_env.close()
        eval_env = None
        del model
        model = None

        eval_env = make_eval_env(env_kwargs, output_dir)
        model = make_model_helper(
            eval_env,
            algo_cls=algo_cls,
            folder_name=str(output_dir),
            model_path=output_dir / "models" / BEST_MODEL_FILE,
            policy_params=policy_params,
            device="cpu",
        )
        eval_results = evaluate(
            model=model,
            env=eval_env,
            n_episodes=CONFIG["env_config"]["n_test_episodes"],
            deterministic=True,
        )
        eval_results_path = output_dir / "eval_results.json"
        save_json(eval_results_path, eval_results)

        stats = summarize_eval_results(eval_results)
        return {
            "seed": seed,
            "train_folder_path": str(output_dir),
            "eval_results_path": str(eval_results_path),
            **stats,
        }
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()
        del model
        del eval_env
        del train_env


def save_summary(seed_summaries: list[dict[str, Any]]) -> None:
    rewards = [entry["reward_mean"] for entry in seed_summaries]
    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "seeds": SEEDS,
        "config": CONFIG,
        "seed_summaries": seed_summaries,
        "across_seed_reward_mean": mean(rewards),
        "across_seed_reward_std": stdev(rewards) if len(rewards) > 1 else 0.0,
        "across_seed_reward_min": min(rewards),
        "across_seed_reward_max": max(rewards),
    }
    save_json(RESULTS_ROOT / "summary.json", aggregate)

    with (RESULTS_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as file_pointer:
        writer = csv.DictWriter(file_pointer, fieldnames=list(seed_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(seed_summaries)

    np.savez(
        RESULTS_ROOT / "summary_rewards.npz",
        seeds=np.array([entry["seed"] for entry in seed_summaries]),
        reward_means=np.array(rewards),
        reward_stds=np.array([entry["reward_std"] for entry in seed_summaries]),
    )


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    seed_summaries = []
    for seed in SEEDS:
        print(f"Training seed {seed}")
        summary = train_and_evaluate_seed(seed)
        seed_summaries.append(summary)
        print(
            f"seed={seed} reward_mean={summary['reward_mean']:.3f} "
            f"reward_std={summary['reward_std']:.3f}"
        )

    save_summary(seed_summaries)
    print(f"Saved aggregate results to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
