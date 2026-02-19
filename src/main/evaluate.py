import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import gymnasium as gym
import highway_env

from src.common.file_utils import _json_default, ensure_dir, TRAINING_METADATA_FILE
from utils import (
    ensure_dir,
    set_global_seed,
    ALGORITHM_MAP,
    build_env,
    load_model,
    load_training_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stable-Baselines3 policy on the highway environment."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Directory with training artifacts (expects training_metadata.json).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to rollout.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to use deterministic actions during evaluation.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment while evaluating. May slow down execution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base random seed applied to the evaluation environment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cap on steps per episode to avoid extremely long rollouts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file where evaluation metrics will be stored.",
    )
    return parser.parse_args()


def rollout_episode(
    model: BaseAlgorithm,
    env: gym.Env,
    *,
    env_seed: int,
    deterministic: bool,
) -> Tuple[float, int, List[Dict[str, Any]]]:
    assert(env_seed >= int(1e6))
    episode_reward = 0.0
    steps = 0
    infos: List[Dict[str, Any]] = []
    obs, info = env.reset(seed=env_seed)
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        steps += 1
        infos.append(info)
        if terminated or truncated:
            break
    return episode_reward, steps, infos

def evaluate(
    model: BaseAlgorithm,
    env: gym.Env,
    n_episodes: int,
    *,
    deterministic: bool,
    env_seed: int,
    **kwargs,
) -> List[Dict[str, Any]]:
    episodes_stats: List[Dict[str, Any]] = []
    for idx in range(n_episodes):
        env.reset(seed=env_seed)
        reward, length, infos = rollout_episode(
            model,
            env,
            env_seed=env_seed,
            deterministic=deterministic,
        )
        episodes_stats.append(
            {
                "episode": idx,
                "reward": reward,
                "length": length,
                "seed": env_seed,
                "infos": infos,
            }
        )
    return episodes_stats

def show_eval_results(eval_results: List[Dict[str, Any]]):
    if not eval_results:
        print("No evaluation results to display.")
        return

    episode_count = len(eval_results)
    rewards: List[float] = []
    lengths: List[float] = []
    speeds: List[float] = []
    crashes = 0

    for entry in eval_results:
        rewards.append(entry["reward"])
        lengths.append(entry["length"])
        infos = entry["infos"]
        for info in infos:
            speed = info["speed"]
            speeds.append(speed)
            if info["crashed"]:
                crashes += 1

    def format_stats(values: List[float]) -> str:
        if not values:
            return "n/a"
        mean = sum(values) / len(values)
        return f"mean={mean:.2f}, min={min(values):.2f}, max={max(values):.2f}"

    print(f"Evaluated {episode_count} episodes")
    print(f"Reward stats:\t{format_stats(rewards)}")
    print(f"Length stats:\t{format_stats(lengths)}")
    print(f"Speed  stats:\t{format_stats(speeds)}")
    print(f"Crashes observed: {crashes}")

    sample_count = min(3, episode_count)
    print("Sample episodes:")
    for entry in eval_results[:sample_count]:
        print(
            f"  episode={entry["episode"]}, reward={entry["reward"]}, length={entry["length"]}, seed={entry["seed"]}"
        )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    metadata = load_training_metadata(run_dir)

    if args.seed is not None:
        set_global_seed(args.seed)

    model_path = metadata["model_path"]
    algo_name = metadata["model_class"]
    env_id = metadata["env_id"]
    env_config = metadata["env_config"]

    model = load_model(model_path, algo_name)
    env = build_env(env_id, env_config)
    episodes = args.episodes
    seed = args.seed
    max_steps = args.max_steps

    results = evaluate(
        model,
        env,
        episodes,
        deterministic=args.deterministic,
        env_seed=seed,
    )
    env.close()

    summary = aggregate_metrics(results)

    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  w{key}: {value}")

    if args.output:
        ensure_dir(args.output.expanduser().resolve().parent)
        payload = {
            "summary": summary,
            "episodes": results,
            "model_path": str(model_path),
            "env_id": env_id,
            "deterministic": args.deterministic,
        }
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
