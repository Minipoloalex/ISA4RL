"""Evaluate a Stable-Baselines3 agent trained with ``train.py``.

The script restores a saved model, rebuilds the matching highway-env instance,
and rolls out multiple evaluation episodes while tracking per-episode rewards
and optional environment-specific metrics (e.g., crashes).  Results are printed
to stdout and can be written to JSON for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install gymnasium") from exc

try:  # ensures highway environments are registered
    import highway_env  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError("highway-env is required: pip install highway-env") from exc


from utils import (
    _json_default,
    ensure_dir,
    set_global_seed,
    ALGORITHM_MAP,
    TRAINING_METADATA_FILE,
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
    deterministic: bool,
    max_steps: Optional[int],
) -> Tuple[float, int, List[Dict[str, Any]]]:
    episode_reward = 0.0
    steps = 0
    infos: List[Dict[str, Any]] = []
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        steps += 1
        infos.append(info)
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break
    return episode_reward, steps, infos


def evaluate(
    model: BaseAlgorithm,
    env: gym.Env,
    n_episodes: int,
    max_steps: Optional[int],
    *,
    deterministic: bool,
    seed: Optional[int],
    **kwargs,
) -> List[Dict[str, Any]]:
    episodes_stats: List[Dict[str, Any]] = []
    for idx in range(n_episodes):
        if seed is not None:
            env.reset(seed=seed + idx)
        reward, length, infos = rollout_episode(
            model,
            env,
            deterministic=deterministic,
            max_steps=max_steps,
        )
        episodes_stats.append(
            {
                "episode": idx,
                "reward": reward,
                "length": length,
                "infos": infos,
            }
        )
    return episodes_stats


def aggregate_metrics(
    per_episode: Iterable[Dict[str, Any]],
) -> Dict[str, Union[float, int]]:
    rewards = np.array([entry["reward"] for entry in per_episode], dtype=np.float64)
    lengths = np.array([entry["length"] for entry in per_episode], dtype=np.int32)
    summary: Dict[str, Union[float, int]] = {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }
    bool_fields = {"crashed", "is_success", "success"}
    for field in bool_fields:
        values = [entry.get(field) for entry in per_episode if field in entry]
        if values:
            summary[f"{field}_rate"] = float(np.mean(values))
    return summary


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

    results = evaluate(model, env, episodes, max_steps, deterministic=True, seed=seed)
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
