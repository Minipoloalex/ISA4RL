import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import gymnasium as gym

from typing import List, Dict, Any
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

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
    infos.append(info)
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
) -> List[Dict[str, Any]]:
    base_seed = int(1e6)
    episodes_stats: List[Dict[str, Any]] = []
    for i in range(n_episodes):
        cur_seed = base_seed + i
        reward, length, infos = rollout_episode(
            model,
            env,
            env_seed=cur_seed,
            deterministic=deterministic,
        )
        episodes_stats.append(
            {
                "episode": i,
                "reward": reward,
                "length": length,
                "seed": cur_seed,
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
