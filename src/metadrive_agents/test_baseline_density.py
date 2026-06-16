"""Probe MetaDrive IDM baseline behavior across traffic densities.

Run from this directory with:
    uv run test_baseline_density.py
"""

from __future__ import annotations

from panda3d.core import loadPrcFileData
loadPrcFileData("", "notify-level-linmath error")

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.envs.my_metadrive_env import MyMetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from scipy.stats import kurtosis


BASE_SEED = int(1e6)
DEFAULT_DENSITIES = [0, 0.05, 0.5]
DEFAULT_CONFIG_PATH = Path("../../results/metadrive/BqCQXSFQIRMzHjxZXydg/instance_config.json")


@dataclass
class EpisodeResult:
    density: float
    episode: int
    seed: int
    reward: float
    length: int
    crashed: bool
    timeout: bool
    mean_speed: float
    speed_kurtosis: float
    vehicle_count_mean: float
    vehicle_count_max: int
    termination: str
    speeds: list[float]


def load_base_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as file:
        instance_config = json.load(file)

    env_config = copy.deepcopy(instance_config["env_config"]["config"])
    env_config["agent_policy"] = IDMPolicy
    env_config["manual_control"] = False
    env_config["use_render"] = True
    return env_config


def placeholder_action(env: gym.Env) -> Any:
    action_space = env.action_space

    if isinstance(action_space, gym.spaces.Box):
        return np.zeros(action_space.shape, dtype=action_space.dtype)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return np.asarray(action_space.nvec // 2, dtype=action_space.dtype)
    if isinstance(action_space, gym.spaces.Discrete):
        base_env = env.unwrapped
        steering_dim = base_env.config["discrete_steering_dim"]
        throttle_dim = base_env.config["discrete_throttle_dim"]
        action = (throttle_dim // 2) * steering_dim + (steering_dim // 2)
        if action >= action_space.n:
            raise ValueError(f"Invalid placeholder action {action} for {action_space}.")
        return action

    raise ValueError(f"Unsupported action space type: {type(action_space)}")


def get_speed(info: dict[str, Any], env: gym.Env) -> float:
    if "speed" in info:
        return float(info["speed"]) / 3.6

    ego = getattr(env.unwrapped, "vehicle", None)
    if ego is not None and hasattr(ego, "speed"):
        return float(ego.speed)

    raise KeyError("Could not read ego speed from MetaDrive info or vehicle.")


def count_other_vehicles(env: gym.Env) -> int:
    base_env = env.unwrapped
    ego = getattr(base_env, "vehicle", None)
    engine = getattr(base_env, "engine", None)
    traffic_manager = getattr(engine, "traffic_manager", None)
    vehicles = getattr(traffic_manager, "vehicles", None)

    if vehicles is None:
        return 0

    if isinstance(vehicles, dict):
        iterable = vehicles.values()
    else:
        iterable = vehicles

    return sum(1 for vehicle in iterable if vehicle is not ego)


def termination_reason(info: dict[str, Any], terminated: bool, truncated: bool) -> str:
    if truncated:
        return "timeout"
    if not terminated:
        return "running"

    for key in ["crash", "crashed", "arrive_dest", "out_of_road", "max_step"]:
        if bool(info.get(key, False)):
            return key
    return "terminated"


def run_episode(env: gym.Env, density: float, episode: int) -> EpisodeResult:
    seed = BASE_SEED + episode
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    policy = env.unwrapped.engine.get_policy(env.unwrapped.agent.name)
    if not isinstance(policy, IDMPolicy):
        raise RuntimeError(f"Expected IDMPolicy, got {policy.__class__.__name__}.")

    action = placeholder_action(env)
    total_reward = 0.0
    speeds: list[float] = []
    vehicle_counts: list[int] = []
    crashed = False
    timeout = False
    reason = "running"

    for step in range(1, env.unwrapped.config["horizon"] + 1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        crashed = crashed or bool(info.get("crashed", False) or info.get("crash", False))
        timeout = timeout or bool(truncated)
        speeds.append(get_speed(dict(info), env))
        vehicle_counts.append(count_other_vehicles(env))

        if terminated or truncated:
            reason = termination_reason(dict(info), terminated, truncated)
            break

    speed_array = np.asarray(speeds, dtype=float)
    vehicle_count_array = np.asarray(vehicle_counts, dtype=float)
    speed_kurtosis = 0.0
    if speed_array.size >= 2:
        value = float(kurtosis(speed_array, nan_policy="omit"))
        if np.isfinite(value):
            speed_kurtosis = value

    return EpisodeResult(
        density=density,
        episode=episode,
        seed=seed,
        reward=total_reward,
        length=len(speeds),
        crashed=crashed,
        timeout=timeout,
        mean_speed=float(np.mean(speed_array)) if speed_array.size else 0.0,
        speed_kurtosis=speed_kurtosis,
        vehicle_count_mean=float(np.mean(vehicle_count_array)) if vehicle_count_array.size else 0.0,
        vehicle_count_max=int(np.max(vehicle_count_array)) if vehicle_count_array.size else 0,
        termination=reason,
        speeds=speeds,
    )


def summarize(results: list[EpisodeResult]) -> dict[str, float]:
    rewards = np.asarray([result.reward for result in results], dtype=float)
    lengths = np.asarray([result.length for result in results], dtype=float)
    episode_mean_speeds = np.asarray([result.mean_speed for result in results], dtype=float)
    all_speed_kurtosis = np.asarray([result.speed_kurtosis for result in results], dtype=float)
    all_speeds = np.asarray(
        [speed for result in results for speed in result.speeds],
        dtype=float,
    )

    step_speed_kurtosis = 0.0
    if all_speeds.size >= 2:
        value = float(kurtosis(all_speeds, nan_policy="omit"))
        if np.isfinite(value):
            step_speed_kurtosis = value

    return {
        "episodes": float(len(results)),
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "collision_rate": float(np.mean([result.crashed for result in results])),
        "timeout_rate": float(np.mean([result.timeout for result in results])),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "step_speed_mean": float(np.mean(all_speeds)),
        "step_speed_std": float(np.std(all_speeds)),
        "step_speed_kurtosis": step_speed_kurtosis,
        "episode_mean_speed_mean": float(np.mean(episode_mean_speeds)),
        "episode_speed_kurtosis_mean": float(np.mean(all_speed_kurtosis)),
        "episode_speed_kurtosis_values": float(len(set(all_speed_kurtosis.tolist()))),
        "vehicle_count_mean": float(np.mean([result.vehicle_count_mean for result in results])),
        "vehicle_count_max": float(np.max([result.vehicle_count_max for result in results])),
    }


def print_results(density: float, results: list[EpisodeResult]) -> None:
    summary = summarize(results)
    print()
    print(f"traffic_density={density:.2f}")
    for key, value in summary.items():
        print(f"  {key}: {value:.6g}")
    print("  episodes:")
    for result in results:
        print(
            "    "
            f"episode={result.episode:02d} seed={result.seed} "
            f"reward={result.reward:.6f} length={result.length:03d} "
            f"crashed={int(result.crashed)} timeout={int(result.timeout)} "
            f"mean_speed={result.mean_speed:.6f} "
            f"speed_kurtosis={result.speed_kurtosis:.6f} "
            f"vehicles_mean={result.vehicle_count_mean:.3f} "
            f"vehicles_max={result.vehicle_count_max} "
            f"termination={result.termination}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=DEFAULT_DENSITIES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    base_config = load_base_config(args.config)
    print(f"Base config: {args.config}")
    print(f"Map config: {base_config['map_config']}")
    print(f"Episode seeds: {BASE_SEED}..{BASE_SEED + args.episodes - 1}")

    for density in args.densities:
        config = copy.deepcopy(base_config)
        config["traffic_density"] = float(density)
        env = MyMetaDriveEnv(config)
        try:
            results = [
                run_episode(env, float(density), episode)
                for episode in range(args.episodes)
            ]
        finally:
            env.close()
        print_results(float(density), results)


if __name__ == "__main__":
    main()
