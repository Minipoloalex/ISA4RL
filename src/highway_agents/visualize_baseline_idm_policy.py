"""
A test script to visualize the baseline IDM policy used by metafeature probes.
"""

import argparse
import time
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from methods.utils.metafeature_utils import (
    ensure_idm_vehicle,
    is_highway_env,
    is_merge_env,
    make_idm_baseline_policy,
)


def _lanes_count(env_id: str, args: argparse.Namespace) -> int:
    if args.lanes_count is not None:
        return args.lanes_count
    if env_id == "highway-fast-v0":
        return 4
    if env_id == "merge-generic-v0":
        return 2
    raise ValueError(f"Unsupported environment for IDM visualization: {env_id}")


def _vehicles_count(env_id: str, args: argparse.Namespace) -> int:
    if args.vehicles_count is not None:
        return args.vehicles_count
    if args.traffic_density is None:
        return 2

    vehicles_count = round(args.traffic_density * _lanes_count(env_id, args) * args.duration)
    return max(0, vehicles_count)


def _build_config(env_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "duration": args.duration,
        "vehicles_count": _vehicles_count(env_id, args),
    }

    if env_id == "highway-fast-v0":
        config["lanes_count"] = _lanes_count(env_id, args)
        if args.traffic_density is not None:
            config["vehicles_density"] = args.traffic_density
    elif env_id == "merge-generic-v0":
        config.update(
            {
                "lanes_count": _lanes_count(env_id, args),
                "length_before_merge": args.length_before_merge,
                "merge_length_converge": args.merge_length_converge,
                "merge_length_parallel": args.merge_length_parallel,
                "length_after_merge": args.length_after_merge,
            }
        )
    else:
        raise ValueError(f"Unsupported environment for IDM visualization: {env_id}")

    return config


def _vehicle_summary(env: gym.Env) -> Dict[str, Any]:
    vehicle = env.unwrapped.vehicle
    return {
        "speed": float(vehicle.speed),
        "target_speed": float(vehicle.target_speed),
        "lane_index": vehicle.lane_index,
        "target_lane_index": vehicle.target_lane_index,
        "crashed": bool(vehicle.crashed),
    }


def run_episode(env: gym.Env, seed: int, sleep_seconds: float, print_every: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    ensure_idm_vehicle(env)

    if not (is_highway_env(env.unwrapped) or is_merge_env(env.unwrapped)):
        raise ValueError(f"Unsupported IDM baseline environment: {env.spec.id}")

    policy = make_idm_baseline_policy(env)
    initial = _vehicle_summary(env)
    total_reward = 0.0
    steps = 0
    speeds = []
    rewards = []
    terminated = False
    truncated = False

    print(
        f"  initial speed={initial['speed']:.2f} target_speed={initial['target_speed']:.2f} "
        f"lane={initial['lane_index']} target_lane={initial['target_lane_index']}"
    )

    env.render()
    while not (terminated or truncated):
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        vehicle = env.unwrapped.vehicle

        total_reward += float(reward)
        steps += 1
        speeds.append(float(vehicle.speed))
        rewards.append(float(reward))

        if print_every > 0 and (steps == 1 or steps % print_every == 0 or terminated or truncated):
            print(
                f"  step={steps:03d} reward={float(reward):.3f} "
                f"speed={float(vehicle.speed):.2f} target_speed={float(vehicle.target_speed):.2f} "
                f"lane={vehicle.lane_index} target_lane={vehicle.target_lane_index} "
                f"crashed={bool(vehicle.crashed)}"
            )

        env.render()
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    final = _vehicle_summary(env)
    return {
        "return": total_reward,
        "steps": steps,
        "mean_reward": float(np.mean(rewards)),
        "mean_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "final_speed": final["speed"],
        "target_speed": final["target_speed"],
        "crashed": final["crashed"],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the baseline IDM controller for highway-fast-v0 or merge-generic-v0."
    )
    parser.add_argument("--env-id", choices=["both", "highway-fast-v0", "merge-generic-v0"], default="both")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--traffic-density", type=float, default=None)
    parser.add_argument("--lanes-count", type=int, default=None)
    parser.add_argument("--length-before-merge", type=float, default=150.0)
    parser.add_argument("--merge-length-converge", type=float, default=80.0)
    parser.add_argument("--merge-length-parallel", type=float, default=80.0)
    parser.add_argument("--length-after-merge", type=float, default=300.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.03)
    args = parser.parse_args(argv)

    env_ids = ["highway-fast-v0", "merge-generic-v0"] if args.env_id == "both" else [args.env_id]
    for env_id in env_ids:
        env_config = _build_config(env_id, args)
        env = gym.make(env_id, render_mode="human", config=env_config)
        try:
            for episode in range(args.episodes):
                episode_seed = args.seed + episode
                density = env_config["vehicles_count"] / _lanes_count(env_id, args) / args.duration
                print(
                    f"episode={episode} env_id={env_id} seed={episode_seed} "
                    f"vehicles_count={env_config['vehicles_count']} traffic_density={density:.4f}"
                )
                result = run_episode(env, episode_seed, args.sleep, args.print_every)
                print(
                    f"episode={episode} env_id={env_id} seed={episode_seed} "
                    f"return={result['return']:.3f} steps={result['steps']} "
                    f"mean_reward={result['mean_reward']:.3f} mean_speed={result['mean_speed']:.2f} "
                    f"max_speed={result['max_speed']:.2f} final_speed={result['final_speed']:.2f} "
                    f"target_speed={result['target_speed']:.2f} crashed={result['crashed']} "
                    f"terminated={result['terminated']} truncated={result['truncated']}"
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
