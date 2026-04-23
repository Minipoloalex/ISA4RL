"""
A test script to visualize the performance of the heuristic parking policy.
"""

import argparse
import time
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401

from methods.utils.metafeature_utils import make_parking_geometric_policy


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "duration": args.duration,
        "parking_spots": args.parking_spots,
        "vehicles_count": args.vehicles_count,
        "add_walls": not args.no_walls,
        "action": {
            "type": "ContinuousAction",
            "steering_range": (-args.steering_range, args.steering_range),
        },
    }
    return config


def run_episode(env: gym.Env, seed: int, sleep_seconds: float) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    policy = make_parking_geometric_policy(env)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    env.render()
    while not (terminated or truncated):
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        env.render()
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    return {
        "return": total_reward,
        "steps": steps,
        "success": bool(info.get("is_success", False)),
        "crashed": bool(info.get("crashed", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the non-learned geometric policy for highway-env parking-v0."
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--parking-spots", type=int, default=14)
    parser.add_argument("--vehicles-count", type=int, default=0)
    parser.add_argument("--no-walls", action="store_true")
    parser.add_argument("--steering-range", type=float, default=0.7853981633974483)
    parser.add_argument("--sleep", type=float, default=0.03)
    args = parser.parse_args(argv)

    env = gym.make("parking-v0", render_mode="human", config=_build_config(args))
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            result = run_episode(env, episode_seed, args.sleep)
            print(
                f"episode={episode} seed={episode_seed} "
                f"return={result['return']:.3f} steps={result['steps']} "
                f"success={result['success']} crashed={result['crashed']} "
                f"truncated={result['truncated']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
