"""
A test script to visualize the heuristic lane-keeping policy.
"""

import argparse
import time
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from methods.utils.metafeature_utils import make_lane_keeping_observation_policy


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "duration": args.duration,
        "state_noise": args.state_noise,
        "derivative_noise": args.derivative_noise,
        "action": {
            "type": "ContinuousAction",
            "longitudinal": False,
            "lateral": True,
            "dynamical": True,
            "steering_range": (-args.steering_range, args.steering_range),
        },
    }
    return config


def _observed_tracking_errors(obs: Dict[str, np.ndarray]) -> Dict[str, float]:
    state = np.asarray(obs["state"], dtype=float).reshape(-1)
    reference_state = np.asarray(obs["reference_state"], dtype=float).reshape(-1)
    return {
        "lateral_error": float(state[0] - reference_state[0]),
        "heading_error": float(state[1] - reference_state[1]),
    }


def run_episode(env: gym.Env, seed: int, sleep_seconds: float) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    policy = make_lane_keeping_observation_policy(env)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    abs_lateral_errors = []
    abs_heading_errors = []
    abs_actions = []

    env.render()
    while not (terminated or truncated):
        errors = _observed_tracking_errors(obs)
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        abs_lateral_errors.append(abs(errors["lateral_error"]))
        abs_heading_errors.append(abs(errors["heading_error"]))
        abs_actions.append(abs(float(action[0])))

        env.render()
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    return {
        "return": total_reward,
        "steps": steps,
        "mean_abs_lateral_error": float(np.mean(abs_lateral_errors)),
        "mean_abs_heading_error": float(np.mean(abs_heading_errors)),
        "mean_abs_action": float(np.mean(abs_actions)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the observation-based heuristic policy for highway-env lane-keeping-v0."
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--state-noise", type=float, default=0.05)
    parser.add_argument("--derivative-noise", type=float, default=0.05)
    parser.add_argument("--steering-range", type=float, default=0.3490658503988659)
    parser.add_argument("--sleep", type=float, default=0.03)
    args = parser.parse_args(argv)

    env = gym.make("lane-keeping-v0", render_mode="human", config=_build_config(args))
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            result = run_episode(env, episode_seed, args.sleep)
            print(
                f"episode={episode} seed={episode_seed} "
                f"return={result['return']:.3f} steps={result['steps']} "
                f"mean_abs_lateral_error={result['mean_abs_lateral_error']:.3f} "
                f"mean_abs_heading_error={result['mean_abs_heading_error']:.3f} "
                f"mean_abs_action={result['mean_abs_action']:.3f} "
                f"truncated={result['truncated']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
