import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from carla_agents.gymdrive_adapter import register_carla_env
from methods.metafeatures import _extract_carla_min_ttc


REPO_ROOT = Path(__file__).resolve().parents[2]
CARLA_EVAL_CONFIGS = REPO_ROOT / "config" / "eval-configs" / "carla.json"


def load_eval_configs() -> List[Dict[str, Any]]:
    with CARLA_EVAL_CONFIGS.open("r") as f:
        return json.load(f)


def default_traffic_config_index(configs: List[Dict[str, Any]]) -> int:
    for index, config in enumerate(configs):
        env_config = config["env_config"]["config"]
        if int(env_config["max_traffic_vehicles"]) > 0:
            return index
    raise ValueError("No CARLA eval config with traffic was found.")


def make_zero_action(action_space: gym.Space) -> Any:
    if isinstance(action_space, gym.spaces.Discrete):
        return 0
    if isinstance(action_space, gym.spaces.Box):
        return np.zeros(action_space.shape, dtype=action_space.dtype)
    raise TypeError(f"Unsupported CARLA action space type: {type(action_space)}")


def finite_or_inf(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def live_min_ttc_samples(
    env: gym.Env,
    steps: int,
) -> List[float]:
    samples = []
    action = make_zero_action(env.action_space)
    for step in range(steps):
        traffic_states = env.unwrapped.get_active_traffic_vehicle_states()
        min_ttc = _extract_carla_min_ttc(env)
        samples.append(min_ttc)
        ego_state = env.unwrapped.get_current_vehicle_state()
        print(
            "step=",
            step,
            "traffic=",
            len(traffic_states),
            "ego_speed_m_s=",
            f"{ego_state['speed']:.3f}",
            "min_ttc_s=",
            finite_or_inf(min_ttc),
        )

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print(f"episode ended at step={step}: terminated={terminated} truncated={truncated}")
            break
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live CARLA-GymDrive smoke test for methods.metafeatures "
            "CARLA min TTC extraction."
        )
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help=(
            "Index in config/eval-configs/carla.json. Defaults to the first "
            "config with max_traffic_vehicles > 0."
        ),
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Scenario variant index within the selected config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Maximum number of simulator steps to sample.",
    )
    parser.add_argument(
        "--no-autopilot",
        action="store_true",
        help="Disable CARLA behavior-agent autopilot for the ego vehicle.",
    )
    parser.add_argument(
        "--require-finite",
        action="store_true",
        help="Fail unless at least one sampled min TTC is finite.",
    )
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="Disable VAE observation wrapping for faster local TTC smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_carla_env()

    configs = load_eval_configs()
    config_index = (
        default_traffic_config_index(configs)
        if args.config_index is None
        else args.config_index
    )
    selected = copy.deepcopy(configs[config_index]["env_config"])
    env_config = selected["config"]
    if int(env_config["max_traffic_vehicles"]) <= 0:
        raise ValueError(
            f"Selected config index {config_index} has max_traffic_vehicles="
            f"{env_config['max_traffic_vehicles']}. Choose a traffic config."
        )
    if args.no_vae:
        env_config["use_vae"] = False

    scenario_names = env_config["scenario_names"]
    scenario_name = scenario_names[args.scenario_index]
    print(
        "config_index=",
        config_index,
        "config_id=",
        env_config["config_id"],
        "scenario_name=",
        scenario_name,
        "max_traffic_vehicles=",
        env_config["max_traffic_vehicles"],
        "autopilot=",
        not args.no_autopilot,
    )

    env = gym.make(selected["env_id"], config=env_config)
    try:
        env.unwrapped.enable_autopilot(not args.no_autopilot)
        env.reset(options={"scenario_name": scenario_name})
        traffic_states = env.unwrapped.get_active_traffic_vehicle_states()
        if not traffic_states:
            raise AssertionError(
                "No live CARLA traffic actors were found after reset. "
                "The selected scenario cannot test traffic TTC."
            )

        samples = live_min_ttc_samples(env, args.steps)
        if args.require_finite and not any(math.isfinite(value) for value in samples):
            raise AssertionError("No finite min TTC was observed.")
        print("Live CARLA min TTC smoke test passed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
