import itertools
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common.file_utils import EVAL_CONFIGS_PATH, TRAIN_CONFIGS_PATH, read_json, save_json

from carla_agents.gymdrive_adapter import ENV_ID


REPO_ROOT = Path(__file__).resolve().parents[4]
CARLA_GYMDRIVE_ROOT = REPO_ROOT / "CARLA-GymDrive"
CARLA_TRAIN_CONFIGS = CARLA_GYMDRIVE_ROOT / "src" / "config" / "isa_carla_train_configs.json"
CARLA_SCENARIOS = CARLA_GYMDRIVE_ROOT / "src" / "config" / "isa_carla_scenarios.json"
CARLA_SENSORS = CARLA_GYMDRIVE_ROOT / "src" / "config" / "vae_rgb_sensors.json"
CARLA_ALGO_CONFIGS = REPO_ROOT / "config" / "carla-algo-configs.json"

TRAIN_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5
N_TEST_EPISODES = 50
TIME_LIMIT = 60


def carla_env_config(train_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "env_id": ENV_ID,
        "train_timesteps": TRAIN_TIMESTEPS,
        "eval_freq": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
        "n_test_episodes": N_TEST_EPISODES,
        "config": {
            "config_id": train_config["config_id"],
            "scenario_group": train_config["scenario_group"],
            "source_trigger_type": train_config["source_trigger_type"],
            "situation": train_config["situation"],
            "weather_condition": train_config["weather_condition"],
            "max_traffic_vehicles": train_config["max_traffic_vehicles"],
            "action_space": train_config["action_space"],
            "continuous_actions": train_config["continuous_actions"],
            "scenario_names": list(train_config["scenario_names"]),
            "scenario_count": train_config["scenario_count"],
            "max_steps": 400,
            "scenarios_file": str(CARLA_SCENARIOS.relative_to(REPO_ROOT)),
            "sensor_config": str(CARLA_SENSORS.relative_to(REPO_ROOT)),
            "time_limit": TIME_LIMIT,
            "initialize_server": False,
            "random_weather": False,
            "random_traffic": False,
            "synchronous_mode": True,
            "show_sensor_data": False,
            "has_traffic": True,
            "autopilot": False,
            "verbose": False,
            "use_vae": True,
            "vae_model": "vae_64",
            "vae_root": "CARLA-GymDrive/vae_repo/vae/log_dir",
            "vae_device": "auto",
            "keep_rgb": False,
            "sample_latent": False,
        },
    }

# TODO: maybe use in metafeatures.py if estimating is necessary for CARLA env
# def carla_lane_count_estimate(train_config: Dict[str, Any]) -> int:
#     scenario_group = train_config["scenario_group"]
#     if scenario_group == "road":
#         return 1
#     if scenario_group in ("junction", "highway_merge"):
#         return 2
#     raise ValueError(f"Unknown CARLA scenario group: {scenario_group}")


def carla_algo_configs() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "discrete": [
            {
                "algo": "dqn",
                "action_space": "Discrete",
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0001,
                "buffer_size": 10_000,
                "learning_starts": 500,
                "batch_size": 32,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 2_000,
                "exploration_fraction": 0.1,
                "exploration_final_eps": 0.01,
            },
            {
                "algo": "ppo",
                "action_space": "Discrete",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0003,
                "n_steps": 512,
                "batch_size": 64,
                "gamma": 0.99,
            },
            {
                "algo": "a2c",
                "action_space": "Discrete",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0007,
                "n_steps": 5,
                "gamma": 0.99,
            },
        ],
        "continuous": [
            {
                "algo": "ppo",
                "action_space": "Continuous",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0003,
                "n_steps": 512,
                "batch_size": 64,
                "gamma": 0.99,
            },
            {
                "algo": "a2c",
                "action_space": "Continuous",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0007,
                "n_steps": 5,
                "gamma": 0.99,
            },
            {
                "algo": "sac",
                "action_space": "Continuous",
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0003,
                "buffer_size": 50_000,
                "learning_starts": 1_000,
                "batch_size": 256,
                "gamma": 0.99,
            },
        ],
    }


def build_all_configs() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_configs = read_json(CARLA_TRAIN_CONFIGS)
    algo_configs = carla_algo_configs()
    save_json(CARLA_ALGO_CONFIGS, algo_configs)

    env_configs = [carla_env_config(config) for config in source_configs.values()]
    train_configs = []
    for env_config, algo_config in itertools.product(env_configs, sum(algo_configs.values(), [])):
        action_space = env_config["config"]["action_space"]
        if algo_config["action_space"].lower() != action_space:
            continue
        train_configs.append(
            {
                "env_config": deepcopy(env_config),
                "algo_config": deepcopy(algo_config),
                "timestamp": time.time_ns(),
            }
        )

    eval_configs = [
        {
            "env_config": env_config,
            "timestamp": time.time_ns(),
        }
        for env_config in env_configs
    ]
    return train_configs, eval_configs


def main() -> None:
    train_configs, eval_configs = build_all_configs()
    save_json(TRAIN_CONFIGS_PATH("carla"), train_configs)
    save_json(EVAL_CONFIGS_PATH("carla"), eval_configs)
    print(f"CARLA train configs: {len(train_configs)}")
    print(f"CARLA eval configs: {len(eval_configs)}")


if __name__ == "__main__":
    main()
