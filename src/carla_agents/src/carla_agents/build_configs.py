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

TRAIN_TIMESTEPS = 200_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
N_TEST_EPISODES = 50
CARLA_MAX_STEPS_BY_SCENARIO_GROUP = {
    "highway": 400,
    "highway_merge": 400,
    "road": 600,
    "intersection": 600,
    "junction": 600,
}

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
            "max_steps": CARLA_MAX_STEPS_BY_SCENARIO_GROUP[
                train_config["scenario_group"]
            ],
            "scenarios_file": str(CARLA_SCENARIOS.relative_to(REPO_ROOT)),
            "sensor_config": str(CARLA_SENSORS.relative_to(REPO_ROOT)),
            "initialize_server": True,
            "offscreen_rendering": True,
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

def carla_algo_configs() -> Dict[str, List[Dict[str, Any]]]:
    # CARLA uses the dictionary observation stream produced by
    # VaeObservationWrapper/RlObservationWrapper, so all policies are
    # MultiInputPolicy. Hyperparameters are adapted from the non-image baselines
    # in config/algo-configs.json / config/rlzoo-algo-hyperparams:
    # DQN and discrete PPO follow CartPole-style discrete control, discrete A2C
    # follows LunarLander-style discrete control, and continuous PPO/A2C/SAC
    # follow BipedalWalker-style continuous control. Changes from those source
    # baselines are commented next to the affected parameters.
    return {
        "discrete": [
            {
                "algo": "dqn",
                "action_space": "Discrete",
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.0023,
                "batch_size": 64,
                "buffer_size": 100_000,
                "learning_starts": 1_000,
                "gamma": 0.99,
                "target_update_interval": 10,
                "train_freq": 256,
                "gradient_steps": 128,
                "exploration_fraction": 0.16,
                "exploration_final_eps": 0.04,
                "policy_kwargs": "dict(net_arch=[256, 256])",
            },
            {
                "algo": "ppo",
                "action_space": "Discrete",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                # CartPole PPO uses n_envs=8 and n_steps=32, i.e. 256 samples
                # per rollout. CARLA is constrained to n_envs=1, so n_steps=256
                # preserves the same rollout batch size.
                "n_steps": 256,
                "batch_size": 256,
                "gae_lambda": 0.8,
                "gamma": 0.98,
                "n_epochs": 20,
                "ent_coef": 0.0,
                "learning_rate": "lin_0.001",
                "clip_range": "lin_0.2",
            },
            {
                "algo": "a2c",
                "action_space": "Discrete",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "gamma": 0.995,
                # LunarLander A2C uses n_envs=8 and n_steps=5, i.e. 40 samples
                # per rollout. CARLA is constrained to n_envs=1, so n_steps=40
                # preserves the same rollout batch size.
                "n_steps": 40,
                "learning_rate": "lin_0.00083",
                "ent_coef": 0.00001,
            },
        ],
        "continuous": [
            {
                "algo": "ppo",
                "action_space": "Continuous",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "n_steps": 2048,
                "batch_size": 64,
                "gae_lambda": 0.95,
                "gamma": 0.999,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "learning_rate": 0.0003,
                "clip_range": 0.18,
            },
            {
                "algo": "a2c",
                "action_space": "Continuous",
                "n_envs": 1,
                "normalize": False,
                "policy": "MultiInputPolicy",
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
                # BipedalWalker A2C uses n_envs=16 and n_steps=8, i.e. 128
                # samples per rollout. CARLA is constrained to n_envs=1, so
                # n_steps=128 preserves the same rollout batch size.
                "n_steps": 128,
                "gae_lambda": 0.9,
                "vf_coef": 0.4,
                "gamma": 0.99,
                "use_rms_prop": True,
                "normalize_advantage": False,
                "learning_rate": "lin_0.00096",
                "use_sde": True,
                "policy_kwargs": "dict(log_std_init=-2, ortho_init=False)",
            },
            {
                "algo": "sac",
                "action_space": "Continuous",
                "normalize": False,
                "policy": "MultiInputPolicy",
                "learning_rate": 0.00073,
                "buffer_size": 300_000,
                "batch_size": 256,
                "ent_coef": "auto",
                "gamma": 0.98,
                "tau": 0.02,
                "train_freq": 64,
                "gradient_steps": 64,
                "learning_starts": 10_000,
                "use_sde": True,
                "policy_kwargs": "dict(log_std_init=-3, net_arch=[400, 300])",
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
            "env_config": deepcopy(env_config),
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
