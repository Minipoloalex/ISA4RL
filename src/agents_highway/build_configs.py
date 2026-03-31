import numpy as np
import matplotlib.pyplot as plt
import yaml

import itertools
from typing import List, Dict, Any, Iterable, Tuple, Sequence, Optional, Callable
from pathlib import Path
from pprint import pprint
import time
from copy import deepcopy
import logging
from math import ceil

logger = logging.getLogger(__name__)

from common.config_utils import CONFIG
from common.env_utils import ENVS, ALLOW_OBS, ENV_ACTION_SPACE, D, C
from common.file_utils import (
    save_json,
    read_json,
    ensure_dir,
    TRAIN_CONFIGS_PATH,
    EVAL_CONFIGS_PATH,
    ENV_CONFIG_PATH,
    ALGO_CONFIG_PATH,
    OBS_CONFIG_PATH,
    ALGO_CONFIG_HYPERPARAMS_PATH,
)
from env_fixed_configs import *

ALGO_HYPERPAMETER_ENVS_ACTION_SPACE = {
    "bipedalwalker-v3": C,
    "cartpole-v1": D,
    "atari": D,
}
ALGO_HYPERPAMETER_ENVS = list(ALGO_HYPERPAMETER_ENVS_ACTION_SPACE.keys())

ALGO_FILES = ["a2c", "ppo", "dqn", "sac"]
ALLOW_ACTION_SPACE = {
    "a2c": [D, C],
    "ppo": [D, C],
    "dqn": [D],
    "sac": [C],
}
ALGO_KEYS_TO_DROP = ["n_timesteps", "frame_stack", "env_wrapper"]

OBS_CNN = ["GrayscaleObservation"]
OBS_MLP = ["Kinematics", "TimeToCollision", "KinematicsGoal", "OccupancyGrid", "AttributesObservation", "ExitObservation"]

def log_configs(env_description: str, configs: List[CONFIG]):
    logger.debug(f"Number of {env_description} configs: {len(configs)}")

def build_highway_configs() -> List[CONFIG]:
    LANE_COUNTS = np.linspace(2, 5, 4, dtype=int)
    LANE_CAPACITY = 10 # capacity of each lane in the fixed duration
    VEHICLE_DENSITIES = [np.float64(0.0)] + list(np.linspace(0.5, 2.5, 9, dtype=float))
    MAX_VEHICLES_COUNT = 50
    DURATION = 40
    configs = []
    for (lane_cnt, density) in itertools.product(LANE_COUNTS, VEHICLE_DENSITIES):
        config = deepcopy(HIGHWAY_FIXED_CONFIGS)
        """
        Example good configs (tested to see if make sense):
        lanes_count: 5, vehicles_count: 50, vehicles_density: 2, duration: 40        
        lanes_count: 5, vehicles_count: 10, vehicles_density: 0.5, duration: 40
        lanes_count: 2, vehicles_count: 5, vehicles_density: 0.5, duration: 40 (decent)
        """
        # High densities make the ego not be able to traverse everything, so we can upper bound the vehicle count
        veh_cnt = min(MAX_VEHICLES_COUNT, ceil(lane_cnt * LANE_CAPACITY * (density / 2)))
        assert(density > 0 or veh_cnt == 0)
        config["config"].update({
            "lanes_count": lane_cnt,
            "vehicles_density": density,
            "vehicles_count": veh_cnt,
            "duration": DURATION,
            "ego_spacing": 2,
        })
        configs.append(config)
    log_configs("Highway", configs)
    return configs

def build_merge_configs() -> List[CONFIG]:
    LANES_COUNT = np.linspace(2, 4, 3, dtype=int)
    MERGE_LENGTHS = np.linspace(50, 125, 4, dtype=int)
    configs = []
    for (lanes_cnt, merge_length) in itertools.product(LANES_COUNT, MERGE_LENGTHS):
        max_vehicle_cnt = 10 + (lanes_cnt - 2) * 5
        VEHICLE_COUNTS = np.linspace(0, max_vehicle_cnt, max_vehicle_cnt // 5 + 1, dtype=int)
        for veh_cnt in VEHICLE_COUNTS:
            config = deepcopy(MERGE_FIXED_CONFIGS)
            config["config"].update({
                "lanes_count": lanes_cnt,
                "vehicles_count": veh_cnt,
                "merge_length_parallel": merge_length,
            })
            configs.append(config)
    return configs

def build_roundabout_configs() -> List[CONFIG]:
    LANE_COUNTS = np.linspace(2, 4, 3, dtype=int)
    RADIUS = np.linspace(20, 40, 3, dtype=int)
    configs = []
    for lanes_idx, lanes_cnt in enumerate(LANE_COUNTS):
        for radius_idx, radius in enumerate(RADIUS):
            max_vehicle_count = min(20, 10 + (lanes_idx + radius_idx) * 5)
            VEHICLE_COUNTS = np.linspace(0, max_vehicle_count, max_vehicle_count // 5 + 1, dtype=int)
            for veh_cnt in VEHICLE_COUNTS:
                duration = 15 + radius / 10 + (lanes_cnt - 2)  # min: 17, max: 21
                config = deepcopy(ROUNDABOUT_FIXED_CONFIGS)
                config["config"].update({
                    "vehicles_count": veh_cnt,
                    "roundabout_lanes": lanes_cnt,
                    "roundabout_radius": radius,
                    "duration": duration,
                })
                configs.append(config)
    return configs

def build_u_turn_configs() -> List[CONFIG]:
    return [
        deepcopy(U_TURN_FIXED_CONFIGS)
    ]

def build_two_way_configs() -> List[CONFIG]:
    return [
        deepcopy(TWO_WAY_FIXED_CONFIGS)
    ]

def build_exit_configs() -> List[CONFIG]:
    # Similar idea to highway, with additional parameters
    EXIT_POSITIONS = np.linspace(300, 700, 3, dtype=int)
    EXIT_LENGTHS = np.linspace(50, 150, 3, dtype=int)

    LANE_COUNTS = np.linspace(2, 5, 4, dtype=int)
    LANE_CAPACITY = 10 # capacity of each lane in the fixed duration
    VEHICLE_DENSITIES = list(np.linspace(0.5, 2, 4, dtype=float))

    DURATION_HIGHWAY = 40
    configs = []
    for (exit_pos, exit_length, lane_cnt, veh_density) in itertools.product(
        EXIT_POSITIONS, EXIT_LENGTHS, LANE_COUNTS, VEHICLE_DENSITIES,
    ):
        config = deepcopy(EXIT_FIXED_CONFIGS)
        exit_sz = exit_pos + exit_length
        road_length = exit_sz + 500

        # Match example default config of (400 + 100) / 25 = 20 close to 18
        # Note how speeds are [18, 24, 30] in exit-v0
        duration = exit_sz // 25    # range of values: [14, 34]

        # Matches highway formula and takes into account duration to avoid unnecessary vehicles
        veh_cnt = ceil(lane_cnt * LANE_CAPACITY * (veh_density / 2) * duration / DURATION_HIGHWAY)  # max value: 43
        assert(veh_density > 0 or veh_cnt == 0)
        config["config"].update({
            "exit_position": exit_pos,
            "exit_length": exit_length,
            "road_length": road_length,
            "duration": duration,
        })
        configs.append(config)
    return configs

def build_lane_keeping_configs() -> List[CONFIG]:
    STEERING_RANGES = np.linspace(20, 60, 5, dtype=int)
    DURATIONS = np.linspace(100, 200, 2, dtype=int)
    NOISES = np.linspace(0, 0.2, 5, dtype=float)

    configs = []
    for (steering, duration, noise) in itertools.product(STEERING_RANGES, DURATIONS, NOISES):
        config = deepcopy(LANE_KEEPING_FIXED_CONFIGS)
        ang = np.deg2rad(steering)
        config["config"]["action"]["steering_range"] = [-ang, ang]
        config["config"].update({
            "duration": duration,
            "state_noise": noise,
            "derivative_noise": noise,
        })
        configs.append(config)

    log_configs("Lane Keeping", configs)
    return configs

def build_racetrack_configs() -> List[CONFIG]:
    STEERING_RANGES = np.linspace(30, 60, 3, dtype=int)
    BASIC_VEHICLE_COUNTS = np.linspace(0, 10, 3, dtype=int)
    configs = []
    for (steering, veh_cnt) in itertools.product(STEERING_RANGES, BASIC_VEHICLE_COUNTS):
        config = deepcopy(BASIC_RACETRACK_FIXED_CONFIGS)
        ang = np.deg2rad(steering)
        config["config"]["action"]["steering_range"] = [-ang, ang]
        config.update({
            "vehicles_count": veh_cnt,
        })
        configs.append(config)

    ROAD_LENGTHS = np.linspace(100, 200, 3, dtype=int)
    LANES_COUNT = np.linspace(2, 4, 3, dtype=int)
    for (steering, (length_idx, road_length), (lanes_idx, lanes_count)) in itertools.product(
        STEERING_RANGES, enumerate(ROAD_LENGTHS), enumerate(LANES_COUNT),
    ):
        aux = lanes_idx * 2 + length_idx    # to check if there's enough space for more vehicles
        max_veh_cnt = 20 if aux >= 3 else 10
        oval_veh_cnts = np.linspace(0, max_veh_cnt, max_veh_cnt // 10 + 1, dtype=int)
        for veh_cnt in oval_veh_cnts:
            config = deepcopy(OVAL_RACETRACK_FIXED_CONFIGS)
            ang = np.deg2rad(steering)
            config["config"]["action"]["steering_range"] = [-ang, ang]

            all_but_2nd = [i for i in range(1, lanes_count+1) if i != 2]
            BLOCK_LANES = [[], [1]]
            if all_but_2nd not in BLOCK_LANES:
                BLOCK_LANES.append(all_but_2nd)

            for blocks in BLOCK_LANES:
                config["config"].update({
                    "lanes_count": lanes_count,
                    "vehicles_count": veh_cnt,
                    "length": road_length,
                    "block_lanes": blocks,
                })
                configs.append(config)
    log_configs("Racetrack", configs)
    return configs

def build_parking_configs() -> List[CONFIG]:
    PARKING_SPOTS = np.linspace(1, 16, 6, dtype=int)
    STEERING_RANGES = np.linspace(30, 60, 3, dtype=int)
    ADD_WALLS = [True, False]
    OCCUPANCIES = [0, 0.25, 0.5, 0.75]
    configs = []
    for (spots, steering, walls) in itertools.product(PARKING_SPOTS, STEERING_RANGES, ADD_WALLS):
        veh_cnts = set()
        actual_spots = spots * 2
        for occ in OCCUPANCIES:
            veh_cnts.add(int(occ * actual_spots))
        for veh_cnt in veh_cnts:
            config = deepcopy(PARKING_FIXED_CONFIGS)
            ang = np.deg2rad(steering)
            config["config"]["action"]["steering_range"] = [-ang, ang]
            config["config"].update({
                "parking_spots": spots,
                "vehicles_count": veh_cnt,
                "add_walls": walls,
            })
            configs.append(config)
    log_configs("Parking", configs)
    return configs

def extract_algo_configs():
    aggregated: Dict[str, Dict[str, Any]] = {}

    for algo in ALGO_FILES:
        config_path = ALGO_CONFIG_HYPERPARAMS_PATH / f"{algo}.yml"
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        if not isinstance(raw_config, dict):
            raise ValueError(
                f"Unexpected structure in {config_path}: expected a mapping."
            )

        filtered = {
            env_name: params
            for env_name, params in raw_config.items()
            if isinstance(env_name, str) and env_name.lower() in ALGO_HYPERPAMETER_ENVS
        }
        if filtered:
            aggregated[algo] = filtered

    ALGO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(ALGO_CONFIG_PATH, aggregated)
    return aggregated


def get_algo_configs():
    algo_configs = read_json(ALGO_CONFIG_PATH)
    configs = []
    for algo, hyperparam_config in algo_configs.items():
        for original_env, hyperparams in hyperparam_config.items():
            action_space = ALGO_HYPERPAMETER_ENVS_ACTION_SPACE[original_env.lower()]
            for key in ALGO_KEYS_TO_DROP:
                hyperparams.pop(key, None)
            configs.append(
                {
                    "algo": algo,
                    "action_space": action_space,
                    **hyperparams,
                }
            )
    return configs

def get_obs_configs():
    return read_json(OBS_CONFIG_PATH)

def get_env_configs(env_id: str) -> List[CONFIG]:
    return read_json(ENV_CONFIG_PATH(env_id))

def get_env_obs_configs(env_id: str, all_obs_configs: List[CONFIG]):
    allowed = ALLOW_OBS[env_id]
    return [
        config
        for config in all_obs_configs
        if config["type"] in allowed
    ]

def get_env_algo_configs(env_id: str, all_algo_configs: List[CONFIG]):
    env_action_space = ENV_ACTION_SPACE[env_id]
    return [
        config
        for config in all_algo_configs
        if config["action_space"] == env_action_space
    ]

def valid_config(algo_config: CONFIG, obs_config: CONFIG) -> bool:
    # we can assume None is never anything
    return (
        algo_config["policy"] == "CnnPolicy"
        and obs_config["type"] in OBS_CNN
    ) or (
        algo_config["policy"] == "MlpPolicy"
        and obs_config["type"] in OBS_MLP
    )

def build_all_configs(
    env_configs: List[CONFIG],
    obs_configs: List[CONFIG],
    algo_configs: List[CONFIG],
) -> Tuple[List[CONFIG], List[CONFIG]]:
    run_configs = [
        {
            "env_config": config[0],
            "obs_config": config[1],
            "algo_config": config[2],
            "timestamp": time.time_ns(),
        }
        for config in itertools.product(
            env_configs, obs_configs, algo_configs,
        )
        if valid_config(config[2], config[1])
    ]
    eval_configs = [
        {
            "env_config": config[0],
            "obs_config": config[1],
            "timestamp": time.time_ns(),
        }
        for config in itertools.product(
            env_configs, obs_configs,
        )
    ]
    return run_configs, eval_configs

def build_configs(builder: Callable[[], List[CONFIG]], save_path: Path, name: str) -> List[CONFIG]:
    configs = builder()
    save_json(save_path, configs)
    print(f"{name} environment configs: {len(configs)}")
    return configs

if __name__ == "__main__":
    build_configs(build_highway_configs, ENV_CONFIG_PATH("highway"), "Highway")
    build_configs(build_merge_configs, ENV_CONFIG_PATH("merge"), "Merge")
    build_configs(build_roundabout_configs, ENV_CONFIG_PATH("roundabout"), "Roundabout")
    build_configs(build_lane_keeping_configs, ENV_CONFIG_PATH("lane-keeping"), "Lane Keeping")
    build_configs(build_racetrack_configs, ENV_CONFIG_PATH("racetrack"), "Racetrack")
    build_configs(build_parking_configs, ENV_CONFIG_PATH("parking"), "Parking")
    build_configs(build_exit_configs, ENV_CONFIG_PATH("exit"), "Exit")

    algo_configs = extract_algo_configs()
    obs_configs = get_obs_configs()
    for config in obs_configs:
        name = config["type"]
        assert((name in OBS_MLP) ^ (name in OBS_CNN))

    print(f"\nAlgo Configs: {len(algo_configs)}")
    print(f"\nObservation Configs: {len(obs_configs)}")
    print("\n\n")
    all_train_configs = []
    all_eval_configs = []
    for env in ENVS:
        env_obs_configs = get_env_obs_configs(env, obs_configs)
        env_algo_configs = get_env_algo_configs(env, get_algo_configs())
        env_run_configs, env_eval_configs = build_all_configs(
            get_env_configs(env), env_obs_configs, env_algo_configs,
        )
        save_json(TRAIN_CONFIGS_PATH(env), env_run_configs)
        save_json(EVAL_CONFIGS_PATH(env), env_eval_configs)
        all_train_configs.extend(env_run_configs)
        all_eval_configs.extend(env_eval_configs)
        print(f">>>> {env}")
        print(f"Train configs: {len(env_run_configs)}")
        print(f"Eval configs: {len(env_eval_configs)}")
        print()

    print(f"\n\nTotal number of train configs: {len(all_train_configs)}")
    # print("Example config:")
    # pprint(all_run_configs[0])

    print(f"\n\nTotal number of eval configs: {len(all_eval_configs)}")
    # print("Example config:")
    # pprint(all_eval_configs[0])
