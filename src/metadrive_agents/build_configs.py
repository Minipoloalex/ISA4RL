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
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.sensors.rgb_camera import RGBCamera

logger = logging.getLogger(__name__)
from env_fixed_configs import METADRIVE_FIXED_CONFIGS

from common.config_utils import CONFIG
from common.env_utils import METADRIVE_ENVS, ALLOW_OBS, ENV_ACTION_SPACE, D, C, K, TTC, KG, OG, A, E, GS
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
from common.build_configs_utils import (
    CNN,
    MLP,
    MULTI_INPUT,
    build_configs,
    extract_algo_configs,
    get_algo_configs,
    get_env_configs,
)

# For image, use this in my version of MetaDriveEnv with an if based on the observation
# IMG_OBS_CONFIG = {
#     "image_observation": True,
#     "sensors": dict(rgb_camera=(RGBCamera, 200, 100)),
#     "stack_size": 3,
# }

def build_metadrive_configs() -> List[CONFIG]:
    # VEHICLE_MODELS = ["default", "s", "xl"]

    INTERSECTION_PROB_DIST = {"StdInterSection": 1/3, "StdTInterSection": 1/3, "Roundabout": 1/3}

    MAPS = ["rORY", "SC", "TXT", INTERSECTION_PROB_DIST]
    # human tested valid times for completion, assuming maximum occupancy (traffic density = 0.3)
    MAP_HORIZONS = [800, 500, 500, 400]

    VEHICLE_MODELS = ["default"]
    LANE_NUMS = np.linspace(2, 5, 4, dtype=int)
    LANE_WIDTHS = [3.5]

    TRAFFIC_DENSITIES = np.linspace(0.0, 0.3, 7, dtype=float)
    DISCRETE_ACTION = [True, False]

    configs = []
    for (map_idx, metadrive_map), veh_model, lane_num, lane_width, veh_density, discrete_action in itertools.product(
        enumerate(MAPS), VEHICLE_MODELS, LANE_NUMS, LANE_WIDTHS, TRAFFIC_DENSITIES, DISCRETE_ACTION,
    ):
        config = deepcopy(METADRIVE_FIXED_CONFIGS)
        horizon = MAP_HORIZONS[map_idx]
        if type(metadrive_map) is dict:
            n_blocks = 1
            map_type = MapGenerateMethod.BIG_BLOCK_NUM
            sequence = n_blocks
            config["config"].update({
                # Needs to be used to instantiate PGBlockDistConfig later
                "block_dist_config": INTERSECTION_PROB_DIST,
            })
        else:
            n_blocks = len(metadrive_map)
            map_type = MapGenerateMethod.BIG_BLOCK_SEQUENCE
            sequence = metadrive_map

        if metadrive_map == "rORY" and lane_num >= 5:
            logger.warning("Skipping pathological MetaDrive map config: map=%s lane_num=%s", metadrive_map, lane_num)
            continue

        config["config"].update({
            "traffic_density": veh_density,
            "discrete_action": discrete_action,
            "horizon": horizon,
            "image_observation": False, # The image observation is too slow per timestep
        })
        config["config"]["map_config"] = {
            BaseMap.GENERATE_TYPE: map_type,
            BaseMap.GENERATE_CONFIG: sequence,  # block num / block ID sequence
            BaseMap.LANE_WIDTH: lane_width,
            BaseMap.LANE_NUM: lane_num,
        }
        config["config"]["vehicle_config"] = {"vehicle_model": veh_model}
        configs.append(config)

    return configs

def validate_algo_config(algo_config: CONFIG):
    algo_config.pop("normalize", None)
    # algo_config["policy"] = MULTI_INPUT
    return algo_config

def valid_config(env_config: CONFIG, algo_config: CONFIG):
    if algo_config["policy"] == CNN:
        return False

    return (
        (env_config["config"]["discrete_action"] and algo_config["action_space"] == D) or
        (not env_config["config"]["discrete_action"] and algo_config["action_space"] == C)
    )

def build_all_configs(
    env_configs: List[CONFIG],
    algo_configs: List[CONFIG],
) -> Tuple[List[CONFIG], List[CONFIG]]:
    run_configs = [
        {
            "env_config": env,
            # "obs_config": obs,
            "algo_config": validate_algo_config(deepcopy(algo)),
            "timestamp": time.time_ns(),
        }
        for (env, algo) in itertools.product(
            env_configs, algo_configs,
        )
        if valid_config(env, algo)
    ]
    eval_configs = [
        {
            "env_config": env,
            "timestamp": time.time_ns(),
        }
        for (env,) in itertools.product(
            env_configs,
        )
    ]
    return run_configs, eval_configs

if __name__ == "__main__":
    env = "metadrive"
    build_configs(build_metadrive_configs, ENV_CONFIG_PATH(env), "Metadrive")

    algo_configs = extract_algo_configs()
    print(f"\nAlgo Configs: {len(algo_configs)}")
    print("\n\n")

    algo_configs = get_algo_configs()
    env_run_configs, env_eval_configs = build_all_configs(
        get_env_configs(env), algo_configs,
    )

    save_json(TRAIN_CONFIGS_PATH(env), env_run_configs)
    save_json(EVAL_CONFIGS_PATH(env), env_eval_configs)
    
    print(f">>>> {env}")
    print(f"Train configs: {len(env_run_configs)}")
    print(f"Eval configs: {len(env_eval_configs)}")
    print()
