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
from common.env_utils import ENVS, ALLOW_OBS, ENV_ACTION_SPACE, D, C, K, TTC, KG, OG, A, E, GS
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

MLP = "MlpPolicy"
CNN = "CnnPolicy"
MULTI_INPUT = "MultiInputPolicy"

OBS_POLICY = {
    K: MLP,
    TTC: MLP,
    KG: MULTI_INPUT,
    OG: MLP,
    E: MLP,
    A: MULTI_INPUT,
    GS: CNN,
}

def validate_policy_type(algo_config: CONFIG, obs_config: CONFIG):
    if algo_config["policy"] == MLP and OBS_POLICY[obs_config["type"]] == MULTI_INPUT:
        algo_config["policy"] = MULTI_INPUT
    return algo_config

def log_configs(env_description: str, configs: List[CONFIG]):
    logger.debug(f"Number of {env_description} configs: {len(configs)}")

# When the values are randomized, these are the bounds (base_map.py)
MAX_LANE_WIDTH = 4.5
MIN_LANE_WIDTH = 3.0
MAX_LANE_NUM = 3
MIN_LANE_NUM = 2

CURVE = "C"

def build_metadrive_configs() -> List[CONFIG]:
    MAP_NUM_CONFIG = MapGenerateMethod.BIG_BLOCK_NUM
    MAP_SEQUENCE_CONFIG = MapGenerateMethod.BIG_BLOCK_SEQUENCE
    configs = []
    LANE_NUMS = np.linspace(2, 3, 2, dtype=int)
    LANE_WIDTHS = np.linspace(3, 4.5, 4, dtype=float)
    # map_config={
    #     BaseMap.GENERATE_TYPE: MAP_NUM_CONFIG,
    #     BaseMap.GENERATE_CONFIG: 3,
    #     BaseMap.LANE_WIDTH: 3.5,
    #     BaseMap.LANE_NUM: 3,
    # }
    MAPS = [
        CURVE,
    ]
    random_lane_num = [True, False]
    random_lane_width = [True, False]
    random_vehicle_model = [True, False]
    vehicle_models = ["s", "m", "l", "xl", "default"]
    random_traffic = [True, False]
    TRAFFIC_DENSITY = np.linspace(0.0, 1.0, 6, dtype=float)
    DISCRETE_ACTION = [True, False]
    DISCRETE_STEERING_DIM = np.linspace(5, 10, 2, dtype=int)
    DISCRETE_THROTTLE_DIM = np.linspace(5, 10, 2, dtype=int)
    # HORIZON =  
    return configs


if __name__ == "__main__":
    # build_configs(build_highway_configs, ENV_CONFIG_PATH("highway"), "Highway")
    pass
