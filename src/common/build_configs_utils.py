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
from common.env_utils import D, C, K, TTC, KG, OG, A, E, GS
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


def get_env_configs(env_id: str) -> List[CONFIG]:
    return read_json(ENV_CONFIG_PATH(env_id))


def build_configs(builder: Callable[[], List[CONFIG]], save_path: Path, name: str) -> List[CONFIG]:
    configs = builder()
    save_json(save_path, configs)
    print(f"{name} environment configs: {len(configs)}")
    return configs
