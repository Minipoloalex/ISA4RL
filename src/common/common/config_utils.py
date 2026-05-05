from copy import deepcopy

from typing import Dict, Any, Callable, Optional, List
from .file_utils import *
from .general_utils import generate_random_string

CONFIG = Dict[str, Any]
USELESS_ENV_CONFIG_PARAMS: List[str] = ["n_test_episodes", "n_eval_episodes", "eval_freq", "train_timesteps"]

def get_config_map(base_dir: Path, config_path_func: Callable[[Path], Path], remove_params: bool = False):
    instance_config_map = {}
    for folder in os.listdir(base_dir):
        folder_path = base_dir / folder
        config_path = config_path_func(folder_path)

        if os.path.isdir(folder_path) and os.path.exists(config_path):
            ic = read_json(config_path)
            if remove_params:
                ic = remove_useless_params(ic)
            instance_config_map[folder] = ic
    return instance_config_map

def get_new_random_id() -> str:
    return generate_random_string(20)

def remove_useless_params(config_dict):
    fixed_config = deepcopy(config_dict)
    for param in USELESS_ENV_CONFIG_PARAMS:
        fixed_config["env_config"].pop(param, None)
    return fixed_config

def __instance_config(env_config: CONFIG, obs_config: Optional[CONFIG]):
    fixed_config = remove_useless_params({
        "env_config": env_config,
        "obs_config": obs_config,
    })
    return fixed_config

def get_instance_id(base_dir: Path, env_config: CONFIG, obs_config: Optional[CONFIG]) -> str:
    # env_config: (mistakenly) includes stuff like timesteps, eval_freq, n_eval_episodes and n_test_episodes
    # since n_test_episodes was modified, this required a bad fix
    instance_config_map = get_config_map(base_dir, RESULTS_INSTANCE_CONFIG_FILE, remove_params=True)
    ic = __instance_config(env_config, obs_config)
    for id, conf in instance_config_map.items():
        if conf == ic:
            return id
    return get_new_random_id()

def get_algo_id(instance_folder_path: Path, algo_config: CONFIG) -> str:
    train_path = RESULTS_TRAIN_FOLDER_PATH(instance_folder_path)
    algo_config_map = get_config_map(train_path, RESULTS_TRAIN_CONFIG_FILE)
    for id, conf in algo_config_map.items():
        if conf == algo_config:
            return id
    return get_new_random_id()

def save_instance_config(folder: Path, env_config: CONFIG, obs_config: Optional[CONFIG]) -> None:
    ic = __instance_config(env_config, obs_config)
    file = RESULTS_INSTANCE_CONFIG_FILE(folder)
    if not file.exists():
        save_json(file, ic)

def save_algo_config(folder: Path, algo_config: Dict[str, Any]) -> None:
    file = RESULTS_TRAIN_CONFIG_FILE(folder)
    if not file.exists():
        save_json(file, algo_config)
