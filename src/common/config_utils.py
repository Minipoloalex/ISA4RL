from typing import Dict, Any, Tuple, Callable
from .file_utils import *
from .general_utils import generate_random_string

CONFIG = Dict[str, Any]

def get_config_map(base_dir: Path, config_path_func: Callable[[Path], Path]):
    instance_config_map = {}
    for folder in os.listdir(base_dir):
        folder_path = base_dir / folder
        config_path = config_path_func(folder_path)

        if os.path.isdir(folder_path) and os.path.exists(config_path):
            instance_config_map[folder] = read_json(config_path)
    return instance_config_map

def get_new_random_id() -> str:
    return generate_random_string(20)

def __instance_config(env_config: CONFIG, obs_config: CONFIG):
    return {
        "env_config": env_config,
        "obs_config": obs_config,
    }

def get_instance_id(base_dir: Path, env_config: CONFIG, obs_config: CONFIG) -> str:
    instance_config_map = get_config_map(base_dir, RESULTS_INSTANCE_CONFIG_FILE)
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

def save_instance_config(folder: Path, env_config: CONFIG, obs_config: CONFIG) -> None:
    ic = __instance_config(env_config, obs_config)
    file = RESULTS_INSTANCE_CONFIG_FILE(folder)
    if not file.exists():
        save_json(file, ic)

def save_algo_config(folder: Path, algo_config: Dict[str, Any]) -> None:
    file = RESULTS_TRAIN_CONFIG_FILE(folder)
    if not file.exists():
        save_json(file, algo_config)
