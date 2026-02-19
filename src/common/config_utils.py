from typing import Dict, Any, Tuple
from file_utils import *

CONFIG = Dict[str, Any]

# Map from (env_id, obs_id, algo_id) -> train id
_TRAIN_ID_CACHE: Dict[Tuple[int, int, int], int] | None = None

def get_all_train_configs() -> List[CONFIG]:
    return read_json(TRAIN_CONFIG_PATH)

def get_all_instance_configs() -> List[CONFIG]:
    return read_json(INSTANCE_CONFIG_PATH)

def get_all_eval_configs() -> List[CONFIG]:
    return read_json(EVAL_CONFIG_PATH)

def map_to_train_id(orig_env_id: int, obs_id: int, algo_id: int) -> int:
    """Map from (env_id, obs_id, algo_id) to train_id"""
    global _TRAIN_ID_CACHE
    if _TRAIN_ID_CACHE is None:
        _TRAIN_ID_CACHE = {}
        for config in get_all_train_configs():
            env_cfg = config["env_config"]
            obs_cfg = config["obs_config"]
            algo_cfg = config["algo_config"]
            key = (
                env_cfg["orig_id"],
                obs_cfg["id"],
                algo_cfg["id"],
            )
            if key in _TRAIN_ID_CACHE:
                raise ValueError(f"Duplicate training config key detected: {key}")
            _TRAIN_ID_CACHE[key] = config["id"]

    key = (orig_env_id, obs_id, algo_id)
    try:
        return _TRAIN_ID_CACHE[key]
    except KeyError as exc:
        raise KeyError(f"Missing training config for env={orig_env_id}, obs={obs_id}, algo={algo_id}") from exc
