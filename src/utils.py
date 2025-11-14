import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pprint import pprint

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from configs import RunConfig, InstanceConfig
import gymnasium as gym

CONFIG = Dict[str, Any]
AlgorithmName = str

ALGORITHM_MAP: Dict[AlgorithmName, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

BASE_OUTPUT_PATH = Path("results")
BASE_CONFIG_PATH = Path("config")
BASE_IMAGES_PATH = Path("images")

TRAIN_FOLDER = "train"
METAFEATURES_FOLDER = "metafeatures"

MODEL_FILE = "model.zip"

TRAIN_CONFIG_PATH = BASE_CONFIG_PATH / "train_configs.json"
INSTANCE_CONFIG_PATH = BASE_CONFIG_PATH / "instance_configs.json"
EVAL_CONFIG_PATH = BASE_CONFIG_PATH / "eval_configs.json"
HIGHWAY_CONFIG_PATH = BASE_CONFIG_PATH / "highway-configs.json"
ROUNDABOUT_CONFIG_PATH = BASE_CONFIG_PATH / "roundabout-configs.json"
MERGE_CONFIG_PATH = BASE_CONFIG_PATH / "merge-configs.json"
ALGO_CONFIG_PATH = BASE_CONFIG_PATH / "algo-configs.json"
OBS_CONFIG_PATH = BASE_CONFIG_PATH / "obs-configs.json"

ALGO_CONFIG_HYPERPARAMS_PATH = BASE_CONFIG_PATH / "rlzoo-algo-hyperparams"

EVALUATION_RESULTS_BASE_PATH = "eval_results"
TRAINING_METADATA_FILE = "training_metadata.json"
METAFEATURES_RESULTS_FILE = "metafeatures.json"
EVALUATION_RESULTS_FILE = lambda seed: f"seed_{seed}.json"

TRAIN_TIMESTEPS = int(1e5)

# Mostly refer to the environment
DISCARD_POLICY_PARAMS = ["n_envs", "algo", "env_wrapper", "frame_stack", "normalize", "id"]

# Map from (env_id, obs_id, algo_id) -> train id
_TRAIN_ID_CACHE: Dict[Tuple[int, int, int], int] | None = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_json(file: Path):
    with file.open("r", encoding="utf-8") as fp:
        return json.load(fp)

def save_json(file: Path, results: Dict[str, Any] | List[Dict[str, Any]] | List[Any]):
    with file.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=_json_default)

def ensure_dir(path: str | Path) -> None:
    if type(path) is str:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True) # type: ignore


def discretize(obs: np.ndarray, clip_range: Tuple[float, float] = (-5.0, 5.0), bins_per_dim: int = 15) -> Tuple[int, ...]:
    obs = np.asarray(obs, dtype=np.float32)
    low, high = clip_range
    obs = np.clip(obs, low, high)
    edges = np.linspace(low, high, bins_per_dim + 1)
    return tuple(np.digitize(obs, edges))


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32).ravel()
    if isinstance(obs, (list, tuple)):
        return np.asarray(obs, dtype=np.float32).ravel()
    if isinstance(obs, (int, float)):
        return np.asarray([obs], dtype=np.float32)
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def _normalize_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return tuple(action.astype(np.float32).ravel().tolist())
    if isinstance(action, (list, tuple)):
        return tuple(np.asarray(action, dtype=np.float32).ravel().tolist())
    if isinstance(action, (np.integer, int)):  # type: ignore[arg-type]
        return int(action)
    if isinstance(action, (np.floating, float)):  # type: ignore[arg-type]
        return float(action)
    return action


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _interpolate_range_value(lo: Any, hi: Any, weight: float) -> Any:
    if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):  # type: ignore[arg-type]
        raw = float(lo + (hi - lo) * weight)
        clipped = max(lo, min(hi, _round_half_up(raw)))
        return int(clipped)
    raw = float(lo + (hi - lo) * weight)
    return float(round(max(lo, min(hi, raw)), 6))


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):  # type: ignore[attr-defined]
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int32, np.int64)): # type: ignore
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): # type: ignore
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def build_env(env_id: str, config: Dict[str, Any]) -> gym.Env:
    env_kwargs: Dict[str, Any] = {
        "config": config
    }
    return gym.make(env_id, **env_kwargs)

def _make_train_env(
    env_id: str,
    env_config: Dict[str, Any],
    env_cnt: int,
    vec_cls: type[DummyVecEnv] | type[SubprocVecEnv],
    vec_kwargs: Dict[str, Any] | None,
) -> VecEnv:
    env_kwargs = {"config": env_config.copy(), "render_mode": None}
    return make_vec_env(
        env_id,
        n_envs=env_cnt,
        vec_env_cls=vec_cls,
        env_kwargs=env_kwargs,
        vec_env_kwargs=vec_kwargs,
    )

def _make_eval_env(env_id: str, env_config: Dict[str, Any]) -> gym.Env:
    env_kwargs = {"config": env_config.copy()}
    return gym.make(env_id, max_episode_steps=None, disable_env_checker=None, **env_kwargs)

def _make_model(
    env: VecEnv,
    *,
    algo_cls: type[BaseAlgorithm],
    folder_name: str,
    policy_params: Dict[str, Any],
    device: str,
) -> BaseAlgorithm:
    model_path = Path(folder_name) / MODEL_FILE
    if _nonempty_file_in(model_path):
        custom_objects = {}
        lr = policy_params.get("learning_rate")
        if lr is not None and type(lr) is not str:
            custom_objects["learning_rate"] = lr
        model = algo_cls.load(str(model_path), env=env, device=device, custom_objects=custom_objects)
        model.tensorboard_log = folder_name
        return model
    return algo_cls(env=env, tensorboard_log=folder_name, device=device, **policy_params)


def get_env_id(env: gym.Env):
    assert(env.spec is not None)
    return env.spec.id


def map_algo_name_to_class(algo_name: str) -> type[BaseAlgorithm]:
    algo_cls = ALGORITHM_MAP.get(algo_name.lower())
    if algo_cls is None:
        raise KeyError(
            f"Unknown algorithm '{algo_name}'. Expected one of {sorted(ALGORITHM_MAP)}."
        )
    return algo_cls

def load_model(model_path: Path, algo_name: str) -> BaseAlgorithm:
    algo_cls = map_algo_name_to_class(algo_name)
    return algo_cls.load(str(model_path))

def load_training_metadata(run_dir: Path) -> Dict[str, Any]:
    metadata_path = run_dir / TRAINING_METADATA_FILE
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def save_eval_results(results: List[Dict[str, Any]], folder_name: str, eval_seed: int):
    folder_path = Path(folder_name) / EVALUATION_RESULTS_BASE_PATH
    ensure_dir(folder_path)

    filepath = folder_path / EVALUATION_RESULTS_FILE(eval_seed)
    save_json(filepath, results)

def save_extract_results(results, folder_name: str):
    ensure_dir(folder_name)
    filepath = Path(folder_name) / METAFEATURES_RESULTS_FILE
    save_json(filepath, results)

def get_all_train_configs() -> List[CONFIG]:
    return read_json(TRAIN_CONFIG_PATH)

def get_all_instance_configs() -> List[CONFIG]:
    return read_json(INSTANCE_CONFIG_PATH)

def get_all_eval_configs() -> List[CONFIG]:
    return read_json(EVAL_CONFIG_PATH)

def _parse_policy_kwargs(raw_value: Any) -> Any:
    """Convert string-encoded policy kwargs into a Python object."""
    if not isinstance(raw_value, str):
        return raw_value
    allowed_names = {
        "RMSpropTFLike": RMSpropTFLike,
        "dict": dict,
    }
    try:
        return eval(raw_value, {"__builtins__": {}}, allowed_names)
    except Exception as exc:  # pragma: no cover - config errors surface at runtime
        raise ValueError(f"Failed to parse policy_kwargs='{raw_value}': {exc}") from exc

def _linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func


def _resolve_schedule_placeholders(value: Any) -> Any:
    """Recursively replace lin_* placeholders with schedule callables."""
    if isinstance(value, str) and value.startswith("lin_"):
        parts = value.split("_")
        assert(len(parts) == 2)
        numeric_value = _coerce_numeric(parts[1])
        if numeric_value is None:
            raise ValueError(f"Invalid linear schedule specification: '{value}'")
        return _linear_schedule(numeric_value)
    return value

def load_all_run_configs(get_configs: Callable[[], List[CONFIG]]) -> List[RunConfig]:
    """get_configs should be get_all_run_configs or get_all_eval_configs"""
    configs = get_configs()
    run_configs: List[RunConfig] = []
    for config in configs:
        env_config : CONFIG = config["env_config"]
        obs_config : CONFIG = config["obs_config"]
        algo_config: CONFIG = config["algo_config"]

        id: int = config["id"]
        id_env: int = env_config["id"]
        orig_id_env: int = env_config["orig_id"]    # TODO: remove the other one (with .get())
        id_obs: int = obs_config["id"]
        id_algo: int = algo_config["id"]
        train_id: int = map_to_train_id(orig_id_env, id_obs, id_algo)

        train_folder_name: str = str(BASE_OUTPUT_PATH / TRAIN_FOLDER / str(train_id))
        instance_folder_name: str = str(BASE_OUTPUT_PATH / METAFEATURES_FOLDER / f"ENV{id_env}_OBS{id_obs}")

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id: str = env_config["env_id"]
        eval_seed: int = env_config["eval_seed"]

        env_config: CONFIG = env_config["config"]
        env_config["observation"] = obs_config

        algo_name: str = algo_config["algo"]
        n_envs = algo_config.get("n_envs", 1)
        policy = algo_config["policy"]
        for key in DISCARD_POLICY_PARAMS:
            algo_config.pop(key, None)

        for param in algo_config.keys():
            algo_config[param] = _resolve_schedule_placeholders(algo_config[param])

        policy_params: Dict[str, Any] = algo_config
        policy_kwargs = policy_params.get("policy_kwargs")
        if policy_kwargs is not None:
            policy_params["policy_kwargs"] = _parse_policy_kwargs(policy_kwargs)
        algo_cls = map_algo_name_to_class(algo_name)

        vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
        vec_env_kwargs = None if n_envs == 1 else {"start_method": "spawn"}
        device = "cuda" if policy == "CnnPolicy" else "cpu"

        run_configs.append(
            RunConfig(
                id=id,
                train_id=train_id,
                id_env_config=id_env,
                orig_id_env_config=orig_id_env,
                id_obs_config=id_obs,
                id_algo_config=id_algo,
                instance_folder_name=instance_folder_name,
                train_folder_name=train_folder_name,
                make_env=partial(_make_train_env, env_id, env_config, n_envs, vec_env_cls, vec_env_kwargs),
                make_eval_env=partial(_make_eval_env, env_id, env_config),
                make_model=partial(
                    _make_model,
                    algo_cls=algo_cls,
                    folder_name=train_folder_name,
                    policy_params=policy_params,
                    device=device,
                ),
                timesteps=TRAIN_TIMESTEPS,
                train_seed=0,
                eval_seed=eval_seed,
            )
        )
    return run_configs

def load_all_instance_configs() -> List[InstanceConfig]:
    configs = get_all_instance_configs()
    instance_configs: List[InstanceConfig] = []
    for config in configs:
        env_config : Dict[str, Any] = config["env_config"]
        obs_config : Dict[str, Any] = config["obs_config"]

        id: int = config["id"]
        id_env: int = env_config["id"]
        orig_id_env: int = env_config["orig_id"]
        id_obs: int = obs_config["id"]
        instance_folder_name: str = str(BASE_OUTPUT_PATH / METAFEATURES_FOLDER / f"ENV{id_env}_OBS{id_obs}")

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id: str = env_config["env_id"]
        env_config: Dict[str, Any] = env_config["config"]
        env_config["observation"] = obs_config

        instance_configs.append(
            InstanceConfig(
                id=id,
                id_env_config=id_env,
                orig_id_env_config=orig_id_env,
                id_obs_config=id_obs,
                make_eval_env=partial(_make_eval_env, env_id, env_config),
                eval_seed=1,
                instance_folder_name=instance_folder_name,
            )
        )
    return instance_configs

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

def _nonempty_file_in(filepath: Path) -> bool:
    """Return True if filepath exists, is a regular file, and is non-empty."""
    try:
        return filepath.is_file() and filepath.stat().st_size > 0
    except OSError:
        return False

def is_trained(config: RunConfig) -> bool:
    """Trained iff training metadata artifact exists."""
    return _nonempty_file_in(Path(config.train_folder_name) / TRAINING_METADATA_FILE)

def is_evaluated(config: RunConfig) -> bool:
    """Evaluated iff evaluation results artifact exists."""
    return _nonempty_file_in(
        Path(config.train_folder_name)
        / EVALUATION_RESULTS_BASE_PATH
        / EVALUATION_RESULTS_FILE(config.eval_seed)
    )

def is_extracted(config: InstanceConfig) -> bool:
    """Extracted iff metafeatures result artifact exists."""
    return _nonempty_file_in(Path(config.instance_folder_name) / METAFEATURES_RESULTS_FILE)

def unwrap_first_env(vec_env: VecEnv) -> Optional[gym.Env]:
    """Return the first underlying gym.Env from a VecEnv when accessible.

    Subproc-based VecEnvs keep environments in separate processes, making the
    raw env objects unpicklable across process boundaries. In that case we fall
    back to returning ``None`` and callers are expected to query attributes via
    ``VecEnv.get_attr`` instead.
    """
    current: VecEnv = vec_env
    # Unwrap nested VecEnv wrappers if present (e.g. VecMonitor -> VecNormalize -> VecEnvBase)
    for _ in range(10):
        envs = getattr(current, "envs", None)
        if envs:
            return envs[0].unwrapped
        next_vec = getattr(current, "venv", None)
        if next_vec is None:
            break
        current = next_vec
    return None


def vectorize_env(env: Union[gym.Env, VecEnv]) -> Tuple[VecEnv, Optional[gym.Env]]:
    """Ensure we operate on a VecEnv while keeping a handle to the base env."""
    if isinstance(env, VecEnv):
        return env, unwrap_first_env(env)
    return DummyVecEnv([lambda: env]), env

def annotate_ids(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            **d,
            "id": id,
        }
        for id, d in enumerate(lst)
    ]
