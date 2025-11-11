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

AlgorithmName = str
ALGORITHM_MAP: Dict[AlgorithmName, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

BASE_OUTPUT_PATH = Path("results")
BASE_CONFIG_PATH = Path("config")
BASE_IMAGES_PATH = Path("images")

MODEL_FILE = "model.zip"

RUN_CONFIG_PATH = BASE_CONFIG_PATH / "configs.json"
INSTANCE_CONFIG_PATH = BASE_CONFIG_PATH / "instance_configs.json"
HIGHWAY_CONFIG_PATH = BASE_CONFIG_PATH / "highway-configs.json"
ROUNDABOUT_CONFIG_PATH = BASE_CONFIG_PATH / "roundabout-configs.json"
MERGE_CONFIG_PATH = BASE_CONFIG_PATH / "merge-configs.json"
ALGO_CONFIG_PATH = BASE_CONFIG_PATH / "algo-configs.json"
OBS_CONFIG_PATH = BASE_CONFIG_PATH / "obs-configs.json"

ALGO_CONFIG_HYPERPARAMS_PATH = BASE_CONFIG_PATH / "rlzoo-algo-hyperparams"

TRAINING_METADATA_FILE = "training_metadata.json"
EVALUATION_RESULTS_FILE = "eval_results.json"
METAFEATURES_RESULTS_FILE = "metafeatures.json"

TRAIN_TIMESTEPS = int(1e5)

# Mostly refer to the environment
DISCARD_POLICY_PARAMS = ["n_envs", "algo", "env_wrapper", "frame_stack", "normalize"]


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

def save_eval_results(results: List[Dict[str, Any]], folder_name: str):
    ensure_dir(folder_name)
    filepath = Path(folder_name) / EVALUATION_RESULTS_FILE
    save_json(filepath, results)

def save_extract_results(results, folder_name: str):
    ensure_dir(folder_name)
    filepath = Path(folder_name) / METAFEATURES_RESULTS_FILE
    save_json(filepath, results)

def get_all_configs() -> List[Dict[str, Any]]:
    return read_json(RUN_CONFIG_PATH)

def get_all_instance_configs() -> List[Dict[str, Any]]:
    return read_json(INSTANCE_CONFIG_PATH)

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

def load_all_run_configs() -> List[RunConfig]:
    configs = get_all_configs()
    run_configs: List[RunConfig] = []
    for config in configs:
        env_config : Dict[str, Any] = config["env_config"]
        obs_config : Dict[str, Any] = config["obs_config"]
        algo_config: Dict[str, Any] = config["algo_config"]

        id: int = config["id"]
        id_env: int = env_config["id"]
        id_obs: int = obs_config["id"]
        id_algo: int = algo_config["id"]
        folder_name: str = str(BASE_OUTPUT_PATH / str(id))

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id: str = env_config["env_id"]
        env_config: Dict[str, Any] = env_config["config"]
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

        def env_factory(
            env_id: str = env_id,
            env_config: Dict[str, Any] = env_config,
            env_cnt: int = n_envs,
            vec_cls: type[DummyVecEnv] | type[SubprocVecEnv] = vec_env_cls,
            vec_kwargs: Dict[str, Any] | None = vec_env_kwargs,
        ) -> VecEnv:
            env_kwargs = {"config": env_config.copy(), "render_mode": None}
            return make_vec_env(
                env_id,
                n_envs=env_cnt,
                vec_env_cls=vec_cls,
                env_kwargs=env_kwargs,
                vec_env_kwargs=vec_kwargs,
            )

        def eval_env_factory(
            env_id: str = env_id,
            env_config: Dict[str, Any] = env_config,
        ) -> gym.Env:
            env_kwargs = {"config": env_config.copy()}
            return gym.make(env_id, max_episode_steps=None, disable_env_checker=None, **env_kwargs)

        def model_factory(
            env: VecEnv,
            *,
            algo_cls: type[BaseAlgorithm] = algo_cls,
            tensorboard_log: str = folder_name,
            policy_params: Dict[str, Any] = policy_params,
            device: str = device,
        ) -> BaseAlgorithm:
            model_path = Path(folder_name) / MODEL_FILE
            if _nonempty_file_in(model_path):
                model = algo_cls.load(str(model_path), env=env, device=device)
                model.tensorboard_log = tensorboard_log
                return model
            return algo_cls(env=env, tensorboard_log=tensorboard_log, device=device, **policy_params)

        run_configs.append(
            RunConfig(
                id=id,
                id_env_config=id_env,
                id_obs_config=id_obs,
                id_algo_config=id_algo,
                folder_name=folder_name,
                make_env=env_factory,
                make_eval_env=eval_env_factory,
                make_model=model_factory,
                timesteps=TRAIN_TIMESTEPS,
                train_seed=0,
                eval_seed=1,
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
        id_obs: int = obs_config["id"]
        folder_name: str = str(BASE_OUTPUT_PATH / str(id))

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id: str = env_config["env_id"]
        env_config: Dict[str, Any] = env_config["config"]
        env_config["observation"] = obs_config

        def eval_env_factory(
            env_id: str = env_id,
            env_config: Dict[str, Any] = env_config,
        ) -> gym.Env:
            env_kwargs = {"config": env_config.copy()}
            return gym.make(env_id, max_episode_steps=None, disable_env_checker=None, **env_kwargs)

        instance_configs.append(
            InstanceConfig(
                id=id,
                id_env_config=id_env,
                id_obs_config=id_obs,
                make_eval_env=eval_env_factory,
                eval_seed=1,
            )
        )
    return instance_configs

def _nonempty_file_in(filepath: Path) -> bool:
    """Return True if filepath exists, is a regular file, and is non-empty."""
    try:
        return filepath.is_file() and filepath.stat().st_size > 0
    except OSError:
        return False

def is_trained(config: RunConfig) -> bool:
    """Trained iff training metadata artifact exists."""
    return _nonempty_file_in(Path(config.folder_name) / TRAINING_METADATA_FILE)

def is_evaluated(config: RunConfig) -> bool:
    """Evaluated iff evaluation results artifact exists."""
    return _nonempty_file_in(Path(config.folder_name) / EVALUATION_RESULTS_FILE)

def is_extracted(config: RunConfig) -> bool:
    """Extracted iff metafeatures result artifact exists."""
    return _nonempty_file_in(Path(config.folder_name) / METAFEATURES_RESULTS_FILE)

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
