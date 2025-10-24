import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from run_config import RunConfig
import gymnasium as gym

AlgorithmName = str
ALGORITHM_MAP: Dict[AlgorithmName, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

BASE_OUTPUT_PATH = Path("results")
CONFIG_FILE = "configs.json"
MODEL_FILE = "model.zip"

TRAINING_METADATA_FILE = "training_metadata.json"
EVALUATION_RESULTS_FILE = "eval_results.json"
METAFEATURES_RESULTS_FILE = "metafeatures_result.json"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_json(file: Path):
    with file.open("w", encoding="utf-8") as fp:
        return json.load(fp)

def save_json(file: Path, results):
    with file.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=_json_default)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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

def save_eval_results(results, folder_name: str):
    filepath = Path(folder_name) / EVALUATION_RESULTS_FILE
    save_json(filepath, results)

def save_extract_results(results, folder_name: str):
    filepath = Path(folder_name) / METAFEATURES_RESULTS_FILE
    save_json(filepath, results)


def load_all_configs() -> List[RunConfig]:
    config_data = read_json(Path(CONFIG_FILE))
    configs = []
    for config_id, config in config_data:
        env = build_env(config["env_id"], config["env_config"])
        algo_builder = map_algo_name_to_class(config["model"])
        algo = algo_builder(
            env=env,
            tensorboard_log=config["folder_name"],
            **config["policy_params"],
        )
        configs.append(RunConfig(
            id=config_id,
            folder_name=config["folder_name"],
            env=env,
            model=algo,
            timesteps=config["timesteps"],
            seed=config["seed"],
        ))
    return configs

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

