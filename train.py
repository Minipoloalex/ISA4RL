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

import numpy as np
import yaml
import gymnasium as gym
import pandas as pd
import highway_env

from utils import (
    set_global_seed,
    ensure_dir,
    discretize,
    _flatten_obs,
    _normalize_action,
    _round_half_up,
    _interpolate_range_value,
    _coerce_numeric,
    _json_default,
    METADATA_FILE,
    get_env_id,
)
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure

def train(
    env: gym.Env,
    model: BaseAlgorithm,
    timesteps: int,
    folder_name: str,
    *,
    seed: Optional[int] = None,
    callback: Optional[BaseCallback] = None,
    log_interval: int = 1,
    progress_bar: bool = False,
) -> Path:
    """Train a Stable-Baselines3 model and persist artifacts to disk.

    Args:
        env: Gymnasium environment already configured with the desired dynamics.
        model: Instantiated Stable-Baselines3 algorithm ready for training.
        timesteps: Number of environment steps used for learning.
        folder_name: Target directory where logs and the trained model are written.
        seed: Optional integer seed to make training reproducible.
        callback: Optional SB3 callback invoked during learning.
        progress_bar: Whether to display Stable-Baselines3's tqdm progress bar.

    Returns:
        Path: Filesystem path to the saved model archive (``.zip``).

    Raises:
        ValueError: If ``timesteps`` is not a positive integer.
    """
    output_dir = Path(folder_name).expanduser()
    ensure_dir(output_dir)

    logs_dir = output_dir / "logs"
    ensure_dir(logs_dir)

    tensorboard_dir = logs_dir / "tensorboard"
    ensure_dir(tensorboard_dir)

    if seed is not None:
        set_global_seed(seed)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(seed)

    model.set_env(env)

    # Configure logging so that we always emit CSV + TensorBoard traces.
    logger: Logger = configure(str(logs_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    model.tensorboard_log = str(tensorboard_dir)

    learn_kwargs: Dict[str, Any] = {
        "total_timesteps": timesteps,
        "log_interval": log_interval,
    }
    if callback is not None:
        learn_kwargs["callback"] = callback
    if progress_bar:
        learn_kwargs["progress_bar"] = True

    before = time.perf_counter()
    model.learn(**learn_kwargs)
    elapsed = time.perf_counter() - before
    print(f"Training finished in {elapsed:.2f}s for {timesteps} timesteps.")

    model_path = output_dir / "model"
    model.save(model_path)

    metadata = {
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "timestamp": time.time(),
        "model_path": str(model_path) + ".zip",
        "logs_dir": str(logs_dir),
        "env_config": env.unwrapped.config, # type: ignore
        "env_id": get_env_id(env),
        "model_class": model.__class__.__name__,
    }
    with (output_dir / METADATA_FILE).open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, default=_json_default, indent=2)

    return model_path


if __name__ == "__main__":
    env = gym.make("highway-fast-v0")
    model = PPO("MlpPolicy", env, device="cpu")
    train(env, model, int(5e4), "first_test", seed=0, progress_bar=True)
