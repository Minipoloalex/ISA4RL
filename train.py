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

import numpy as np
import torch
import yaml
import gymnasium as gym
import pandas as pd
import highway_env
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

torch.set_num_threads(1)

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
    TRAINING_METADATA_FILE,
    MODEL_FILE,
    get_env_id,
    vectorize_env,
    unwrap_first_env,
)

def train(
    env: VecEnv,
    model: BaseAlgorithm,
    timesteps: int,
    folder_name: str,
    *,
    seed: Optional[int] = None,
    callback: Optional[BaseCallback] = None,
    progress_bar: bool = False,
    **kwargs,
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
    """
    output_dir = Path(folder_name).expanduser()
    ensure_dir(output_dir)

    logs_dir = output_dir / "logs"
    ensure_dir(logs_dir)

    tensorboard_dir = logs_dir / "tensorboard"
    ensure_dir(tensorboard_dir)

    base_env = unwrap_first_env(env)
    if seed is not None:
        set_global_seed(seed)
        try:
            env.seed(seed)
        except TypeError:
            print("env.reset() has no seed parameter")
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
    }
    if callback is not None:
        learn_kwargs["callback"] = callback
    if progress_bar:
        learn_kwargs["progress_bar"] = True

    before = time.perf_counter()
    model.learn(**learn_kwargs)
    elapsed = time.perf_counter() - before
    print(f"Training finished in {elapsed:.2f}s ({elapsed / 60:.2f}min) for {timesteps} timesteps.")

    model_path = output_dir / MODEL_FILE
    model.save(model_path)

    metadata = {
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "timestamp": time.time(),
        "model_path": str(model_path),
        "logs_dir": str(logs_dir),
        "env_config": base_env.unwrapped.config, # type: ignore
        "env_id": get_env_id(base_env),
        "model_class": model.__class__.__name__,
    }
    with (output_dir / TRAINING_METADATA_FILE).open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, default=_json_default, indent=2)

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment with training a PPO agent on highway-fast-v0"
    )
    parser.add_argument(
        "-n", "--name",
        dest="run_name",
        required=True,
        help="Name of the output folder/test run"
    )
    args = parser.parse_args()

    env = make_vec_env(
        "highway-fast-v0",
        n_envs=8,
        seed=0,
        vec_env_cls=SubprocVecEnv,
    )
    model = PPO("MlpPolicy", env, device="cpu",
        n_steps=32,
        batch_size=256,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.0,
    )
    train(env, model, int(2e5), args.run_name, seed=0, progress_bar=True)
