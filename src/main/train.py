import argparse
import json
import math
import os
import random
import time
from datetime import datetime
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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv

from common.file_utils import (
    TENSORBOARD_FOLDER, 
    LOGS_FOLDER,
    MODELS_FOLDER,
    RESULTS_TRAIN_METADATA_PATH,
    MODEL_FILE,
    BEST_MODEL_FILE,
    BEST_VEC_NORMALIZE_FILE,
    VEC_NORMALIZE_FILE,
)
torch.set_num_threads(1)

from utils.general_utils import (
    set_global_seed,
    ensure_dir,
    discretize,
    _flatten_obs,
    _normalize_action,
    _round_half_up,
    _interpolate_range_value,
    _coerce_numeric,
)
from utils.sb3_utils import get_env_id, unwrap_first_env, find_vec_normalize
from common.file_utils import _json_default

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.looked = False

    def _on_step(self) -> bool:
        if not self.looked:
            self.vec_normalize = find_vec_normalize(self.training_env)
            self.looked = True

        # Find the wrapper in the training env and save its stats
        if self.vec_normalize is not None:
            self.vec_normalize.save(self.save_path)
        return True

def train(
    env: VecEnv,
    model: BaseAlgorithm,
    timesteps: int,
    folder_name: str,
    eval_env: VecEnv,
    n_eval_episodes: int,
    eval_freq: int,
    *,
    seed: Optional[int] = None,
    progress_bar: bool = False,
    **kwargs,
) -> Tuple[Path, Path]:
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

    logs_dir = output_dir / LOGS_FOLDER
    ensure_dir(logs_dir)

    tensorboard_dir = logs_dir / TENSORBOARD_FOLDER
    ensure_dir(tensorboard_dir)

    models_dir = output_dir / MODELS_FOLDER
    ensure_dir(models_dir)

    base_env = unwrap_first_env(env)
    if seed is not None:
        set_global_seed(seed)
        env.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(seed)

    model.set_env(env)

    # Configure logging so that we always emit CSV + TensorBoard traces.
    logger: Logger = configure(str(logs_dir), ["csv", "tensorboard"])
    model.set_logger(logger)
    model.tensorboard_log = str(tensorboard_dir)

    learn_kwargs: Dict[str, Any] = {
        "total_timesteps": timesteps,
    }

    best_model_path = models_dir / BEST_MODEL_FILE
    best_vec_normalize_path = models_dir / BEST_VEC_NORMALIZE_FILE
    on_best_callback = SaveVecNormalizeCallback(str(best_vec_normalize_path))
    learn_kwargs["callback"] = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(models_dir),   # auto saves to models_dir / best_model.zip
        log_path=folder_name,
        eval_freq=eval_freq,    # already takes into account the number of vectorized environments
        callback_on_new_best=on_best_callback,
        deterministic=True,
        render=False,
        verbose=1,
    )
    learn_kwargs["progress_bar"] = progress_bar

    # Gather environment metadata without forcing access to the raw env when using
    # SubprocVecEnv, which keeps environments in separate processes
    if base_env is not None:
        env_config = base_env.unwrapped.config  # type: ignore
        env_id = get_env_id(base_env)
    else:
        configs = env.get_attr("config")
        env_config = configs[0]
        specs = env.get_attr("spec")
        env_id = specs[0].id

    before = time.perf_counter()
    model.learn(**learn_kwargs)
    elapsed = time.perf_counter() - before
    print(f"Training finished in {elapsed:.2f}s ({elapsed / 60:.2f} min) for {timesteps} timesteps.")

    final_model_path = models_dir / MODEL_FILE
    model.save(final_model_path)

    final_vec_normalize_path = models_dir / VEC_NORMALIZE_FILE
    vec_normalize = find_vec_normalize(env)
    if vec_normalize is not None:
        vec_normalize.save(str(final_vec_normalize_path))    

    metadata = {
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "logs_dir": str(logs_dir),
        "env_config": env_config,
        "env_id": env_id,
        "model_class": model.__class__.__name__,
    }
    metadata_output_path = RESULTS_TRAIN_METADATA_PATH(output_dir)
    with metadata_output_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, default=_json_default, indent=2)

    return best_model_path, best_vec_normalize_path
