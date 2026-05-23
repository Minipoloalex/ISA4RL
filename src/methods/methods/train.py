import argparse
import json
import logging
import math
import os
import random
import re
import shutil
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
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger as SB3Logger, configure
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
from methods.utils.general_utils import (
    set_global_seed,
    ensure_dir,
    discretize,
    _flatten_obs,
    _normalize_action,
    _round_half_up,
    _interpolate_range_value,
    _coerce_numeric,
)
from methods.utils.sb3_utils import get_env_id, unwrap_first_env, find_vec_normalize
from common.file_utils import _json_default
from methods.evaluate import CHECKPOINT_SELECTION_BASE_SEED, evaluate

try:
    import highway_env
except:
    pass
try:
    import metadrive
except:
    pass

torch.set_num_threads(1)

logger = logging.getLogger(__name__)

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
    eval_env: Optional[VecEnv],
    n_eval_episodes: int,
    eval_freq: int,
    *,
    seed: Optional[int] = None,
    progress_bar: bool = False,
    post_training_eval_env: Optional[VecEnv] = None,
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
    sb3_logger: SB3Logger = configure(str(logs_dir), ["csv", "tensorboard"])
    model.set_logger(sb3_logger)
    model.tensorboard_log = str(tensorboard_dir)

    learn_kwargs: Dict[str, Any] = {
        "total_timesteps": timesteps,
    }

    best_model_path = models_dir / BEST_MODEL_FILE
    best_vec_normalize_path = models_dir / BEST_VEC_NORMALIZE_FILE
    checkpoint_dir = models_dir / "checkpoints"
    if eval_env is not None:
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
    else:
        ensure_dir(checkpoint_dir)
        learn_kwargs["callback"] = CheckpointCallback(
            save_freq=eval_freq,
            save_path=str(checkpoint_dir),
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=False,    # there won't be any vec normalize for carla
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
    logger.info(
        "Training finished in %.2fs (%.2f min) for %d timesteps.",
        elapsed,
        elapsed / 60,
        timesteps,
    )
    final_model_path = models_dir / MODEL_FILE
    model.save(final_model_path)

    final_vec_normalize_path = models_dir / VEC_NORMALIZE_FILE
    vec_normalize = find_vec_normalize(env)
    if vec_normalize is not None:
        vec_normalize.save(str(final_vec_normalize_path))

    checkpoint_evaluation_results = None
    if post_training_eval_env is not None:
        logger.info("Starting post-training checkpoint evaluation.")
        checkpoint_evaluation_results = evaluate_checkpoints_after_training(
            model=model,
            model_paths=get_checkpoint_model_paths(
                checkpoint_dir,
                final_model_path,
                max_step=timesteps,
            ),
            eval_env=post_training_eval_env,
            n_eval_episodes=n_eval_episodes,
            best_model_path=best_model_path,
        )
        best_checkpoint_evaluation = max(
            checkpoint_evaluation_results,
            key=lambda checkpoint_evaluation: checkpoint_evaluation["mean_reward"],
        )
        logger.info(
            "Finished evaluating %d model checkpoints. Best checkpoint: %s "
            "(mean_reward=%.6f, episodes=%d). Saved best model to %s.",
            len(checkpoint_evaluation_results),
            best_checkpoint_evaluation["model_path"],
            best_checkpoint_evaluation["mean_reward"],
            len(best_checkpoint_evaluation["episode_rewards"]),
            best_model_path,
        )

    env_config_dict = (
        env_config if type(env_config) is dict else env.get_attr("orig_config")[0]
    )
    metadata = {
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(final_model_path),
        "best_model_path": str(best_model_path)
        if eval_env is not None or post_training_eval_env is not None
        else None,
        "checkpoint_evaluations": checkpoint_evaluation_results,
        "logs_dir": str(logs_dir),
        "env_config": env_config_dict,
        "env_id": env_id,
        "model_class": model.__class__.__name__,
    }
    metadata_output_path = RESULTS_TRAIN_METADATA_PATH(output_dir)
    with metadata_output_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, default=_json_default, indent=2)

    if eval_env is None and post_training_eval_env is None:
        return final_model_path, final_vec_normalize_path
    return best_model_path, best_vec_normalize_path


def get_checkpoint_model_paths(
    checkpoint_dir: Path,
    final_model_path: Path,
    *,
    max_step: int,
) -> List[Path]:
    checkpoint_paths = sorted(
        (
            path
            for path in checkpoint_dir.glob("model_*_steps.zip")
            if _checkpoint_step(path) < max_step
        ),
        key=_checkpoint_step,
    )
    return [*checkpoint_paths, final_model_path]


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)_steps\.zip", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {path}")
    return int(match.group(1))


def evaluate_checkpoints_after_training(
    *,
    model: BaseAlgorithm,
    model_paths: Sequence[Path],
    eval_env: VecEnv,
    n_eval_episodes: int,
    best_model_path: Path,
) -> List[Dict[str, Any]]:
    if not model_paths:
        raise ValueError(
            "No checkpoint models were available for post-training evaluation."
        )

    candidate_model = None
    best_mean_reward = -math.inf
    best_candidate_path = None
    checkpoint_evaluation_results: List[Dict[str, Any]] = []

    try:
        for model_path in model_paths:
            if candidate_model is not None:
                del candidate_model
                candidate_model = None
            candidate_model = model.__class__.load(
                str(model_path),
                env=eval_env,
                device=model.device,
            )
            episode_results = evaluate(
                model=candidate_model,
                env=eval_env,
                n_episodes=n_eval_episodes,
                base_seed=CHECKPOINT_SELECTION_BASE_SEED,
                deterministic=True,
            )
            rewards = [episode_result["reward"] for episode_result in episode_results]
            mean_reward = float(np.mean(rewards))
            result = {
                "model_path": str(model_path),
                "mean_reward": mean_reward,
                "base_seed": CHECKPOINT_SELECTION_BASE_SEED,
                "episode_rewards": rewards,
                "episode_lengths": [
                    episode_result["length"] for episode_result in episode_results
                ],
                "episode_seeds": [
                    episode_result["seed"] for episode_result in episode_results
                ],
            }
            checkpoint_evaluation_results.append(result)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_candidate_path = model_path

        if best_candidate_path is None:
            raise RuntimeError("Could not select a best checkpoint model.")
        shutil.copy2(best_candidate_path, best_model_path)
        return checkpoint_evaluation_results
    finally:
        if candidate_model is not None:
            del candidate_model
