import datetime
import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque, defaultdict
from multiprocessing import get_context, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm
from pprint import pprint
from functools import partial

import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sklearn.linear_model import Ridge
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from configs import TrainConfig, InstanceConfig
from train import train
from evaluate import evaluate, show_eval_results
from metafeatures import extract_metafeatures as compute_metafeatures
from common.file_utils import save_json
from utils.load_config_utils import (
    is_trained,
    is_evaluated,
    is_extracted,
    RESULTS_TRAIN_METADATA_PATH,
    RESULTS_METAFEATURES_PATH,
    RESULTS_EVALUATION_PATH,
)
from multiprocessing import get_context, cpu_count

def train_agents(train_configs: List[TrainConfig]):
    train_configs = [config for config in train_configs if not is_trained(config)]
    for config in tqdm(train_configs, total=len(train_configs)):
        train_env = config.ensure_train_env()
        model = config.ensure_model()
        eval_env = config.ensure_eval_env()
        train(
            env=train_env,
            model=model,
            timesteps=config.timesteps,
            folder_name=str(config.train_folder_path),
            eval_env=eval_env,
            n_eval_episodes=config.n_eval_episodes,
            eval_freq=config.eval_freq,
            seed=0,
            progress_bar=True,
        )
        config.close()


def eval_agents(train_configs: List[TrainConfig]):
    eval_configs = [
        config
        for config in train_configs
        if is_trained(config) and not is_evaluated(config)
    ]
    for config in tqdm(eval_configs, total=len(eval_configs)):
        eval_env=config.ensure_eval_env()
        model=config.ensure_model()
        eval_results=evaluate(
            model=model,
            env=eval_env,
            n_episodes=config.n_test_episodes,
            deterministic=True,
        )
        save_json(RESULTS_EVALUATION_PATH(config.train_folder_path), eval_results)
        config.close()

        show_eval_results(eval_results)


def _extract_and_save(config: InstanceConfig) -> None:
    extract_results = compute_metafeatures(config)
    save_json(RESULTS_METAFEATURES_PATH(config.instance_folder_path), extract_results)
    config.close()


def extract_metafeatures(instance_configs: List[InstanceConfig], workers: int):
    assert(workers > 0)
    metafeature_configs = [
        config for config in instance_configs if not is_extracted(config)
    ]
    desc = f"Extracting metafeatures (workers={workers})"
    if workers == 1:
        for config in tqdm(
            metafeature_configs, total=len(metafeature_configs), desc=desc
        ):
            _extract_and_save(config)
        return

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_extract_and_save, metafeature_configs),
            total=len(metafeature_configs),
            desc=desc,
        ):
            pass
