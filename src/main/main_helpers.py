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

from configs import RunConfig, InstanceConfig
from train import train
from evaluate import evaluate, show_eval_results
from metafeatures import extract_metafeatures as compute_metafeatures
from common.config_utils import get_all_train_configs, get_all_eval_configs
from common.file_utils import (
    save_eval_results,
    save_extract_results,
)
from utils.load_config_utils import (
    load_all_run_configs,
    load_all_instance_configs,
    is_trained,
    is_evaluated,
    is_extracted,
)
from multiprocessing import get_context, cpu_count

def train_agents(run_configs: List[RunConfig]):
    train_configs = [config for config in run_configs if not is_trained(config)]
    for config in tqdm(train_configs, total=len(train_configs)):
        train_env = config.ensure_train_env()
        model = config.ensure_model()
        eval_env = config.ensure_eval_env()
        train(
            env=train_env,
            model=model,
            timesteps=config.timesteps,
            folder_name=config.train_folder_name,
            eval_env=eval_env,
            n_eval_episodes=config.n_eval_episodes,
            eval_freq=config.eval_freq,
            seed=config.train_seed,
            progress_bar=True,
        )
        config.close()


def eval_agents(run_configs: List[RunConfig]):
    eval_configs = [
        config
        for config in run_configs
        if is_trained(config) and not is_evaluated(config)
    ]
    for config in tqdm(eval_configs, total=len(eval_configs)):
        test_env = config.ensure_test_env()
        model = config.ensure_model()
        eval_results = evaluate(
            model=model,
            env=test_env,
            n_episodes=10,
            deterministic=False,
            env_seed=config.eval_seed,
        )
        save_eval_results(eval_results, config.train_folder_name, config.eval_seed)
        config.close()

        show_eval_results(eval_results)


def _extract_and_save(config: InstanceConfig) -> None:
    extract_results = compute_metafeatures(config)
    save_extract_results(extract_results, config.instance_folder_name)
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
