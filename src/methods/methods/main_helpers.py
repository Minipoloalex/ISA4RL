import datetime
import argparse
import json
import math
import os
import random
import time
import logging
from collections import Counter, deque, defaultdict
from multiprocessing import get_context, cpu_count
from pathlib import Path
from typing import Any, Callable, List, Optional
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

from methods.configs import TrainConfig, InstanceConfig
from methods.train import train
from methods.evaluate import evaluate, show_eval_results
from methods.metafeatures import extract_metafeatures as compute_metafeatures
from common.file_utils import save_json, read_json, BASE_RESULTS_PATH, OTHER_RESULTS_PATH, nonempty_file_in
from methods.utils.load_config_utils import (
    is_trained,
    is_evaluated,
    is_extracted,
    RESULTS_TRAIN_METADATA_PATH,
    RESULTS_METAFEATURES_PATH,
    RESULTS_EVALUATION_PATH,
)
from methods.utils.group_utils import find_config_in_folder, is_combination_trained, merge_result_folders
from multiprocessing import get_context, cpu_count

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

def train_agents(train_configs: List[TrainConfig]):
    train_configs = [config for config in train_configs if not is_trained(config)]
    for i, config in tqdm(enumerate(train_configs), total=len(train_configs)):
        try:
            try:
                logger.info(f"open fds before env creation: {len(os.listdir("/proc/self/fd"))}")
            except:
                pass
            train_env = config.ensure_train_env()
            model = config.ensure_model()
            eval_env = config.ensure_eval_env()
            try:
                logger.info(f"open fds after env creation: {len(os.listdir("/proc/self/fd"))}")
            except:
                pass
            logger.info(f"Started training run {i} in path: {config.train_folder_path}")
            logger.info(f"Configuration for algorithm: {model.__str__()}")
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
        finally:
            config.close()
        logger.info(f"Saved training information from run in {config.train_folder_path}")


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


def _extract_and_save(
    config: InstanceConfig,
    requested_groups: Optional[List[str]] = None,
    update_threshold: float = 0.0,
) -> None:
    path = RESULTS_METAFEATURES_PATH(config.instance_folder_path)
    existing_data = read_json(path) if path.is_file() else {}
    extract_results = compute_metafeatures(config, requested_groups, existing_data, update_threshold)
    save_json(path, extract_results)


def extract_metafeatures(instance_configs: List[InstanceConfig], workers: int, requested_groups: Optional[List[str]] = None, update_threshold: float = 0.0):
    assert(workers > 0)
    
    def needs_compute(config: InstanceConfig) -> bool:
        path = RESULTS_METAFEATURES_PATH(config.instance_folder_path)
        if not path.is_file():
            return True
        if update_threshold < 0 and requested_groups is None:
            return False
            
        data = read_json(path) or {}
        feature_groups = data.get("feature_groups", {})
        
        groups_to_check = requested_groups
        if groups_to_check is None:
            # Check standard groups if none specified
            groups_to_check = [
                "env_features",
                "probes",
                "mb_normalized_lipschitz",
                "mb_transition_stochasticity",
                "mb_transition_linearity",
                "mb_action_landscape_ruggedness",
                "mb_state_entropy",
                # "pic",
            ]

        for g in groups_to_check:
            if g not in feature_groups:
                return True
            if feature_groups[g].get("timestamp", 0.0) < update_threshold:
                return True
        return False

    metafeature_configs = [
        config for config in instance_configs if needs_compute(config)
    ]
    desc = f"Extracting metafeatures (workers={workers})"
    
    extract_fn = partial(_extract_and_save, requested_groups=requested_groups, update_threshold=update_threshold)
    
    if workers == 1:
        for config in tqdm(
            metafeature_configs, total=len(metafeature_configs), desc=desc
        ):
            extract_fn(config)
        return

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(extract_fn, metafeature_configs),
            total=len(metafeature_configs),
            desc=desc,
        ):
            pass

def check_agents(train_configs: List[TrainConfig]):
    segs = []   # segments of not trained configurations
    i = 0
    while i < len(train_configs):
        if not is_trained(train_configs[i]):
            lo = i
            hi = i + 1
            while hi < len(train_configs) and not is_trained(train_configs[hi]):
                hi += 1
            segs.append((lo, hi))
            i = hi - 1
        i += 1
    pprint(f"Segments not trained yet: {segs}")


def group_results(
        result_folders: list[str],
    ):
    # Assumptions:
    # 1. The `results` folder contains all the environments and all the algorithms (and it is clean)
    # 2. The folder structure is consistent throughout all the `results` folders
    # 3. The combination `environment + algorithm` has files in exactly one of the `result_folders`
    #    (if the combination ran partially in one and completely in the other, there is a 50-50 chance of breaking,
    #     if the combination is not present in any computer, the folder will be left empty)

    folder = BASE_RESULTS_PATH
    for env_folder in folder.iterdir():
        if env_folder.name == "isa":
            continue
        for env_instance_folder in env_folder.iterdir():
            env_config_file = env_instance_folder / "instance_config.json"
            env_config = read_json(env_config_file)
            for final_results_folder in (env_instance_folder / "train").iterdir():
                algo_config_file = final_results_folder / "algo_config.json"
                algo_config = read_json(algo_config_file)

                # Now, go through the other result folders to merge
                for result_folder in result_folders:
                    result_folder_to_merge = find_config_in_folder(OTHER_RESULTS_PATH(result_folder), (env_config, algo_config))
                    if result_folder_to_merge is None:
                        continue

                    if is_combination_trained(result_folder_to_merge):
                        merge_result_folders(result_folder_to_merge, final_results_folder)
                        break
