import datetime
import argparse
import gc
import json
import math
import os
import random
import time
import logging
import traceback
from collections import Counter, deque, defaultdict
from multiprocessing import get_context, cpu_count, Queue
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
from methods.metafeatures import TIMESTAMP_KEY
from methods.check import check_helper
from common.file_utils import save_json, read_json, BASE_RESULTS_PATH, OTHER_RESULTS_PATH, nonempty_file_in
from methods.utils.load_config_utils import (
    is_trained,
    is_evaluated,
    is_extracted,
    RESULTS_TRAIN_METADATA_PATH,
    RESULTS_METAFEATURES_PATH,
    RESULTS_EVALUATION_PATH,
)
from methods.utils.group_utils import (
    find_config_in_folder,
    find_instance_config_in_folder,
    is_combination_trained,
    is_combination_extracted,
    merge_result_folders,
    merge_metafeature_results,
)
from multiprocessing import get_context, cpu_count

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

def _train_one_agent(config: TrainConfig, run_index: int) -> None:
    train_env = None
    eval_env = None
    model = None
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
        logger.info(f"Started training run {run_index} in path: {config.train_folder_path}")
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
        del model
        del eval_env
        del train_env

def train_agents(train_configs: List[TrainConfig]):
    train_configs = [config for config in train_configs if not is_trained(config)]
    for i, config in tqdm(enumerate(train_configs), total=len(train_configs)):
        _train_one_agent(config, i)
        logger.info(f"Saved training information from run in {config.train_folder_path}")


def eval_agents(train_configs: List[TrainConfig]):
    eval_configs = [
        config
        for config in train_configs
        if is_trained(config) and not is_evaluated(config)
    ]
    for config in tqdm(eval_configs, total=len(eval_configs)):
        _eval_one_agent(config)


def _eval_one_agent(config: TrainConfig) -> None:
    eval_env = None
    model = None
    eval_results = None
    try:
        eval_env = config.ensure_eval_env()
        model = config.ensure_model_for_env(eval_env)
        eval_results = evaluate(
            model=model,
            env=eval_env,
            n_episodes=config.n_test_episodes,
            deterministic=True,
        )
        save_json(RESULTS_EVALUATION_PATH(config.train_folder_path), eval_results)
        show_eval_results(eval_results)
    finally:
        config.close()
        del eval_results
        del model
        del eval_env


def _extract_and_save(
    config: InstanceConfig,
    env_name: str,
    requested_groups: Optional[List[str]] = None,
    update_threshold: float = 0.0,
) -> None:
    path = RESULTS_METAFEATURES_PATH(config.instance_folder_path)
    existing_data = read_json(path) if path.is_file() else {}
    extract_results = compute_metafeatures(env_name, config, requested_groups, existing_data, update_threshold)
    save_json(path, extract_results)
    logger.info(f"Done: {config.instance_folder_path}")


def extract_metafeatures(
    env_name: str,
    instance_configs: List[InstanceConfig],
    workers: int,
    requested_groups: Optional[List[str]] = None,
    update_threshold: float = 0.0,
) -> None:
    assert workers > 0
    
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
            if feature_groups[g][TIMESTAMP_KEY] < update_threshold:
                return True
        return False

    metafeature_configs = [
        config for config in instance_configs if needs_compute(config)
    ]
    desc = f"Extracting metafeatures (workers={workers})"
    
    extract_fn = partial(
        _extract_and_save, env_name=env_name, requested_groups=requested_groups, update_threshold=update_threshold,
    )
    
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

def check_agents(train_configs: List[TrainConfig], instance_configs: List[InstanceConfig]):
    not_trained_segs = check_helper(train_configs, is_trained)
    not_evaled_segs = check_helper(train_configs, is_evaluated)
    not_extracted_segs = check_helper(instance_configs, is_extracted)
    logger.info(f"Total train_configs: {len(train_configs)}, Total instance configs: {len(instance_configs)}")
    logger.info(f"Segments not trained yet: {not_trained_segs}")
    logger.info(f"Segments not evaluated yet: {not_evaled_segs}")
    logger.info(f"Segments not extracted yet: {not_extracted_segs}")
    


def group_results(
        result_folders: list[str],
    ):
    # Assumptions:1
    # 1. The `results` folder contains all the environments and all the algorithms (and it is clean)
    # 2. The folder structure is consistent throughout all the `results` folders
    # 3. The combination `environment + algorithm` has files in exactly one of the `result_folders`
    #    (if the combination ran partially in one and completely in the other, there is a 50-50 chance of breaking,
    #     if the combination is not present in any computer, the folder will be left empty)

    folder = BASE_RESULTS_PATH
    for env_folder in folder.iterdir(): # e.g. exit/
        if env_folder.name == "isa" or not env_folder.is_dir():
            continue
        logger.info(
            f"Started grouping results for environment {env_folder.name} for {len(list(env_folder.iterdir()))} instance folders"
        )
        for env_instance_folder in env_folder.iterdir(): # exit/<id>/
            env_config_file = env_instance_folder / "instance_config.json" # exit/<id>/instance_config.json
            env_config = read_json(env_config_file)

            for result_folder in result_folders:
                result_folder_to_merge = find_instance_config_in_folder(
                    OTHER_RESULTS_PATH(result_folder),
                    env_folder.name,
                    env_config,
                )
                if result_folder_to_merge is None:
                    continue

                if is_combination_extracted(result_folder_to_merge):
                    merge_metafeature_results(
                        RESULTS_METAFEATURES_PATH(result_folder_to_merge),  # <other-results>/exit/<id>/metafeatures.json
                        RESULTS_METAFEATURES_PATH(env_instance_folder),     # results/exit/<id>/metafeatures.json
                    )
                    break
            if not (env_instance_folder / "train").is_dir():
                continue
            for final_results_folder in (env_instance_folder / "train").iterdir(): # exit/<id>/train/<id>/
                algo_config_file = final_results_folder / "algo_config.json" # exit/<id>/train/<id>/algo_config.json
                algo_config = read_json(algo_config_file)

                # Now, go through the other result folders to merge
                for result_folder in result_folders:
                    result_folder_to_merge = find_config_in_folder(
                        OTHER_RESULTS_PATH(result_folder),
                        env_folder.name,
                        env_config,
                        algo_config,
                    )
                    if result_folder_to_merge is None:
                        continue

                    if is_combination_trained(result_folder_to_merge):
                        merge_result_folders(result_folder_to_merge, final_results_folder)
                        break
        logger.info(f"Finished grouping results for environment: {env_folder.name}")
