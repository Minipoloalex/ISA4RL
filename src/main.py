import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm
from pprint import pprint

import numpy as np
import yaml

from configs import RunConfig, InstanceConfig

from train import train
from evaluate import evaluate, show_eval_results
from metafeatures import extract_metafeatures as compute_metafeatures
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
    BASE_OUTPUT_PATH,
    save_eval_results,
    save_extract_results,
    is_trained,
    is_evaluated,
    is_extracted,
    load_all_run_configs,
    load_all_instance_configs,
)

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.linear_model import Ridge
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def train_agents(run_configs: List[RunConfig]):
    train_configs = filter(lambda config: not is_trained(config), run_configs)
    for config in tqdm(train_configs):
        env = config.ensure_env()
        model = config.ensure_model()
        train(
            env=env,
            model=model,
            timesteps=config.timesteps,
            folder_name=config.folder_name,
            seed=config.train_seed,
            progress_bar=True,
        )
        config.close()

def eval_agents(run_configs: List[RunConfig]):
    eval_configs = filter(lambda config: is_trained(config) and not is_evaluated(config), run_configs)
    for config in tqdm(eval_configs):
        eval_env = config.ensure_eval_env()
        model = config.ensure_model()
        eval_results = evaluate(
            model=model,
            env=eval_env,
            n_episodes=10,
            max_steps=None,
            deterministic=True,
            seed=config.eval_seed,
        )
        save_eval_results(eval_results, config.folder_name)
        config.close()

        show_eval_results(eval_results)

def extract_metafeatures(run_configs: List[RunConfig]):
    metafeature_configs = filter(lambda config: not is_extracted(config), run_configs)
    for config in tqdm(metafeature_configs):
        extract_results = compute_metafeatures(config)
        save_extract_results(extract_results, config.folder_name)
        config.close()

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run training, evaluation, or metafeature extraction for highway configs."
    )
    parser.add_argument(
        "--task",
        choices=("train", "evaluate", "extract"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inclusive starting index for selecting run configurations.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive ending index for selecting run configurations. Defaults to all.",
    )
    args = parser.parse_args(argv)

    configs = (
        load_all_instance_configs()
        if args.task == "extract"
        else load_all_run_configs()
    )

    total = len(configs)
    start = max(0, args.start)
    end = total if args.end is None else max(start, min(total, args.end))
    selected = configs[start:end]

    if not selected:
        print("No run configurations selected. Nothing to do.")
        return

    task_map: Dict[str, Callable[[List[InstanceConfig | RunConfig]], None]] = {
        "train": train_agents,
        "evaluate": eval_agents,
        "extract": extract_metafeatures,
    }   # type: ignore
    task_map[args.task](selected)   # type: ignore


if __name__ == "__main__":
    main()
