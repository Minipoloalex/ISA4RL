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

from configs import RunConfig, InstanceConfig

from train import train
from evaluate import evaluate, show_eval_results
from metafeatures import extract_metafeatures as compute_metafeatures
from file_utils import _json_default
from utils import (
    save_eval_results,
    save_extract_results,
    is_trained,
    is_evaluated,
    is_extracted,
    get_all_train_configs,
    get_all_eval_configs,
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
    train_configs = [config for config in run_configs if not is_trained(config)]
    for config in tqdm(train_configs, total=len(train_configs)):
        env = config.ensure_env()
        model = config.ensure_model()
        train(
            env=env,
            model=model,
            timesteps=config.timesteps,
            folder_name=config.train_folder_name,
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
        eval_env = config.ensure_eval_env()
        model = config.ensure_model()
        eval_results = evaluate(
            model=model,
            env=eval_env,
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes to use for metafeature extraction (extract task only).",
    )
    args = parser.parse_args(argv)

    load_configs = {
        "train": partial(load_all_run_configs, get_all_train_configs),
        "evaluate": partial(load_all_run_configs, get_all_eval_configs),
        "extract": load_all_instance_configs,
    }
    configs = load_configs[args.task]()

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
    }  # type: ignore
    if args.task == "extract":
        extract_metafeatures(selected, workers=args.workers)  # type: ignore[arg-type]
    else:
        print("", datetime.datetime.now(), "\n\n\n")
        task_map[args.task](selected)  # type: ignore
        print("\n\n\n", datetime.datetime.now())

if __name__ == "__main__":
    main()
