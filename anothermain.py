
from __future__ import annotations

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
from tqdm import tqdm

import numpy as np
import yaml

from run_config import RunConfig

from train import train
from evaluate import evaluate
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
)

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install gymnasium") from exc

try:
    import highway_env  # noqa: F401  # ensures envs are registered
except ImportError as exc:  # pragma: no cover
    raise ImportError("highway-env is required: pip install highway-env") from exc

try:
    from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required for training probes and portfolio agents."
    ) from exc

try:
    from sklearn.linear_model import Ridge
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError("pandas is required: pip install pandas") from exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for trajectory embeddings.") from exc


# TODO:
# def build_configs() -> List[RunConfig]:

# TODO:
# perform ISA


def train_agents(run_configs: List[RunConfig]):
    train_configs = filter(lambda config: not is_trained(config), run_configs)
    for config in tqdm(train_configs):
        train(**asdict(config), progress_bar=True)

def eval_agents(run_configs):
    eval_configs = filter(lambda config: is_trained(config) and not is_evaluated(config), run_configs)
    for config in tqdm(eval_configs):
        eval_results = evaluate(**asdict(config), n_episodes=10, deterministic=True)
        save_eval_results(eval_results, config.folder_name)

def extract_metafeatures(run_configs):
    metafeature_configs = filter(lambda config: is_trained(config) and not is_extracted(config), run_configs)
    for config in tqdm(metafeature_configs):
        extract_results = extract_metafeatures(**asdict(config))
        save_extract_results(extract_results, config.folder_name)


def main():
    # parse args here
    pass


if __name__ == "__main__":
    main()

"""
An idea would be that the timestamp is the instance id.
Then, there would be a file to map from timestamp to instance configurations.

Based on the timestamp, we can get the agent information:
the name of the folder with the trained agent would have been a function of the timestamp.

We can keep a list of timestamps and environments that we want to run, and another corresponding to those we've already run.
If there's an error somewhere, some configs will be saved, and running again will avoid those.
The same could be said for training agents.
"""

