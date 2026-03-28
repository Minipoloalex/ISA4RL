import datetime
import argparse
from functools import partial
from typing import Optional, Sequence, Dict, Callable, List

import gymnasium as gym
import highway_env
import pandas as pd

from configs import TrainConfig, InstanceConfig
from common.env_utils import ENVS
from utils.load_config_utils import load_env_train_configs, load_env_instance_configs
from main_helpers import train_agents, eval_agents, extract_metafeatures


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
        "--env",
        choices=ENVS,
        help="Highway-env environment to use",
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
    env_name = args.env

    load_configs = {
        "train": partial(load_env_train_configs, env_name),
        "evaluate": partial(load_env_train_configs, env_name),
        "extract": partial(load_env_instance_configs, env_name),
    }
    configs = load_configs[args.task]()

    total = len(configs)
    start = max(0, args.start)
    end = total if args.end is None else max(start, min(total, args.end))
    selected = configs[start:end]

    if not selected:
        print("No run configurations selected. Nothing to do.")
        return

    task_map: Dict[str, Callable[[List[TrainConfig]], None]] = {
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
