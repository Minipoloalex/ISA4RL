import datetime
import argparse
from functools import partial
from typing import Optional, Sequence, Dict, Callable, List

import gymnasium as gym
import pandas as pd

from methods.configs import TrainConfig, InstanceConfig
from common.env_utils import HIGHWAY_ENVS, METADRIVE_ENVS
from common.file_utils import OTHER_RESULTS_PATH
from methods.utils.load_config_utils import load_env_train_configs, load_env_instance_configs
from methods.main_helpers import train_agents, eval_agents, extract_metafeatures, check_agents, group_results

def main(valid_envs: Optional[List[str]], argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run training, evaluation, metafeature extraction, or checks of what has been run for highway-env configs."
    )
    valid_envs = valid_envs if valid_envs is not None else HIGHWAY_ENVS + METADRIVE_ENVS

    parser.add_argument(
        "--task",
        choices=("train", "evaluate", "extract", "check", "group"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--env",
        choices=valid_envs,
        help="Environment to use",
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
        "--result-folders",
        type=str,
        default="",
        help="List of result folders to group into the set of results (comma separated)"
    )
    parser.add_argument(    # for metafeatures
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes to use for metafeature extraction (extract task only).",
    )
    parser.add_argument(    # for metafeatures
        "--metafeature-groups",
        type=str,
        default=None,
        help="Comma-separated list of metafeature groups to compute (e.g. 'env_features,mb_state_entropy'). Default computes all.",
    )
    parser.add_argument(    # for metafeatures
        "--update",
        type=float,
        default=1.0,
        help="Unix timestamp float. Use an helper to convert a date to a timestamp. Groups older than this will be recomputed.",
    )
    args = parser.parse_args(argv)
    env_name = args.env
    result_folders = args.result_folders.split(",")

    load_configs = {
        "train": partial(load_env_train_configs, env_name),
        "evaluate": partial(load_env_train_configs, env_name),
        "extract": partial(load_env_instance_configs, env_name),
        "check": partial(load_env_train_configs, env_name), # used to check what configs have already trained
        "group": lambda : [ 0 ], # unused, just to not crash
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
        "check": check_agents,
    }  # type: ignore
    if args.task == "extract":
        groups = args.metafeature_groups.split(",") if args.metafeature_groups else None
        extract_metafeatures(
            selected, 
            workers=args.workers, 
            requested_groups=groups, 
            update_threshold=args.update
        )  # type: ignore[arg-type]
    elif args.task == "group":
        group_results(result_folders)
    else:
        print("", datetime.datetime.now(), "\n\n\n")
        task_map[args.task](selected)  # type: ignore
        print("\n\n\n", datetime.datetime.now())
        
    print("Exiting")

if __name__ == "__main__":
    main(None)
