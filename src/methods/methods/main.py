import datetime
import argparse
import time
from functools import partial
from typing import Optional, Sequence, Dict, Callable, List
import logging

import gymnasium as gym
import pandas as pd

from methods.configs import TrainConfig, InstanceConfig
from common.env_utils import HIGHWAY_ENVS, METADRIVE_ENVS
from common.file_utils import OTHER_RESULTS_PATH
from methods.utils.load_config_utils import load_env_train_configs, load_env_instance_configs
from methods.main_helpers import train_agents, eval_agents, extract_metafeatures, check_agents, group_results

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

def main(valid_envs: Optional[List[str]], argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run training, evaluation, metafeature extraction, or checks of what has been run for highway-env configs."
    )
    valid_envs = valid_envs if valid_envs is not None else HIGHWAY_ENVS + METADRIVE_ENVS

    # check is used to check the progress of training/evaluating/extracting
    parser.add_argument(
        "-t",
        "--task",
        choices=("train", "evaluate", "extract", "check", "group"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "-e",
        "--env",
        choices=valid_envs,
        help="Environment to use",
    )
    parser.add_argument(
        "-S",
        "--start",
        type=int,
        default=0,
        help="Inclusive starting index for selecting run configurations.",
    )
    parser.add_argument(
        "-E",
        "--end",
        type=int,
        default=None,
        help="Exclusive ending index for selecting run configurations. Defaults to all.",
    )
    parser.add_argument(
        "-rf",
        "--result-folders",
        type=str,
        default="",
        help="List of result folders to group into the set of results (comma separated)"
    )
    parser.add_argument(    # for metafeatures and evaluation
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes to use for evaluation and metafeature extraction.",
    )
    parser.add_argument(    # for evaluation
        "-re",
        "--repeat-evaluation",
        action="store_true",
        help="Evaluate trained configurations even when evaluation results already exist.",
    )
    parser.add_argument(    # for metafeatures
        "-mg",
        "--metafeature-groups",
        type=str,
        default=None,
        help="Comma-separated list of metafeature groups to compute (e.g. 'env_features,mb_state_entropy'). Default computes all.",
    )
    parser.add_argument(    # for metafeatures
        "-u",
        "--update",
        type=float,
        default=time.time(),
        help="Unix timestamp float. Use an helper to convert a date to a timestamp. Groups older than this will be recomputed.",
    )
    args = parser.parse_args(argv)
    env_name = args.env
    result_folders = args.result_folders.split(",")

    logger.info(f"Running task {args.task} for environment {env_name}")

    load_configs = {
        "train": partial(load_env_train_configs, env_name),
        "evaluate": partial(load_env_train_configs, env_name),
        "extract": partial(load_env_instance_configs, env_name),
        "check": lambda : [ 0 ], # unused, just to not crash
        "group": lambda : [ 0 ], # unused, just to not crash
    }
    configs = load_configs[args.task]()

    total = len(configs)
    start = max(0, args.start)
    end = total if args.end is None else max(start, min(total, args.end))
    selected = configs[start:end]

    if not selected:
        logger.info("No run configurations selected. Nothing to do.")
        return

    task_map: Dict[str, Callable[[List[TrainConfig]], None]] = {
        "train": train_agents,
    }  # type: ignore
    if args.task == "extract":
        groups = args.metafeature_groups.split(",") if args.metafeature_groups else None
        extract_metafeatures(
            env_name,
            selected,
            workers=args.workers,
            requested_groups=groups,
            update_threshold=args.update,
        )  # type: ignore[arg-type]
    elif args.task == "group":
        group_results(result_folders)
    elif args.task == "check":
        logger.info(f"Checking trained, evaluated and extracted instances\n\n\n")
        for env in valid_envs:
            logger.info(f"Checking for env {env}")
            check_agents(load_env_train_configs(env), load_env_instance_configs(env))
            logger.info(f"Finished checking\n")
    elif args.task == "evaluate":
        logger.info(f"Starting task {args.task}\n\n\n")
        eval_agents(
            selected,
            workers=args.workers,
            repeat_evaluation=args.repeat_evaluation,
        )  # type: ignore[arg-type]
        logger.info(f"Finished task {args.task}\n\n\n")
    else:
        logger.info(f"Starting task {args.task}\n\n\n")
        task_map[args.task](selected)  # type: ignore
        logger.info(f"Finished task {args.task}\n\n\n")

    logger.info("Exiting")

if __name__ == "__main__":
    main(None)
