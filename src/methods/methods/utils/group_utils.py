from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import shutil

from common.file_utils import (
    save_json,
    read_json,
    BASE_RESULTS_PATH,
    OTHER_RESULTS_PATH,
    nonempty_file_in,
    RESULTS_TRAIN_METADATA_PATH,
    RESULTS_METAFEATURES_PATH,
)
from common.config_utils import CONFIG, remove_useless_params

def find_instance_config_in_folder(folder: Path, env_name: str, target_env_config: CONFIG) -> Optional[Path]:
    target_env_config = remove_useless_params(target_env_config)
    for env_folder in folder.iterdir(): # e.g. results_server/exit/
        if env_folder.name != env_name:
            continue

        for env_instance_folder in env_folder.iterdir(): # results_server/exit/<id>/
            if not env_instance_folder.is_dir():
                continue

            env_config_file = env_instance_folder / "instance_config.json"  # results_server/exit/<id>/instance_config.json
            env_config = read_json(env_config_file)

            env_config = remove_useless_params(env_config)
            if env_config == target_env_config:
                return env_instance_folder
    return None

def find_config_in_folder(folder: Path, env_name: str, target_env_config: CONFIG, target_algo_config: CONFIG) -> Optional[Path]:
    env_instance_folder = find_instance_config_in_folder(folder, env_name, target_env_config)
    if env_instance_folder is None or not (env_instance_folder / "train").is_dir():
        return None
    for results_folder in (env_instance_folder / "train").iterdir(): # results_server/exit/<id>/train/<id>/
        algo_config_file = results_folder / "algo_config.json" # results_server/exit/<id>/train/<id>/algo_config.json
        algo_config = read_json(algo_config_file)

        if algo_config == target_algo_config:
            return results_folder
    return None

def is_combination_trained(path: Path) -> bool:
    return nonempty_file_in(RESULTS_TRAIN_METADATA_PATH(path))

def is_combination_extracted(path: Path) -> bool:
    return nonempty_file_in(RESULTS_METAFEATURES_PATH(path))

def merge_result_folders(src_path: Path, dst_path: Path) -> None:
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

def merge_metafeature_results(src_path: Path, dst_path: Path) -> None:
    shutil.copy2(src_path, dst_path)
