from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from common.file_utils import save_json, read_json, BASE_RESULTS_PATH, OTHER_RESULTS_PATH, nonempty_file_in
from pathlib import Path
from .load_config_utils import (
    RESULTS_TRAIN_METADATA_PATH,
)

def find_config_in_folder(folder: Path, config: tuple[dict[Any, Any], dict[Any, Any]]) -> Optional[Path]:
    for env_folder in folder.iterdir():
        if env_folder.name == "isa":
            continue

        for env_instance_folder in env_folder.iterdir():
            env_config_file = env_instance_folder / "instance_config.json"
            env_config = read_json(env_config_file)

            if env_config != config[0]:
                continue

            for results_folder in (env_instance_folder / "train").iterdir():
                algo_config_file = results_folder / "algo_config.json"
                algo_config = read_json(algo_config_file)

                if algo_config == config[1]:
                    return results_folder
    return None

def is_combination_trained(path: Path) -> bool:
    return nonempty_file_in(RESULTS_TRAIN_METADATA_PATH(path))

def merge_result_folders(src_path: Path, dst_path: Path) -> None:
    # TODO
    print(f"Should copy path {src_path} to path {dst_path}")
    return None
