import json
import os
from pathlib import Path
from typing import Dict, Any, List, Callable
import numpy as np

from common.env_utils import ENVS

# BASE PATHS

base_path = Path(os.getenv("APP_DIR", "../../"))

BASE_RESULTS_PATH = base_path / "results"
OTHER_RESULTS_PATH: Callable[[str], Path] = lambda folder: base_path / folder
BASE_CONFIG_PATH = base_path / "config"
BASE_IMAGES_PATH = base_path / "images"

# CONFIGS

TRAIN_CONFIG_PATH: Path = BASE_CONFIG_PATH / "train_configs.json"
INSTANCE_CONFIG_PATH: Path = BASE_CONFIG_PATH / "instance_configs.json"
EVAL_CONFIG_PATH: Path = BASE_CONFIG_PATH / "eval_configs.json"

ALGO_CONFIG_PATH: Path = BASE_CONFIG_PATH / "algo-configs.json"
OBS_CONFIG_PATH: Path = BASE_CONFIG_PATH / "obs-configs.json"

ENV_CONFIGS_FOLDER: Path = BASE_CONFIG_PATH / "env-configs"
ENV_CONFIG_PATH: Callable[[str], Path] = lambda env: ENV_CONFIGS_FOLDER / f"{env}-configs.json"
# HIGHWAY_CONFIG_PATH: Path = ENV_CONFIG_PATH("highway-fast-v0")
# ROUNDABOUT_CONFIG_PATH: Path = ENV_CONFIG_PATH("roundabout-generic-v0")
# MERGE_CONFIG_PATH: Path = ENV_CONFIG_PATH("merge-generic-v0")

ALGO_CONFIG_HYPERPARAMS_PATH: Path = BASE_CONFIG_PATH / "rlzoo-algo-hyperparams"

TRAIN_CONFIGS_FOLDER: Path = BASE_CONFIG_PATH / "train-configs"
EVAL_CONFIGS_FOLDER: Path = BASE_CONFIG_PATH / "eval-configs"
TRAIN_CONFIGS_PATH: Callable[[str], Path] = lambda env: TRAIN_CONFIGS_FOLDER / f"{env}.json"
EVAL_CONFIGS_PATH: Callable[[str], Path] = lambda env: EVAL_CONFIGS_FOLDER / f"{env}.json"

# RESULTS
RESULTS_ENV_FOLDER_PATH: Callable[[Path, str], Path] = (
    lambda base_results_path, env_name: base_results_path / env_name
)
RESULTS_INSTANCE_FOLDER_PATH: Callable[[Path, str], Path] = lambda env_folder_path, instance_id: env_folder_path / instance_id
RESULTS_TRAIN_FOLDER_PATH: Callable[[Path], Path] = lambda instance_folder_path: instance_folder_path / "train"
RESULTS_TRAIN_ALGO_FOLDER_PATH: Callable[[Path, str], Path] = lambda train_folder_path, algo_id: train_folder_path / algo_id

RESULTS_INSTANCE_CONFIG_FILE: Callable[[Path], Path] = lambda instance_folder_path: instance_folder_path / "instance_config.json"
RESULTS_TRAIN_CONFIG_FILE: Callable[[Path], Path]= lambda train_algo_folder_path: train_algo_folder_path / "algo_config.json"

MODELS_FOLDER = "models"
MODEL_FILE = "model.zip"
BEST_MODEL_FILE = "best_model.zip"
BEST_VEC_NORMALIZE_FILE = "vec_normalize_best.pkl"
VEC_NORMALIZE_FILE = "vec_normalize.pkl"

TENSORBOARD_FOLDER = "tensorboard"
LOGS_FOLDER = "logs"

RESULTS_TRAIN_METADATA_PATH: Callable[[Path], Path] = lambda algo_train_folder_path: algo_train_folder_path / "training_metadata.json"
RESULTS_METAFEATURES_PATH: Callable[[Path], Path] = lambda instance_folder_path: instance_folder_path / "metafeatures.json"
RESULTS_EVALUATION_PATH: Callable[[Path], Path] = lambda algo_train_folder_path: algo_train_folder_path / "eval_results.json"

_folders = [
    BASE_RESULTS_PATH,
    ENV_CONFIGS_FOLDER,
    TRAIN_CONFIGS_FOLDER,
    EVAL_CONFIGS_FOLDER,
    *[RESULTS_ENV_FOLDER_PATH(BASE_RESULTS_PATH, env) for env in ENVS],
]
for folder_path in _folders:
    folder_path.mkdir(parents=True, exist_ok=True)

def read_json(file: Path):
    with file.open("r", encoding="utf-8") as fp:
        return json.load(fp)

def save_json(file: Path, results: Dict[str, Any] | List[Dict[str, Any]] | List[Any]) -> None:
    with file.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=_json_default)

def ensure_dir(path: str | Path) -> None:
    if type(path) is str:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True) # type: ignore

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int32, np.int64)): # type: ignore
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): # type: ignore
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def nonempty_file_in(filepath: Path) -> bool:
    """Return True if filepath exists, is a regular file, and is non-empty."""
    try:
        return filepath.is_file() and filepath.stat().st_size > 0
    except OSError:
        return False
