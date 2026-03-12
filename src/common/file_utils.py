import json
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

base_path = Path(os.getenv("APP_DIR", "../../"))

BASE_OUTPUT_PATH = base_path / "results"
BASE_CONFIG_PATH = base_path / "config"
BASE_IMAGES_PATH = base_path / "images"

TRAIN_FOLDER = "train"
METAFEATURES_FOLDER = "metafeatures"
MODELS_FOLDER = "models"

BEST_MODEL_FILE = "best_model.zip"
MODEL_FILE = "model.zip"

TENSORBOARD_FOLDER = "tensorboard"
LOGS_FOLDER = "logs"

TRAIN_CONFIG_PATH = BASE_CONFIG_PATH / "train_configs.json"
INSTANCE_CONFIG_PATH = BASE_CONFIG_PATH / "instance_configs.json"
EVAL_CONFIG_PATH = BASE_CONFIG_PATH / "eval_configs.json"
HIGHWAY_CONFIG_PATH = BASE_CONFIG_PATH / "highway-configs.json"
ROUNDABOUT_CONFIG_PATH = BASE_CONFIG_PATH / "roundabout-configs.json"
MERGE_CONFIG_PATH = BASE_CONFIG_PATH / "merge-configs.json"
ALGO_CONFIG_PATH = BASE_CONFIG_PATH / "algo-configs.json"
OBS_CONFIG_PATH = BASE_CONFIG_PATH / "obs-configs.json"

ALGO_CONFIG_HYPERPARAMS_PATH = BASE_CONFIG_PATH / "rlzoo-algo-hyperparams"

EVALUATION_RESULTS_BASE_PATH = "eval_results"
TRAINING_METADATA_FILE = "training_metadata.json"
METAFEATURES_RESULTS_FILE = "metafeatures.json"
EVALUATION_RESULTS_FILE = lambda seed: f"seed_{seed}.json"

def read_json(file: Path):
    with file.open("r", encoding="utf-8") as fp:
        return json.load(fp)

def save_json(file: Path, results: Dict[str, Any] | List[Dict[str, Any]] | List[Any]):
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

def load_training_metadata(run_dir: Path) -> Dict[str, Any]:
    metadata_path = run_dir / TRAINING_METADATA_FILE
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def save_eval_results(results: List[Dict[str, Any]], folder_name: str, eval_seed: int):
    folder_path = Path(folder_name) / EVALUATION_RESULTS_BASE_PATH
    ensure_dir(folder_path)

    filepath = folder_path / EVALUATION_RESULTS_FILE(eval_seed)
    save_json(filepath, results)

def save_extract_results(results, folder_name: str):
    ensure_dir(folder_name)
    filepath = Path(folder_name) / METAFEATURES_RESULTS_FILE
    save_json(filepath, results)
