from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import pandas as pd

from common.eval_summary import aggregate_metrics
from common.file_utils import (
    BASE_OUTPUT_PATH,
    EVALUATION_RESULTS_BASE_PATH,
    EVALUATION_RESULTS_FILE,
    METAFEATURES_FOLDER,
    METAFEATURES_RESULTS_FILE,
    TRAIN_FOLDER,
    ensure_dir,
    read_json,
)
from common.config_utils import (
    CONFIG,
    get_all_train_configs,
    get_all_instance_configs,
    get_all_eval_configs,
    map_to_train_id,
)

ISA_OUTPUT_DIR = BASE_OUTPUT_PATH / "isa"
DATASET_FILENAME = "instancespace_dataset.csv"
DEFAULT_METRIC = "mean_reward"
METAFEATURES_TO_DISCARD = {"observation_type_id"}

def _run_dir(run_id: int) -> Path:
    return BASE_OUTPUT_PATH / TRAIN_FOLDER / str(run_id)


def _instance_dir(env_id: int, obs_id: int) -> Path:
    folder = f"ENV{env_id}_OBS{obs_id}"
    return BASE_OUTPUT_PATH / METAFEATURES_FOLDER / folder


def _load_metafeatures(env_id: int, obs_id: int, source: str, debug: bool) -> Optional[Dict[str, float]]:
    file_path = _instance_dir(env_id, obs_id) / METAFEATURES_RESULTS_FILE
    if not file_path.is_file():
        return None
    data = read_json(file_path)
    record: Dict[str, Any] = {
        "instances": data["instance_id"],
        "source": source,
    }
    debug_info = {
        "env_config_id": env_id,
        "obs_config_id": obs_id,
        "env_name": data.get("env_name"),
        "metafeatures_elapsed_seconds": data.get("elapsed_seconds"),
    }
    if debug:
        record.update(debug_info)

    features = data.get("features", {})
    diagnostics = data.get("diagnostics", {}) if debug else {}
    record.update({f"feature_{k}": v for k, v in features.items() if k not in METAFEATURES_TO_DISCARD})
    record.update({f"diag_{k}": v for k, v in diagnostics.items()})
    return record


def _load_metric(run_id: int, metric_key: str, eval_seed: int) -> Optional[float]:
    eval_file = _run_dir(run_id) / EVALUATION_RESULTS_BASE_PATH / EVALUATION_RESULTS_FILE(eval_seed)
    if not eval_file.is_file():
        return None
    per_episode = read_json(eval_file)
    if not per_episode:
        return None
    summary = aggregate_metrics(per_episode)
    value = summary.get(metric_key)
    return float(value) if value is not None else None


def build_dataset(metric_key: str, debug: bool) -> pd.DataFrame:
    configs: List[CONFIG] = get_all_eval_configs()
    instance_rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    missing_meta: List[Tuple[int, int]] = []
    missing_metric: List[int] = []

    for cfg in configs:
        env_cfg = cfg["env_config"]
        obs_cfg = cfg["obs_config"]
        algo_cfg = cfg["algo_config"]

        env_id = int(env_cfg["id"])
        eval_seed = int(env_cfg["eval_seed"])
        orig_env_id = int(env_cfg["orig_id"])
        obs_id = int(obs_cfg["id"])
        algo_name = str(algo_cfg["algo"]).lower()
        algo_id = algo_cfg["id"]
        key = (env_id, obs_id)
        train_id = map_to_train_id(orig_env_id, obs_id, algo_id)

        row = instance_rows.get(key)
        if row is None:
            source = f"{env_cfg["env_id"]}_{obs_cfg["type"]}"
            # source = f"{env_cfg["env_id"]}"
            meta = _load_metafeatures(env_id, obs_id, source, debug)
            if meta is None:
                missing_meta.append(key)
                continue
            row = dict(meta)
            instance_rows[key] = row

        metric_value = _load_metric(train_id, metric_key, eval_seed)
        if metric_value is None:
            missing_metric.append(cfg["id"])
            continue
        column_name = f"algo_{algo_name}_{metric_key}"
        # normalize by IDM performance
        metric_value = metric_value / row["feature_idm_mean_episode_return"] # normalization
        row[column_name] = metric_value

    if missing_meta:
        unique = sorted(set(missing_meta))
        print(f"[isa] Skipped {len(unique)} instances without metafeatures.")
    if missing_metric:
        print(f"[isa] {len(missing_metric)} runs are missing evaluation data.")

    # Normalize, with best algorithm performance = 1
    # for k, v in instance_rows.items():
    #     cols = [col for col in v.keys() if col.startswith("algo_")]
    #     best = max(cols, key=lambda name: v[name])
    #     best_val = v[best]
    #     for col in cols:
    #         instance_rows[k][col] /= best_val

    df = pd.DataFrame(instance_rows.values())
    if debug:
        df.sort_values(["env_config_id", "obs_config_id"], inplace=True)
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an instance-space dataset (one row per instance, columns for metafeatures and algorithm performance)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include diagnostic metrics and others from metafeature extraction.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Evaluation metric field to extract from eval summaries (default: mean_reward).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ISA_OUTPUT_DIR / DATASET_FILENAME,
        help="CSV file where the dataset will be stored.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset = build_dataset(args.metric, args.debug)
    if dataset.empty:
        print("[isa] No instances with both metafeatures and evaluation metrics were found.")
        return
    feature_columns = [col for col in dataset.columns if col.startswith("feature_")]
    algo_columns = [col for col in dataset.columns if col.startswith(f"algo_")]
    leading_columns = [col for col in dataset.columns if col not in algo_columns]
    dataset = dataset[leading_columns + algo_columns]

    output_path = save_dataset(dataset, args.output)
    print(
        "[isa] Dataset ready for instancespace:"
        f" {len(dataset)} instances,"
        f" {len(feature_columns)} (meta-)feature columns,"
        f" {len(algo_columns)} algorithm performance columns."
    )
    print(f"[isa] Saved to '{output_path}'.")


if __name__ == "__main__":
    main()
