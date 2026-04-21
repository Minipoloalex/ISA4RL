import argparse
import logging
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import pandas as pd

from common.eval_summary import aggregate_metrics
from common.file_utils import (
    save_json,
    read_json,
    BASE_RESULTS_PATH,
    nonempty_file_in,
    ensure_dir,
    RESULTS_METAFEATURES_PATH,
    RESULTS_EVALUATION_PATH,
    RESULTS_TRAIN_FOLDER_PATH,
    RESULTS_TRAIN_CONFIG_FILE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

def build_isa_dataset(metric_key: str = "mean_reward", debug: bool = False, output_path: Optional[Path] = None):
    if output_path is None:
        output_path = BASE_RESULTS_PATH / "isa" / "instancespace_dataset.csv"

    instance_rows: Dict[str, Dict[str, Any]] = {}
    missing_meta = 0
    missing_metric = 0

    # Iterate over environments
    for env_folder in BASE_RESULTS_PATH.iterdir():
        if not env_folder.is_dir() or env_folder.name == "isa":
            continue
            
        # Iterate over instances
        for instance_folder in env_folder.iterdir():
            if not instance_folder.is_dir() or instance_folder.name == "train":
                continue
                
            instance_id = instance_folder.name
            
            # Load metafeatures
            meta_path = RESULTS_METAFEATURES_PATH(instance_folder)
            if not nonempty_file_in(meta_path):
                missing_meta += 1
                continue
                
            meta_data = read_json(meta_path)
            
            row: Dict[str, Any] = {"instances": instance_id, "source": env_folder.name}
            
            # Flatten feature groups
            feature_groups = meta_data.get("feature_groups", {})
            for group_name, group_data in feature_groups.items():
                features = group_data.get("features", {})
                for k, v in features.items():
                    if isinstance(v, (int, float, bool)):
                        row[f"feature_{k}"] = v
                        
            if debug:
                for group_name, group_data in feature_groups.items():
                    diagnostics = group_data.get("diagnostics", {})
                    for k, v in diagnostics.items():
                         if isinstance(v, (int, float, bool)):
                             row[f"diag_{group_name}_{k}"] = v

            train_folder = RESULTS_TRAIN_FOLDER_PATH(instance_folder)
            if train_folder.is_dir():
                for algo_folder in train_folder.iterdir():
                    if not algo_folder.is_dir():
                        continue
                    
                    algo_config_path = RESULTS_TRAIN_CONFIG_FILE(algo_folder)
                    if not nonempty_file_in(algo_config_path):
                        continue
                        
                    algo_config = read_json(algo_config_path)
                    algo_name = str(algo_config.get("algo", algo_folder.name)).lower()
                    
                    eval_path = RESULTS_EVALUATION_PATH(algo_folder)
                    if not nonempty_file_in(eval_path):
                        missing_metric += 1
                        continue
                        
                    eval_data = read_json(eval_path)
                    if not eval_data:
                        missing_metric += 1
                        continue
                        
                    summary = aggregate_metrics(eval_data)
                    metric_value = summary.get(metric_key)
                    
                    if metric_value is not None:
                        # normalize by IDM performance
                        idm_metric = row.get(f"feature_idm_{metric_key}")
                        if idm_metric is None and metric_key == "mean_reward":
                            idm_metric = row.get("feature_idm_reward_mean")
                        
                        if idm_metric is not None and idm_metric != 0:
                            metric_value = float(metric_value) / float(idm_metric)
                            
                        column_name = f"algo_{algo_name}_{metric_key}"
                        row[column_name] = float(metric_value)
                    else:
                        missing_metric += 1

            instance_rows[instance_id] = row

    if missing_meta:
        logger.warning(f"[isa] Skipped {missing_meta} instances without metafeatures.")
    if missing_metric:
        logger.warning(f"[isa] {missing_metric} runs are missing evaluation data.")

    df = pd.DataFrame(list(instance_rows.values()))
    
    if df.empty:
        logger.error("[isa] No instances with metafeatures were found.")
        raise ValueError("[isa] No instances with metafeatures were found.")
        
    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    algo_columns = [col for col in df.columns if col.startswith("algo_")]
    leading_columns = [col for col in df.columns if col not in algo_columns]
    df = df[leading_columns + algo_columns]

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    
    logger.info(
        "[isa] Dataset ready for instancespace:"
        f" {len(df)} instances,"
        f" {len(feature_columns)} (meta-)feature columns,"
        f" {len(algo_columns)} algorithm performance columns."
    )
    logger.info(f"[isa] Saved to '{output_path}'.")
    return df

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
        default="mean_reward",
        help="Evaluation metric field to extract from eval summaries (default: mean_reward).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_RESULTS_PATH / "isa" / "instancespace_dataset.csv",
        help="CSV file where the dataset will be stored.",
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    build_isa_dataset(args.metric, args.debug, args.output)

if __name__ == "__main__":
    main()
