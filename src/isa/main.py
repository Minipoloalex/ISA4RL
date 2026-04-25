import argparse
import logging
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import pandas as pd
from isa import run_instance_space_analysis, InstanceSpaceAnalysisError
import datetime

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

def filter_metafeature_columns(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.0,
    report_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Drop metafeatures that are too sparse or have no variation."""
    if max_missing_ratio < 0.0 or max_missing_ratio > 1.0:
        raise ValueError("[isa] max_missing_ratio must be between 0.0 and 1.0.")

    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    if not feature_columns:
        raise ValueError("[isa] No metafeature columns were found.")

    missing_ratios = df[feature_columns].isna().mean()
    unique_counts = df[feature_columns].nunique(dropna=True)

    dropped_features: Dict[str, Dict[str, Any]] = {}
    kept_features = []
    for column in feature_columns:
        reasons = []
        if missing_ratios[column] > max_missing_ratio:
            reasons.append("too_many_missing_values")
        if unique_counts[column] <= 1:
            reasons.append("single_unique_value")

        if reasons:
            dropped_features[column] = {
                "missing_ratio": float(missing_ratios[column]),
                "unique_values": int(unique_counts[column]),
                "reasons": reasons,
            }
        else:
            kept_features.append(column)

    if not kept_features:
        raise ValueError("[isa] Metafeature filtering removed every feature column.")

    if dropped_features:
        logger.warning(
            "[isa] Dropped %s/%s metafeature columns before ISA "
            "(max missing ratio %.2f).",
            len(dropped_features),
            len(feature_columns),
            max_missing_ratio,
        )
        for column, details in dropped_features.items():
            logger.info(
                "[isa] Dropped %s: missing %.2f%%, %s unique values, reasons=%s",
                column,
                details["missing_ratio"] * 100,
                details["unique_values"],
                ",".join(details["reasons"]),
            )

    filtered_columns = [
        col for col in df.columns
        if not col.startswith("feature_") or col in kept_features
    ]
    filtered_df = df[filtered_columns].copy()

    if report_path is not None:
        ensure_dir(report_path.parent)
        save_json(
            report_path,
            {
                "max_missing_ratio": max_missing_ratio,
                "kept_features": kept_features,
                "dropped_features": dropped_features,
            },
        )
        logger.info(f"[isa] Metafeature filter report saved to '{report_path}'.")

    return filtered_df


def build_isa_dataset(
    metric_key: str = "mean_reward",
    debug: bool = False,
    output_path: Optional[Path] = None,
    max_feature_missing: float = 0.0,
    filter_report_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = BASE_RESULTS_PATH / "isa" / "instancespace_dataset.csv"
    if filter_report_path is None:
        filter_report_path = output_path.with_name(f"{output_path.stem}_filter_report.json")

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
    df = filter_metafeature_columns(df, max_feature_missing, filter_report_path)
    feature_columns = [col for col in df.columns if col.startswith("feature_")]

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

def _extensive_metafeature_analysis(df: pd.DataFrame) -> None:
    for col in df.columns:
        if not col.startswith("feature_"):
            continue
        print(f">> {col}")
        nr = df[col].nunique(dropna=False)
        print(f"Unique values: {nr}")
        if nr <= 25:
            print(df[col].unique())
        
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Min: {df[col].min()}, Max: {df[col].max()}")
            print(f"Mean: {df[col].mean():.4f}, Std: {df[col].std():.4f}")
        print(f"Missing values: {df[col].isnull().mean() * 100:.1f}%")
        print("-" * 40)

def _print_summary(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    duplicates = df.duplicated().sum()
    missing_counts = df.isna().sum()
    missing_total = int(missing_counts.sum())
    missing_columns = int((missing_counts > 0).sum())

    print(f"Rows: {n_rows:,}, columns: {n_cols:,}")
    print(f"Duplicate rows: {duplicates:,}")
    print(f"Missing values: {missing_total:,} across {missing_columns} columns")

    top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
    if not top_missing.empty:
        print("\nColumns with most missing values:")
        for column, count in top_missing.items():
            pct = count / n_rows * 100
            print(f"  - {column}: {count:,} ({pct:.2f}%)")

    unique_counts = df.nunique(dropna=False)
    print("\nColumns with lowest cardinality:")
    for column, count in unique_counts.sort_values().head(5).items():
        print(f"  - {column}: {count:,} unique values")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return

    variance = numeric_df.var(numeric_only=True).sort_values(ascending=False).head(5)
    print("\nHigh-variance numeric columns:")
    for column, value in variance.items():
        print(f"  - {column}: variance {value:.2f}")

    std = numeric_df.std(numeric_only=True)
    mean_abs = numeric_df.abs().mean(numeric_only=True).replace(0, pd.NA)
    cv = (std / mean_abs).dropna()
    if not cv.empty:
        print("\nColumns with highest coefficient of variation (std/|mean|):")
        for column, value in cv.sort_values(ascending=False).head(5).items():
            print(f"  - {column}: {value:.2f}")

    print("\n")
    _extensive_metafeature_analysis(df)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISA tool suite: build dataset, run instance space analysis, or analyze metafeatures."
    )
    parser.add_argument(
        "--task",
        choices=["build", "analyze", "metafeatures"],
        required=True,
        help="Task to execute.",
    )
    # common arguments
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_RESULTS_PATH / "isa" / "instancespace_dataset.csv",
        help="Path to the dataset CSV file.",
    )
    
    # build args
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include diagnostic metrics (for build task).",
    )
    parser.add_argument(
        "--metric",
        default="mean_reward",
        help="Evaluation metric field to extract (for build task).",
    )
    parser.add_argument(
        "--max-feature-missing",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed missing-value ratio for each metafeature column "
            "(for build and analyze tasks). Default removes any metafeature with missing values."
        ),
    )
    parser.add_argument(
        "--filter-report",
        type=Path,
        default=None,
        help=(
            "Path to write the metafeature filtering report. For build, defaults to "
            "'<dataset stem>_filter_report.json'. For analyze, defaults inside the output directory."
        ),
    )
    
    # analyze args
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_RESULTS_PATH / "isa" / f"analysis_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Directory where ISA outputs will be stored (for analyze task).",
    )
    parser.add_argument(
        "--options",
        type=Path,
        default=None,
        help="Optional JSON file overriding Instance Space options (for analyze task).",
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.task == "build":
        build_isa_dataset(
            args.metric,
            args.debug,
            args.dataset,
            args.max_feature_missing,
            args.filter_report,
        )
    elif args.task == "analyze":
        try:
            ensure_dir(args.output)
            report_path = args.filter_report
            if report_path is None:
                report_path = args.output / "instancespace_dataset_filter_report.json"
            filtered_dataset_path = args.output / "instancespace_dataset_filtered.csv"
            df = pd.read_csv(str(args.dataset))
            filtered_df = filter_metafeature_columns(df, args.max_feature_missing, report_path)
            filtered_df.to_csv(filtered_dataset_path, index=False)
            run_instance_space_analysis(filtered_dataset_path, args.output, args.options)
        except InstanceSpaceAnalysisError as err:
            print(f"[isa] {err}")
            raise SystemExit(1) from err
    elif args.task == "metafeatures":
        if not args.dataset.is_file():
            print(f"[isa] Dataset not found: '{args.dataset}'")
            return
        df = pd.read_csv(str(args.dataset))
        _print_summary(df)

if __name__ == "__main__":
    main()
