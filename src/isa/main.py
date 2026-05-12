import argparse
import logging
from typing import Dict, Any, Optional, Sequence, List, Collection
from pathlib import Path
import pandas as pd
from isa import DEFAULT_OPTIONS_PATH, run_instance_space_analysis, InstanceSpaceAnalysisError
from metafeature_exclusions import EXCLUDED_METAFEATURE_COLUMNS
import datetime
import re
# import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)

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
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

DEFAULT_DATASET_PATH = BASE_RESULTS_PATH / "isa" / "instancespace_dataset.csv"


def parse_env_names(value: str) -> List[str]:
    envs = [env.strip() for env in value.split(",")]
    envs = [env for env in envs if env]
    if not envs:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list with at least one environment."
        )
    if len(set(envs)) != len(envs):
        raise argparse.ArgumentTypeError("Environment list contains duplicate names.")
    return canonical_env_names(envs)


def _filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    component = component.strip("-")
    if not component:
        raise ValueError(f"Could not create filename component from environment '{value}'.")
    return component


def canonical_env_names(envs: Sequence[str]) -> List[str]:
    return sorted(envs, key=lambda env: (_filename_component(env), env))


def canonical_algorithm_names(algorithms: Sequence[str]) -> List[str]:
    return sorted(algorithm.lower() for algorithm in algorithms)


def default_isa_dataset_path(envs: Sequence[str]) -> Path:
    env_component = "_".join(_filename_component(env) for env in canonical_env_names(envs))
    return BASE_RESULTS_PATH / "isa" / f"instancespace_dataset_{env_component}.csv"


def default_isa_analysis_output_path(envs: Optional[Sequence[str]]) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if envs is None:
        env_component = "all-envs"
    else:
        env_component = "_".join(
            _filename_component(env) for env in canonical_env_names(envs)
        )
    return BASE_RESULTS_PATH / "isa" / f"analysis_{timestamp}_{env_component}"


def isa_dataset_metadata_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_metadata.json")


def resolve_dataset_path(
    dataset_path: Optional[Path],
    envs: Optional[Sequence[str]],
) -> Path:
    if dataset_path is not None:
        return dataset_path
    if envs is not None:
        return default_isa_dataset_path(envs)
    return DEFAULT_DATASET_PATH


def parse_algorithm_names(value: str) -> List[str]:
    algorithms = [algorithm.strip().lower() for algorithm in value.split(",")]
    algorithms = [algorithm for algorithm in algorithms if algorithm]
    if not algorithms:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list with at least one algorithm."
        )
    if len(set(algorithms)) != len(algorithms):
        raise argparse.ArgumentTypeError(
            "Algorithm list contains duplicate names."
        )
    return canonical_algorithm_names(algorithms)


def filter_algorithm_columns(
    df: pd.DataFrame,
    algorithms: Optional[Sequence[str]],
    metric_key: str,
) -> pd.DataFrame:
    if algorithms is None:
        return df

    selected_algorithms = canonical_algorithm_names(algorithms)
    algo_columns = [col for col in df.columns if col.startswith("algo_")]
    selected_algo_columns = [
        f"algo_{algorithm}_{metric_key}" for algorithm in selected_algorithms
    ]
    missing_algo_columns = [
        col for col in selected_algo_columns if col not in algo_columns
    ]
    if missing_algo_columns:
        available_algo_columns = ", ".join(sorted(algo_columns))
        missing = ", ".join(missing_algo_columns)
        raise ValueError(
            "[isa] Requested algorithm performance columns were not found: "
            f"{missing}. Available algorithm columns: {available_algo_columns}"
        )

    non_algo_columns = [col for col in df.columns if col not in algo_columns]
    logger.info(
        "[isa] Using algorithm performance columns for projection: %s",
        ", ".join(selected_algo_columns),
    )
    return df[non_algo_columns + selected_algo_columns].copy()


def filter_metafeature_columns(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.0,
    report_path: Optional[Path] = None,
    near_constant_threshold: float = 1e-12,
) -> pd.DataFrame:
    """Drop metafeatures that are too sparse or have effectively no variation."""
    if max_missing_ratio < 0.0 or max_missing_ratio > 1.0:
        raise ValueError("[isa] max_missing_ratio must be between 0.0 and 1.0.")
    if near_constant_threshold < 0.0:
        raise ValueError("[isa] near_constant_threshold must be non-negative.")

    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    if not feature_columns:
        raise ValueError("[isa] No metafeature columns were found.")

    missing_ratios = df[feature_columns].isna().mean()
    unique_counts = df[feature_columns].nunique(dropna=True)
    feature_min = df[feature_columns].min()
    feature_max = df[feature_columns].max()
    feature_range = feature_max - feature_min

    dropped_features: Dict[str, Dict[str, Any]] = {}
    kept_features = []
    for column in feature_columns:
        reasons = []
        if missing_ratios[column] > max_missing_ratio:
            reasons.append("too_many_missing_values")
        if unique_counts[column] <= 1:
            reasons.append("single_unique_value")
        elif feature_range[column] <= near_constant_threshold:
            reasons.append("near_constant_value")

        if reasons:
            dropped_features[column] = {
                "missing_ratio": float(missing_ratios[column]),
                "unique_values": int(unique_counts[column]),
                "min": float(feature_min[column]),
                "max": float(feature_max[column]),
                "range": float(feature_range[column]),
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
                "[isa] Dropped %s: missing %.2f%%, %s unique values, "
                "range %.3g, reasons=%s",
                column,
                details["missing_ratio"] * 100,
                details["unique_values"],
                details["range"],
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
                "near_constant_threshold": near_constant_threshold,
                "kept_features": kept_features,
                "dropped_features": dropped_features,
            },
        )
        logger.info(f"[isa] Metafeature filter report saved to '{report_path}'.")

    return filtered_df


def exclude_configured_metafeature_columns(
    df: pd.DataFrame,
    excluded_columns: Collection[str] = EXCLUDED_METAFEATURE_COLUMNS,
    force_exclusion: bool = True,
) -> pd.DataFrame:
    """Remove metafeatures configured as diagnostics only from the ISA dataset."""
    invalid_columns = [
        column for column in excluded_columns if not column.startswith("feature_")
    ]
    if invalid_columns:
        invalid = ", ".join(sorted(invalid_columns))
        raise ValueError(
            "[isa] Excluded metafeature columns must use final CSV names starting "
            f"with 'feature_'. Invalid entries: {invalid}"
        )

    missing_columns = [column for column in excluded_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        if force_exclusion:
            raise ValueError(
                "[isa] Configured metafeature exclusions were not found in the dataset: "
                f"{missing}"
            )
        else:
            logger.warning(
                "[isa] Configured metafeature exclusions were not found in the dataset: %s",
                missing,
            )

    if not excluded_columns:
        return df

    logger.info(
        "[isa] Excluding %s configured metafeature columns from ISA dataset: %s",
        len(excluded_columns),
        ", ".join(sorted(excluded_columns)),
    )
    return df.drop(columns=list(excluded_columns)).copy()


def normalize_algorithm_reward(
    agent_metric: float,
    row: Dict[str, Any],
    metric_key: str,
    train_folder: Path,
) -> float:
    baseline_column = f"feature_baseline_{metric_key}"
    random_column = f"feature_random_{metric_key}"

    missing_columns = [
        column for column in (baseline_column, random_column) if column not in row
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            "[isa] Cannot normalize algorithm performance for "
            f"'{train_folder}' because required columns are missing: {missing}"
        )

    baseline_metric = row[baseline_column]
    random_metric = row[random_column]
    if pd.isna(baseline_metric) or pd.isna(random_metric):
        raise ValueError(
            "[isa] Cannot normalize algorithm performance for "
            f"'{train_folder}' because baseline or random metric is missing."
        )

    denominator = float(baseline_metric) - float(random_metric)
    if float(agent_metric) < float(random_metric):
        logger.warning(
            "[isa] Trained agent underperformed random policy for '%s': "
            "agent_%s=%s, random_%s=%s. Normalized score will be below 0.0.",
            train_folder,
            metric_key,
            agent_metric,
            metric_key,
            random_metric,
        )

    if denominator < 0.0:
        logger.warning(
            "[isa] Random policy outperformed baseline heuristic for '%s': "
            "random_%s=%s, baseline_%s=%s. Normalized scores will still be "
            "computed, but their direction is inverted for this instance.",
            train_folder,
            metric_key,
            random_metric,
            metric_key,
            baseline_metric,
        )

    if denominator == 0.0:
        raise ValueError(
            "[isa] Cannot normalize algorithm performance for "
            f"'{train_folder}' because baseline and random {metric_key} are equal "
            f"({baseline_metric})."
        )

    return (float(agent_metric) - float(random_metric)) / denominator


def build_isa_dataset(
    envs: List[str],
    metric_key: str = "mean_reward",
    debug: bool = False,
    output_path: Optional[Path] = None,
    max_feature_missing: float = 0.0,
    filter_report_path: Optional[Path] = None,
    projection_algorithms: Optional[Sequence[str]] = None,
):
    envs = canonical_env_names(envs)
    if projection_algorithms is not None:
        projection_algorithms = canonical_algorithm_names(projection_algorithms)

    if output_path is None:
        output_path = default_isa_dataset_path(envs)
    if filter_report_path is None:
        filter_report_path = output_path.with_name(f"{output_path.stem}_filter_report.json")

    instance_rows: Dict[str, Dict[str, Any]] = {}
    missing_meta = 0
    missing_metric = 0

    # Iterate over environments
    for env_folder in BASE_RESULTS_PATH.iterdir():
        if not env_folder.is_dir() or env_folder.name == "isa" or env_folder.name not in envs:
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
                        metric_value = normalize_algorithm_reward(
                            float(metric_value),
                            row,
                            metric_key,
                            train_folder,
                        )
                            
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
        
    df = filter_algorithm_columns(df, projection_algorithms, metric_key)
    algo_columns = [col for col in df.columns if col.startswith("algo_")]
    excluded_metafeature_columns = sorted(EXCLUDED_METAFEATURE_COLUMNS)
    df = exclude_configured_metafeature_columns(df)
    df = filter_metafeature_columns(df, max_feature_missing, filter_report_path)
    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    diagnostic_columns = [col for col in df.columns if col.startswith("diag_")]

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)

    source_counts = df["source"].value_counts().sort_index().to_dict()
    metadata_path = isa_dataset_metadata_path(output_path)
    save_json(
        metadata_path,
        {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parameters": {
                "envs": list(envs),
                "algorithms": (
                    list(projection_algorithms)
                    if projection_algorithms is not None
                    else None
                ),
                "metric_key": metric_key,
                "debug": debug,
                "max_feature_missing": max_feature_missing,
                "excluded_metafeature_columns": excluded_metafeature_columns,
                "filter_report_path": str(filter_report_path),
                "dataset_path": str(output_path),
                "algorithm_metric_normalization": {
                    "enabled": True,
                    "formula": (
                        "(agent_metric - random_metric) / "
                        "(baseline_metric - random_metric)"
                    ),
                    "agent_metric": metric_key,
                    "baseline_metric_column": f"feature_baseline_{metric_key}",
                    "random_metric_column": f"feature_random_{metric_key}",
                },
            },
            "results": {
                "instances": len(df),
                "source_counts": source_counts,
                "columns": len(df.columns),
                "feature_columns": len(feature_columns),
                "algorithm_columns": len(algo_columns),
                "diagnostic_columns": len(diagnostic_columns),
                "missing_metafeatures": missing_meta,
                "missing_metrics": missing_metric,
                "algorithm_column_names": algo_columns,
                "feature_column_names": feature_columns,
                "diagnostic_column_names": diagnostic_columns,
            },
        },
    )
    
    logger.info(
        "[isa] Dataset ready for instancespace:"
        f" {len(df)} instances,"
        f" {len(feature_columns)} (meta-)feature columns,"
        f" {len(algo_columns)} algorithm performance columns."
    )
    logger.info(f"[isa] Saved to '{output_path}'.")
    logger.info(f"[isa] Metadata saved to '{metadata_path}'.")
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
        "-t",
        "--task",
        choices=["build", "analyze", "metafeatures"],
        required=True,
        help="Task to execute.",
    )
    # common arguments
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to the dataset CSV file. Defaults to a filename that includes "
            "the selected environments when --envs is provided."
        ),
    )
    
    # build args
    parser.add_argument(
        "-g",
        "--debug",
        action="store_true",
        help="Include diagnostic metrics (for build task).",
    )
    parser.add_argument(
        "-e",
        "--envs",
        type=parse_env_names,
        required=True,
        help=(
            "Comma-separated list of environments to use for the ISA dataset and "
            "environment-specific default dataset path."
        ),
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="mean_reward",
        help="Evaluation metric field to extract and use for algorithm column filtering.",
    )
    parser.add_argument(
        "-a",
        "--algorithms",
        type=parse_algorithm_names,
        required=True,
        help=(
            "Comma-separated algorithm names to use as projection performance columns, "
            "for example 'ppo,a2c,dqn'. If omitted, all algorithm columns are kept."
        ),
    )
    parser.add_argument(
        "-f",
        "--max-feature-missing",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed missing-value ratio for each metafeature column "
            "(for build and analyze tasks). Default removes any metafeature with missing values."
        ),
    )
    parser.add_argument(
        "-r",
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
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory where ISA outputs will be stored (for analyze task). Defaults "
            "to 'analysis_<timestamp>_<envs>' under the ISA results folder."
        ),
    )
    parser.add_argument(
        "-p",
        "--options",
        type=Path,
        help="JSON file with Instance Space options (for analyze task).",
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.task == "build":
        build_isa_dataset(
            args.envs,
            args.metric,
            args.debug,
            args.dataset,
            args.max_feature_missing,
            args.filter_report,
            args.algorithms,
        )
    elif args.task == "analyze":
        try:
            dataset_path = resolve_dataset_path(args.dataset, args.envs)
            output_path = (
                args.output
                if args.output is not None
                else default_isa_analysis_output_path(args.envs)
            )
            ensure_dir(output_path)
            report_path = args.filter_report
            if report_path is None:
                report_path = output_path / "instancespace_dataset_filter_report.json"
            filtered_dataset_path = output_path / "instancespace_dataset_filtered.csv"
            df = pd.read_csv(str(dataset_path))
            df = filter_algorithm_columns(df, args.algorithms, args.metric)
            filtered_df = filter_metafeature_columns(df, args.max_feature_missing, report_path)
            filtered_df.to_csv(filtered_dataset_path, index=False)
            run_instance_space_analysis(filtered_dataset_path, output_path, args.options)
        except InstanceSpaceAnalysisError as err:
            print(f"[isa] {err}")
            raise SystemExit(1) from err
    elif args.task == "metafeatures":
        dataset_path = resolve_dataset_path(args.dataset, args.envs)
        if not dataset_path.is_file():
            print(f"[isa] Dataset not found: '{dataset_path}'")
            return
        df = pd.read_csv(str(dataset_path))
        _print_summary(df)

if __name__ == "__main__":
    main()
