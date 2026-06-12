from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from common.config_utils import remove_useless_params


SOURCE_COLUMN_NAMES = ("source", "sources")
logger = logging.getLogger(__name__)


class MergeCsvError(RuntimeError):
    """Raised when two ISA CSV files cannot be merged."""


def read_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_source_column(dataframe: pd.DataFrame, csv_path: Path) -> str:
    source_columns = [column for column in SOURCE_COLUMN_NAMES if column in dataframe.columns]
    if len(source_columns) != 1:
        raise MergeCsvError(
            f"CSV file must contain exactly one source column from "
            f"{SOURCE_COLUMN_NAMES}: '{csv_path}'."
        )
    return source_columns[0]


def read_csv(csv_path: Path) -> pd.DataFrame:
    csv_path = csv_path.expanduser()
    if not csv_path.is_file():
        raise MergeCsvError(f"CSV file not found: '{csv_path}'.")

    dataframe = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "instances" not in dataframe.columns:
        raise MergeCsvError(f"CSV file does not contain an 'instances' column: '{csv_path}'.")
    source_column = get_source_column(dataframe, csv_path)
    if dataframe["instances"].str.strip().eq("").any():
        raise MergeCsvError(f"CSV file contains empty instances in '{csv_path}'.")
    if dataframe[source_column].str.strip().eq("").any():
        raise MergeCsvError(f"CSV file contains empty sources in '{csv_path}'.")
    if dataframe["instances"].duplicated().any():
        duplicated_instances = sorted(
            dataframe.loc[dataframe["instances"].duplicated(), "instances"]
            .astype(str)
            .unique()
        )
        raise MergeCsvError(
            f"CSV file contains duplicated instances in '{csv_path}': "
            f"{', '.join(duplicated_instances)}"
        )

    return dataframe


def validate_results_root(results_root: Path, label: str) -> None:
    if not results_root.is_dir():
        raise MergeCsvError(f"{label} results root not found: '{results_root}'.")


def normalized_config_key(config: dict) -> str:
    normalized_config = remove_useless_params(config)
    return json.dumps(normalized_config, sort_keys=True)


def instance_config_path(results_root: Path, source: str, instance_id: str) -> Path:
    return results_root / source / instance_id / "instance_config.json"


def read_instance_config(results_root: Path, source: str, instance_id: str) -> dict:
    config_path = instance_config_path(results_root, source, instance_id)
    if not config_path.is_file():
        raise MergeCsvError(f"Instance config not found: '{config_path}'.")
    return read_json(config_path)


def build_config_index(
    dataframe: pd.DataFrame,
    results_root: Path,
    source_column: str,
) -> dict[tuple[str, str], str]:
    config_index: dict[tuple[str, str], str] = {}

    for row in dataframe.itertuples(index=False):
        instance_id = getattr(row, "instances")
        source = getattr(row, source_column)
        config = read_instance_config(results_root, source, instance_id)
        key = (source, normalized_config_key(config))

        if key in config_index:
            raise MergeCsvError(
                f"Multiple rows in the secondary CSV match the same normalized config "
                f"for source '{source}': '{config_index[key]}' and '{instance_id}'."
            )
        config_index[key] = instance_id

    return config_index


def align_secondary_instances(
    priority_dataframe: pd.DataFrame,
    secondary_dataframe: pd.DataFrame,
    priority_results_root: Path,
    secondary_results_root: Path,
    priority_source_column: str,
    secondary_source_column: str,
) -> pd.DataFrame:
    secondary_config_index = build_config_index(
        secondary_dataframe,
        secondary_results_root,
        secondary_source_column,
    )
    secondary_by_instance = secondary_dataframe.set_index("instances", drop=False)
    aligned_rows = []
    missing_matches = []
    matched_secondary_instance_ids = set()

    for row in priority_dataframe.itertuples(index=False):
        priority_instance_id = getattr(row, "instances")
        source = getattr(row, priority_source_column)
        config = read_instance_config(priority_results_root, source, priority_instance_id)
        key = (source, normalized_config_key(config))
        secondary_instance_id = secondary_config_index.get(key)

        if secondary_instance_id is None:
            missing_matches.append(f"{source}/{priority_instance_id}")
            continue

        aligned_row = secondary_by_instance.loc[secondary_instance_id].copy()
        aligned_row["instances"] = priority_instance_id
        matched_secondary_instance_ids.add(secondary_instance_id)
        if secondary_source_column != priority_source_column:
            aligned_row[priority_source_column] = source
            aligned_row = aligned_row.drop(labels=[secondary_source_column])
        aligned_rows.append(aligned_row)

    if missing_matches:
        logger.error(
            "Could not find matching secondary configs for %d priority instances: %s",
            len(missing_matches),
            ", ".join(missing_matches),
        )
        raise MergeCsvError(
            "Could not find matching secondary configs for: "
            f"{', '.join(missing_matches)}"
        )

    unmatched_secondary_instances = sorted(
        set(secondary_dataframe["instances"]) - matched_secondary_instance_ids
    )
    if unmatched_secondary_instances:
        logger.error(
            "Could not match %d secondary instances from the priority CSV: %s",
            len(unmatched_secondary_instances),
            ", ".join(unmatched_secondary_instances),
        )
        raise MergeCsvError(
            "Could not match secondary instances from the priority CSV: "
            f"{', '.join(unmatched_secondary_instances)}"
        )

    return pd.DataFrame(aligned_rows).reset_index(drop=True)


def ordered_columns(
    priority_dataframe: pd.DataFrame,
    secondary_dataframe: pd.DataFrame,
    source_column: str,
) -> list[str]:
    columns = ["instances", source_column]
    seen_columns = set(columns)

    for dataframe in (priority_dataframe, secondary_dataframe):
        for column in dataframe.columns:
            if column in seen_columns:
                continue
            columns.append(column)
            seen_columns.add(column)

    return columns


def merge_csvs(
    priority_path: Path,
    secondary_path: Path,
    output_path: Path,
    priority_results_root: Path,
    secondary_results_root: Path,
) -> None:
    priority_dataframe = read_csv(priority_path)
    secondary_dataframe = read_csv(secondary_path)
    priority_source_column = get_source_column(priority_dataframe, priority_path)
    secondary_source_column = get_source_column(secondary_dataframe, secondary_path)
    priority_results_root = priority_results_root.expanduser()
    secondary_results_root = secondary_results_root.expanduser()
    validate_results_root(priority_results_root, "Priority")
    validate_results_root(secondary_results_root, "Secondary")

    secondary_dataframe = align_secondary_instances(
        priority_dataframe,
        secondary_dataframe,
        priority_results_root,
        secondary_results_root,
        priority_source_column,
        secondary_source_column,
    )

    priority_indexed = priority_dataframe.set_index("instances", drop=False)
    secondary_indexed = secondary_dataframe.set_index("instances", drop=False)
    secondary_indexed = secondary_indexed.loc[priority_indexed.index]

    columns = ordered_columns(priority_dataframe, secondary_dataframe, priority_source_column)
    merged_dataframe = pd.DataFrame(index=priority_indexed.index)
    for column in columns:
        if column in priority_indexed.columns:
            merged_dataframe[column] = priority_indexed[column]
            continue
        merged_dataframe[column] = secondary_indexed[column]

    output_path = output_path.expanduser()
    merged_dataframe.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two ISA CSV files by matching instance_config.json files from "
            "their results roots. Values from the priority CSV are kept when both "
            "files contain the same column."
        )
    )
    parser.add_argument(
        "--priority",
        type=Path,
        required=True,
        help="CSV whose values are kept when both files contain the same column.",
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        required=True,
        help="CSV used to fill columns that are not present in the priority CSV.",
    )
    parser.add_argument(
        "--priority-results-root",
        type=Path,
        required=True,
        help="Results root containing the priority CSV instance folders.",
    )
    parser.add_argument(
        "--secondary-results-root",
        type=Path,
        required=True,
        help="Results root containing the secondary CSV instance folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the merged CSV will be written.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    try:
        merge_csvs(
            args.priority,
            args.secondary,
            args.output,
            args.priority_results_root,
            args.secondary_results_root,
        )
    except MergeCsvError as err:
        print(f"[merge_csvs] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
