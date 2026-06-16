from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "isa"


class MergeCsvsRowwiseError(RuntimeError):
    """Raised when CSV files cannot be merged row-wise."""


@dataclass(frozen=True)
class CsvInput:
    csv_path: Path
    source: str


def results_path(csv_path: Path) -> Path:
    csv_path = csv_path.expanduser()
    if csv_path.is_absolute():
        return csv_path

    resolved_path = (RESULTS_DIR / csv_path).resolve()
    resolved_results_dir = RESULTS_DIR.resolve()

    if not resolved_path.is_relative_to(resolved_results_dir):
        raise MergeCsvsRowwiseError(
            f"Relative CSV paths must stay inside '{resolved_results_dir}': '{csv_path}'."
        )

    return resolved_path


def read_csv(csv_path: Path) -> pd.DataFrame:
    csv_path = results_path(csv_path)
    if not csv_path.is_file():
        raise MergeCsvsRowwiseError(f"CSV file not found: '{csv_path}'.")

    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)


def validate_source(source: str) -> None:
    if not source.strip():
        raise MergeCsvsRowwiseError("Source values cannot be empty.")


def output_columns(columns_from_path: Path, source_column: str) -> list[str]:
    columns_from_dataframe = read_csv(columns_from_path)
    columns = list(columns_from_dataframe.columns)

    if not columns:
        raise MergeCsvsRowwiseError(f"Template CSV has no columns: '{columns_from_path}'.")
    if len(columns) != len(set(columns)):
        duplicated_columns = sorted({column for column in columns if columns.count(column) > 1})
        raise MergeCsvsRowwiseError(
            f"Template CSV contains duplicated columns: {', '.join(duplicated_columns)}."
        )
    if source_column not in columns:
        raise MergeCsvsRowwiseError(
            f"Template CSV does not contain source column '{source_column}': "
            f"'{columns_from_path}'."
        )

    return columns


def parse_inputs(raw_inputs: list[list[str]]) -> list[CsvInput]:
    csv_inputs = []
    for csv_path, source in raw_inputs:
        validate_source(source)
        csv_inputs.append(CsvInput(csv_path=Path(csv_path), source=source))

    return csv_inputs


def merge_csvs_rowwise(
    columns_from_path: Path,
    output_path: Path,
    source_column: str,
    csv_inputs: list[CsvInput],
) -> None:
    columns = output_columns(columns_from_path, source_column)
    merged_dataframes = []

    for csv_input in csv_inputs:
        dataframe = read_csv(csv_input.csv_path)
        output_dataframe = dataframe.reindex(columns=columns)
        if csv_input.source != "HighwayEnv":
            output_dataframe[source_column] = csv_input.source
        merged_dataframes.append(output_dataframe)

    merged_dataframe = pd.concat(merged_dataframes, ignore_index=True)
    output_path = results_path(output_path)
    if not output_path.parent.is_dir():
        raise MergeCsvsRowwiseError(f"Output directory not found: '{output_path.parent}'.")

    merged_dataframe.to_csv(output_path, index=False)

    print(
        f"Wrote {len(merged_dataframe)} rows and {len(merged_dataframe.columns)} columns "
        f"to '{output_path}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge CSV files row-wise using the columns from one template CSV. "
            "Each input CSV is appended after replacing its source column with "
            "the source value supplied next to the input path."
        )
    )
    parser.add_argument(
        "--columns-from",
        type=Path,
        required=True,
        help=(
            "CSV whose columns and column order define the output schema. "
            "Relative paths are read from results/."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("CSV", "SOURCE"),
        required=True,
        help=(
            "Input CSV path and the source value to write for all of its rows. "
            "Repeat once per CSV. Relative paths are read from results/."
        ),
    )
    parser.add_argument(
        "--source-column",
        required=True,
        help="Name of the source column that must exist in the template CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the merged CSV will be written. Relative paths are written to results/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        merge_csvs_rowwise(
            columns_from_path=args.columns_from,
            output_path=args.output,
            source_column=args.source_column,
            csv_inputs=parse_inputs(args.input),
        )
    except MergeCsvsRowwiseError as err:
        print(f"[merge_csvs_rowwise] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
