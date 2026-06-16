from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISA_RESULTS_DIR = PROJECT_ROOT / "results" / "isa"


class RemoveCsvColumnsError(RuntimeError):
    """Raised when columns cannot be removed from an ISA CSV."""


def results_isa_path(file_name: str) -> Path:
    file_path = ISA_RESULTS_DIR / file_name
    resolved_path = file_path.resolve()
    resolved_results_dir = ISA_RESULTS_DIR.resolve()

    if not resolved_path.is_relative_to(resolved_results_dir):
        raise RemoveCsvColumnsError(
            f"CSV paths must stay inside '{resolved_results_dir}': '{file_name}'."
        )

    return resolved_path


def column_names(raw_columns: list[str]) -> list[str]:
    columns = []
    for raw_column in raw_columns:
        for column in raw_column.split(","):
            stripped_column = column.strip()
            if not stripped_column:
                raise RemoveCsvColumnsError("Column names cannot be empty.")
            columns.append(stripped_column)

    duplicated_columns = sorted({column for column in columns if columns.count(column) > 1})
    if duplicated_columns:
        raise RemoveCsvColumnsError(
            f"Columns were requested more than once: {', '.join(duplicated_columns)}."
        )

    return columns


def remove_csv_columns(csv_name: str, output_name: str, columns: list[str]) -> None:
    csv_path = results_isa_path(csv_name)
    output_path = results_isa_path(output_name)

    if not csv_path.is_file():
        raise RemoveCsvColumnsError(f"CSV file not found: '{csv_path}'.")
    if not output_path.parent.is_dir():
        raise RemoveCsvColumnsError(f"Output directory not found: '{output_path.parent}'.")

    dataframe = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise RemoveCsvColumnsError(
            f"CSV file does not contain requested columns: {', '.join(missing_columns)}."
        )

    output_dataframe = dataframe.drop(columns=columns)
    output_dataframe.to_csv(output_path, index=False)

    print(
        f"Wrote {len(output_dataframe)} rows and {len(output_dataframe.columns)} columns "
        f"from '{csv_path}' to '{output_path}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove specific columns from an ISA CSV in results/isa."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV file name inside results/isa.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file name inside results/isa.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Column names to remove. Use spaces or commas to separate multiple columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        remove_csv_columns(args.csv, args.output, column_names(args.columns))
    except RemoveCsvColumnsError as err:
        print(f"[remove_csv_columns] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
