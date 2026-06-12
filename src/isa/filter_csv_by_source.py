from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISA_RESULTS_DIR = PROJECT_ROOT / "results" / "isa"


class FilterCsvBySourceError(RuntimeError):
    """Raised when an ISA CSV cannot be filtered by source."""


def results_isa_path(file_name: str) -> Path:
    file_path = ISA_RESULTS_DIR / file_name
    resolved_path = file_path.resolve()
    resolved_results_dir = ISA_RESULTS_DIR.resolve()

    if not resolved_path.is_relative_to(resolved_results_dir):
        raise FilterCsvBySourceError(
            f"CSV paths must stay inside '{resolved_results_dir}': '{file_name}'."
        )

    return resolved_path


def filter_csv_by_source(csv_name: str, source: str, output_name: str) -> None:
    csv_path = results_isa_path(csv_name)
    output_path = results_isa_path(output_name)

    if not csv_path.is_file():
        raise FilterCsvBySourceError(f"CSV file not found: '{csv_path}'.")

    dataframe = pd.read_csv(csv_path)
    if "source" not in dataframe.columns:
        raise FilterCsvBySourceError(f"CSV file does not contain a 'source' column: '{csv_path}'.")

    filtered_dataframe = dataframe[dataframe["source"] == source]
    filtered_dataframe.to_csv(output_path, index=False)

    print(
        f"Wrote {len(filtered_dataframe)} of {len(dataframe)} rows "
        f"from '{csv_path}' to '{output_path}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter an ISA CSV in results/isa by exact source column value."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV file name inside results/isa.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Exact source value to keep.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file name inside results/isa.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        filter_csv_by_source(args.csv, args.source, args.output)
    except FilterCsvBySourceError as err:
        print(f"[filter_csv_by_source] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
