from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metafeature_exclusions import DOMAIN_SPECIFIC_METAFEATURES


class RemoveDomainSpecificMetafeaturesError(RuntimeError):
    """Raised when domain-specific metafeatures cannot be removed from a CSV."""


def remove_domain_specific_metafeatures(
    csv_path: Path,
    output_path: Path,
    allow_missing: bool,
) -> None:
    if not csv_path.is_file():
        raise RemoveDomainSpecificMetafeaturesError(f"CSV file not found: '{csv_path}'.")
    if not output_path.parent.is_dir():
        raise RemoveDomainSpecificMetafeaturesError(
            f"Output directory not found: '{output_path.parent}'."
        )

    dataframe = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    domain_specific_metafeatures = sorted(DOMAIN_SPECIFIC_METAFEATURES)
    missing_columns = [
        column for column in domain_specific_metafeatures
        if column not in dataframe.columns
    ]
    if missing_columns and not allow_missing:
        missing = ", ".join(missing_columns)
        raise RemoveDomainSpecificMetafeaturesError(
            "CSV file does not contain all configured domain-specific "
            f"metafeatures: {missing}."
        )

    columns_to_drop = [
        column for column in domain_specific_metafeatures
        if column in dataframe.columns
    ]
    if not columns_to_drop:
        raise RemoveDomainSpecificMetafeaturesError(
            "CSV file does not contain any configured domain-specific metafeatures."
        )

    output_dataframe = dataframe.drop(columns=columns_to_drop)
    output_dataframe.to_csv(output_path, index=False)

    print(
        f"Wrote {len(output_dataframe)} rows and {len(output_dataframe.columns)} columns "
        f"from '{csv_path}' to '{output_path}'. "
        f"Removed {len(columns_to_drop)} domain-specific metafeature columns."
    )
    print("Removed domain-specific metafeatures:")
    for column in columns_to_drop:
        print(f"- {column}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove configured domain-specific metafeature columns from a CSV."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Drop configured domain-specific metafeatures that are present, "
            "instead of requiring every configured column to exist."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        remove_domain_specific_metafeatures(
            csv_path=args.csv,
            output_path=args.output,
            allow_missing=args.allow_missing,
        )
    except RemoveDomainSpecificMetafeaturesError as err:
        print(f"[remove_domain_specific_metafeatures] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
