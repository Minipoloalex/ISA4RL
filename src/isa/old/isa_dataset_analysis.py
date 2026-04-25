from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

DATASET_PATH = Path("results") / "isa" / "instancespace_dataset.csv"


def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(str(DATASET_PATH))


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

def _extensive_metafeature_analysis(df: pd.DataFrame) -> None:
    for col in df.columns:
        if not col.startswith("feature_"):
            continue
        print(f">> {col}")
        nr = df[col].nunique()
        print(f"Unique values: {nr}")
        if nr <= 25:
            print(df[col].unique())
        print(df[col].mean(), df[col].std())
        print()

def main() -> None:
    df = _load_dataset()
    _print_summary(df)


if __name__ == "__main__":
    main()
