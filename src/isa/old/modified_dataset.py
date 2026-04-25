import pandas as pd
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

COLUMNS_TO_DISCARD: Sequence[str] = ()
IN_PATH = Path("results") / "isa" / "old_instancespace_dataset.csv"
OUT_PATH = Path("results") / "isa" / "version6.csv"
OBSERVATION_KEYWORDS = ["obs", "observation"]
STICKY_KEYWORDS = ["sticky"]
RowPredicate = Callable[[pd.Series], bool]
ROW_FILTERS: Sequence[RowPredicate] = ()

def _matches_observation_metadata(column: str) -> bool:
    lowered = column.lower()
    return any(keyword in lowered for keyword in OBSERVATION_KEYWORDS)

def _matches_sticky_metadata(column: str) -> bool:
    lowered = column.lower()
    return any(keyword in lowered for keyword in STICKY_KEYWORDS)

def _columns_to_drop(columns: Iterable[str]) -> List[str]:
    """Return all columns that, for instance, leak observation-space information.
    Or instead give useless information"""
    auto_columns = [col for col in columns if _matches_observation_metadata(col) or _matches_sticky_metadata(col)]
    manual_columns = [col for col in COLUMNS_TO_DISCARD if col in columns]
    return sorted(set(auto_columns + manual_columns))

def _drop_rows(df: pd.DataFrame, predicates: Sequence[RowPredicate]) -> pd.DataFrame:
    if df.empty or not predicates:
        return df

    def should_drop(row: pd.Series) -> bool:
        for predicate in predicates:
            if predicate(row):
                return True
        return False

    drop_mask = df.apply(should_drop, axis=1)
    if not drop_mask.any():
        return df
    return df.loc[~drop_mask]

def filter_grayscale(row: pd.Series) -> bool:
    return row["source"].endswith("GrayscaleObservation")

def main() -> None:
    ROW_FILTERS = []

    df = pd.read_csv(IN_PATH)
    columns_to_drop = _columns_to_drop(df.columns)
    filtered_df = df.drop(columns=columns_to_drop, errors="ignore")
    filtered_df = _drop_rows(filtered_df, ROW_FILTERS)
    filtered_df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
