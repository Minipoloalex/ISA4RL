import pandas as pd
from pandas import DataFrame
from pathlib import Path

def filter_highway_sources(df: DataFrame, substring: str) -> DataFrame:
    """Return rows whose 'source' column contains the substring `substring`."""
    source = df["source"]
    mask = source.str.contains(substring, case=False, na=False)
    return df[mask]


# highway, roundabout, merge
FILTER = "merge"

inpath = Path("results") / "isa" / "version6.csv"
outpath = Path("results") / "isa" / f"version6_{FILTER}.csv"
df = pd.read_csv(inpath, index_col="instances")


df = filter_highway_sources(df, FILTER)
print(df.head())

df.to_csv(outpath)
