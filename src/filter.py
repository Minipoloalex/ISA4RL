import pandas as pd
from pandas import DataFrame
from pathlib import Path

def filter_highway_sources(df: DataFrame, substring: str) -> DataFrame:
    """Return rows whose 'source' column contains the substring `substring`."""
    source = df["source"]
    mask = source.str.contains(substring, case=False, na=False)
    ans = df.loc[mask].copy()
    ans["source"] = ans["source"].str.partition("_")[2]
    return ans


# highway, roundabout, merge
FILTER = "highway"

inpath = Path("results") / "isa" / "instancespace_dataset.csv"
outpath = Path("results") / "isa" / f"instancespace_dataset_{FILTER}.csv"
df = pd.read_csv(inpath, index_col="instances")


df = filter_highway_sources(df, FILTER)
print(df.head())

df.to_csv(outpath)
