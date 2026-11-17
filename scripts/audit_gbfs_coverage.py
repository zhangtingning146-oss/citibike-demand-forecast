from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 15-minute GBFS collection coverage")
    parser.add_argument("status_parquet", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/gbfs_coverage.csv"))
    args = parser.parse_args()

    df = pd.read_parquet(args.status_parquet, columns=["collected_at_utc", "feed_last_updated_utc"])
    observed = pd.DatetimeIndex(df["collected_at_utc"].drop_duplicates().sort_values())
    if observed.empty:
        raise SystemExit("No observations found")
    expected = pd.date_range(observed.min().floor("15min"), observed.max().ceil("15min"), freq="15min")
    nearest = pd.DataFrame({"expected_at_utc": expected})
    actual = pd.DataFrame({"collected_at_utc": observed}).sort_values("collected_at_utc")
    nearest = pd.merge_asof(
        nearest, actual, left_on="expected_at_utc", right_on="collected_at_utc",
        direction="nearest", tolerance=pd.Timedelta("7min30s"),
    )
    nearest["missing"] = nearest["collected_at_utc"].isna()
    nearest["schedule_delay_seconds"] = (
        nearest["collected_at_utc"] - nearest["expected_at_utc"]
    ).dt.total_seconds()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nearest.to_csv(args.output, index=False)
    print(nearest["missing"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

