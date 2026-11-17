from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

ENDPOINT = "https://data.ny.gov/resource/wujg-7c2s.json"
OUT = Path("data/processed/mta_2024_hourly_average_by_complex.csv")


def main() -> None:
    walk_path = Path("data/processed/citibike_subway_walk_sample.csv")
    if not walk_path.exists():
        raise SystemExit(f"Missing walking-distance sample: {walk_path}")
    walk = pd.read_csv(walk_path, dtype={"subway_complex_id": "string"})
    requested_ids = sorted(walk["subway_complex_id"].dropna().unique())
    id_filter = ",".join(f"'{value}'" for value in requested_ids)
    select = (
        "station_complex_id, station_complex, "
        "date_extract_hh(transit_timestamp) as hour_of_day, "
        "date_extract_dow(transit_timestamp) as day_of_week, "
        "sum(ridership) as total_entries, "
        "count(distinct transit_timestamp) as observed_hour_count"
    )
    group = (
        "station_complex_id, station_complex, "
        "date_extract_hh(transit_timestamp), date_extract_dow(transit_timestamp)"
    )
    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(4), wait=wait_exponential_jitter(multiplier=2, max=30), reraise=True,
    )
    def fetch_quarter(start: str, end: str) -> list[dict]:
        params = {
            "$select": select,
            "$where": (
                "borough = 'Manhattan' AND transit_mode = 'subway' AND "
                f"station_complex_id in ({id_filter}) AND "
                f"transit_timestamp >= '{start}' AND transit_timestamp < '{end}'"
            ),
            "$group": group,
            "$order": "station_complex_id, day_of_week, hour_of_day",
            "$limit": 50000,
        }
        response = requests.get(
            ENDPOINT, params=params, timeout=150,
            headers={"User-Agent": "citibike-demand-course-project/0.1 (academic research)"},
        )
        response.raise_for_status()
        return response.json()

    quarter_bounds = [
        ("2024-01-01T00:00:00", "2024-04-01T00:00:00"),
        ("2024-04-01T00:00:00", "2024-07-01T00:00:00"),
        ("2024-07-01T00:00:00", "2024-10-01T00:00:00"),
        ("2024-10-01T00:00:00", "2025-01-01T00:00:00"),
    ]
    quarter_frames = [pd.DataFrame(fetch_quarter(start, end)) for start, end in quarter_bounds]
    frame = pd.concat(quarter_frames, ignore_index=True)
    for column in ("hour_of_day", "day_of_week", "total_entries", "observed_hour_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.groupby(
        ["station_complex_id", "station_complex", "hour_of_day", "day_of_week"], as_index=False
    )[["total_entries", "observed_hour_count"]].sum()
    frame["average_entries_per_hour"] = frame["total_entries"] / frame["observed_hour_count"]
    frame["station_complex_id"] = frame["station_complex_id"].astype("string")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)

    matched_path = Path("data/processed/citibike_subway_walk_with_ridership_sample.csv")
    matched_complexes = 0
    walk_complexes = 0
    walk_complexes = walk["subway_complex_id"].nunique()
    matched = walk.merge(
        frame, left_on="subway_complex_id", right_on="station_complex_id", how="left"
    )
    matched_complexes = matched.loc[matched["average_entries_per_hour"].notna(), "subway_complex_id"].nunique()
    matched.to_csv(matched_path, index=False)

    note = {
        "source": ENDPOINT,
        "dataset": "MTA Subway Hourly Ridership: 2020-2024",
        "method": "Sum fare-category rows, then divide by distinct timestamps within complex, hour-of-day, and day-of-week.",
        "rows": len(frame),
        "station_complexes": int(frame["station_complex_id"].nunique()),
        "walk_sample_complexes": int(walk_complexes),
        "walk_sample_complexes_matched": int(matched_complexes),
    }
    Path("data/processed/mta_ridership_match_report.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()

