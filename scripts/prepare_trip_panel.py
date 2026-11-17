from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd

REQUIRED = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id", "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
]


def analyze_zip(path: Path, manhattan_ids: set[str], collect_panel: bool) -> tuple[dict, pd.DataFrame]:
    rows = 0
    missing = Counter()
    start_ids: set[str] = set()
    end_ids: set[str] = set()
    parts: list[pd.DataFrame] = []
    earliest = None
    latest = None
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"CRC failure in {path}: {bad_member}")
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        for member in members:
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, dtype={"start_station_id": "string", "end_station_id": "string"}, chunksize=250_000):
                    absent = set(REQUIRED) - set(chunk.columns)
                    if absent:
                        raise ValueError(f"{member} missing fields: {sorted(absent)}")
                    rows += len(chunk)
                    missing.update(chunk[REQUIRED].isna().sum().to_dict())
                    start_ids.update(chunk["start_station_id"].dropna().astype(str))
                    end_ids.update(chunk["end_station_id"].dropna().astype(str))
                    starts = pd.to_datetime(chunk["started_at"], errors="coerce")
                    ends = pd.to_datetime(chunk["ended_at"], errors="coerce")
                    valid_times = pd.concat([starts, ends]).dropna()
                    if not valid_times.empty:
                        earliest = valid_times.min() if earliest is None else min(earliest, valid_times.min())
                        latest = valid_times.max() if latest is None else max(latest, valid_times.max())
                    if collect_panel:
                        s = pd.DataFrame({
                            "station_id": chunk["start_station_id"],
                            "hour": starts.dt.floor("h"), "ride_out": 1, "ride_in": 0,
                        })
                        e = pd.DataFrame({
                            "station_id": chunk["end_station_id"],
                            "hour": ends.dt.floor("h"), "ride_out": 0, "ride_in": 1,
                        })
                        events = pd.concat([s, e], ignore_index=True)
                        events = events[events["station_id"].isin(manhattan_ids)].dropna(subset=["hour"])
                        parts.append(events.groupby(["station_id", "hour"], as_index=False)[["ride_in", "ride_out"]].sum())
    aggregated = pd.DataFrame(columns=["station_id", "hour", "ride_in", "ride_out"])
    if parts:
        aggregated = pd.concat(parts).groupby(["station_id", "hour"], as_index=False)[["ride_in", "ride_out"]].sum()
    report = {
        "zip_file": path.name,
        "zip_bytes": path.stat().st_size,
        "csv_members": members,
        "row_count": rows,
        "earliest_timestamp": str(earliest),
        "latest_timestamp": str(latest),
        "missing_values": dict(missing),
        "distinct_start_station_ids": len(start_ids),
        "distinct_end_station_ids": len(end_ids),
        "start_ids_matching_current_manhattan": len(start_ids & manhattan_ids),
        "end_ids_matching_current_manhattan": len(end_ids & manhattan_ids),
    }
    return report, aggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbfs-info", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("zips", nargs="+", type=Path)
    args = parser.parse_args()

    info = pd.read_parquet(args.gbfs_info)
    boundary = gpd.read_file(args.boundary).to_crs(4326)
    stations = gpd.GeoDataFrame(info, geometry=gpd.points_from_xy(info["lon"], info["lat"]), crs=4326)
    manhattan = stations[stations.geometry.within(boundary.geometry.iloc[0])].copy()
    manhattan_ids = set(manhattan["short_name"].dropna().astype(str))
    reports: list[dict] = []
    panel_source = pd.DataFrame()
    for path in args.zips:
        collect = path.name.startswith("202411")
        report, aggregate = analyze_zip(path, manhattan_ids, collect)
        reports.append(report)
        if collect:
            panel_source = aggregate

    if not panel_source.empty:
        start = pd.Timestamp("2024-11-25 00:00:00")
        end = pd.Timestamp("2024-11-30 23:00:00")
        window = panel_source[panel_source["hour"].between(start, end)]
        top = (window.assign(total=window["ride_in"] + window["ride_out"])
               .groupby("station_id")["total"].sum().nlargest(20).index)
        index = pd.MultiIndex.from_product(
            [top, pd.date_range(start, end, freq="h")], names=["station_id", "hour"]
        )
        panel = (window[window["station_id"].isin(top)]
                 .set_index(["station_id", "hour"])
                 .reindex(index, fill_value=0).reset_index())
        panel["net_flow"] = panel["ride_in"] - panel["ride_out"]
        names = manhattan[["short_name", "station_id", "name", "lat", "lon"]].rename(columns={
            "short_name": "station_id", "station_id": "gbfs_station_id", "name": "station_name"
        })
        panel = panel.merge(names, on="station_id", how="left")
        out_panel = Path("data/processed/manhattan_station_hour_panel_sample.csv")
        out_panel.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_panel, index=False)
        reports.append({"panel_sample_rows": len(panel), "panel_sample_stations": len(top), "panel_path": str(out_panel)})

    out = Path("data/processed/trip_data_quality_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

