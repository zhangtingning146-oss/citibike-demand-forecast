from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

BASE = "https://data.ctbk.dev/gbfs/avail/raw/day/{day}.parquet"


def probe(day: date) -> tuple[date, int, int | None]:
    url = BASE.format(day=day.isoformat())
    response = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
    total = None
    if response.status_code == 206 and response.headers.get("Content-Range"):
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    return day, response.status_code, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    parser.add_argument("--gbfs-info", type=Path)
    args = parser.parse_args()
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        probes = list(pool.map(probe, days))
    available = [(d, size) for d, status, size in probes if status == 206]
    if not available:
        raise SystemExit("No public ctbk.dev daily Parquet files found")

    first, last = available[0][0], available[-1][0]
    missing = [d.isoformat() for d, status, _ in probes if first <= d <= last and status != 206]
    sample_day, sample_size = available[-1]
    sample_path = Path("data/raw/ctbk") / f"availability_{sample_day}.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists() or sample_path.stat().st_size != sample_size:
        with requests.get(BASE.format(day=sample_day), stream=True, timeout=120) as response:
            response.raise_for_status()
            with sample_path.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)

    pf = pq.ParquetFile(sample_path)
    columns = pf.schema_arrow.names
    station_col = next((c for c in ("station_id", "gbfs_station_id", "id") if c in columns), None)
    sample = pd.read_parquet(sample_path, columns=[station_col] if station_col else None)
    ctbk_ids = set(sample[station_col].dropna().astype(str)) if station_col else set()
    gbfs_ids: set[str] = set()
    if args.gbfs_info and args.gbfs_info.exists():
        gbfs_ids = set(pd.read_parquet(args.gbfs_info, columns=["station_id"])["station_id"].astype(str))
    report = {
        "url_template": BASE,
        "first_available_day": first.isoformat(),
        "last_available_day": last.isoformat(),
        "available_days": len(available),
        "missing_days_within_span": missing,
        "latest_sample_day": sample_day.isoformat(),
        "latest_sample_bytes": sample_size,
        "latest_sample_rows": pf.metadata.num_rows,
        "latest_sample_row_groups": pf.metadata.num_row_groups,
        "schema": str(pf.schema_arrow),
        "station_id_column": station_col,
        "distinct_ctbk_station_ids": len(ctbk_ids),
        "distinct_current_gbfs_ids": len(gbfs_ids),
        "ctbk_ids_matching_current_gbfs": len(ctbk_ids & gbfs_ids),
        "ctbk_ids_not_in_current_gbfs": len(ctbk_ids - gbfs_ids),
    }
    out = Path("data/processed/ctbk_history_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

