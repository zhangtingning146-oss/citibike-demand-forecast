from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ctbk.dev minute inventory to hourly risk labels")
    parser.add_argument("--input", type=Path, default=Path("data/raw/ctbk/daily"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/inventory_hourly"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect("work/inventory_build.duckdb")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    reports = []
    for source in sorted(args.input.glob("*.parquet")):
        day = source.stem
        year, month = day[:4], day[5:7]
        out_dir = args.output / f"year={year}" / f"month={month}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"day={day}.parquet"
        if target.exists():
            stats = con.execute(
                "SELECT count(*) rows, count(DISTINCT gbfs_station_id) stations, "
                "min(observed_minutes) min_minutes, max(observed_minutes) max_minutes "
                "FROM read_parquet(?)", [str(target)]
            ).fetchone()
            reports.append({"day": day, "rows": stats[0], "stations": stats[1], "min_minutes": stats[2], "max_minutes": stats[3]})
            continue
        con.execute(f"""
            COPY (
              WITH base AS (
                SELECT *,
                  date_trunc('hour', to_timestamp(ts)) AS hour_start_utc,
                  timezone('America/New_York', date_trunc('hour', to_timestamp(ts))) AS hour_local,
                  (is_installed = 1 AND is_renting = 1 AND is_returning = 1) AS operating,
                  (num_bikes_available + num_bikes_disabled + num_docks_available + num_docks_disabled) AS observed_capacity
                FROM read_parquet('{sql_path(source)}')
              )
              SELECT
                station_id AS gbfs_station_id,
                hour_start_utc,
                hour_local,
                count(DISTINCT ts)::USMALLINT AS feed_generations,
                count(DISTINCT date_trunc('minute', to_timestamp(ts)))::USMALLINT AS observed_minutes,
                count(*) FILTER (WHERE operating)::USMALLINT AS operating_observations,
                avg(num_bikes_available) FILTER (WHERE operating) AS mean_bikes_available,
                avg(num_docks_available) FILTER (WHERE operating) AS mean_docks_available,
                min(num_bikes_available) FILTER (WHERE operating)::INTEGER AS min_bikes_available,
                min(num_docks_available) FILTER (WHERE operating)::INTEGER AS min_docks_available,
                sum((num_bikes_available = 0)::INTEGER) FILTER (WHERE operating)::USMALLINT AS empty_observations,
                sum((num_docks_available = 0)::INTEGER) FILTER (WHERE operating)::USMALLINT AS full_observations,
                avg((num_bikes_available = 0)::INTEGER) FILTER (WHERE operating) AS empty_share,
                avg((num_docks_available = 0)::INTEGER) FILTER (WHERE operating) AS full_share,
                bool_or(num_bikes_available = 0 AND operating) AS empty_any,
                bool_or(num_docks_available = 0 AND operating) AS full_any,
                median(observed_capacity) FILTER (WHERE is_installed = 1)::INTEGER AS capacity_observed,
                max(last_reported)::BIGINT AS last_reported_max
              FROM base
              GROUP BY station_id, hour_start_utc, hour_local
              ORDER BY station_id, hour_start_utc
            ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """)
        stats = con.execute(
            "SELECT count(*) rows, count(DISTINCT gbfs_station_id) stations, "
            "min(observed_minutes) min_minutes, max(observed_minutes) max_minutes "
            "FROM read_parquet(?)", [str(target)]
        ).fetchone()
        reports.append({"day": day, "rows": stats[0], "stations": stats[1], "min_minutes": stats[2], "max_minutes": stats[3]})
        print(f"{day}: rows={stats[0]:,} stations={stats[1]:,}", flush=True)
    con.close()
    report_path = Path("data/processed/inventory_hourly_quality.json")
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

