from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


SAFE_COLUMNS = """
station_id, hour_local, local_date, local_hour, iso_day_of_week,
station_name, lat, lon, ride_out, ride_in, net_outflow,
ride_out_classic, ride_out_electric, ride_out_other,
ride_in_classic, ride_in_electric, ride_in_other,
ride_out_member, ride_out_casual, ride_in_member, ride_in_casual,
zero_filled_active_hour,
dst_ambiguous_collapsed, temperature_2m_c, apparent_temperature_c,
relative_humidity_2m_pct, precipitation_mm, rain_mm, snowfall_cm,
wind_speed_10m_max_ms, wind_gusts_10m_max_ms, cloud_cover_pct, weather_code,
holiday_name, is_federal_holiday, is_weekend, is_workday,
days_from_thanksgiving, is_thanksgiving, is_thanksgiving_eve,
is_day_after_thanksgiving, is_thanksgiving_window_3d, time_split,
capacity_observed, empty_share, full_share, empty_any, full_any,
inventory_observed_minutes, ride_out_lag_1h, ride_in_lag_1h,
ride_out_lag_24h, ride_in_lag_24h, ride_out_lag_168h, ride_in_lag_168h,
ride_out_mean_previous_24h, ride_in_mean_previous_24h,
ride_out_mean_previous_168h, ride_in_mean_previous_168h
"""


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model-ready Python Parquet and Stata CSV datasets")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/exports"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.catalog), read_only=True)

    safe_parquet = args.output / "python_model_panel.parquet"
    if safe_parquet.exists() and not args.force:
        print(f"skip existing {safe_parquet}")
    else:
        con.execute(f"COPY (SELECT {SAFE_COLUMNS} FROM model_features) TO '{qpath(safe_parquet)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        print(f"wrote {safe_parquet}")

    stata_dir = args.output / "stata_csv"
    stata_dir.mkdir(parents=True, exist_ok=True)
    for year in range(2023, 2027):
        target = stata_dir / f"citibike_station_hour_{year}.csv"
        if target.exists() and not args.force:
            print(f"skip existing {target}")
            continue
        con.execute(f"COPY (SELECT {SAFE_COLUMNS} FROM model_features WHERE year(hour_local)={year} ORDER BY station_id, hour_local) TO '{qpath(target)}' (HEADER, DELIMITER ',')")
        print(f"wrote {target}")
    con.close()


if __name__ == "__main__":
    main()
