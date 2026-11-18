from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from dateutil.tz import datetime_ambiguous, datetime_exists, gettz
from pandas.tseries.holiday import USFederalHolidayCalendar


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def thanksgiving(year: int) -> date:
    current = date(year, 11, 1)
    first_thursday = current + timedelta(days=(3 - current.weekday()) % 7)
    return first_thursday + timedelta(days=21)


def build_calendar(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.DataFrame({"local_date": pd.date_range(start, end, freq="D")})
    federal = USFederalHolidayCalendar().holidays(start=start, end=end, return_name=True)
    federal_map = {timestamp.date(): name for timestamp, name in federal.items()}
    days["holiday_name"] = days["local_date"].dt.date.map(federal_map)
    days["is_federal_holiday"] = days["holiday_name"].notna()
    days["is_weekend"] = days["local_date"].dt.day_of_week >= 5
    days["is_workday"] = ~days["is_weekend"] & ~days["is_federal_holiday"]
    days["thanksgiving_date"] = days["local_date"].dt.year.map(thanksgiving)
    days["days_from_thanksgiving"] = (
        days["local_date"].dt.date - days["thanksgiving_date"]
    ).map(lambda delta: delta.days)
    days["is_thanksgiving"] = days["days_from_thanksgiving"].eq(0)
    days["is_thanksgiving_eve"] = days["days_from_thanksgiving"].eq(-1)
    days["is_day_after_thanksgiving"] = days["days_from_thanksgiving"].eq(1)
    days["is_thanksgiving_window_3d"] = days["days_from_thanksgiving"].between(-3, 3)
    days["month"] = days["local_date"].dt.month.astype("uint8")
    days["iso_day_of_week"] = days["local_date"].dt.isocalendar().day.astype("uint8")
    days["local_date"] = days["local_date"].dt.date

    hours = pd.DataFrame({
        "hour_local": pd.date_range(f"{start} 00:00:00", f"{end} 23:00:00", freq="h")
    })
    tz = gettz("America/New_York")
    hours["dst_hour_exists"] = hours["hour_local"].map(lambda value: datetime_exists(value.to_pydatetime(), tz))
    hours["dst_ambiguous_collapsed"] = hours["hour_local"].map(
        lambda value: datetime_ambiguous(value.to_pydatetime(), tz)
    )
    hours = hours[hours["dst_hour_exists"]].copy()
    return days, hours


def main() -> None:
    parser = argparse.ArgumentParser(description="Join trips, weather, calendar, and inventory; build leakage-safe lags")
    parser.add_argument("--panel-root", type=Path, default=Path("data/processed/panel/hourly_trips"))
    parser.add_argument("--weather", type=Path, default=Path("data/raw/weather/open_meteo_hourly_2023_2026.parquet"))
    parser.add_argument("--inventory-root", type=Path, default=Path("data/processed/inventory_hourly"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features/model_features_v2"))
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.rglob("*.parquet")):
        raise SystemExit(f"Refusing to overwrite existing feature dataset: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    panel_glob = sql_path(args.panel_root / "year=*" / "month=*" / "part.parquet")
    station_month_glob = sql_path(args.panel_root / "year=*" / "month=*" / "station_month.parquet")
    inventory_glob = sql_path(args.inventory_root / "year=*" / "month=*" / "day=*.parquet")
    con = duckdb.connect(str(args.catalog))
    Path("work/duckdb_feature_temp").mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA temp_directory='work/duckdb_feature_temp'")
    con.execute(f"CREATE OR REPLACE VIEW trip_panel AS SELECT * FROM read_parquet('{panel_glob}', hive_partitioning=true)")
    con.execute(f"CREATE OR REPLACE VIEW station_month_history AS SELECT * FROM read_parquet('{station_month_glob}', hive_partitioning=true)")

    con.execute("""
        CREATE OR REPLACE TABLE station_history AS
        SELECT station_id,
          arg_min(station_name, hour_local) AS first_station_name,
          arg_max(station_name, hour_local) AS latest_station_name,
          min(hour_local) AS first_observed_hour,
          max(hour_local) AS last_observed_hour,
          count(DISTINCT date_trunc('month', hour_local))::INTEGER AS observed_active_months,
          count(*)::BIGINT AS active_envelope_hours,
          sum((ride_out + ride_in))::BIGINT AS total_station_events,
          any_value(gbfs_station_id_current) FILTER (WHERE gbfs_station_id_current IS NOT NULL) AS gbfs_station_id_current,
          max(capacity_current)::INTEGER AS capacity_current,
          bool_or(matched_current_gbfs) AS matched_current_gbfs
        FROM trip_panel
        GROUP BY station_id
    """)
    con.execute("COPY station_history TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(Path("data/processed/station_history.parquet"))])

    weather = pd.read_parquet(args.weather)
    con.register("weather_raw_df", weather)
    con.execute("""
        CREATE OR REPLACE TABLE weather_local AS
        SELECT hour_local,
          avg(temperature_2m) AS temperature_2m_c,
          avg(apparent_temperature) AS apparent_temperature_c,
          avg(relative_humidity_2m) AS relative_humidity_2m_pct,
          sum(precipitation) AS precipitation_mm,
          sum(rain) AS rain_mm,
          sum(snowfall) AS snowfall_cm,
          max(wind_speed_10m) AS wind_speed_10m_max_ms,
          max(wind_gusts_10m) AS wind_gusts_10m_max_ms,
          avg(cloud_cover) AS cloud_cover_pct,
          arg_max(weather_code, precipitation) AS weather_code,
          count(*)::UTINYINT AS weather_utc_hours_collapsed
        FROM weather_raw_df
        GROUP BY hour_local
    """)

    calendar_days, calendar_hours = build_calendar("2023-01-01", "2026-10-31")
    con.register("calendar_day_df", calendar_days)
    con.register("calendar_hour_df", calendar_hours)
    con.execute("CREATE OR REPLACE TABLE calendar_day AS SELECT * FROM calendar_day_df")
    con.execute("CREATE OR REPLACE TABLE calendar_hour AS SELECT * FROM calendar_hour_df")

    inventory_files = list(args.inventory_root.rglob("day=*.parquet"))
    if inventory_files:
        con.execute(f"""
            CREATE OR REPLACE VIEW inventory_hourly AS
            SELECT * FROM read_parquet('{inventory_glob}', hive_partitioning=true)
        """)
        con.execute("""
            CREATE OR REPLACE TABLE inventory_local AS
            SELECT gbfs_station_id, hour_local,
              sum(feed_generations)::INTEGER AS feed_generations,
              sum(observed_minutes)::INTEGER AS inventory_observed_minutes,
              avg(mean_bikes_available) AS mean_bikes_available,
              avg(mean_docks_available) AS mean_docks_available,
              min(min_bikes_available)::INTEGER AS min_bikes_available,
              min(min_docks_available)::INTEGER AS min_docks_available,
              max(empty_share) AS empty_share,
              max(full_share) AS full_share,
              bool_or(empty_any) AS empty_any,
              bool_or(full_any) AS full_any,
              max(capacity_observed)::INTEGER AS capacity_observed,
              count(*)::UTINYINT AS inventory_utc_hours_collapsed
            FROM inventory_hourly
            GROUP BY gbfs_station_id, hour_local
        """)
    else:
        con.execute("""
            CREATE OR REPLACE TABLE inventory_local (
              gbfs_station_id VARCHAR, hour_local TIMESTAMP, feed_generations INTEGER,
              inventory_observed_minutes INTEGER, mean_bikes_available DOUBLE,
              mean_docks_available DOUBLE, min_bikes_available INTEGER,
              min_docks_available INTEGER, empty_share DOUBLE, full_share DOUBLE,
              empty_any BOOLEAN, full_any BOOLEAN, capacity_observed INTEGER,
              inventory_utc_hours_collapsed UTINYINT
            )
        """)

    con.execute("""
        CREATE OR REPLACE TEMP VIEW enriched0 AS
        SELECT p.*, sh.first_observed_hour, sh.last_observed_hour,
          sh.observed_active_months, sh.total_station_events,
          ch.dst_ambiguous_collapsed,
          w.* EXCLUDE (hour_local),
          c.* EXCLUDE (local_date, month, iso_day_of_week),
          i.* EXCLUDE (gbfs_station_id, hour_local),
          CASE
            WHEN c.is_thanksgiving_window_3d AND year(p.local_date) IN (2024, 2025)
              THEN 'thanksgiving_holdout'
            WHEN p.local_date < DATE '2024-10-01' THEN 'train_initial'
            WHEN p.local_date < DATE '2025-07-01' THEN 'validation_regular'
            ELSE 'test_regular'
          END AS time_split,
          lag(p.hour_local) OVER (PARTITION BY p.station_id ORDER BY p.hour_local) AS previous_panel_hour
        FROM trip_panel p
        JOIN calendar_hour ch USING (hour_local)
        LEFT JOIN weather_local w USING (hour_local)
        LEFT JOIN calendar_day c USING (local_date)
        LEFT JOIN station_history sh USING (station_id)
        LEFT JOIN inventory_local i
          ON i.gbfs_station_id = p.gbfs_station_id_current AND i.hour_local = p.hour_local
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW enriched1 AS
        SELECT *,
          CASE WHEN previous_panel_hour IS NULL OR date_diff('hour', previous_panel_hour, hour_local) != 1 THEN 1 ELSE 0 END AS segment_start
        FROM enriched0
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW enriched2 AS
        SELECT *,
          sum(segment_start) OVER (PARTITION BY station_id ORDER BY hour_local ROWS UNBOUNDED PRECEDING) AS active_segment
        FROM enriched1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW lagged AS
        SELECT *,
          lag(hour_local, 1) OVER win AS lag_hour_1,
          lag(hour_local, 24) OVER win AS lag_hour_24,
          lag(hour_local, 168) OVER win AS lag_hour_168,
          lag(ride_out, 1) OVER win AS ride_out_lag_1h_raw,
          lag(ride_in, 1) OVER win AS ride_in_lag_1h_raw,
          lag(ride_out, 24) OVER win AS ride_out_lag_24h_raw,
          lag(ride_in, 24) OVER win AS ride_in_lag_24h_raw,
          lag(ride_out, 168) OVER win AS ride_out_lag_168h_raw,
          lag(ride_in, 168) OVER win AS ride_in_lag_168h_raw,
          avg(ride_out) OVER (win ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS ride_out_mean_previous_24h,
          avg(ride_in) OVER (win ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS ride_in_mean_previous_24h,
          avg(ride_out) OVER (win ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING) AS ride_out_mean_previous_168h,
          avg(ride_in) OVER (win ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING) AS ride_in_mean_previous_168h
        FROM enriched2
        WINDOW win AS (PARTITION BY station_id, active_segment ORDER BY hour_local)
    """)
    output = sql_path(args.output)
    con.execute(f"""
        COPY (
          SELECT * EXCLUDE (
            previous_panel_hour, segment_start,
            lag_hour_1, lag_hour_24, lag_hour_168,
            ride_out_lag_1h_raw, ride_in_lag_1h_raw,
            ride_out_lag_24h_raw, ride_in_lag_24h_raw,
            ride_out_lag_168h_raw, ride_in_lag_168h_raw
          ),
          CASE WHEN date_diff('hour', lag_hour_1, hour_local) = 1 THEN ride_out_lag_1h_raw END AS ride_out_lag_1h,
          CASE WHEN date_diff('hour', lag_hour_1, hour_local) = 1 THEN ride_in_lag_1h_raw END AS ride_in_lag_1h,
          CASE WHEN date_diff('hour', lag_hour_24, hour_local) = 24 THEN ride_out_lag_24h_raw END AS ride_out_lag_24h,
          CASE WHEN date_diff('hour', lag_hour_24, hour_local) = 24 THEN ride_in_lag_24h_raw END AS ride_in_lag_24h,
          CASE WHEN date_diff('hour', lag_hour_168, hour_local) = 168 THEN ride_out_lag_168h_raw END AS ride_out_lag_168h,
          CASE WHEN date_diff('hour', lag_hour_168, hour_local) = 168 THEN ride_in_lag_168h_raw END AS ride_in_lag_168h,
          year(hour_local)::INTEGER AS panel_year,
          month(hour_local)::UTINYINT AS panel_month
          FROM lagged
        ) TO '{output}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (panel_year, panel_month), ROW_GROUP_SIZE 100000)
    """)
    feature_glob = sql_path(args.output / "panel_year=*" / "panel_month=*" / "*.parquet")
    con.execute(f"CREATE OR REPLACE VIEW model_features AS SELECT * FROM read_parquet('{feature_glob}', hive_partitioning=true)")
    con.execute("""
        CREATE OR REPLACE VIEW econometric_panel AS
        SELECT * EXCLUDE (
          active_segment, gbfs_station_id_current, capacity_current,
          matched_current_gbfs, first_observed_hour, last_observed_hour,
          observed_active_months, total_station_events
        ) FROM model_features
    """)
    con.execute("""
        CREATE OR REPLACE VIEW ann_gru_features AS
        SELECT * EXCLUDE (
          active_segment, gbfs_station_id_current, capacity_current,
          matched_current_gbfs, first_observed_hour, last_observed_hour,
          observed_active_months, total_station_events
        ) FROM model_features
    """)
    con.execute("""
        CREATE OR REPLACE VIEW fixed_date_features AS
        SELECT * EXCLUDE (
          active_segment, gbfs_station_id_current, capacity_current,
          matched_current_gbfs, first_observed_hour, last_observed_hour,
          observed_active_months, total_station_events,
          feed_generations, inventory_observed_minutes, mean_bikes_available,
          mean_docks_available, min_bikes_available, min_docks_available,
          empty_share, full_share, empty_any, full_any, capacity_observed,
          inventory_utc_hours_collapsed,
          ride_out_lag_1h, ride_in_lag_1h, ride_out_lag_24h, ride_in_lag_24h,
          ride_out_lag_168h, ride_in_lag_168h,
          ride_out_mean_previous_24h, ride_in_mean_previous_24h,
          ride_out_mean_previous_168h, ride_in_mean_previous_168h
        ) FROM model_features
    """)
    checks = con.execute("""
        SELECT count(*) rows, count(DISTINCT station_id) stations,
          min(hour_local) first_hour, max(hour_local) last_hour,
          sum((net_outflow != ride_out - ride_in)::INTEGER) bad_net_flow,
          sum(dst_ambiguous_collapsed::INTEGER) ambiguous_dst_rows,
          sum((temperature_2m_c IS NULL)::INTEGER) missing_weather_rows,
          sum((ride_out_lag_24h IS NULL)::INTEGER) missing_lag24_rows
        FROM model_features
    """).fetchdf()
    report = checks.iloc[0].to_dict()
    report["time_split_counts"] = con.execute(
        "SELECT time_split, count(*) rows FROM model_features GROUP BY time_split ORDER BY time_split"
    ).fetchdf().to_dict("records")
    Path("data/processed/stage2_feature_quality.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    con.close()


if __name__ == "__main__":
    main()

