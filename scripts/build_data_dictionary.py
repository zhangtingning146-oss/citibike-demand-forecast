from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


DESCRIPTIONS = {
    "station_id": "Citi Bike trip-file station ID (public short name, stored as text)",
    "hour_local": "Naive New York civil hour used as the panel time key",
    "local_date": "New York local calendar date",
    "local_hour": "Hour of local day, 0-23",
    "iso_day_of_week": "ISO weekday, Monday=1 and Sunday=7",
    "station_name": "Most representative trip-file station name in the station-month",
    "lat": "Trip-weighted station-month latitude",
    "lon": "Trip-weighted station-month longitude",
    "ride_out": "Trips starting at the station in the hour",
    "ride_in": "Trips ending at the station in the hour",
    "net_outflow": "ride_out minus ride_in",
    "zero_filled_active_hour": "True when the active station-hour had no observed start or end trip",
    "active_start_hour": "First observed trip hour for this station in this month",
    "active_end_hour": "Last observed trip hour for this station in this month",
    "gbfs_station_id_current": "Current live GBFS UUID mapped through short_name; reference only",
    "capacity_current": "Capacity in the 2026 live GBFS snapshot; not historical capacity",
    "matched_current_gbfs": "Whether the historical station ID matches the current live snapshot",
    "month": "Hive source partition month string",
    "year": "Hive source partition year",
    "first_observed_hour": "First observed panel hour across the full dataset; audit only",
    "last_observed_hour": "Last observed panel hour across the full dataset; audit only",
    "observed_active_months": "Count of active months across the full dataset; audit only",
    "total_station_events": "All ride starts plus ends across the full dataset; audit only",
    "dst_ambiguous_collapsed": "True for the repeated fall-back civil hour collapsed across two UTC hours",
    "temperature_2m_c": "Hourly 2 m air temperature in degrees Celsius",
    "apparent_temperature_c": "Hourly apparent temperature in degrees Celsius",
    "relative_humidity_2m_pct": "Hourly relative humidity at 2 m, percent",
    "precipitation_mm": "Hourly total precipitation, millimetres",
    "rain_mm": "Hourly rain, millimetres",
    "snowfall_cm": "Hourly snowfall, centimetres",
    "wind_speed_10m_max_ms": "Maximum 10 m wind speed represented in the local civil hour, m/s",
    "wind_gusts_10m_max_ms": "Maximum 10 m wind gust represented in the local civil hour, m/s",
    "cloud_cover_pct": "Mean cloud cover, percent",
    "weather_code": "Open-Meteo/WMO weather code",
    "weather_utc_hours_collapsed": "Number of UTC weather hours represented in the local civil hour",
    "holiday_name": "US federal holiday name where applicable",
    "is_federal_holiday": "US federal holiday indicator",
    "is_weekend": "Saturday or Sunday indicator",
    "is_workday": "Neither weekend nor US federal holiday",
    "thanksgiving_date": "Fourth Thursday of November for the row year",
    "days_from_thanksgiving": "Local date minus that year's Thanksgiving date",
    "is_thanksgiving": "Thanksgiving date indicator",
    "is_thanksgiving_eve": "Day before Thanksgiving indicator",
    "is_day_after_thanksgiving": "Day after Thanksgiving indicator",
    "is_thanksgiving_window_3d": "Within three days before or after Thanksgiving, inclusive",
    "feed_generations": "Count of source inventory feed generations aggregated into the hour",
    "inventory_observed_minutes": "Count of source minute slots observed in the inventory hour",
    "mean_bikes_available": "Mean rentable bikes during observed inventory samples",
    "mean_docks_available": "Mean available docks during observed inventory samples",
    "min_bikes_available": "Minimum rentable bikes during observed inventory samples",
    "min_docks_available": "Minimum available docks during observed inventory samples",
    "empty_share": "Share of observed inventory samples with zero rentable bikes",
    "full_share": "Share of observed inventory samples with zero available docks",
    "empty_any": "At least one empty observation in the hour",
    "full_any": "At least one full observation in the hour",
    "capacity_observed": "Maximum contemporaneously observed bikes plus docks in the inventory hour",
    "inventory_utc_hours_collapsed": "Number of UTC inventory hours represented in the local civil hour",
    "time_split": "Chronological train, validation, test, or Thanksgiving holdout assignment",
    "active_segment": "Internal run ID for consecutive active station-hours",
    "panel_month": "Output partition month",
    "panel_year": "Output partition year",
}


def describe_ride_column(name: str) -> str:
    direction = "starting" if "_out_" in name else "ending"
    kind = name.rsplit("_", 1)[-1]
    return f"Trips {direction} at the station in the hour, {kind} category"


def classify(name: str) -> tuple[str, str, str]:
    if name.startswith("ride_") and ("_lag_" in name or "_mean_previous_" in name):
        return "derived prior trip panel", "rolling_only", "predictor"
    if name.startswith("ride_") and name not in {"ride_out", "ride_in"}:
        return "official Citi Bike trip files", "observed_outcome", "outcome_component"
    if name in {"ride_out", "ride_in", "net_outflow"}:
        return "official Citi Bike trip files", "observed_outcome", "target"
    if name.startswith(("temperature_", "apparent_", "relative_", "precipitation", "rain_", "snowfall", "wind_", "cloud_", "weather_")):
        return "Open-Meteo historical archive", "historical_observed; forecast_vintage_required_for_future", "predictor"
    if name in {"feed_generations", "inventory_observed_minutes", "mean_bikes_available", "mean_docks_available", "min_bikes_available", "min_docks_available", "empty_share", "full_share", "empty_any", "full_any", "capacity_observed", "inventory_utc_hours_collapsed"}:
        role = "risk_label" if name in {"empty_share", "full_share", "empty_any", "full_any"} else "inventory_observation"
        return "ctbk.dev historical GBFS archive", "observed_same_hour; not_fixed_date", role
    if name in {"gbfs_station_id_current", "capacity_current", "matched_current_gbfs", "first_observed_hour", "last_observed_hour", "observed_active_months", "total_station_events"}:
        return "current GBFS or full-period audit", "reference_only", "exclude_from_model"
    if name.startswith("is_") or name in {"holiday_name", "thanksgiving_date", "days_from_thanksgiving", "local_hour", "iso_day_of_week", "local_date"}:
        return "derived calendar", "known_in_advance", "predictor"
    if name in {"time_split", "active_segment", "month", "year", "panel_month", "panel_year", "zero_filled_active_hour", "dst_ambiguous_collapsed"}:
        return "derived pipeline metadata", "audit_or_split", "metadata"
    return "official Citi Bike trip files / derived panel", "historically_observed", "identifier_or_attribute"


def identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the formal Stage 2 field dictionary with missingness")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("outputs/STAGE2_DATA_DICTIONARY.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.catalog), read_only=True)
    schema = con.execute("DESCRIBE model_features").fetchdf()
    columns = schema["column_name"].tolist()
    null_sql = ", ".join(f"sum(({identifier(col)} IS NULL)::BIGINT) AS {identifier(col)}" for col in columns)
    row_count = con.execute("SELECT count(*) FROM model_features").fetchone()[0]
    nulls = con.execute(f"SELECT {null_sql} FROM model_features").fetchone()
    records = []
    for index, (name, dtype) in enumerate(zip(columns, schema["column_type"])):
        source, availability, role = classify(name)
        description = DESCRIPTIONS.get(name)
        if description is None and name.startswith("ride_") and not ("_lag_" in name or "_mean_previous_" in name):
            description = describe_ride_column(name)
        elif description is None and "_lag_" in name:
            description = f"Prior {name.rsplit('_', 1)[-1]} value of {name.split('_lag_')[0]}, only when the civil-hour gap is exact"
        elif description is None and "_mean_previous_" in name:
            description = f"Trailing mean of {name.split('_mean_previous_')[0]} using only preceding rows in the active segment"
        elif description is None:
            description = name.replace("_", " ")
        missing = int(nulls[index])
        records.append({
            "field": name,
            "duckdb_type": dtype,
            "description": description,
            "source": source,
            "availability": availability,
            "model_role": role,
            "missing_rows": missing,
            "missing_rate": missing / row_count,
        })
    pd.DataFrame(records).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {len(records)} fields to {args.output}")
    con.close()


if __name__ == "__main__":
    main()
