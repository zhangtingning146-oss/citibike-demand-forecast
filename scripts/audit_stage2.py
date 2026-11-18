from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the formal Stage 2 panel and feature store")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/audits"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.catalog), read_only=True)
    panel = con.execute("""
        SELECT count(*) AS rows,
          count(DISTINCT station_id) AS stations,
          min(hour_local) AS first_hour,
          max(hour_local) AS last_hour,
          sum(ride_out) AS ride_out,
          sum(ride_in) AS ride_in,
          sum(net_outflow) AS net_outflow,
          sum((net_outflow != ride_out - ride_in)::INTEGER) AS bad_net_identity_rows,
          count(*) - count(DISTINCT station_id || '|' || cast(hour_local AS VARCHAR)) AS duplicate_keys,
          sum(zero_filled_active_hour::INTEGER) AS zero_filled_active_hours,
          sum((lat IS NULL OR lon IS NULL)::INTEGER) AS missing_coordinate_rows,
          sum((NOT matched_current_gbfs)::INTEGER) AS unmatched_current_gbfs_rows
        FROM trip_panel
    """).fetchdf().iloc[0].to_dict()

    monthly = con.execute("""
        SELECT year(hour_local) AS year, month(hour_local) AS month,
          count(*) AS rows, count(DISTINCT station_id) AS stations,
          sum(ride_out) AS ride_out, sum(ride_in) AS ride_in,
          sum(zero_filled_active_hour::INTEGER) AS zero_filled_active_hours,
          sum((NOT matched_current_gbfs)::INTEGER) AS unmatched_current_gbfs_rows
        FROM trip_panel GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchdf()
    monthly.to_csv(args.output / "monthly_panel_coverage.csv", index=False)

    station_changes = con.execute("""
        WITH ordered AS (
          SELECT *,
            lag(lat) OVER (PARTITION BY station_id ORDER BY active_start_hour) AS prior_lat,
            lag(lon) OVER (PARTITION BY station_id ORDER BY active_start_hour) AS prior_lon,
            lag(station_name) OVER (PARTITION BY station_id ORDER BY active_start_hour) AS prior_name
          FROM station_month_history
        ), distances AS (
          SELECT *, CASE WHEN prior_lat IS NOT NULL THEN
            2 * 6371000 * asin(sqrt(
              pow(sin(radians(lat - prior_lat) / 2), 2) +
              cos(radians(prior_lat)) * cos(radians(lat)) *
              pow(sin(radians(lon - prior_lon) / 2), 2)
            )) END AS move_m
          FROM ordered
        )
        SELECT station_id, station_name, prior_name, active_start_hour,
          lat, lon, prior_lat, prior_lon, move_m
        FROM distances
        WHERE move_m >= 100 OR (prior_name IS NOT NULL AND station_name != prior_name)
        ORDER BY move_m DESC NULLS LAST, station_id, active_start_hour
    """).fetchdf()
    station_changes.to_csv(args.output / "station_location_name_changes.csv", index=False)

    features = con.execute("""
        SELECT count(*) AS rows, count(DISTINCT station_id) AS stations,
          min(hour_local) AS first_hour, max(hour_local) AS last_hour,
          sum((temperature_2m_c IS NULL)::INTEGER) AS missing_weather_rows,
          sum((ride_out_lag_24h IS NULL)::INTEGER) AS missing_lag24_rows,
          sum((capacity_observed IS NULL)::INTEGER) AS missing_contemporaneous_capacity_rows,
          sum((inventory_observed_minutes IS NOT NULL)::INTEGER) AS inventory_labeled_rows,
          sum(dst_ambiguous_collapsed::INTEGER) AS ambiguous_dst_rows
        FROM model_features
    """).fetchdf().iloc[0].to_dict()
    split_counts = con.execute("""
        SELECT time_split, count(*) AS rows, count(DISTINCT station_id) AS stations,
          min(hour_local) AS first_hour, max(hour_local) AS last_hour
        FROM model_features GROUP BY 1 ORDER BY 1
    """).fetchdf()
    split_counts.to_csv(args.output / "time_split_coverage.csv", index=False)

    inventory = con.execute("""
        SELECT count(*) AS station_hours, count(DISTINCT gbfs_station_id) AS stations,
          min(hour_local) AS first_hour, max(hour_local) AS last_hour,
          min(inventory_observed_minutes) AS min_observed_minutes,
          max(inventory_observed_minutes) AS max_observed_minutes,
          sum((inventory_observed_minutes < 45)::INTEGER) AS low_coverage_station_hours,
          sum(empty_any::INTEGER) AS empty_risk_station_hours,
          sum(full_any::INTEGER) AS full_risk_station_hours
        FROM inventory_local
    """).fetchdf().iloc[0].to_dict()

    report = {
        "trip_panel": panel,
        "features": features,
        "inventory": inventory,
        "station_change_records": len(station_changes),
        "notes": {
            "zero_fill_scope": "Only from each station-month's first to last observed trip hour.",
            "activity_limitation": "Trip records cannot identify temporary mid-month closures; those hours remain in the active envelope.",
            "capacity_current": "Current GBFS capacity is reference-only and must not be used as a historical predictor.",
            "weather": "Historical observed weather is valid for training/explanation; future evaluation must use forecast vintages available at the prediction origin.",
        },
    }
    (args.output / "stage2_quality_report.json").write_text(
        json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=json_default))
    con.close()


if __name__ == "__main__":
    main()
