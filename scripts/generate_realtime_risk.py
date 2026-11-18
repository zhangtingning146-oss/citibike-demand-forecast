from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from train_inventory_risk import CATEGORICAL, NUMERIC


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a separate next-hour risk warning from the latest live GBFS snapshot")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--station-information", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--profile-dir", type=Path, default=Path("data/processed/stage3/models/final_2026"))
    parser.add_argument("--risk-dir", type=Path, default=Path("data/processed/stage3/risk"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage3/realtime_risk"))
    args = parser.parse_args()
    status = pd.read_parquet(args.status)
    latest_feed = status["feed_last_updated_utc"].max()
    status = status[status["feed_last_updated_utc"].eq(latest_feed)].copy()
    info = pd.read_parquet(args.station_information)
    info_time = "info_collected_at_utc" if "info_collected_at_utc" in info.columns else "collected_at_utc"
    info = info.sort_values(info_time).drop_duplicates("station_id", keep="last")
    status = status.rename(columns={"station_id": "gbfs_station_id"})
    info = info.rename(columns={"station_id": "gbfs_station_id"})
    status = status.merge(info[["gbfs_station_id", "name", "short_name", "lat", "lon", "capacity"]], on="gbfs_station_id", how="left")
    con = duckdb.connect(str(args.catalog), read_only=True)
    manhattan = con.execute("SELECT DISTINCT station_id, gbfs_station_id_current FROM model_features WHERE gbfs_station_id_current IS NOT NULL").fetchdf()
    con.close()
    status = status.merge(manhattan, left_on="gbfs_station_id", right_on="gbfs_station_id_current", how="inner")
    collected = pd.Timestamp(status["collected_at_utc"].max())
    target_hour_utc = collected.ceil("h")
    target_local = target_hour_utc.tz_convert("America/New_York")
    status["target_hour_utc"] = target_hour_utc
    status["target_local_hour"] = target_local.hour
    status["target_iso_dow"] = target_local.isoweekday()
    status["mean_bikes_available"] = status["num_bikes_available"].astype(float)
    status["mean_docks_available"] = status["num_docks_available"].astype(float)
    status["min_bikes_available"] = status["num_bikes_available"].astype(float)
    status["min_docks_available"] = status["num_docks_available"].astype(float)
    status["capacity_observed"] = status["capacity"].fillna(
        status["num_bikes_available"] + status["num_docks_available"]
    ).astype(float)
    month_profile = pd.read_parquet(args.profile_dir / "profile_month.parquet")
    week_profile = pd.read_parquet(args.profile_dir / "profile_week.parquet")
    month_match = month_profile[
        month_profile["profile_month"].eq(target_local.month)
        & month_profile["iso_day_of_week"].eq(target_local.isoweekday())
        & month_profile["local_hour"].eq(target_local.hour)
    ][["station_id", "profile_out", "profile_in"]]
    week_match = week_profile[
        week_profile["iso_day_of_week"].eq(target_local.isoweekday())
        & week_profile["local_hour"].eq(target_local.hour)
    ][["station_id", "week_out", "week_in"]]
    status = status.merge(month_match, on="station_id", how="left").merge(week_match, on="station_id", how="left")
    status["profile_out"] = status["profile_out"].fillna(status["week_out"]).fillna(0)
    status["profile_in"] = status["profile_in"].fillna(status["week_in"]).fillna(0)
    net = status["profile_out"] - status["profile_in"]
    projected_bikes = status["mean_bikes_available"] - net
    status["current_inventory_empty_warning"] = status["mean_bikes_available"] <= 2
    status["current_inventory_full_warning"] = status["mean_docks_available"] <= 2
    status["inventory_plus_profile_empty_warning"] = projected_bikes <= 0
    status["inventory_plus_profile_full_warning"] = projected_bikes >= status["capacity_observed"]
    for risk in ("empty", "full"):
        model = joblib.load(args.risk_dir / f"{risk}_risk_logistic.joblib")
        status[f"logistic_probability_{risk}"] = model.predict_proba(status[NUMERIC + CATEGORICAL])[:, 1]
        status[f"logistic_warning_{risk}"] = status[f"logistic_probability_{risk}"] >= 0.5
    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"risk_{generated}.parquet"
    status.to_parquet(output, index=False, compression="zstd")
    status.sort_values(["logistic_probability_empty", "logistic_probability_full"], ascending=False).to_csv(
        output.with_suffix(".csv"), index=False
    )
    metadata = {
        "generated_at_utc": generated,
        "source_feed_last_updated_utc": str(latest_feed),
        "source_collected_at_utc": str(collected),
        "target_hour_utc": str(target_hour_utc),
        "rows": len(status),
        "separate_from_fixed_date_forecast": True,
        "warning": "This uses the latest live snapshot. It is not a feature in the long-ahead Thanksgiving forecast.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
