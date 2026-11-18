from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import numpy as np
import pandas as pd
import torch

from citibike_pipeline.stage3_models import (
    FEATURE_NAMES, DemandGRU, DemandMLP, Standardizer, build_numeric_features,
    predict_gru, predict_mlp,
)
from run_stage3_experiments import WEATHER_RENAME, regression_matrix


TARGET_DATE = pd.Timestamp("2026-11-26")


def choose_model(metrics_path: Path) -> tuple[str, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    eligible = ["historical_profile", "panel_weather_holiday", "ann_full", "gru_exogenous"]
    subset = metrics[
        metrics["model"].isin(eligible)
        & metrics["scenario"].eq("thanksgiving_window")
        & metrics["target"].isin(["ride_out", "ride_in"])
    ]
    ranking = subset.groupby("model", as_index=False)["wape"].mean().sort_values("wape")
    if ranking.empty:
        raise RuntimeError("No 2025 Thanksgiving-window model comparison was found")
    return str(ranking.iloc[0]["model"]), ranking


def load_weather(path: Path) -> tuple[pd.DataFrame, dict]:
    weather = pd.read_parquet(path).rename(columns=WEATHER_RENAME)
    weather["hour_utc"] = pd.to_datetime(weather["hour_utc"], utc=True)
    meta_path = path.with_suffix(".source.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return weather, metadata


def build_target_frame(catalog: Path, model_dir: Path, weather: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(catalog), read_only=True)
    stations = con.execute("""
        SELECT station_id,
          arg_max(station_name, hour_local) AS station_name,
          arg_max(lat, hour_local) AS lat,
          arg_max(lon, hour_local) AS lon
        FROM model_features
        WHERE hour_local >= TIMESTAMP '2026-10-01'
          AND hour_local < TIMESTAMP '2026-11-01'
        GROUP BY station_id ORDER BY station_id
    """).fetchdf()
    con.close()
    local_hours = pd.date_range("2026-11-26 00:00:00", "2026-11-26 23:00:00", freq="h")
    frame = stations.merge(pd.DataFrame({"hour_local": local_hours}), how="cross")
    frame["local_date"] = TARGET_DATE.date()
    frame["local_hour"] = frame["hour_local"].dt.hour.astype("uint8")
    frame["iso_day_of_week"] = TARGET_DATE.isoweekday()
    frame["panel_month"] = 11
    frame["hour_utc"] = frame["hour_local"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    frame["is_weekend"] = False
    frame["is_federal_holiday"] = True
    frame["is_workday"] = False
    frame["is_thanksgiving"] = True
    frame["is_thanksgiving_eve"] = False
    frame["is_day_after_thanksgiving"] = False
    frame["days_from_thanksgiving"] = 0

    month_profile = pd.read_parquet(model_dir / "profile_month.parquet")
    week_profile = pd.read_parquet(model_dir / "profile_week.parquet")
    global_profile = pd.read_parquet(model_dir / "profile_global.parquet")
    frame = frame.merge(
        month_profile,
        left_on=["station_id", "panel_month", "iso_day_of_week", "local_hour"],
        right_on=["station_id", "profile_month", "iso_day_of_week", "local_hour"],
        how="left",
    )
    frame = frame.merge(
        week_profile,
        on=["station_id", "iso_day_of_week", "local_hour"], how="left",
    )
    frame = frame.merge(
        global_profile,
        left_on=["panel_month", "iso_day_of_week", "local_hour"],
        right_on=["profile_month", "iso_day_of_week", "local_hour"],
        how="left", suffixes=("", "_global"),
    )
    frame["profile_out"] = frame["profile_out"].fillna(frame["week_out"]).fillna(frame["global_out"]).fillna(0)
    frame["profile_in"] = frame["profile_in"].fillna(frame["week_in"]).fillna(frame["global_in"]).fillna(0)
    frame["profile_observations"] = frame["profile_observations"].fillna(frame["week_observations"]).fillna(0)
    frame["new_station_fallback"] = frame["profile_observations"].eq(0)
    keep_weather = ["hour_utc", *WEATHER_RENAME.values()]
    frame = frame.merge(weather[keep_weather], on="hour_utc", how="left", validate="many_to_one")
    missing = frame[list(WEATHER_RENAME.values())].isna().sum()
    if missing.any():
        raise RuntimeError(f"Weather forecast does not cover all target hours: {missing.to_dict()}")
    return frame.sort_values(["station_id", "hour_local"]).reset_index(drop=True)


def predict_selected(frame: pd.DataFrame, selected: str, model_dir: Path, metadata: dict) -> np.ndarray:
    if selected == "historical_profile":
        return frame[["profile_out", "profile_in"]].to_numpy(dtype=np.float32)
    if selected == "panel_weather_holiday":
        coefficients = pd.read_csv(model_dir / "panel_weather_holiday_coefficients.csv")
        x = regression_matrix(frame, include_holiday=True)
        result = []
        for target, profile in (("ride_out", "profile_out"), ("ride_in", "profile_in")):
            params = coefficients[coefficients["target"].eq(target)].set_index("term")["coefficient"]
            residual = x[params.index].to_numpy() @ params.to_numpy()
            result.append(np.clip(np.expm1(np.log1p(frame[profile].to_numpy()) + residual), 0, None))
        return np.column_stack(result)

    mapping = metadata["station_mapping"]
    station = frame["station_id"].map(mapping).fillna(0).to_numpy(dtype=np.int64)
    standardizer = Standardizer.from_dict(metadata["standardizer"])
    x = standardizer.transform(build_numeric_features(frame))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected == "ann_full":
        model = DemandMLP(len(mapping) + 1, len(FEATURE_NAMES)).to(device)
        model.load_state_dict(torch.load(model_dir / "ann_full.pt", map_location=device, weights_only=True))
        return predict_mlp(model, x, station)
    if selected == "gru_exogenous":
        model = DemandGRU(len(mapping) + 1, len(FEATURE_NAMES)).to(device)
        model.load_state_dict(torch.load(model_dir / "gru_exogenous.pt", map_location=device, weights_only=True))
        n_stations = frame["station_id"].nunique()
        sequence = x.reshape(n_stations, 24, x.shape[1])
        station_sequence = station.reshape(n_stations, 24)[:, 0]
        return predict_gru(model, sequence, station_sequence).reshape(-1, 2)
    raise ValueError(selected)


def uncertainty_from_2025(predictions_path: Path, model: str) -> tuple[float, float, float]:
    frame = pd.read_parquet(predictions_path)
    frame = frame[frame["days_from_thanksgiving"].between(-3, 3)]
    error_out = np.abs(frame["ride_out"] - frame[f"{model}_ride_out"])
    error_in = np.abs(frame["ride_in"] - frame[f"{model}_ride_in"])
    actual_net = frame["ride_out"] - frame["ride_in"]
    predicted_net = frame[f"{model}_net_outflow"]
    return float(error_out.quantile(0.90)), float(error_in.quantile(0.90)), float(np.abs(actual_net - predicted_net).quantile(0.90))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally freeze the 2026 Thanksgiving fixed-date forecast")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/processed/stage3/models/final_2026"))
    parser.add_argument("--metrics", type=Path, default=Path("data/processed/stage3/results/backtest_2025/model_metrics.csv"))
    parser.add_argument("--backtest-predictions", type=Path, default=Path("data/processed/stage3/results/backtest_2025/predictions.parquet"))
    parser.add_argument("--weather-file", type=Path)
    parser.add_argument("--status", choices=("preliminary", "final"), default="preliminary")
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage3/forecasts/2026-11-26"))
    args = parser.parse_args()
    if args.weather_file is None:
        candidates = sorted(Path("data/raw/weather_vintages").glob("live_20261126_retrieved_*.parquet"))
        if not candidates:
            raise SystemExit("No live 2026-11-26 weather forecast file found")
        args.weather_file = candidates[-1]
    weather, weather_meta = load_weather(args.weather_file)
    retrieved = pd.Timestamp(weather_meta["retrieved_at_utc"]).tz_convert("America/New_York")
    if args.status == "final" and retrieved.date() not in {pd.Timestamp("2026-11-24").date(), pd.Timestamp("2026-11-25").date()}:
        raise SystemExit("Final freeze requires a weather forecast actually retrieved on Nov 24 or Nov 25 New York time")
    selected, ranking = choose_model(args.metrics)
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    frame = build_target_frame(args.catalog, args.model_dir, weather)
    prediction = predict_selected(frame, selected, args.model_dir, metadata)
    q_out, q_in, q_net = uncertainty_from_2025(args.backtest_predictions, selected)
    frame["predicted_ride_out"] = prediction[:, 0]
    frame["predicted_ride_in"] = prediction[:, 1]
    frame["predicted_net_outflow"] = prediction[:, 0] - prediction[:, 1]
    frame["ride_out_lower_90"] = np.clip(prediction[:, 0] - q_out, 0, None)
    frame["ride_out_upper_90"] = prediction[:, 0] + q_out
    frame["ride_in_lower_90"] = np.clip(prediction[:, 1] - q_in, 0, None)
    frame["ride_in_upper_90"] = prediction[:, 1] + q_in
    frame["net_outflow_lower_90"] = frame["predicted_net_outflow"] - q_net
    frame["net_outflow_upper_90"] = frame["predicted_net_outflow"] + q_net
    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{args.status}_{generated}"
    args.output.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "station_id", "station_name", "lat", "lon", "hour_local", "hour_utc",
        "predicted_ride_out", "predicted_ride_in", "predicted_net_outflow",
        "ride_out_lower_90", "ride_out_upper_90", "ride_in_lower_90",
        "ride_in_upper_90", "net_outflow_lower_90", "net_outflow_upper_90",
        "temperature_2m_c", "precipitation_mm", "snowfall_cm", "wind_speed_10m_max_ms",
        "profile_out", "profile_in", "profile_observations", "new_station_fallback",
    ]
    frame[output_columns].to_parquet(args.output / f"{prefix}.parquet", index=False, compression="zstd")
    frame[output_columns].to_csv(args.output / f"{prefix}.csv", index=False)
    pressure = frame.groupby(["station_id", "station_name", "lat", "lon"], as_index=False).agg(
        peak_absolute_net_outflow=("predicted_net_outflow", lambda value: float(np.max(np.abs(value)))),
        daily_net_outflow=("predicted_net_outflow", "sum"),
        daily_ride_out=("predicted_ride_out", "sum"),
        daily_ride_in=("predicted_ride_in", "sum"),
    ).sort_values("peak_absolute_net_outflow", ascending=False)
    pressure.head(50).to_csv(args.output / f"{prefix}_top_pressure.csv", index=False)
    manifest = {
        "status": args.status, "target_local_date": "2026-11-26",
        "generated_at_utc": generated, "selected_model": selected,
        "model_ranking_2025_thanksgiving_window": ranking.to_dict("records"),
        "model_version": metadata["model_version"],
        "training_start": metadata["training_start"],
        "training_end_exclusive": metadata["training_end_exclusive"],
        "weather_file": str(args.weather_file), "weather_metadata": weather_meta,
        "rows": len(frame), "stations": int(frame["station_id"].nunique()),
        "uncertainty_method": "Symmetric empirical band using the 90th-percentile absolute error from the strict 2025 Thanksgiving ±3-day backtest; not claimed as finite-sample conformal coverage.",
        "uncertainty_q": {"ride_out": q_out, "ride_in": q_in, "net_outflow": q_net},
        "not_actual_validation": "2026-11 official trips are not published; this file is a forecast, not a verified outcome.",
        "separate_from_realtime_risk": True,
    }
    (args.output / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
