from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from citibike_pipeline.stage3_models import (
    BASIC_FEATURES,
    FEATURE_NAMES,
    DemandGRU,
    DemandMLP,
    Standardizer,
    build_numeric_features,
    canonical_hour_utc,
    demand_metrics,
    predict_gru,
    predict_mlp,
    save_metadata,
    seed_everything,
    train_gru,
    train_mlp,
)


SEED = 20261118
WEATHER_RENAME = {
    "temperature_2m": "temperature_2m_c",
    "apparent_temperature": "apparent_temperature_c",
    "relative_humidity_2m": "relative_humidity_2m_pct",
    "precipitation": "precipitation_mm",
    "snowfall": "snowfall_cm",
    "wind_speed_10m": "wind_speed_10m_max_ms",
    "wind_gusts_10m": "wind_gusts_10m_max_ms",
    "cloud_cover": "cloud_cover_pct",
}


def sql_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def sql_path(value: Path) -> str:
    return str(value.resolve()).replace("\\", "/").replace("'", "''")


def create_profiles(con: duckdb.DuckDBPyConnection, train_start: pd.Timestamp, cutoff: pd.Timestamp) -> None:
    where = (
        f"hour_local >= TIMESTAMP '{sql_timestamp(train_start)}' AND "
        f"hour_local < TIMESTAMP '{sql_timestamp(cutoff)}' AND "
        "NOT coalesce(dst_ambiguous_collapsed, false)"
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE profile_month AS
        SELECT station_id, month(hour_local)::UTINYINT AS profile_month,
          iso_day_of_week, local_hour,
          avg(ride_out)::DOUBLE AS profile_out,
          avg(ride_in)::DOUBLE AS profile_in,
          count(*)::INTEGER AS profile_observations
        FROM model_features WHERE {where}
        GROUP BY 1,2,3,4
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE profile_week AS
        SELECT station_id, iso_day_of_week, local_hour,
          avg(ride_out)::DOUBLE AS week_out,
          avg(ride_in)::DOUBLE AS week_in,
          count(*)::INTEGER AS week_observations
        FROM model_features WHERE {where}
        GROUP BY 1,2,3
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE profile_global AS
        SELECT month(hour_local)::UTINYINT AS profile_month,
          iso_day_of_week, local_hour,
          avg(ride_out)::DOUBLE AS global_out,
          avg(ride_in)::DOUBLE AS global_in
        FROM model_features WHERE {where}
        GROUP BY 1,2,3
    """)


def load_frame(
    con: duckdb.DuckDBPyConnection,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sample_mod: int | None = None,
    sample_by_day: bool = False,
    include_attributes: bool = False,
) -> pd.DataFrame:
    sample = ""
    if sample_mod:
        key = "p.local_date" if sample_by_day else "p.hour_local"
        sample = f"AND hash(p.station_id, {key}) % {sample_mod} = 0"
    attributes = ", p.station_name, p.lat, p.lon" if include_attributes else ""
    query = f"""
        SELECT p.station_id, p.hour_local, p.local_date, p.local_hour,
          p.iso_day_of_week, month(p.hour_local)::UTINYINT AS panel_month,
          p.ride_out, p.ride_in, p.net_outflow,
          p.temperature_2m_c, p.apparent_temperature_c,
          p.relative_humidity_2m_pct, p.precipitation_mm, p.snowfall_cm,
          p.wind_speed_10m_max_ms, p.wind_gusts_10m_max_ms, p.cloud_cover_pct,
          p.is_weekend, p.is_workday, p.is_federal_holiday,
          p.is_thanksgiving, p.is_thanksgiving_eve,
          p.is_day_after_thanksgiving, p.days_from_thanksgiving,
          coalesce(pm.profile_out, pw.week_out, pg.global_out, 0)::DOUBLE AS profile_out,
          coalesce(pm.profile_in, pw.week_in, pg.global_in, 0)::DOUBLE AS profile_in,
          coalesce(pm.profile_observations, pw.week_observations, 0)::INTEGER AS profile_observations,
          (pm.station_id IS NULL AND pw.station_id IS NULL) AS new_station_fallback
          {attributes}
        FROM model_features p
        LEFT JOIN profile_month pm
          ON pm.station_id=p.station_id AND pm.profile_month=month(p.hour_local)
          AND pm.iso_day_of_week=p.iso_day_of_week AND pm.local_hour=p.local_hour
        LEFT JOIN profile_week pw
          ON pw.station_id=p.station_id AND pw.iso_day_of_week=p.iso_day_of_week
          AND pw.local_hour=p.local_hour
        LEFT JOIN profile_global pg
          ON pg.profile_month=month(p.hour_local)
          AND pg.iso_day_of_week=p.iso_day_of_week AND pg.local_hour=p.local_hour
        WHERE p.hour_local >= TIMESTAMP '{sql_timestamp(start)}'
          AND p.hour_local < TIMESTAMP '{sql_timestamp(end)}'
          AND NOT coalesce(p.dst_ambiguous_collapsed, false)
          {sample}
        ORDER BY p.station_id, p.hour_local
    """
    frame = con.execute(query).fetchdf()
    frame["hour_utc"] = canonical_hour_utc(frame["hour_local"])
    if frame["hour_utc"].isna().any():
        raise RuntimeError("Ambiguous or nonexistent local hours remained after filtering")
    return frame


def override_weather(frame: pd.DataFrame, forecast_path: Path) -> pd.DataFrame:
    weather = pd.read_parquet(forecast_path)
    weather["hour_utc"] = pd.to_datetime(weather["hour_utc"], utc=True)
    weather = weather.rename(columns=WEATHER_RENAME)
    keep = ["hour_utc", *WEATHER_RENAME.values()]
    merged = frame.drop(columns=list(WEATHER_RENAME.values())).merge(weather[keep], on="hour_utc", how="left", validate="many_to_one")
    missing = merged[list(WEATHER_RENAME.values())].isna().sum()
    if missing.any():
        raise RuntimeError(f"Archived forecast does not cover evaluation hours: {missing.to_dict()}")
    return merged


def station_index(frame: pd.DataFrame) -> tuple[dict[str, int], np.ndarray]:
    stations = sorted(frame["station_id"].dropna().unique().tolist())
    mapping = {station: index + 1 for index, station in enumerate(stations)}
    encoded = frame["station_id"].map(mapping).fillna(0).to_numpy(dtype=np.int64)
    return mapping, encoded


def encode_station(frame: pd.DataFrame, mapping: dict[str, int]) -> np.ndarray:
    return frame["station_id"].map(mapping).fillna(0).to_numpy(dtype=np.int64)


REG_WEATHER = [
    "temperature_10c", "temperature_sq", "humidity_100", "log_precipitation",
    "snowfall_cm", "wind_10ms", "cloud_100",
]
REG_HOLIDAY = [
    "is_federal_holiday", "is_thanksgiving", "is_thanksgiving_eve",
    "is_day_after_thanksgiving", "thanksgiving_distance",
]


def regression_matrix(frame: pd.DataFrame, include_holiday: bool) -> pd.DataFrame:
    temperature = frame["temperature_2m_c"].to_numpy(dtype=float)
    data = {
        "temperature_10c": temperature / 10,
        "temperature_sq": (temperature / 10) ** 2,
        "humidity_100": frame["relative_humidity_2m_pct"].to_numpy(dtype=float) / 100,
        "log_precipitation": np.log1p(frame["precipitation_mm"].to_numpy(dtype=float)),
        "snowfall_cm": frame["snowfall_cm"].to_numpy(dtype=float),
        "wind_10ms": frame["wind_speed_10m_max_ms"].to_numpy(dtype=float) / 10,
        "cloud_100": frame["cloud_cover_pct"].to_numpy(dtype=float) / 100,
    }
    if include_holiday:
        for column in REG_HOLIDAY[:-1]:
            data[column] = frame[column].astype(float).to_numpy()
        data["thanksgiving_distance"] = frame["days_from_thanksgiving"].clip(-7, 7).to_numpy(dtype=float) / 7
    return sm.add_constant(pd.DataFrame(data), has_constant="add")


def fit_regression(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    include_holiday: bool,
    coefficient_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    sample = train
    if len(sample) > 600_000:
        sample = sample.sample(600_000, random_state=SEED)
    x_train = regression_matrix(sample, include_holiday)
    x_eval = regression_matrix(evaluation, include_holiday)
    predictions = []
    coefficient_rows = []
    for target, profile in (("ride_out", "profile_out"), ("ride_in", "profile_in")):
        residual = np.log1p(sample[target].to_numpy(dtype=float)) - np.log1p(sample[profile].to_numpy(dtype=float))
        fit = sm.OLS(residual, x_train).fit(
            cov_type="cluster", cov_kwds={"groups": sample["local_date"].astype(str).to_numpy()}
        )
        prediction = np.expm1(np.log1p(evaluation[profile].to_numpy(dtype=float)) + fit.predict(x_eval))
        predictions.append(np.clip(prediction, 0, None))
        for term in x_train.columns:
            coefficient_rows.append({
                "target": target, "term": term, "coefficient": fit.params[term],
                "cluster_robust_se": fit.bse[term], "p_value": fit.pvalues[term],
                "nobs": int(fit.nobs), "r_squared_residual": fit.rsquared,
            })
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
    return predictions[0], predictions[1]


def complete_day_sequences(
    frame: pd.DataFrame,
    x: np.ndarray,
    station_encoded: np.ndarray,
    with_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    work = frame[["station_id", "local_date", "local_hour"]].copy()
    work["row_position"] = np.arange(len(frame))
    summary = work.groupby(["station_id", "local_date"], sort=False).agg(
        rows=("row_position", "size"), min_hour=("local_hour", "min"),
        max_hour=("local_hour", "max"), distinct_hours=("local_hour", "nunique")
    )
    keys = summary.query("rows == 24 and min_hour == 0 and max_hour == 23 and distinct_hours == 24").reset_index()[["station_id", "local_date"]]
    selected = work.merge(keys, on=["station_id", "local_date"], how="inner").sort_values(["station_id", "local_date", "local_hour"])
    positions = selected["row_position"].to_numpy(dtype=np.int64)
    n_sequences = len(positions) // 24
    x_sequence = x[positions].reshape(n_sequences, 24, x.shape[1])
    station_sequence = station_encoded[positions][::24]
    y_sequence = None
    if with_target:
        y = frame[["ride_out", "ride_in"]].to_numpy(dtype=np.float32)
        y_sequence = y[positions].reshape(n_sequences, 24, 2)
    return x_sequence, station_sequence, y_sequence, positions


def metric_rows(
    evaluation: pd.DataFrame,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    training_weather_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_low, temp_high = training_weather_reference["temperature_2m_c"].quantile([0.05, 0.95]).tolist()
    precipitation_high = training_weather_reference["precipitation_mm"].quantile(0.95)
    scenarios = {
        "all": np.ones(len(evaluation), dtype=bool),
        "workday": evaluation["is_workday"].to_numpy(dtype=bool),
        "weekend": evaluation["is_weekend"].to_numpy(dtype=bool),
        "thanksgiving_window": evaluation["days_from_thanksgiving"].between(-3, 3).to_numpy(),
        "thanksgiving_day": evaluation["is_thanksgiving"].to_numpy(dtype=bool),
        "extreme_weather": (
            (evaluation["temperature_2m_c"] <= temp_low)
            | (evaluation["temperature_2m_c"] >= temp_high)
            | (evaluation["precipitation_mm"] >= precipitation_high)
        ).to_numpy(),
    }
    overall = []
    station_rows = []
    top_rows = []
    actual_out = evaluation["ride_out"].to_numpy(dtype=float)
    actual_in = evaluation["ride_in"].to_numpy(dtype=float)
    for model, (pred_out, pred_in) in predictions.items():
        for scenario, mask in scenarios.items():
            if mask.sum() == 0:
                continue
            metrics = demand_metrics(actual_out[mask], actual_in[mask], pred_out[mask], pred_in[mask])
            for target, values in metrics.items():
                overall.append({"model": model, "scenario": scenario, "target": target, "rows": int(mask.sum()), **values})
        temp = evaluation[["station_id", "ride_out", "ride_in"]].copy()
        temp["pred_out"] = pred_out
        temp["pred_in"] = pred_in
        for station, group in temp.groupby("station_id"):
            values = demand_metrics(
                group["ride_out"].to_numpy(), group["ride_in"].to_numpy(),
                group["pred_out"].to_numpy(), group["pred_in"].to_numpy(),
            )
            station_rows.append({
                "model": model, "station_id": station, "rows": len(group),
                "ride_out_mae": values["ride_out"]["mae"], "ride_out_wape": values["ride_out"]["wape"],
                "ride_in_mae": values["ride_in"]["mae"], "ride_in_wape": values["ride_in"]["wape"],
                "net_mae": values["net_outflow"]["mae"],
            })
        actual_pressure = np.abs(actual_out - actual_in)
        predicted_pressure = np.abs(pred_out - pred_in)
        rank_frame = pd.DataFrame({"hour_utc": evaluation["hour_utc"], "actual": actual_pressure, "predicted": predicted_pressure})
        for k in (10, 20):
            precisions = []
            for _, group in rank_frame.groupby("hour_utc", sort=False):
                if len(group) < k:
                    continue
                actual_top = set(group.nlargest(k, "actual").index)
                predicted_top = set(group.nlargest(k, "predicted").index)
                precisions.append(len(actual_top & predicted_top) / k)
            top_rows.append({"model": model, "k": k, "mean_precision_at_k": float(np.mean(precisions)), "hours": len(precisions)})
    return pd.DataFrame(overall), pd.DataFrame(station_rows), pd.DataFrame(top_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-date Citi Bike demand backtests")
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--weather-root", type=Path, default=Path("data/raw/weather_vintages"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/stage3"))
    parser.add_argument("--sample-mod", type=int, default=8)
    parser.add_argument("--ann-epochs", type=int, default=4)
    parser.add_argument("--gru-epochs", type=int, default=3)
    args = parser.parse_args()
    seed_everything(SEED)
    cutoff = pd.Timestamp(f"{args.year}-11-01")
    train_start = pd.Timestamp("2023-01-01")
    evaluation_end = pd.Timestamp(f"{args.year}-12-01")
    run_name = f"backtest_{args.year}"
    model_dir = args.output_root / "models" / run_name
    result_dir = args.output_root / "results" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.catalog), read_only=False)
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    create_profiles(con, train_start, cutoff)
    con.execute(f"COPY profile_month TO '{sql_path(model_dir / 'profile_month.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print("Loading deterministic training sample", flush=True)
    train = load_frame(con, train_start, cutoff, sample_mod=args.sample_mod)
    training_rows = len(train)
    print(f"training rows={training_rows:,}", flush=True)
    print("Loading complete evaluation month", flush=True)
    evaluation = load_frame(con, cutoff, evaluation_end, include_attributes=True)
    evaluation = override_weather(evaluation, args.weather_root / f"previous_runs_day2_{args.year}11.parquet")
    print(f"evaluation rows={len(evaluation):,}", flush=True)

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "historical_profile": (
            evaluation["profile_out"].to_numpy(dtype=float),
            evaluation["profile_in"].to_numpy(dtype=float),
        )
    }
    predictions["panel_weather"] = fit_regression(
        train, evaluation, False, result_dir / "panel_weather_coefficients.csv"
    )
    predictions["panel_weather_holiday"] = fit_regression(
        train, evaluation, True, result_dir / "panel_weather_holiday_coefficients.csv"
    )

    mapping, train_station = station_index(train)
    eval_station = encode_station(evaluation, mapping)
    x_train_all = build_numeric_features(train)
    x_eval_all = build_numeric_features(evaluation)
    y_train = train[["ride_out", "ride_in"]].to_numpy(dtype=np.float32)
    training_history = {}
    model_specs = {}
    ann_full_eval = None
    full_standardizer = None
    for label, selected_names, epochs in (
        ("ann_basic", [name for name in FEATURE_NAMES if name in BASIC_FEATURES], max(2, args.ann_epochs - 1)),
        ("ann_full", FEATURE_NAMES, args.ann_epochs),
    ):
        indices = [FEATURE_NAMES.index(name) for name in selected_names]
        standardizer = Standardizer.fit(x_train_all[:, indices])
        x_train = standardizer.transform(x_train_all[:, indices])
        x_eval = standardizer.transform(x_eval_all[:, indices])
        model, history = train_mlp(
            x_train, train_station, y_train, len(mapping) + 1,
            model_dir / f"{label}.pt", epochs=epochs, seed=SEED,
        )
        prediction = predict_mlp(model, x_eval, eval_station)
        predictions[label] = (prediction[:, 0], prediction[:, 1])
        training_history[label] = history
        model_specs[label] = {
            "feature_names": selected_names, "standardizer": standardizer.to_dict(),
            "epochs": epochs,
        }
        if label == "ann_full":
            ann_full_eval = prediction
            full_standardizer = standardizer
        del x_train, x_eval

    training_weather_reference = train[["temperature_2m_c", "precipitation_mm"]].copy()
    del x_train_all, y_train, train
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading station-day sample for GRU", flush=True)
    train_day = load_frame(con, train_start, cutoff, sample_mod=args.sample_mod, sample_by_day=True)
    train_day_station = encode_station(train_day, mapping)
    x_day_all = build_numeric_features(train_day)
    x_day = full_standardizer.transform(x_day_all)
    x_sequence, station_sequence, y_sequence, _ = complete_day_sequences(train_day, x_day, train_day_station)
    gru, gru_history = train_gru(
        x_sequence, station_sequence, y_sequence, len(mapping) + 1,
        model_dir / "gru_exogenous.pt", epochs=args.gru_epochs, seed=SEED,
    )
    x_eval_full = full_standardizer.transform(x_eval_all)
    x_eval_sequence, eval_station_sequence, _, positions = complete_day_sequences(
        evaluation, x_eval_full, eval_station, with_target=False
    )
    gru_sequence_prediction = predict_gru(gru, x_eval_sequence, eval_station_sequence)
    gru_prediction = ann_full_eval.copy()
    gru_prediction[positions] = gru_sequence_prediction.reshape(-1, 2)
    predictions["gru_exogenous"] = (gru_prediction[:, 0], gru_prediction[:, 1])
    training_history["gru_exogenous"] = gru_history
    model_specs["gru_exogenous"] = {
        "feature_names": FEATURE_NAMES, "standardizer": full_standardizer.to_dict(),
        "epochs": args.gru_epochs, "sequence_hours": 24,
        "fallback_for_incomplete_days": "ann_full",
    }

    overall, per_station, top_k = metric_rows(evaluation, predictions, training_weather_reference)
    overall.to_csv(result_dir / "model_metrics.csv", index=False)
    per_station.to_csv(result_dir / "station_metrics.csv", index=False)
    top_k.to_csv(result_dir / "topk_metrics.csv", index=False)
    prediction_frame = evaluation[[
        "station_id", "station_name", "lat", "lon", "hour_local", "hour_utc",
        "ride_out", "ride_in", "net_outflow", "is_thanksgiving",
        "days_from_thanksgiving", "temperature_2m_c", "precipitation_mm",
        "new_station_fallback", "profile_observations",
    ]].copy()
    for model, (pred_out, pred_in) in predictions.items():
        prediction_frame[f"{model}_ride_out"] = pred_out
        prediction_frame[f"{model}_ride_in"] = pred_in
        prediction_frame[f"{model}_net_outflow"] = pred_out - pred_in
    prediction_frame.to_parquet(result_dir / "predictions.parquet", index=False, compression="zstd")

    metadata = {
        "run_name": run_name, "training_start": str(train_start),
        "training_cutoff_exclusive": str(cutoff), "evaluation_start": str(cutoff),
        "evaluation_end_exclusive": str(evaluation_end),
        "training_sample_modulus": args.sample_mod, "training_rows": training_rows,
        "evaluation_rows": len(evaluation), "station_count_train": len(mapping),
        "unknown_station_rows_evaluation": int((eval_station == 0).sum()),
        "weather_vintage": str(args.weather_root / f"previous_runs_day2_{args.year}11.parquet"),
        "weather_lead_hours": 48, "ambiguous_fall_hours_removed": True,
        "target_lags_used": False, "seed": SEED,
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "station_mapping": mapping, "models": model_specs, "training_history": training_history,
    }
    save_metadata(model_dir / "metadata.json", metadata)
    print(overall.query("scenario == 'all'").to_string(index=False), flush=True)
    con.close()


if __name__ == "__main__":
    main()
