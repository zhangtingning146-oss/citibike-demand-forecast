from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Stage 3 backtest and ablation results")
    parser.add_argument("--root", type=Path, default=Path("data/processed/stage3/results"))
    args = parser.parse_args()
    frames = []
    top_frames = []
    station_frames = []
    for year in (2024, 2025):
        base = args.root / f"backtest_{year}"
        metrics = pd.read_csv(base / "model_metrics.csv")
        metrics.insert(0, "backtest_year", year)
        frames.append(metrics)
        top = pd.read_csv(base / "topk_metrics.csv")
        top.insert(0, "backtest_year", year)
        top_frames.append(top)
        station = pd.read_csv(base / "station_metrics.csv")
        station.insert(0, "backtest_year", year)
        station_frames.append(station)
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(args.root / "model_comparison.csv", index=False)
    pd.concat(top_frames, ignore_index=True).to_csv(args.root / "topk_comparison.csv", index=False)
    pd.concat(station_frames, ignore_index=True).to_csv(args.root / "station_comparison.csv", index=False)

    baseline = comparison[comparison["model"].eq("historical_profile")][
        ["backtest_year", "scenario", "target", "wape"]
    ].rename(columns={"wape": "baseline_wape"})
    ablation = comparison.merge(baseline, on=["backtest_year", "scenario", "target"], how="left")
    ablation["wape_improvement_vs_profile"] = ablation["baseline_wape"] - ablation["wape"]
    ablation.to_csv(args.root / "ablation_results.csv", index=False)

    predictions = pd.read_parquet(args.root / "backtest_2025" / "predictions.parquet")
    window = predictions[predictions["days_from_thanksgiving"].between(-3, 3)].copy()
    focus = window.groupby(["station_id", "station_name"], as_index=False).agg(
        actual_ride_out=("ride_out", "sum"), actual_ride_in=("ride_in", "sum"),
        actual_net_outflow=("net_outflow", "sum"),
    )
    focus["activity"] = focus["actual_ride_out"] + focus["actual_ride_in"]
    focus["absolute_imbalance"] = focus["actual_net_outflow"].abs()
    focus = focus.sort_values(["activity", "absolute_imbalance"], ascending=False).head(100)
    focus.to_csv(args.root / "focus_stations_2025_thanksgiving.csv", index=False)

    thanksgiving_day = predictions[predictions["is_thanksgiving"]].copy()
    hourly_rows = []
    models = sorted({column.removesuffix("_ride_out") for column in thanksgiving_day.columns if column.endswith("_ride_out")})
    for model in models:
        for hour, group in thanksgiving_day.groupby(pd.to_datetime(thanksgiving_day["hour_local"]).dt.hour):
            for target in ("ride_out", "ride_in"):
                actual = group[target].to_numpy(dtype=float)
                predicted = group[f"{model}_{target}"].to_numpy(dtype=float)
                error = predicted - actual
                hourly_rows.append({
                    "model": model, "local_hour": int(hour), "target": target,
                    "rows": len(group), "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "wape": float(np.sum(np.abs(error)) / max(np.sum(actual), 1.0)),
                })
    pd.DataFrame(hourly_rows).to_csv(args.root / "thanksgiving_2025_hourly_errors.csv", index=False)

    eligible = ["historical_profile", "panel_weather_holiday", "ann_full", "gru_exogenous"]
    ranking = comparison[
        comparison["backtest_year"].eq(2025)
        & comparison["scenario"].eq("thanksgiving_window")
        & comparison["target"].isin(["ride_out", "ride_in"])
        & comparison["model"].isin(eligible)
    ].groupby("model", as_index=False)["wape"].mean().sort_values("wape")
    summary = {
        "selected_final_model": str(ranking.iloc[0]["model"]),
        "selection_rule": "Lowest mean ride-in/out WAPE in the strict 2025 Thanksgiving ±3-day evaluation among the four prespecified core candidates.",
        "ranking": ranking.to_dict("records"),
    }
    (args.root / "model_selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
