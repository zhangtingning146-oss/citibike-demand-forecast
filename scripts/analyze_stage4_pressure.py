from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["historical_profile", "panel_weather_holiday", "ann_full", "gru_exogenous"]


def precision_at_k(frame: pd.DataFrame, score: pd.Series, k: int) -> float:
    values = []
    work = frame.assign(_score=score)
    for _, group in work.groupby("hour_utc", sort=False):
        if len(group) < k:
            continue
        actual = set(group.nlargest(k, "actual_pressure").index)
        predicted = set(group.nlargest(k, "_score").index)
        values.append(len(actual & predicted) / k)
    return float(np.mean(values))


def diagnose_year(path: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["actual_total"] = frame["ride_out"] + frame["ride_in"]
    frame["actual_pressure"] = (frame["ride_out"] - frame["ride_in"]).abs()
    positive = frame["actual_total"].gt(0)
    frame["demand_group"] = "zero"
    frame.loc[positive, "demand_group"] = pd.qcut(
        frame.loc[positive, "actual_total"], q=5, labels=["q1_low", "q2", "q3", "q4", "q5_high"], duplicates="drop"
    ).astype(str)
    pressure_cut = frame["actual_pressure"].quantile(0.90)
    frame["imbalance_group"] = np.where(frame["actual_pressure"].ge(pressure_cut), "top10_pressure", "other")

    bias_rows = []
    decomposition_rows = []
    for model in MODELS:
        error_out = frame[f"{model}_ride_out"] - frame["ride_out"]
        error_in = frame[f"{model}_ride_in"] - frame["ride_in"]
        error_net = error_out - error_in
        decomposition_rows.append({
            "year": year,
            "model": model,
            "ride_out_mae": float(error_out.abs().mean()),
            "ride_in_mae": float(error_in.abs().mean()),
            "net_mae": float(error_net.abs().mean()),
            "ride_out_bias": float(error_out.mean()),
            "ride_in_bias": float(error_in.mean()),
            "net_bias": float(error_net.mean()),
            "out_in_error_correlation": float(np.corrcoef(error_out, error_in)[0, 1]),
            "net_error_to_sum_component_error": float(error_net.abs().sum() / (error_out.abs().sum() + error_in.abs().sum())),
            "net_wape_denominator": float(frame["actual_pressure"].sum()),
            "ride_volume_denominator": float(frame["actual_total"].sum()),
            "net_to_volume_denominator_ratio": float(frame["actual_pressure"].sum() / frame["actual_total"].sum()),
        })
        for dimension in ("demand_group", "imbalance_group"):
            for group, subset in frame.groupby(dimension, observed=True):
                eo = subset[f"{model}_ride_out"] - subset["ride_out"]
                ei = subset[f"{model}_ride_in"] - subset["ride_in"]
                en = eo - ei
                bias_rows.append({
                    "year": year, "model": model, "dimension": dimension, "group": str(group),
                    "rows": len(subset), "actual_out_mean": float(subset["ride_out"].mean()),
                    "actual_in_mean": float(subset["ride_in"].mean()),
                    "out_bias": float(eo.mean()), "in_bias": float(ei.mean()), "net_bias": float(en.mean()),
                    "out_mae": float(eo.abs().mean()), "in_mae": float(ei.abs().mean()), "net_mae": float(en.abs().mean()),
                })

    ranking_rows = []
    ann_volume = np.sqrt(np.maximum(frame["ann_full_ride_out"] + frame["ann_full_ride_in"], 0))
    panel_pressure = (frame["panel_weather_holiday_ride_out"] - frame["panel_weather_holiday_ride_in"]).abs()
    for model in MODELS:
        score = (frame[f"{model}_ride_out"] - frame[f"{model}_ride_in"]).abs()
        for k in (10, 20):
            ranking_rows.append({"year": year, "method": model, "k": k, "precision_at_k": precision_at_k(frame, score, k)})
    for weight in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        score = panel_pressure + weight * ann_volume
        for k in (10, 20):
            ranking_rows.append({
                "year": year, "method": f"panel_pressure_plus_{weight:g}_sqrt_ann_volume",
                "k": k, "precision_at_k": precision_at_k(frame, score, k),
            })
    return pd.DataFrame(bias_rows), pd.DataFrame(decomposition_rows), pd.DataFrame(ranking_rows)


def make_forecast_map(forecast_path: Path, output: Path) -> None:
    frame = pd.read_csv(forecast_path)
    stations = frame.groupby(["station_id", "station_name", "lat", "lon"], as_index=False).agg(
        daily_out=("predicted_ride_out", "sum"), daily_in=("predicted_ride_in", "sum"),
        peak_pressure=("predicted_net_outflow", lambda value: float(np.max(np.abs(value)))),
    )
    stations["daily_net"] = stations["daily_out"] - stations["daily_in"]
    top = stations.nlargest(20, "peak_pressure")
    width, height, margin = 760, 980, 55
    lon_min, lon_max = stations["lon"].min(), stations["lon"].max()
    lat_min, lat_max = stations["lat"].min(), stations["lat"].max()
    x = lambda value: margin + (value - lon_min) / (lon_max - lon_min) * (width - 2 * margin)
    y = lambda value: height - margin - (value - lat_min) / (lat_max - lat_min) * (height - 2 * margin)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Preliminary Citi Bike demand pressure — Thanksgiving 2026</text>',
        '<text x="380" y="50" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">Point size: peak hourly | color: daily net outflow (red) / inflow (blue)</text>',
    ]
    for row in stations.itertuples():
        elements.append(f'<circle cx="{x(row.lon):.1f}" cy="{y(row.lat):.1f}" r="2.2" fill="#aeb8c2" fill-opacity="0.45"/>')
    scale = max(float(top["daily_net"].abs().max()), 1.0)
    top_ids = set(top.nlargest(8, "peak_pressure")["station_id"])
    for row in top.itertuples():
        strength = min(abs(row.daily_net) / scale, 1.0)
        color = f'rgb({int(170 + 85 * strength)},70,70)' if row.daily_net >= 0 else f'rgb(60,90,{int(170 + 85 * strength)})'
        radius = 4 + row.peak_pressure * 0.7
        cx, cy = x(row.lon), y(row.lat)
        elements.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.75" stroke="#222" stroke-width="0.7"/>')
        if row.station_id in top_ids:
            elements.append(f'<text x="{cx + radius + 3:.1f}" y="{cy + 3:.1f}" font-family="Arial" font-size="9" fill="#222">{escape(str(row.station_name))}</text>')
    elements.extend([
        '<text x="380" y="965" text-anchor="middle" font-family="Arial" font-size="10" fill="#555">Geographic scatter only; preliminary demand pressure is not an inventory-risk alert.</text>',
        '</svg>',
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 demand-pressure diagnostics using saved strict backtests")
    parser.add_argument("--results", type=Path, default=Path("data/processed/stage3/results"))
    parser.add_argument("--forecast", type=Path, default=Path("data/processed/stage3/forecasts/2026-11-26/preliminary_20261118T161247Z.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage4/pressure"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    biases, decompositions, rankings = [], [], []
    for year in (2024, 2025):
        result = diagnose_year(args.results / f"backtest_{year}" / "predictions.parquet", year)
        biases.append(result[0]); decompositions.append(result[1]); rankings.append(result[2])
    bias = pd.concat(biases, ignore_index=True)
    decomposition = pd.concat(decompositions, ignore_index=True)
    ranking = pd.concat(rankings, ignore_index=True)
    validation = ranking[ranking["year"].eq(2024) & ranking["method"].str.startswith("panel_pressure_plus_")]
    # Limited intervention capacity makes Top 10 the prespecified primary target;
    # Top 20 breaks ties. 2025 is never consulted for selection.
    selection = validation.pivot(index="method", columns="k", values="precision_at_k").reset_index()
    selected = selection.sort_values([10, 20, "method"], ascending=[False, False, True]).iloc[0]["method"]
    ranking["selected_on_2024_validation"] = ranking["method"].eq(selected)
    bias.to_csv(args.output / "bias_by_demand_group.csv", index=False)
    decomposition.to_csv(args.output / "net_error_decomposition.csv", index=False)
    ranking.to_csv(args.output / "pressure_ranking_comparison.csv", index=False)
    make_forecast_map(args.forecast, args.output / "thanksgiving_2026_preliminary_pressure_map.svg")
    summary = {
        "selection_data": "2024-11 strict backtest only",
        "selection_rule": "Maximize Top-10 precision first, then Top-20 precision as the tie-breaker.",
        "locked_ranking_method": selected,
        "independent_evaluation_data": "2025-11 strict backtest",
        "interpretation": "Absolute panel net flow plus a demand-volatility buffer based on sqrt(ANN total demand).",
    }
    (args.output / "pressure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
