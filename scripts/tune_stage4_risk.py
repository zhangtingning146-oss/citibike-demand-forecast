from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_inventory_risk import CATEGORICAL, NUMERIC, classification_metrics


def load_frame(catalog: Path) -> pd.DataFrame:
    con = duckdb.connect(str(catalog), read_only=True)
    frame = con.execute("""
        WITH manhattan_stations AS (
          SELECT DISTINCT gbfs_station_id_current AS gbfs_station_id
          FROM model_features WHERE gbfs_station_id_current IS NOT NULL
        ), demand_profile AS (
          SELECT gbfs_station_id_current AS gbfs_station_id, iso_day_of_week, local_hour,
            avg(ride_out)::DOUBLE AS profile_out, avg(ride_in)::DOUBLE AS profile_in
          FROM model_features
          WHERE hour_local < TIMESTAMP '2026-09-01'
            AND gbfs_station_id_current IS NOT NULL
            AND NOT coalesce(dst_ambiguous_collapsed, false)
          GROUP BY 1,2,3
        ), eligible AS (
          SELECT * FROM inventory_hourly
          WHERE observed_minutes >= 45 AND operating_observations >= 45 AND capacity_observed > 0
        )
        SELECT origin.gbfs_station_id, origin.hour_start_utc AS origin_hour_utc,
          target.hour_start_utc AS target_hour_utc, target.hour_local AS target_hour_local,
          extract(hour FROM target.hour_local)::INTEGER AS target_local_hour,
          extract(isodow FROM target.hour_local)::INTEGER AS target_iso_dow,
          origin.mean_bikes_available, origin.mean_docks_available,
          origin.min_bikes_available, origin.min_docks_available, origin.capacity_observed,
          coalesce(dp.profile_out, 0) AS profile_out, coalesce(dp.profile_in, 0) AS profile_in,
          target.empty_any AS target_empty_any, target.full_any AS target_full_any
        FROM eligible origin
        JOIN manhattan_stations ms ON ms.gbfs_station_id=origin.gbfs_station_id
        JOIN eligible target ON target.gbfs_station_id=origin.gbfs_station_id
          AND target.hour_start_utc=origin.hour_start_utc + INTERVAL 1 HOUR
        LEFT JOIN demand_profile dp ON dp.gbfs_station_id=origin.gbfs_station_id
          AND dp.iso_day_of_week=extract(isodow FROM target.hour_local)
          AND dp.local_hour=extract(hour FROM target.hour_local)
        WHERE origin.hour_start_utc >= TIMESTAMPTZ '2026-04-07 00:00:00+00'
          AND target.hour_start_utc < TIMESTAMPTZ '2026-11-01 00:00:00+00'
    """).fetchdf()
    con.close()
    frame["target_hour_utc"] = pd.to_datetime(frame["target_hour_utc"], utc=True)
    return frame


def new_model() -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("features", transformer),
        ("classifier", LogisticRegression(max_iter=300, class_weight="balanced", random_state=20261118)),
    ])


def threshold_table(actual: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        rows.append({"threshold": threshold, **classification_metrics(actual, (probability >= threshold).astype(int), probability)})
    return pd.DataFrame(rows)


def mean_precision_at_k(frame: pd.DataFrame, label: str, score: pd.Series, k: int) -> float:
    values = []
    for _, group in frame.assign(_score=score).groupby("target_hour_utc", sort=False):
        if len(group) < k:
            continue
        selected = group.nlargest(k, "_score")
        values.append(float(selected[label].mean()))
    return float(np.mean(values))


def ranking_scores(frame: pd.DataFrame, risk: str, alpha: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    capacity = frame["capacity_observed"].clip(lower=1)
    net = frame["profile_out"] - frame["profile_in"]
    if risk == "empty":
        current = -frame["mean_bikes_available"] / capacity
        demand_only = net
        projected = -(frame["mean_bikes_available"] - alpha * net) / capacity
    else:
        current = -frame["mean_docks_available"] / capacity
        demand_only = -net
        projected = -(frame["mean_docks_available"] + alpha * net) / capacity
    return current, demand_only, projected


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune risk thresholds on September and evaluate once on October")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage4/risk"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.catalog)
    train = frame[frame["target_hour_utc"] < pd.Timestamp("2026-09-01", tz="UTC")].copy()
    validation = frame[frame["target_hour_utc"].between(pd.Timestamp("2026-09-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"), inclusive="left")].copy()
    test = frame[frame["target_hour_utc"] >= pd.Timestamp("2026-10-01", tz="UTC")].copy()
    metrics, threshold_rows, ranking_rows, selected = [], [], [], {}
    for risk, label in (("empty", "target_empty_any"), ("full", "target_full_any")):
        model = new_model()
        model.fit(train[NUMERIC + CATEGORICAL], train[label].astype(int))
        validation_probability = model.predict_proba(validation[NUMERIC + CATEGORICAL])[:, 1]
        table = threshold_table(validation[label].astype(int).to_numpy(), validation_probability)
        best = table.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
        locked_threshold = float(best["threshold"])
        table.insert(0, "risk", risk)
        threshold_rows.append(table)

        # Refit after threshold selection; October remains untouched.
        model.fit(pd.concat([train, validation])[NUMERIC + CATEGORICAL], pd.concat([train, validation])[label].astype(int))
        probability = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
        actual = test[label].astype(int).to_numpy()
        for name, threshold in (("default_0.50", 0.5), ("validation_locked", locked_threshold)):
            metrics.append({"risk": risk, "method": name, "threshold": threshold, **classification_metrics(actual, (probability >= threshold).astype(int), probability)})
        joblib.dump({"model": model, "threshold": locked_threshold}, args.output / f"{risk}_risk_validation_locked.joblib")

        validation_candidates = []
        for alpha in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            _, _, projected = ranking_scores(validation, risk, alpha)
            score = np.mean([mean_precision_at_k(validation, label, projected, k) for k in (10, 20)])
            validation_candidates.append((score, alpha))
        alpha = max(validation_candidates)[1]
        selected[risk] = {"threshold": locked_threshold, "projected_margin_alpha": alpha}
        current, demand_only, projected = ranking_scores(test, risk, alpha)
        for name, score in (("current_inventory_margin", current), ("demand_profile_only", demand_only), ("projected_margin", projected)):
            for k in (10, 20):
                ranking_rows.append({
                    "risk": risk, "method": name, "k": k,
                    "mean_precision_at_k": mean_precision_at_k(test, label, score, k),
                    "alpha_selected_on_validation": alpha if name == "projected_margin" else np.nan,
                })
    pd.concat(threshold_rows, ignore_index=True).to_csv(args.output / "validation_threshold_curves.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.output / "test_threshold_metrics.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(args.output / "test_risk_ranking.csv", index=False)
    metadata = {
        "train": "2026-04-07 through 2026-08-31", "validation": "2026-09", "test": "2026-10",
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_used_for_threshold_selection": False, "selected": selected,
        "ranking_interpretation": "Projected normalized inventory margin after alpha times the pre-September historical demand net flow.",
    }
    (args.output / "stage4_risk_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
