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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "mean_bikes_available", "mean_docks_available", "min_bikes_available",
    "min_docks_available", "capacity_observed", "profile_out", "profile_in",
]
CATEGORICAL = ["target_local_hour", "target_iso_dow"]


def classification_metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray | None = None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    result = {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "true_negative": tn, "actual_risks": int(actual.sum()),
        "predicted_risks": int(predicted.sum()),
    }
    if probability is not None and len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probability))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe one-hour empty/full risk warnings")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage3/risk"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.catalog), read_only=True)
    query = """
        WITH manhattan_stations AS (
          SELECT DISTINCT gbfs_station_id_current AS gbfs_station_id
          FROM model_features WHERE gbfs_station_id_current IS NOT NULL
        ), demand_profile AS (
          SELECT gbfs_station_id_current AS gbfs_station_id,
            iso_day_of_week, local_hour,
            avg(ride_out)::DOUBLE AS profile_out,
            avg(ride_in)::DOUBLE AS profile_in
          FROM model_features
          WHERE hour_local < TIMESTAMP '2026-10-01'
            AND gbfs_station_id_current IS NOT NULL
            AND NOT coalesce(dst_ambiguous_collapsed, false)
          GROUP BY 1,2,3
        ), eligible AS (
          SELECT * FROM inventory_hourly
          WHERE observed_minutes >= 45 AND operating_observations >= 45
            AND capacity_observed > 0
        )
        SELECT origin.gbfs_station_id,
          origin.hour_start_utc AS origin_hour_utc,
          target.hour_start_utc AS target_hour_utc,
          target.hour_local AS target_hour_local,
          extract(hour FROM target.hour_local)::INTEGER AS target_local_hour,
          extract(isodow FROM target.hour_local)::INTEGER AS target_iso_dow,
          origin.observed_minutes AS origin_observed_minutes,
          target.observed_minutes AS target_observed_minutes,
          origin.mean_bikes_available, origin.mean_docks_available,
          origin.min_bikes_available, origin.min_docks_available,
          origin.capacity_observed,
          origin.empty_any AS origin_empty_any, origin.full_any AS origin_full_any,
          coalesce(dp.profile_out, 0) AS profile_out,
          coalesce(dp.profile_in, 0) AS profile_in,
          target.empty_any AS target_empty_any,
          target.full_any AS target_full_any
        FROM eligible origin
        JOIN manhattan_stations ms ON ms.gbfs_station_id=origin.gbfs_station_id
        JOIN eligible target
          ON target.gbfs_station_id=origin.gbfs_station_id
          AND target.hour_start_utc=origin.hour_start_utc + INTERVAL 1 HOUR
        LEFT JOIN demand_profile dp
          ON dp.gbfs_station_id=origin.gbfs_station_id
          AND dp.iso_day_of_week=extract(isodow FROM target.hour_local)
          AND dp.local_hour=extract(hour FROM target.hour_local)
        WHERE origin.hour_start_utc >= TIMESTAMPTZ '2026-04-07 00:00:00+00'
          AND target.hour_start_utc < TIMESTAMPTZ '2026-11-01 00:00:00+00'
    """
    frame = con.execute(query).fetchdf()
    con.close()
    cutoff = pd.Timestamp("2026-10-01", tz="UTC")
    train = frame[frame["target_hour_utc"] < cutoff].copy()
    test = frame[frame["target_hour_utc"] >= cutoff].copy()
    print(f"risk rows train={len(train):,} test={len(test):,}", flush=True)

    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    metrics = []
    predictions = test[["gbfs_station_id", "origin_hour_utc", "target_hour_utc"]].copy()
    for risk, label in (("empty", "target_empty_any"), ("full", "target_full_any")):
        actual = test[label].astype(int).to_numpy()
        current_warning = (
            test["mean_bikes_available"].to_numpy() <= 2
            if risk == "empty" else test["mean_docks_available"].to_numpy() <= 2
        ).astype(int)
        net = test["profile_out"].to_numpy() - test["profile_in"].to_numpy()
        projected_bikes = test["mean_bikes_available"].to_numpy() - net
        projected_warning = (
            projected_bikes <= 0 if risk == "empty"
            else projected_bikes >= test["capacity_observed"].to_numpy()
        ).astype(int)
        for model, warning in (("current_inventory", current_warning), ("inventory_plus_profile_net", projected_warning)):
            metrics.append({"risk": risk, "model": model, **classification_metrics(actual, warning)})

        model = Pipeline([
            ("features", transformer),
            ("classifier", LogisticRegression(max_iter=300, class_weight="balanced", random_state=20261118)),
        ])
        model.fit(train[NUMERIC + CATEGORICAL], train[label].astype(int))
        probability = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
        warning = (probability >= 0.5).astype(int)
        metrics.append({"risk": risk, "model": "logistic_state_plus_profile", **classification_metrics(actual, warning, probability)})
        joblib.dump(model, args.output / f"{risk}_risk_logistic.joblib")
        predictions[f"actual_{risk}"] = actual
        predictions[f"current_inventory_{risk}"] = current_warning
        predictions[f"inventory_plus_profile_net_{risk}"] = projected_warning
        predictions[f"logistic_probability_{risk}"] = probability
        predictions[f"logistic_warning_{risk}"] = warning

    pd.DataFrame(metrics).to_csv(args.output / "risk_metrics.csv", index=False)
    predictions.to_parquet(args.output / "risk_predictions.parquet", index=False, compression="zstd")
    report = {
        "forecast_horizon": "one hour ahead",
        "origin_definition": "All features come from the completed preceding inventory hour or a pre-October demand profile.",
        "quality_filter": "Both origin and target have >=45 observed minutes, >=45 operating observations, and positive capacity.",
        "training_end_exclusive_utc": str(cutoff),
        "train_rows": len(train), "test_rows": len(test),
        "limitations": [
            "Hourly mean inventory is less timely than a true last snapshot; production should substitute the latest live GBFS state.",
            "Profile net flow ignores rebalancing, suppressed demand, failed rentals/returns, and trips in progress.",
            "The target is any empty/full sample in the next hour, not duration or customer harm.",
        ],
    }
    (args.output / "risk_run_metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
