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
    FEATURE_NAMES, Standardizer, build_numeric_features, seed_everything,
    train_gru, train_mlp,
)
from run_stage3_experiments import (
    REG_HOLIDAY, complete_day_sequences, create_profiles, encode_station,
    load_frame, regression_matrix, sql_path,
)


SEED = 20261118


def fit_final_regression(train: pd.DataFrame, output: Path) -> None:
    sample = train.sample(min(len(train), 600_000), random_state=SEED)
    x = regression_matrix(sample, include_holiday=True)
    rows = []
    for target, profile in (("ride_out", "profile_out"), ("ride_in", "profile_in")):
        residual = np.log1p(sample[target].to_numpy(dtype=float)) - np.log1p(sample[profile].to_numpy(dtype=float))
        fit = sm.OLS(residual, x).fit(
            cov_type="cluster", cov_kwds={"groups": sample["local_date"].astype(str).to_numpy()}
        )
        for term in x.columns:
            rows.append({
                "target": target, "term": term, "coefficient": fit.params[term],
                "cluster_robust_se": fit.bse[term], "p_value": fit.pvalues[term],
                "nobs": int(fit.nobs), "r_squared_residual": fit.rsquared,
            })
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final legal long-ahead demand models through 2026-10-31")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage3/models/final_2026"))
    parser.add_argument("--sample-mod", type=int, default=8)
    parser.add_argument("--ann-epochs", type=int, default=4)
    parser.add_argument("--gru-epochs", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    cutoff = pd.Timestamp("2026-11-01")
    train_start = pd.Timestamp("2023-01-01")
    con = duckdb.connect(str(args.catalog), read_only=False)
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    create_profiles(con, train_start, cutoff)
    for table in ("profile_month", "profile_week", "profile_global"):
        con.execute(f"COPY {table} TO '{sql_path(args.output / (table + '.parquet'))}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    all_stations = [row[0] for row in con.execute("SELECT DISTINCT station_id FROM profile_week ORDER BY station_id").fetchall()]
    mapping = {station: index + 1 for index, station in enumerate(all_stations)}

    train = load_frame(con, train_start, cutoff, sample_mod=args.sample_mod)
    print(f"final training rows={len(train):,}", flush=True)
    fit_final_regression(train, args.output / "panel_weather_holiday_coefficients.csv")
    station = encode_station(train, mapping)
    x_all = build_numeric_features(train)
    standardizer = Standardizer.fit(x_all)
    x = standardizer.transform(x_all)
    y = train[["ride_out", "ride_in"]].to_numpy(dtype=np.float32)
    _, ann_history = train_mlp(
        x, station, y, len(mapping) + 1, args.output / "ann_full.pt",
        epochs=args.ann_epochs, seed=SEED,
    )
    del x, x_all, y, train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_day = load_frame(con, train_start, cutoff, sample_mod=args.sample_mod, sample_by_day=True)
    station_day = encode_station(train_day, mapping)
    x_day = standardizer.transform(build_numeric_features(train_day))
    x_sequence, station_sequence, y_sequence, _ = complete_day_sequences(train_day, x_day, station_day)
    _, gru_history = train_gru(
        x_sequence, station_sequence, y_sequence, len(mapping) + 1,
        args.output / "gru_exogenous.pt", epochs=args.gru_epochs, seed=SEED,
    )
    metadata = {
        "model_version": "stage3_final_2026_v1",
        "training_start": str(train_start),
        "training_end_exclusive": str(cutoff),
        "legal_information_set": "No trip observations after 2026-10-31 and no same-target-hour inventory.",
        "sample_modulus": args.sample_mod, "seed": SEED,
        "feature_names": FEATURE_NAMES, "standardizer": standardizer.to_dict(),
        "station_mapping": mapping, "ann_epochs": args.ann_epochs,
        "gru_epochs": args.gru_epochs, "ann_history": ann_history,
        "gru_history": gru_history, "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    con.close()


if __name__ == "__main__":
    main()
