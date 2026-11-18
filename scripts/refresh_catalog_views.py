from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


EXCLUDE_LEAKY_HISTORY = """
active_segment, gbfs_station_id_current, capacity_current,
matched_current_gbfs, first_observed_hour, last_observed_hour,
observed_active_months, total_station_events
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh leakage-scoped model views without rebuilding Parquet")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/citibike_stage2_final.duckdb"))
    args = parser.parse_args()
    con = duckdb.connect(str(args.catalog))
    con.execute(f"CREATE OR REPLACE VIEW econometric_panel AS SELECT * EXCLUDE ({EXCLUDE_LEAKY_HISTORY}) FROM model_features")
    con.execute(f"CREATE OR REPLACE VIEW ann_gru_features AS SELECT * EXCLUDE ({EXCLUDE_LEAKY_HISTORY}) FROM model_features")
    con.execute(f"""
        CREATE OR REPLACE VIEW fixed_date_features AS
        SELECT * EXCLUDE (
          {EXCLUDE_LEAKY_HISTORY},
          feed_generations, inventory_observed_minutes, mean_bikes_available,
          mean_docks_available, min_bikes_available, min_docks_available,
          empty_share, full_share, empty_any, full_any, capacity_observed,
          inventory_utc_hours_collapsed,
          ride_out_lag_1h, ride_in_lag_1h, ride_out_lag_24h, ride_in_lag_24h,
          ride_out_lag_168h, ride_in_lag_168h,
          ride_out_mean_previous_24h, ride_in_mean_previous_24h,
          ride_out_mean_previous_168h, ride_in_mean_previous_168h
        ) FROM model_features
    """)
    for view in ("econometric_panel", "ann_gru_features", "fixed_date_features"):
        columns = con.execute(f"DESCRIBE {view}").fetchdf()
        print(f"{view}: {len(columns)} columns")
    con.close()


if __name__ == "__main__":
    main()
