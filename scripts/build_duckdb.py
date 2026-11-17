from __future__ import annotations

from pathlib import Path

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def main() -> None:
    out = Path("data/processed/citibike.duckdb")
    out.parent.mkdir(parents=True, exist_ok=True)
    info = sorted(Path("data/raw/gbfs").glob("station_information_*.parquet"))[-1]
    status = sorted(Path("data/raw/gbfs").glob("station_status_*.parquet"))[-1]
    tables = {
        "dim_station_current": (info, "parquet"),
        "gbfs_status_local_sample": (status, "parquet"),
        "station_hour_panel_sample": (Path("data/processed/manhattan_station_hour_panel_sample.csv"), "csv"),
        "subway_walk_sample": (Path("data/processed/citibike_subway_walk_sample.csv"), "csv"),
        "mta_hourly_average_sample": (Path("data/processed/mta_2024_hourly_average_by_complex.csv"), "csv"),
    }
    con = duckdb.connect(str(out))
    for name, (path, kind) in tables.items():
        if kind == "parquet":
            source = f"read_parquet('{sql_path(path)}')"
        else:
            source = f"read_csv_auto('{sql_path(path)}', header=true, sample_size=-1)"
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {source}")
    con.execute("""
        CREATE OR REPLACE VIEW station_hour_panel_with_balance AS
        SELECT *, ride_in - ride_out AS checked_net_flow
        FROM station_hour_panel_sample
    """)
    checks = con.execute("""
        SELECT
          (SELECT count(*) FROM dim_station_current) AS stations,
          (SELECT count(*) FROM gbfs_status_local_sample) AS gbfs_rows,
          (SELECT count(*) FROM station_hour_panel_sample) AS panel_rows,
          (SELECT count(*) FROM subway_walk_sample) AS walk_rows,
          (SELECT count(*) FROM mta_hourly_average_sample) AS mta_rows,
          (SELECT count(*) FROM station_hour_panel_sample WHERE net_flow != ride_in - ride_out) AS bad_net_flow
    """).fetchdf()
    print(checks.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()

