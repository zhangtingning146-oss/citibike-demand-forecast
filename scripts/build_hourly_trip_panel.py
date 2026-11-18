from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import duckdb
import geopandas as gpd
import pandas as pd


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def safe_unlink(path: Path, temp_root: Path) -> None:
    resolved = path.resolve()
    root = temp_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside temp root: {resolved}")
    resolved.unlink(missing_ok=True)


def month_from_zip(path: Path) -> str:
    prefix = path.name[:6]
    if len(prefix) != 6 or not prefix.isdigit():
        raise ValueError(f"Cannot infer YYYYMM from {path.name}")
    return prefix


def add_member(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    source = (
        f"read_csv_auto('{sql_path(csv_path)}', header=true, sample_size=100000, "
        "all_varchar=true, ignore_errors=false, null_padding=true)"
    )
    con.execute(f"""
        INSERT INTO hourly_parts
        SELECT
            trim(start_station_id) AS station_id,
            date_trunc('hour', try_cast(started_at AS TIMESTAMP)) AS hour_local,
            any_value(start_station_name) AS station_name,
            median(try_cast(start_lat AS DOUBLE)) AS lat,
            median(try_cast(start_lng AS DOUBLE)) AS lon,
            count(*)::BIGINT AS ride_out,
            0::BIGINT AS ride_in,
            count(*) FILTER (WHERE lower(rideable_type) = 'classic_bike')::BIGINT AS ride_out_classic,
            count(*) FILTER (WHERE lower(rideable_type) = 'electric_bike')::BIGINT AS ride_out_electric,
            count(*) FILTER (WHERE lower(rideable_type) NOT IN ('classic_bike','electric_bike'))::BIGINT AS ride_out_other,
            0::BIGINT AS ride_in_classic, 0::BIGINT AS ride_in_electric, 0::BIGINT AS ride_in_other,
            count(*) FILTER (WHERE lower(member_casual) = 'member')::BIGINT AS ride_out_member,
            count(*) FILTER (WHERE lower(member_casual) = 'casual')::BIGINT AS ride_out_casual,
            0::BIGINT AS ride_in_member, 0::BIGINT AS ride_in_casual
        FROM {source}
        WHERE nullif(trim(start_station_id), '') IS NOT NULL
          AND try_cast(started_at AS TIMESTAMP) IS NOT NULL
        GROUP BY 1, 2
        UNION ALL
        SELECT
            trim(end_station_id) AS station_id,
            date_trunc('hour', try_cast(ended_at AS TIMESTAMP)) AS hour_local,
            any_value(end_station_name) AS station_name,
            median(try_cast(end_lat AS DOUBLE)) AS lat,
            median(try_cast(end_lng AS DOUBLE)) AS lon,
            0::BIGINT AS ride_out,
            count(*)::BIGINT AS ride_in,
            0::BIGINT AS ride_out_classic, 0::BIGINT AS ride_out_electric, 0::BIGINT AS ride_out_other,
            count(*) FILTER (WHERE lower(rideable_type) = 'classic_bike')::BIGINT AS ride_in_classic,
            count(*) FILTER (WHERE lower(rideable_type) = 'electric_bike')::BIGINT AS ride_in_electric,
            count(*) FILTER (WHERE lower(rideable_type) NOT IN ('classic_bike','electric_bike'))::BIGINT AS ride_in_other,
            0::BIGINT AS ride_out_member, 0::BIGINT AS ride_out_casual,
            count(*) FILTER (WHERE lower(member_casual) = 'member')::BIGINT AS ride_in_member,
            count(*) FILTER (WHERE lower(member_casual) = 'casual')::BIGINT AS ride_in_casual
        FROM {source}
        WHERE nullif(trim(end_station_id), '') IS NOT NULL
          AND try_cast(ended_at AS TIMESTAMP) IS NOT NULL
        GROUP BY 1, 2
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build partitioned Manhattan station-hour trip panel")
    parser.add_argument("--trips-dir", type=Path, default=Path("data/raw/trips"))
    parser.add_argument("--boundary", type=Path, default=Path("data/raw/osm/manhattan_boundary.geojson"))
    parser.add_argument("--gbfs-info", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/panel/hourly_trips"))
    parser.add_argument("--start", default="202301")
    parser.add_argument("--end", default="202610")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    temp_root = Path("work/trip_extract")
    temp_root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    boundary = gpd.read_file(args.boundary).to_crs(4326).geometry.iloc[0]
    if args.gbfs_info is None:
        args.gbfs_info = sorted(Path("data/raw/gbfs").glob("station_information_*.parquet"))[-1]

    con = duckdb.connect("work/hourly_trip_build.duckdb")
    Path("work/duckdb_temp").mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA temp_directory='work/duckdb_temp'")
    con.execute(f"CREATE OR REPLACE TEMP TABLE dim_current AS SELECT * FROM read_parquet('{sql_path(args.gbfs_info)}')")

    zip_paths = sorted(args.trips_dir.glob("*-citibike-tripdata.zip"))
    zip_paths = [
        path for path in zip_paths
        if path.name[:6].isdigit() and args.start <= month_from_zip(path) <= args.end
    ]
    if not zip_paths:
        raise SystemExit("No monthly Citi Bike ZIP files found")

    for zip_path in zip_paths:
        ym = month_from_zip(zip_path)
        year, month = ym[:4], ym[4:]
        out_dir = args.output / f"year={year}" / f"month={month}"
        out_file = out_dir / "part.parquet"
        station_file = out_dir / "station_month.parquet"
        if out_file.exists() and station_file.exists() and not args.force:
            print(f"{ym}: skip existing", flush=True)
            continue
        print(f"{ym}: processing {zip_path.name}", flush=True)
        month_start = datetime(int(year), int(month), 1)
        if int(month) == 12:
            month_end = datetime(int(year) + 1, 1, 1)
        else:
            month_end = datetime(int(year), int(month) + 1, 1)
        con.execute("DROP TABLE IF EXISTS hourly_parts")
        con.execute("""
            CREATE TEMP TABLE hourly_parts (
              station_id VARCHAR, hour_local TIMESTAMP, station_name VARCHAR,
              lat DOUBLE, lon DOUBLE,
              ride_out BIGINT, ride_in BIGINT,
              ride_out_classic BIGINT, ride_out_electric BIGINT, ride_out_other BIGINT,
              ride_in_classic BIGINT, ride_in_electric BIGINT, ride_in_other BIGINT,
              ride_out_member BIGINT, ride_out_casual BIGINT,
              ride_in_member BIGINT, ride_in_casual BIGINT
            )
        """)
        with ZipFile(zip_path) as archive:
            members = [item for item in archive.infolist() if item.filename.lower().endswith(".csv")]
            for index, member in enumerate(members, start=1):
                csv_path = temp_root / f"{ym}_{index}.csv"
                with archive.open(member) as source, csv_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                add_member(con, csv_path)
                safe_unlink(csv_path, temp_root)
                print(f"  member {index}/{len(members)} aggregated", flush=True)

        con.execute("""
            CREATE OR REPLACE TEMP TABLE observed AS
            SELECT
              station_id, hour_local,
              arg_max(station_name, ride_out + ride_in) AS station_name,
              sum(lat * (ride_out + ride_in)) / nullif(sum(ride_out + ride_in) FILTER (WHERE lat IS NOT NULL), 0) AS lat,
              sum(lon * (ride_out + ride_in)) / nullif(sum(ride_out + ride_in) FILTER (WHERE lon IS NOT NULL), 0) AS lon,
              sum(ride_out)::BIGINT AS ride_out, sum(ride_in)::BIGINT AS ride_in,
              sum(ride_out_classic)::BIGINT AS ride_out_classic,
              sum(ride_out_electric)::BIGINT AS ride_out_electric,
              sum(ride_out_other)::BIGINT AS ride_out_other,
              sum(ride_in_classic)::BIGINT AS ride_in_classic,
              sum(ride_in_electric)::BIGINT AS ride_in_electric,
              sum(ride_in_other)::BIGINT AS ride_in_other,
              sum(ride_out_member)::BIGINT AS ride_out_member,
              sum(ride_out_casual)::BIGINT AS ride_out_casual,
              sum(ride_in_member)::BIGINT AS ride_in_member,
              sum(ride_in_casual)::BIGINT AS ride_in_casual
            FROM hourly_parts
            WHERE hour_local >= ? AND hour_local < ?
            GROUP BY station_id, hour_local
        """, [month_start, month_end])
        station_month = con.execute("""
            SELECT station_id,
              arg_max(station_name, ride_out + ride_in) AS station_name,
              sum(lat * (ride_out + ride_in)) / nullif(sum(ride_out + ride_in) FILTER (WHERE lat IS NOT NULL), 0) AS lat,
              sum(lon * (ride_out + ride_in)) / nullif(sum(ride_out + ride_in) FILTER (WHERE lon IS NOT NULL), 0) AS lon,
              min(hour_local) AS active_start_hour,
              max(hour_local) AS active_end_hour,
              sum(ride_out + ride_in)::BIGINT AS station_events
            FROM observed
            GROUP BY station_id
        """).fetchdf()
        points = gpd.GeoDataFrame(
            station_month,
            geometry=gpd.points_from_xy(station_month["lon"], station_month["lat"]),
            crs=4326,
        )
        manhattan = points[points.geometry.within(boundary)].drop(columns="geometry")
        con.register("manhattan_station_month_df", manhattan)
        con.execute("CREATE OR REPLACE TEMP TABLE station_month AS SELECT * FROM manhattan_station_month_df")
        out_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY station_month TO '{sql_path(station_file)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.execute(f"""
            COPY (
              WITH grid AS (
                SELECT sm.*, unnest(generate_series(sm.active_start_hour, sm.active_end_hour, INTERVAL 1 HOUR)) AS hour_local
                FROM station_month sm
              )
              SELECT
                g.station_id,
                g.hour_local,
                cast(g.hour_local AS DATE) AS local_date,
                extract(hour FROM g.hour_local)::UTINYINT AS local_hour,
                extract(isodow FROM g.hour_local)::UTINYINT AS iso_day_of_week,
                g.station_name,
                g.lat, g.lon,
                coalesce(o.ride_out, 0)::BIGINT AS ride_out,
                coalesce(o.ride_in, 0)::BIGINT AS ride_in,
                (coalesce(o.ride_out, 0) - coalesce(o.ride_in, 0))::BIGINT AS net_outflow,
                coalesce(o.ride_out_classic, 0)::BIGINT AS ride_out_classic,
                coalesce(o.ride_out_electric, 0)::BIGINT AS ride_out_electric,
                coalesce(o.ride_out_other, 0)::BIGINT AS ride_out_other,
                coalesce(o.ride_in_classic, 0)::BIGINT AS ride_in_classic,
                coalesce(o.ride_in_electric, 0)::BIGINT AS ride_in_electric,
                coalesce(o.ride_in_other, 0)::BIGINT AS ride_in_other,
                coalesce(o.ride_out_member, 0)::BIGINT AS ride_out_member,
                coalesce(o.ride_out_casual, 0)::BIGINT AS ride_out_casual,
                coalesce(o.ride_in_member, 0)::BIGINT AS ride_in_member,
                coalesce(o.ride_in_casual, 0)::BIGINT AS ride_in_casual,
                (o.station_id IS NULL) AS zero_filled_active_hour,
                g.active_start_hour, g.active_end_hour,
                dc.station_id AS gbfs_station_id_current,
                dc.capacity::INTEGER AS capacity_current,
                (dc.station_id IS NOT NULL) AS matched_current_gbfs
              FROM grid g
              LEFT JOIN observed o USING (station_id, hour_local)
              LEFT JOIN dim_current dc ON trim(dc.short_name) = g.station_id
              ORDER BY g.station_id, g.hour_local
            ) TO '{sql_path(out_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """)
        counts = con.execute("SELECT count(*) rows, count(DISTINCT station_id) stations FROM read_parquet(?)", [str(out_file)]).fetchone()
        print(f"{ym}: rows={counts[0]:,} stations={counts[1]:,}", flush=True)

    con.close()


if __name__ == "__main__":
    main()

