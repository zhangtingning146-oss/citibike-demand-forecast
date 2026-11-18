# Citi Bike station supply-demand project

This repository contains the first-stage data pipeline for Manhattan Citi Bike hourly ride-in, ride-out, net-flow, and empty/full risk research. The final case date is Thanksgiving, 2026-11-26. This stage obtains and validates data only; it does not train panel regression, ANN, or GRU models.

## Reproduce the environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Python 3.13.5, Git 2.47.1, VS Code 1.144.0, and DuckDB Python 2.0.0 were verified locally on 2026-11-18. The standalone DuckDB CLI was not present; the project uses the isolated Python package.

## Reliable GBFS collection

Run one local collection:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\collect_gbfs.py --strict
```

The GitHub Actions workflow runs every 15 minutes at minutes 07, 22, 37, and 52. It downloads the current UTC day's assets from a GitHub Release tagged `gbfs-data`, retries the official feeds, validates the snapshot, deduplicates by source feed timestamp and station UUID, and uploads daily Parquet plus a run log back to the release. Release assets avoid committing high-frequency data into Git history. GitHub schedules may start late; use `scripts/audit_gbfs_coverage.py` to distinguish observed and missing 15-minute slots. Missing intervals cannot be reconstructed from later live feeds.

Production repository: https://github.com/zhangtingning146-oss/citibike-demand-forecast. Two cloud collections completed successfully on 2026-11-18 at 04:39 and 04:55 UTC. The second run downloaded the first run's assets, appended a new source generation, deduplicated, uploaded, and was read back. The resulting daily status asset had 5,038 rows, two feed times, and zero duplicate station/feed-time keys.

## Reproduce the acquired datasets

The main scripts are:

- `scripts/check_ctbk_history.py`: probe the public ctbk.dev daily archive, download one sample day, and test UUID compatibility.
- `scripts/fetch_mta_entrances.py`: page through the 2024 MTA entrance mirror and record official provenance.
- `scripts/fetch_osm_walk_test.py`: obtain the Manhattan OSM walking graph and route Citi Bike stations to entry-allowed subway entrances.
- `scripts/prepare_trip_panel.py`: CRC-check and scan official monthly ZIPs, report fields and missing values, and build the station-hour sample.
- `scripts/fetch_mta_hourly_average.py`: aggregate 2024 MTA entries for the subway complexes in the walking sample.
- `scripts/build_duckdb.py`: build the local analytical database and run row-count/net-flow checks.

Large files under `data/` are intentionally ignored by Git. Source URLs and the acquisition date are in `config/sources.json`; field definitions and leakage constraints are in `docs/DATA_DICTIONARY.md`.

