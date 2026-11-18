# Citi Bike station supply-demand project

This repository contains the reproducible data pipeline for Manhattan Citi Bike hourly ride-in, ride-out, net-outflow, and empty/full risk research. The final case date is Thanksgiving, 2026-11-26. Stage 2 builds the formal dataset but does not train panel regression, ANN, or GRU models.

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

The dedicated GitHub Actions scheduler calls the reusable collector every 15 minutes. The collector downloads the current UTC day's assets from a GitHub Release tagged `gbfs-data`, retries the official feeds, validates the snapshot, deduplicates by source feed timestamp and station UUID, and uploads daily Parquet plus a run log back to the release. Release assets avoid committing high-frequency data into Git history. GitHub schedules may start late; use `scripts/audit_gbfs_coverage.py` to distinguish observed and missing 15-minute slots. Missing intervals cannot be reconstructed from later live feeds.

Production repository: https://github.com/zhangtingning146-oss/citibike-demand-forecast. Durable assets are stored in the `gbfs-data` release rather than Git history. See the current stage report for the distinction between manual and genuine `schedule` events.

## Build the formal Stage 2 dataset

The main execution order is:

```powershell
.\.venv\Scripts\python.exe scripts\download_trip_history.py
.\.venv\Scripts\python.exe scripts\extract_annual_trip_archives.py
.\.venv\Scripts\python.exe scripts\build_hourly_trip_panel.py
.\.venv\Scripts\python.exe scripts\fetch_hourly_weather.py
.\.venv\Scripts\python.exe scripts\download_ctbk_inventory.py
.\.venv\Scripts\python.exe scripts\build_inventory_labels.py
.\.venv\Scripts\python.exe scripts\build_model_features.py
.\.venv\Scripts\python.exe scripts\refresh_catalog_views.py
.\.venv\Scripts\python.exe scripts\audit_stage2.py
.\.venv\Scripts\python.exe scripts\export_model_datasets.py
.\.venv\Scripts\python.exe scripts\build_data_dictionary.py
```

The formal trip panel is partitioned Parquet under `data/processed/panel/hourly_trips`. Model features are under `data/processed/features/model_features_v2`, with SQL views in `data/processed/citibike_stage2_final.duckdb`. Python receives a Zstandard-compressed Parquet export; Stata receives yearly CSV files and `stata/import_stage2_panel.do`. Large data are intentionally ignored by Git.

The views have different information sets:

- `fixed_date_features`: excludes same-hour inventory and rolling-demand fields.
- `econometric_panel`: leakage-scoped formal panel for explanatory/forecast experiments.
- `ann_gru_features`: the same safe station attributes plus rolling fields and labeled outcomes.
- `model_features`: full audit table, including reference-only fields; do not feed every column to a model.

## Reproduce the acquired datasets

The main scripts are:

- `scripts/check_ctbk_history.py`: probe the public ctbk.dev daily archive, download one sample day, and test UUID compatibility.
- `scripts/fetch_mta_entrances.py`: page through the 2024 MTA entrance mirror and record official provenance.
- `scripts/fetch_osm_walk_test.py`: obtain the Manhattan OSM walking graph and route Citi Bike stations to entry-allowed subway entrances.
- `scripts/prepare_trip_panel.py`: CRC-check and scan official monthly ZIPs, report fields and missing values, and build the station-hour sample.
- `scripts/fetch_mta_hourly_average.py`: aggregate 2024 MTA entries for the subway complexes in the walking sample.
- `scripts/build_duckdb.py`: build the local analytical database and run row-count/net-flow checks.

Large files under `data/` are intentionally ignored by Git. Source URLs and the acquisition date are in `config/sources.json`; field definitions and leakage constraints are in `docs/DATA_DICTIONARY.md`.

