# Citi Bike station supply-demand project

This repository contains the reproducible data and model pipeline for Manhattan Citi Bike hourly ride-in, ride-out, net-outflow, and empty/full risk research. The final case date is Thanksgiving, 2026-11-26. Stage 3 adds strict 2024/2025 backtests, panel regression, PyTorch ANN/GRU models, inventory-risk evaluation, and a guarded forecast-freezing workflow.

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

The intended GitHub Actions scheduler calls the reusable collector every 15 minutes. The collector downloads the current UTC day's assets from a GitHub Release tagged `gbfs-data`, retries the official feeds, validates the snapshot, deduplicates by source feed timestamp and station UUID, and uploads daily Parquet plus a run log back to the release. Release assets avoid committing high-frequency data into Git history. Native GitHub `schedule` events have not fired in this repository; only manual dispatch and persistence are verified. `docs/GBFS_EXTERNAL_SCHEDULER.md` documents the minimum-permission fallback and the required two-run validation. Missing intervals cannot be reconstructed from later live feeds.

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

## Reproduce Stage 3 models

Install the GPU/model extras without changing the lightweight GBFS workflow environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe scripts\run_stage3_experiments.py --year 2024
.\.venv\Scripts\python.exe scripts\run_stage3_experiments.py --year 2025
.\.venv\Scripts\python.exe scripts\summarize_stage3_results.py
.\.venv\Scripts\python.exe scripts\train_final_demand_models.py
.\.venv\Scripts\python.exe scripts\train_inventory_risk.py
```

The fixed-date ANN and exogenous GRU deliberately use no recent actual demand lags, because official November trips are unavailable when the Thanksgiving forecast must be made. Archived 48-hour-ahead weather forecasts are used for backtests. A forecast can be marked `final` only with a weather file retrieved on New York local November 24 or 25; earlier runs remain `preliminary`. See `docs/STAGE3_EXECUTION_REPORT.md` and `docs/STAGE3_DATA_DICTIONARY.md`.

## Stage 4 operational checks

Stage 4 keeps the Stage 3 models and evaluation boundaries unchanged. It adds demand-bias and net-error diagnostics, a pressure ranking selected on 2024 and evaluated on 2025, inventory-risk thresholds selected on September 2026 and evaluated once on October, and immutable forecast bundles with raw weather copies and SHA-256 checksums.

```powershell
.\.venv\Scripts\python.exe scripts\analyze_stage4_pressure.py
.\.venv\Scripts\python.exe scripts\tune_stage4_risk.py
.\.venv\Scripts\python.exe scripts\generate_thanksgiving_forecast.py --weather-file <file> --dry-run
```

Native GitHub scheduling is now verified but severely sparse rather than 15-minute. See `docs/STAGE4_EXECUTION_REPORT.md` and `docs/GBFS_EXTERNAL_SCHEDULER.md` for the audited coverage and external-scheduler handoff.

