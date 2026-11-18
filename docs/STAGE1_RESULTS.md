# Stage 1 execution results

Run date: 2026-11-18 (Asia/Shanghai). GBFS timestamps are stored in UTC.

## What ran successfully

- Existing environment: Python 3.13.5, VS Code 1.144.0, and Git 2.47.1 were found. DuckDB was absent and DuckDB Python 2.0.0 was installed inside `.venv`.
- Official GBFS: station information and station status each returned 2,519 unique IDs with exact set equality. Required inventory fields had no missing values or negative counts. There were 73 non-operating rows and 67 station-level time errors, mainly old sentinel `last_reported` values. These rows remain flagged instead of being imputed.
- Local GBFS collector: one real snapshot was persisted. Retry, schema checks, severe timestamp checks, daily Parquet, collection log, UUID/source-generation deduplication, and coverage audit code are present. Unit tests pass.
- Cloud GBFS collector: the public repository is `https://github.com/zhangtingning146-oss/citibike-demand-forecast`. GitHub Actions is active on a 15-minute schedule at minutes 07, 22, 37, and 52. Cloud run `44238829943` completed successfully on 2026-11-18 at 04:39 UTC. Release tag `gbfs-data` contained four readable assets. The status file contained 2,519 unique stations and zero duplicate `(feed_last_updated_utc, station_id)` keys; the cloud collection log recorded `result=success`.
- ctbk.dev: anonymous direct daily Parquet downloads work. Probed coverage is 2026-04-07 through 2026-11-16, 218 available days, with six missing dates inside the span: 2026-05-01, 2026-05-12, 2026-06-08, 2026-06-09, 2026-09-07, and 2026-09-11. The 2026-11-16 sample has 3,639,955 rows; all 2,519 station UUIDs match current GBFS.
- Official trip data: `202610-citibike-tripdata.zip` and `202411-citibike-tripdata.zip` downloaded and passed ZIP CRC checks. They contain 5,088,365 and 3,710,134 rows, respectively. Both use the documented 13-column schema. Missing station IDs exist and are reported, not filled.
- Station-hour panel: 2,880 zero-filled rows for 20 Manhattan stations across 144 hours around the 2024 Thanksgiving week. `net_flow = ride_in - ride_out` is checked.
- MTA/OSM feasibility: 2,120 MTA entrance/exit rows were obtained. Current GBFS contains 690 Manhattan stations; 840 Manhattan subway entrances allow entry. The Manhattan walking graph was downloaded with OSMnx. Twelve Citi Bike samples each have the nearest three entrances by routed walking distance (36 rows).
- MTA ridership match: all 12 distinct subway complexes in the walking sample matched `station_complex_id` in the official hourly dataset. The output has 2,016 complex × weekday × hour historical-average rows for 2024.

## Important limitations

- GitHub Actions cron is best-effort and can be delayed. The collector stores real acquisition time and provides a missing-slot audit; it cannot recreate an unobserved live snapshot.
- The 2026-10 trip archive includes a small number of ride end timestamps on 2026-11-01. The 2024-11 archive includes a small number of timestamps before November. Month labels are source-file labels, not strict timestamp filters.
- ctbk.dev is a public third-party archive, not an official Citi Bike historical inventory product, and its six observed missing days must stay explicit.
- The direct MTA Socrata CSV download was intermittently denied from this network. Entrances were retrieved from NYC Department of City Planning's ArcGIS mirror of the MTA 2024 dataset; the official MTA catalog remains the authoritative provenance.
- Weather and holiday feature tables are not yet built. Future realized weather or ridership must never enter the 2026-11-26 forecast feature set.

