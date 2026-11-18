# Data dictionary

All timestamps ending in `_utc` are timezone-aware UTC. Preserve Citi Bike identifiers as text: the live GBFS `station_id` is a UUID, while trip files use the public `short_name` such as `6526.01`. Join trip station IDs to `dim_station_current.short_name`, not to the UUID directly.

## GBFS station status

| Field | Type | Meaning | Quality rule |
|---|---|---|---|
| `collected_at_utc` | timestamp | Time the HTTP response finished locally | The true acquisition time; never replace with schedule time |
| `request_started_at_utc` | timestamp | Time the HTTP request began | Used for request latency checks |
| `feed_last_updated_utc` | timestamp | GBFS envelope `last_updated` | Reject feed more than 5 minutes in the future or 1 hour stale |
| `station_id` | text UUID | Current GBFS station identifier | Unique with feed timestamp and must exist in station information |
| `num_bikes_available` | integer | Rentable bikes of all types | Nonnegative |
| `num_ebikes_available` | integer | Rentable e-bikes | Nonnegative; subset of available bikes |
| `num_bikes_disabled` | integer | Bikes present but unavailable | Nonnegative |
| `num_docks_available` | integer | Docks accepting returns | Nonnegative |
| `num_docks_disabled` | integer | Docks not accepting returns | Nonnegative |
| `is_installed` | 0/1 | Station equipment installed | Retained as published |
| `is_renting` | 0/1 | Rentals allowed | Retained as published |
| `is_returning` | 0/1 | Returns allowed | Retained as published |
| `operating` | boolean | All three status flags equal 1 | Derived, not an official field |
| `station_last_reported_utc` | timestamp | Station-level `last_reported` | Old sentinel values are flagged, not imputed |
| `feed_age_seconds` | number | Collection time minus feed update time | Operational freshness measure |
| `station_report_age_seconds` | number | Collection time minus station report time | Values over one day are flagged |
| `feed_time_error` | boolean | Feed timestamp outside accepted range | Severe collection failure |
| `station_time_error` | boolean | Station timestamp outside accepted range | Row retained with flag |
| `snapshot_sha256` | text | Hash of canonical station rows | Detects identical source snapshots |

Daily files deduplicate on `(feed_last_updated_utc, station_id)`. A retry that sees the same source generation does not create another analytical observation. Collection logs remain one row per workflow attempt.

## Formal trip station-hour panel

The formal panel is `data/processed/panel/hourly_trips/year=YYYY/month=MM/part.parquet`. It covers all Manhattan stations found in each official trip month, not the earlier 20-station feasibility sample.

| Field group | Meaning |
|---|---|
| `station_id`, `hour_local` | Unique panel key. The station ID remains text; the hour is the published New York civil hour. |
| `station_name`, `lat`, `lon` | Station-month attributes derived from trip records; coordinates are trip-weighted within month. |
| `ride_out`, `ride_in` | Hourly starts and ends. |
| `net_outflow` | `ride_out - ride_in`. This sign convention is used throughout Stage 2. |
| `ride_{out,in}_{classic,electric,other}` | Counts by published `rideable_type`. |
| `ride_{out,in}_{member,casual}` | Counts by published user category. |
| `zero_filled_active_hour` | No start or end in an hour between that station-month's first and last observed trip hour. |
| `active_start_hour`, `active_end_hour` | Observed station-month envelope used for conservative zero filling. |
| `gbfs_station_id_current`, `capacity_current`, `matched_current_gbfs` | Current-snapshot reference fields only. They are excluded from model-facing views because they are not historical attributes. |

The active envelope avoids filling months before launch or after retirement. Trip files alone cannot identify a temporary mid-month closure, so those internal hours cannot be certified as operating. Monthly `station_month.parquet` files retain the station version, location, and envelope used to build each grid.

## Weather, calendar, lags, and inventory

The exact 81-field dictionary, including DuckDB types, sources, availability class, model role, and measured missing counts, is generated as `outputs/STAGE2_DATA_DICTIONARY.csv`.

| Field group | Source and rule |
|---|---|
| Temperature, humidity, rain, snow, wind, cloud, weather code | Open-Meteo historical archive. Valid as observed training covariates; a future prediction must replace them with a forecast vintage available at issue time. |
| Holiday and Thanksgiving fields | Deterministic calendar features known in advance. |
| `ride_*_lag_*`, `ride_*_mean_previous_*` | Constructed within consecutive active station segments. They use preceding rows only and are for rolling forecasts, not a long-ahead fixed-date forecast. |
| `empty_share`, `full_share`, `empty_any`, `full_any` | ctbk.dev historical GBFS outcomes. They are risk labels, not contemporaneous predictors. |
| `capacity_observed`, bikes/docks, observation counts | Same-hour inventory observations. They describe coverage/state and are unavailable for a fixed-date future forecast. |
| `time_split` | Chronological experiment assignment; never use as a predictor. |

The repeated fall-back local hour is collapsed and flagged; the nonexistent spring-forward hour is removed from model features. UTC is retained in raw weather and inventory products.

## Subway walking accessibility

| Field | Meaning |
|---|---|
| `gbfs_station_id` | Citi Bike live-feed UUID |
| `citibike_short_name` | Trip-compatible Citi Bike station ID |
| `rank` | Rank among three shortest routed subway entrances |
| `walk_distance_m` | OSM walking-network shortest-path distance in metres |
| `subway_complex_id` | MTA station-complex ID used by hourly ridership |
| `subway_stop_name` | MTA complex/stop label from the entrance dataset |
| `subway_routes` | Published daytime routes |
| `entrance_type` | Stair, elevator, easement, or other MTA entrance type |

Only entrances with `borough=M` and `entry_allowed=YES` are eligible. The MTA hourly feature is the 2024 historical mean for a station complex, day of week, and hour of day. It sums fare-category rows before dividing by distinct hourly timestamps.

## Forecast leakage rule

For the 2026-11-26 case, use only information known by each forecast issue time. Do not use realized weather, realized subway entries, realized trip counts, or later GBFS inventory. Weather features must come from a forecast vintage or from historical observations when training past dates. Holiday indicators are calendar-known and may be created in advance.

