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

## Trip station-hour panel

| Field | Type | Meaning |
|---|---|---|
| `station_id` | text | Citi Bike trip station ID; matches current GBFS `short_name` where available |
| `hour` | local naive timestamp | Hour in the source trip timestamp (America/New_York convention published by Citi Bike) |
| `ride_in` | integer | Trips ending at the station in the hour |
| `ride_out` | integer | Trips starting at the station in the hour |
| `net_flow` | integer | `ride_in - ride_out` |
| `gbfs_station_id` | text UUID | Current live-feed station ID mapped through `short_name` |
| `station_name` | text | Current station name |
| `lat`, `lon` | number | Current GBFS coordinates |

The sample is a complete zero-filled grid for 20 high-volume Manhattan stations from 2024-11-25 00:00 through 2024-11-30 23:00. It contains 2,880 rows.

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

