from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from citibike_pipeline.gbfs import (
    append_deduplicated_parquet,
    discover_feed_urls,
    fetch_json,
    save_json,
    station_information_frame,
    station_status_frame,
    validate_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one Citi Bike GBFS station snapshot")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/gbfs"))
    parser.add_argument("--strict", action="store_true", help="return nonzero after recording a failed run")
    args = parser.parse_args()

    run_started = datetime.now(timezone.utc)
    day = run_started.strftime("%Y-%m-%d")
    status_path = args.output_dir / f"station_status_{day}.parquet"
    log_path = args.output_dir / f"collection_log_{day}.parquet"
    info_path = args.output_dir / f"station_information_{day}.parquet"
    report_path = args.output_dir / f"quality_report_{day}.json"
    log_row: dict[str, object] = {
        "run_started_at_utc": pd.Timestamp(run_started),
        "run_finished_at_utc": pd.NaT,
        "result": "failed",
        "error_type": None,
        "error_message": None,
        "rows_received": 0,
        "rows_after_dedup": 0,
    }
    failed = False

    try:
        urls, discovery = discover_feed_urls()
        info_result = fetch_json(urls["station_information"])
        status_result = fetch_json(urls["station_status"])
        info = station_information_frame(info_result)
        status = station_status_frame(status_result)
        quality = validate_snapshot(info, status)
        severe = [
            quality["duplicate_status_ids"], quality["status_ids_missing_from_info"],
            quality["missing_required_status_values"], quality["negative_inventory_rows"],
            quality["feed_time_error_rows"],
        ]
        if any(severe):
            raise ValueError(f"Severe GBFS quality failure: {quality}")

        info.to_parquet(info_path, index=False, compression="zstd")
        _, after = append_deduplicated_parquet(
            status, status_path, ["feed_last_updated_utc", "station_id"]
        )
        quality.update({
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "discovery_url": discovery.source_url,
            "station_information_url": info_result.source_url,
            "station_status_url": status_result.source_url,
            "status_rows_after_dedup": after,
        })
        save_json(report_path, quality)
        log_row.update({"result": "success", "rows_received": len(status), "rows_after_dedup": after})
        print(json.dumps(quality, indent=2))
    except Exception as exc:  # record operational failure before optionally failing CI
        failed = True
        log_row["error_type"] = type(exc).__name__
        log_row["error_message"] = str(exc)[:2000]
        print(f"GBFS collection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        log_row["run_finished_at_utc"] = pd.Timestamp(datetime.now(timezone.utc))
        append_deduplicated_parquet(
            pd.DataFrame([log_row]), log_path, ["run_started_at_utc"]
        )

    return 1 if failed and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())

