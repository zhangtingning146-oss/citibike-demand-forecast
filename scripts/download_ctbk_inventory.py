from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import requests

BASE = "https://data.ctbk.dev/gbfs/avail/raw/day/{day}.parquet"


def days_between(start: date, end: date):
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def download_day(day: date, output_dir: Path) -> dict:
    day_text = day.isoformat()
    url = BASE.format(day=day_text)
    path = output_dir / f"{day_text}.parquet"
    for attempt in range(1, 5):
        try:
            probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
            if probe.status_code == 404:
                return {"day": day_text, "status": "not_available"}
            probe.raise_for_status()
            total = int(probe.headers.get("Content-Range", "/0").rsplit("/", 1)[-1])
            if path.exists() and path.stat().st_size == total:
                metadata = pq.ParquetFile(path).metadata
                return {"day": day_text, "status": "existing_valid", "bytes": total, "rows": metadata.num_rows}
            temp = path.with_suffix(".parquet.part")
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temp.open("wb") as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temp.stat().st_size != total:
                raise IOError(f"size {temp.stat().st_size} != {total}")
            metadata = pq.ParquetFile(temp).metadata
            temp.replace(path)
            return {"day": day_text, "status": "downloaded_valid", "bytes": total, "rows": metadata.num_rows}
        except Exception as exc:
            if attempt == 4:
                return {"day": day_text, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(min(20, 2**attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public ctbk.dev daily GBFS Parquet")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 7))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 16))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/raw/ctbk/daily"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_day, day, args.output): day for day in days_between(args.start, args.end)}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"{record['day']}: {record['status']}", flush=True)
    records.sort(key=lambda item: item["day"])
    manifest = args.output.parent / "ctbk_download_manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = {
        "requested_days": len(records),
        "available_days": sum(item["status"] in {"existing_valid", "downloaded_valid"} for item in records),
        "missing_days": [item["day"] for item in records if item["status"] == "not_available"],
        "failed_days": [item for item in records if item["status"] == "failed"],
        "bytes": sum(item.get("bytes", 0) for item in records),
        "rows": sum(item.get("rows", 0) for item in records),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

