from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests

BUCKET = "https://s3.amazonaws.com/tripdata"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
USER_AGENT = "citibike-demand-course-project/0.2 (academic research)"


def iter_months(start: str, end: str):
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def find_month_object(session: requests.Session, ym: str) -> tuple[str, int] | None:
    prefix = f"{ym}-citibike-tripdata"
    response = session.get(
        BUCKET, params={"list-type": "2", "prefix": prefix}, timeout=60
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    choices = []
    for node in root.findall("s:Contents", NS):
        key = node.findtext("s:Key", namespaces=NS)
        size = int(node.findtext("s:Size", namespaces=NS) or 0)
        if key and key.endswith(".zip") and "citibike-tripdata" in key:
            choices.append((key, size))
    exact = [item for item in choices if item[0] == f"{ym}-citibike-tripdata.zip"]
    if exact:
        return exact[0]
    if len(choices) == 1:
        return choices[0]
    return None


def find_annual_object(session: requests.Session, year: str) -> tuple[str, int] | None:
    key = f"{year}-citibike-tripdata.zip"
    response = session.get(
        BUCKET, params={"list-type": "2", "prefix": key}, timeout=60
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    for node in root.findall("s:Contents", NS):
        found_key = node.findtext("s:Key", namespaces=NS)
        if found_key == key:
            return found_key, int(node.findtext("s:Size", namespaces=NS) or 0)
    return None


def valid_zip(path: Path, expected_size: int) -> bool:
    if not path.exists() or path.stat().st_size != expected_size:
        return False
    try:
        with ZipFile(path) as archive:
            return bool(archive.infolist()) and archive.testzip() is None
    except BadZipFile:
        return False


def download(session: requests.Session, url: str, path: Path, expected_size: int) -> None:
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 6):
        existing = part.stat().st_size if part.exists() else 0
        if existing >= expected_size:
            part.unlink()
            existing = 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                content_range = response.headers.get("Content-Range", "")
                range_matches = content_range.startswith(f"bytes {existing}-")
                if existing and response.status_code == 206 and range_matches:
                    mode = "ab"
                else:
                    response.raise_for_status()
                    mode = "wb"
                    existing = 0
                with part.open(mode) as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual_size = part.stat().st_size
            if actual_size != expected_size:
                if actual_size > expected_size:
                    part.unlink()
                raise IOError(f"size {actual_size} != expected {expected_size}")
            part.replace(path)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == 5:
                raise
            print(f"retry {attempt}/5 for {path.name}: {exc}", flush=True)
            time.sleep(min(30, 2**attempt))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Citi Bike monthly trip ZIPs")
    parser.add_argument("--start", default="2023-01")
    parser.add_argument("--end", default="2026-10")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/trips"))
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    records = []
    handled_annual_years: set[str] = set()
    for ym in iter_months(args.start, args.end):
        found = find_month_object(session, ym)
        if not found:
            year = ym[:4]
            annual = find_annual_object(session, year) if year not in handled_annual_years else None
            if annual:
                handled_annual_years.add(year)
                key, size = annual
                path = args.output_dir / Path(key).name
                url = f"{BUCKET}/{key}"
                state = "existing_valid" if valid_zip(path, size) else "available"
                print(f"{year} annual: {size / 1024**2:,.1f} MiB {state}", flush=True)
                record = {"month": year, "key": key, "url": url, "bytes": size, "status": state}
                if not args.inventory_only and state != "existing_valid":
                    download(session, url, path, size)
                    if not valid_zip(path, size):
                        raise RuntimeError(f"ZIP CRC validation failed: {path}")
                    record["status"] = "downloaded_valid"
                if not args.inventory_only and path.exists():
                    record["sha256"] = sha256(path)
                records.append(record)
                continue
            if year in handled_annual_years:
                print(f"{ym}: included in {year} annual archive", flush=True)
                continue
            records.append({"month": ym, "status": "not_published"})
            print(f"{ym}: not published", flush=True)
            continue
        key, size = found
        path = args.output_dir / Path(key).name
        url = f"{BUCKET}/{key}"
        state = "existing_valid" if valid_zip(path, size) else "available"
        print(f"{ym}: {size / 1024**2:,.1f} MiB {state}", flush=True)
        record = {"month": ym, "key": key, "url": url, "bytes": size, "status": state}
        if not args.inventory_only and state != "existing_valid":
            download(session, url, path, size)
            if not valid_zip(path, size):
                raise RuntimeError(f"ZIP CRC validation failed: {path}")
            record["status"] = "downloaded_valid"
        if not args.inventory_only and path.exists():
            record["sha256"] = sha256(path)
        records.append(record)

    manifest = args.output_dir / "trip_download_manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    total = sum(item.get("bytes", 0) for item in records if "bytes" in item)
    print(f"months={len(records)} total_available={total / 1024**3:,.2f} GiB manifest={manifest}")


if __name__ == "__main__":
    main()

