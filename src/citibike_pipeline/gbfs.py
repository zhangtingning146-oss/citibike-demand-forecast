from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

DISCOVERY_URL = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
USER_AGENT = "citibike-demand-course-project/0.1 (academic data collection)"


@dataclass(frozen=True)
class FetchResult:
    payload: dict[str, Any]
    requested_at_utc: datetime
    received_at_utc: datetime
    source_url: str


@retry(
    retry=retry_if_exception_type((requests.RequestException, ValueError, KeyError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(multiplier=2, max=30),
    reraise=True,
)
def fetch_json(url: str, timeout_seconds: int = 45) -> FetchResult:
    requested = datetime.now(timezone.utc)
    response = requests.get(
        url,
        timeout=(10, timeout_seconds),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    response.raise_for_status()
    payload = response.json()
    received = datetime.now(timezone.utc)
    if not isinstance(payload, dict) or "data" not in payload or "last_updated" not in payload:
        raise ValueError(f"Unexpected GBFS envelope from {url}")
    return FetchResult(payload, requested, received, url)


def discover_feed_urls(discovery_url: str = DISCOVERY_URL) -> tuple[dict[str, str], FetchResult]:
    result = fetch_json(discovery_url)
    language_map = result.payload["data"]
    feeds = language_map.get("en") or next(iter(language_map.values()))
    urls = {item["name"]: item["url"] for item in feeds["feeds"]}
    for required in ("station_information", "station_status"):
        if required not in urls:
            raise KeyError(f"GBFS discovery is missing {required}")
    return urls, result


def _epoch_to_utc(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="s", utc=True, errors="coerce")


def station_information_frame(result: FetchResult) -> pd.DataFrame:
    rows = result.payload["data"]["stations"]
    frame = pd.DataFrame(rows)
    required = {"station_id", "short_name", "name", "lat", "lon", "capacity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"station_information missing columns: {sorted(missing)}")
    frame["station_id"] = frame["station_id"].astype("string")
    frame["short_name"] = frame["short_name"].astype("string")
    frame["info_feed_last_updated_utc"] = _epoch_to_utc(result.payload["last_updated"])
    frame["info_collected_at_utc"] = pd.Timestamp(result.received_at_utc)
    return frame


def station_status_frame(result: FetchResult) -> pd.DataFrame:
    rows = result.payload["data"]["stations"]
    frame = pd.DataFrame(rows)
    required = {
        "station_id", "num_bikes_available", "num_docks_available",
        "is_installed", "is_renting", "is_returning", "last_reported",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"station_status missing columns: {sorted(missing)}")

    collected = pd.Timestamp(result.received_at_utc)
    feed_updated = _epoch_to_utc(result.payload["last_updated"])
    frame["station_id"] = frame["station_id"].astype("string")
    frame["collected_at_utc"] = collected
    frame["request_started_at_utc"] = pd.Timestamp(result.requested_at_utc)
    frame["feed_last_updated_utc"] = feed_updated
    frame["station_last_reported_utc"] = _epoch_to_utc(frame["last_reported"])
    frame["feed_age_seconds"] = (collected - feed_updated).total_seconds()
    frame["station_report_age_seconds"] = (
        collected - frame["station_last_reported_utc"]
    ).dt.total_seconds()
    frame["feed_time_error"] = (
        (frame["feed_age_seconds"] < -300) | (frame["feed_age_seconds"] > 3600)
    )
    frame["station_time_error"] = (
        (frame["station_report_age_seconds"] < -300)
        | (frame["station_report_age_seconds"] > 86400)
    )
    frame["operating"] = (
        frame["is_installed"].eq(1)
        & frame["is_renting"].eq(1)
        & frame["is_returning"].eq(1)
    )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    frame["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    frame["source_url"] = result.source_url
    keep = [
        "collected_at_utc", "request_started_at_utc", "feed_last_updated_utc",
        "station_id", "num_bikes_available", "num_ebikes_available",
        "num_bikes_disabled", "num_docks_available", "num_docks_disabled",
        "is_installed", "is_renting", "is_returning", "operating",
        "station_last_reported_utc", "feed_age_seconds",
        "station_report_age_seconds", "feed_time_error", "station_time_error",
        "snapshot_sha256", "source_url",
    ]
    for optional in ("num_ebikes_available", "num_bikes_disabled", "num_docks_disabled"):
        if optional not in frame:
            frame[optional] = pd.NA
    return frame[keep]


def validate_snapshot(info: pd.DataFrame, status: pd.DataFrame) -> dict[str, Any]:
    info_ids = set(info["station_id"].dropna())
    status_ids = set(status["station_id"].dropna())
    numeric = ["num_bikes_available", "num_docks_available"]
    return {
        "info_station_count": len(info),
        "status_station_count": len(status),
        "duplicate_info_ids": int(info["station_id"].duplicated().sum()),
        "duplicate_status_ids": int(status["station_id"].duplicated().sum()),
        "status_ids_missing_from_info": len(status_ids - info_ids),
        "info_ids_missing_from_status": len(info_ids - status_ids),
        "missing_required_status_values": int(status[["station_id", *numeric]].isna().sum().sum()),
        "negative_inventory_rows": int(status[numeric].lt(0).any(axis=1).sum()),
        "non_operating_rows": int((~status["operating"]).sum()),
        "feed_time_error_rows": int(status["feed_time_error"].sum()),
        "station_time_error_rows": int(status["station_time_error"].sum()),
    }


def append_deduplicated_parquet(
    frame: pd.DataFrame,
    path: Path,
    subset: list[str],
) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    before = 0
    if path.exists():
        existing = pd.read_parquet(path)
        before = len(existing)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.drop_duplicates(subset=subset, keep="last").sort_values(subset)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)
    return before, len(frame)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)

