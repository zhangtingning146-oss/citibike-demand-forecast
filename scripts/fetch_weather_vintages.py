from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LIVE_URL = "https://api.open-meteo.com/v1/forecast"
BASE_VARIABLES = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "weather_code", "cloud_cover",
    "wind_speed_10m", "wind_gusts_10m",
]


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(multiplier=2, max=30),
    reraise=True,
)
def fetch(url: str, params: dict) -> dict:
    response = requests.get(
        url, params=params, timeout=180,
        headers={"User-Agent": "citibike-demand-course-project/0.3 (academic research)"},
    )
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError(payload)
    return payload


def normalize(payload: dict, suffix: str | None, source: str, retrieved_at: str) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({"hour_utc": pd.to_datetime(hourly["time"], utc=True)})
    for variable in BASE_VARIABLES:
        key = f"{variable}_{suffix}" if suffix else variable
        values = hourly.get(key)
        frame[variable] = values if values is not None else pd.NA
    frame["hour_local_aware"] = frame["hour_utc"].dt.tz_convert("America/New_York")
    frame["hour_local"] = frame["hour_local_aware"].dt.tz_localize(None)
    frame["utc_offset_minutes"] = frame["hour_local_aware"].map(
        lambda value: int(value.utcoffset().total_seconds() // 60)
    )
    frame["weather_vintage_source"] = source
    frame["retrieved_at_utc"] = retrieved_at
    return frame.drop(columns="hour_local_aware")


def save(payload: dict, frame: pd.DataFrame, target: Path, request_url: str, params: dict, timing: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    meta = {
        "source_url": request_url,
        "request_parameters": params,
        "retrieved_at_utc": str(frame["retrieved_at_utc"].iloc[0]),
        "rows": len(frame),
        "first_hour_utc": str(frame["hour_utc"].min()),
        "last_hour_utc": str(frame["hour_utc"].max()),
        "weather_information_timing": timing,
        "hourly_units": payload.get("hourly_units"),
        "missing_values": frame.isna().sum().to_dict(),
    }
    target.with_suffix(".source.json").write_text(
        json.dumps(meta, indent=2, default=int) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch archived two-day forecasts and the current Thanksgiving forecast")
    parser.add_argument("--output", type=Path, default=Path("data/raw/weather_vintages"))
    parser.add_argument("--latitude", type=float, default=40.7580)
    parser.add_argument("--longitude", type=float, default=-73.9855)
    parser.add_argument("--live-target", default="2026-11-26")
    args = parser.parse_args()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for year in (2024, 2025):
        suffix = "previous_day2"
        params = {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "start_date": f"{year}-11-01",
            "end_date": f"{year}-12-01",
            "hourly": ",".join(f"{name}_{suffix}" for name in BASE_VARIABLES),
            "timezone": "GMT",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        payload = fetch(PREVIOUS_RUNS_URL, params)
        frame = normalize(payload, suffix, "open_meteo_previous_runs_day2", retrieved_at)
        save(
            payload, frame, args.output / f"previous_runs_day2_{year}11.parquet",
            PREVIOUS_RUNS_URL, params,
            "Each value is the forecast archived 48 hours before its valid time.",
        )
        print(f"{year}: {len(frame)} archived forecast hours")

    target = pd.Timestamp(args.live_target)
    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "start_date": target.strftime("%Y-%m-%d"),
        "end_date": (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly": ",".join(BASE_VARIABLES),
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    try:
        payload = fetch(LIVE_URL, params)
        frame = normalize(payload, None, "open_meteo_live_forecast", retrieved_at)
        stamp = datetime.fromisoformat(retrieved_at).strftime("%Y%m%dT%H%M%SZ")
        target_path = args.output / f"live_{target.strftime('%Y%m%d')}_retrieved_{stamp}.parquet"
        save(
            payload, frame, target_path, LIVE_URL, params,
            "Live forecast retrieved at the recorded UTC timestamp; preliminary until explicitly frozen.",
        )
        print(f"live: {len(frame)} forecast hours -> {target_path}")
    except (requests.RequestException, ValueError) as exc:
        print(f"live forecast unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
