from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
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
def fetch(params: dict) -> dict:
    response = requests.get(
        URL, params=params, timeout=180,
        headers={"User-Agent": "citibike-demand-course-project/0.2 (academic research)"},
    )
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly NYC historical weather from Open-Meteo")
    parser.add_argument("--start", default="2023-01-01")
    # UTC needs one trailing day to cover the final New York local evening.
    parser.add_argument("--end", default="2026-11-01")
    parser.add_argument("--latitude", type=float, default=40.7580)
    parser.add_argument("--longitude", type=float, default=-73.9855)
    parser.add_argument("--output", type=Path, default=Path("data/raw/weather/open_meteo_hourly_2023_2026.parquet"))
    args = parser.parse_args()

    params = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "start_date": args.start,
        "end_date": args.end,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
        "timeformat": "unixtime",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    payload = fetch(params)
    frame = pd.DataFrame(payload["hourly"])
    frame["hour_start_utc"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    frame["hour_local_aware"] = frame["hour_start_utc"].dt.tz_convert("America/New_York")
    frame["hour_local"] = frame["hour_local_aware"].dt.tz_localize(None)
    frame["utc_offset_minutes"] = frame["hour_local_aware"].map(
        lambda value: int(value.utcoffset().total_seconds() // 60)
    )
    frame["weather_source"] = "open_meteo_historical_archive"
    frame["weather_information_timing"] = "historical_observation_or_reanalysis"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="hour_local_aware").to_parquet(args.output, index=False, compression="zstd")
    meta = {
        "source_url": URL,
        "request_parameters": params,
        "generationtime_ms": payload.get("generationtime_ms"),
        "timezone": payload.get("timezone"),
        "hourly_units": payload.get("hourly_units"),
        "rows": len(frame),
        "first_hour_utc": str(frame["hour_start_utc"].min()),
        "last_hour_utc": str(frame["hour_start_utc"].max()),
        "missing_values": frame.isna().sum().to_dict(),
        "forecast_leakage_warning": (
            "Historical weather is valid for training/explanatory analysis. "
            "A future case must use a forecast vintage available at its issue time."
        ),
    }
    args.output.with_suffix(".source.json").write_text(
        json.dumps(meta, indent=2, default=int) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, default=int))


if __name__ == "__main__":
    main()

