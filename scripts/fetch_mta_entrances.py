from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

SERVICE = (
    "https://services6.arcgis.com/yG5s3afENB5iO9fj/arcgis/rest/services/"
    "MTA_Subway_Entrances_and_Exits__2024_20250108/FeatureServer/0/query"
)
OUT = Path("data/raw/mta/MTA_Subway_Entrances_and_Exits_2024.csv")
SOURCE_NOTE = Path("data/raw/mta/MTA_Subway_Entrances_and_Exits_2024.source.json")


def main() -> None:
    records: list[dict] = []
    offset = 0
    while True:
        response = requests.get(
            SERVICE,
            params={
                "where": "1=1", "outFields": "*", "outSR": 4326,
                "f": "json", "resultOffset": offset, "resultRecordCount": 2000,
                "orderByFields": "ObjectId",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        features = payload.get("features", [])
        for feature in features:
            row = dict(feature["attributes"])
            geometry = feature.get("geometry") or {}
            row.setdefault("entrance_longitude", geometry.get("x"))
            row.setdefault("entrance_latitude", geometry.get("y"))
            records.append(row)
        if len(features) < 2000:
            break
        offset += len(features)

    frame = pd.DataFrame(records).drop_duplicates(subset=["ObjectId"]).sort_values("ObjectId")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    SOURCE_NOTE.write_text(json.dumps({
        "official_catalog": "https://data.ny.gov/d/i9wp-a4ja",
        "dataset": "MTA Subway Entrances and Exits: 2024",
        "retrieval_mirror": SERVICE,
        "mirror_owner": "NYC Department of City Planning ArcGIS",
        "rows": len(frame),
        "columns": list(frame.columns),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(frame):,} rows to {OUT}")


if __name__ == "__main__":
    main()

