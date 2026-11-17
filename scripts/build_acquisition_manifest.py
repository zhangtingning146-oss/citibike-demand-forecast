from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    "data/raw/gbfs/*",
    "data/raw/trips/*.zip",
    "data/raw/mta/*",
    "data/raw/ctbk/*.parquet",
    "data/raw/osm/*",
    "data/processed/*",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = sorted({path for pattern in PATTERNS for path in ROOT.glob(pattern) if path.is_file()})
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": stat.st_size,
            "sha256": sha256(path),
            "file_modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    out = ROOT / "data/processed/acquisition_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} manifest rows to {out}")


if __name__ == "__main__":
    main()

