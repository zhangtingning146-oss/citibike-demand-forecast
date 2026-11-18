from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract official nested monthly Citi Bike ZIPs")
    parser.add_argument("annual_zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw/trips"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.annual_zip) as annual:
        members = [item for item in annual.infolist() if item.filename.lower().endswith(".zip") and "__MACOSX" not in item.filename]
        for index, member in enumerate(members, start=1):
            name = Path(member.filename).name
            target = args.output / name
            if target.exists() and target.stat().st_size == member.file_size:
                with ZipFile(target) as monthly:
                    if monthly.testzip() is None:
                        print(f"{name}: existing valid", flush=True)
                        continue
            temp = target.with_suffix(".zip.part")
            with annual.open(member) as source, temp.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            if temp.stat().st_size != member.file_size:
                raise IOError(f"{name}: extracted size mismatch")
            with ZipFile(temp) as monthly:
                bad = monthly.testzip()
                if bad:
                    raise BadZipFile(f"{name}: CRC failure in {bad}")
            temp.replace(target)
            print(f"{name}: extracted {index}/{len(members)}", flush=True)


if __name__ == "__main__":
    main()
