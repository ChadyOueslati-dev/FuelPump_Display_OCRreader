"""Run every photo in a folder through the pipeline and summarise the results.

Usage:
    python scripts/batch.py "data/Pump Displays_Large"
    python scripts/batch.py "data/Pump Displays_Large" --limit 20
    python scripts/batch.py "data/Pump Displays_Large" --only car_000.jpg pump_003.jpg
    python scripts/batch.py some/folder --csv results.csv

No labels are needed: this shows what the app returns for each photo, not
whether it is right. Use scripts/evaluate.py with a labels CSV for accuracy.
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ocr.engine import detect  # noqa: E402
from app.ocr.preprocess import load_image  # noqa: E402
from app.pipeline import read_display  # noqa: E402
from app.settings import load_validation_config  # noqa: E402

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--limit", type=int, help="only the first N photos")
    parser.add_argument("--only", nargs="+", help="only these file names")
    parser.add_argument("--csv", type=Path, help="also write the results to this CSV file")
    args = parser.parse_args()

    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)
    if args.only:
        photos = [p for p in photos if p.name in set(args.only)]
    if args.limit:
        photos = photos[: args.limit]
    if not photos:
        print(f"error: no photos found in {args.folder}", file=sys.stderr)
        return 2

    config = load_validation_config()
    statuses, reasons, rows = Counter(), Counter(), []
    started = time.perf_counter()
    for path in photos:
        image = load_image(path)
        if image is None:
            print(f"{path.name:<16} unreadable file")
            continue
        t = time.perf_counter()
        result = read_display(image, config, detect)
        seconds = time.perf_counter() - t
        statuses[result.status] += 1
        reasons.update(result.reasons)
        liters = str(result.liters.value) if result.liters else ""
        checked = ""
        if result.reading is not None:
            checked = f"total {result.reading.amount.value}, price {result.reading.price.value}"
        print(
            f"{path.name:<16} {result.status:<9} {liters:<9} {seconds:5.1f}s  "
            f"{checked or ', '.join(result.reasons)}",
            flush=True,
        )
        rows.append([path.name, result.status, liters, checked, ";".join(result.reasons), f"{seconds:.1f}"])

    total = sum(statuses.values())
    print("\n" + "=" * 60)
    print(f"{total} photo(s) in {time.perf_counter() - started:.0f} s")
    for status in ("accepted", "review", "retake"):
        print(f"  {status:<9} {statuses[status]:>4}  ({statuses[status] / total:.0%})")
    if reasons:
        print("Reasons (retake and review):")
        for reason, count in reasons.most_common():
            print(f"  {reason:<24} {count}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "status", "liters", "checked_against", "reasons", "seconds"])
            writer.writerows(rows)
        print(f"Results written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
