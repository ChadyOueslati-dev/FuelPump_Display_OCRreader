"""Run every labelled photo through the pipeline and report accuracy.

Usage:
    python scripts/evaluate.py data/labels.csv
    python scripts/evaluate.py data/labels.csv --photos some/other/folder

The CSV needs the columns filename and volume (the true liters); amount and
price are optional. Photos are looked
up next to the CSV unless --photos is given. Exits with code 1 if any wrong liters
value was accepted.
"""

import argparse
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import FIELDS, Report  # noqa: E402
from app.ocr.engine import detect  # noqa: E402
from app.ocr.preprocess import load_image  # noqa: E402
from app.pipeline import read_display  # noqa: E402
from app.settings import load_validation_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path, help="CSV with filename, amount, volume, price")
    parser.add_argument("--photos", type=Path, help="photo folder (default: the CSV's folder)")
    args = parser.parse_args()
    photos = args.photos or args.labels.parent

    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing = {"filename", "volume"} - set(rows[0].keys() if rows else [])
    if missing:
        print(f"error: {args.labels} is empty or lacks columns: {sorted(missing)}", file=sys.stderr)
        return 2

    config = load_validation_config()
    report = Report()
    skipped = []
    for row in rows:
        name = row["filename"].strip()
        try:
            # Only the liters (volume) is required; total and price may be blank.
            truth = {
                f: Decimal(row[f].strip()) if (row.get(f) or "").strip() else None for f in FIELDS
            }
            if truth["volume"] is None:
                raise InvalidOperation
        except InvalidOperation:
            skipped.append(f"{name}: label is not a number")
            continue
        path = photos / name
        image = load_image(path) if path.is_file() else None
        if image is None:
            skipped.append(f"{name}: file not found or not a readable image")
            continue

        result = read_display(image, config, detect)
        report.add(name, result, truth)
        if result.liters is not None:
            print(f"{name}: {result.status}  liters {result.liters.value}  (label {truth['volume']})")
        else:
            print(f"{name}: {result.status}  ({', '.join(result.reasons)})")

    print("\n" + "=" * 60)
    print(report.format())
    if skipped:
        print(f"\nSkipped {len(skipped)} row(s):")
        for line in skipped:
            print(f"  {line}")
    return 1 if report.wrong_liters else 0


if __name__ == "__main__":
    sys.exit(main())
