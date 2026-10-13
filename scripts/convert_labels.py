"""Convert a labels.csv written by the image generator into this project's format.

Usage:
    python scripts/convert_labels.py generator_labels.csv [more.csv ...] --out "data/Pump Displays_Large/labels.csv"

The generator writes: file, grade, price_per_l, amount_sar, litres, ...
evaluate.py wants:    filename, amount, volume, price
"""

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for source in args.sources:
        with open(source, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append([row["file"], row["amount_sar"], row["litres"], row["price_per_l"]])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "amount", "volume", "price"])
        writer.writerows(rows)
    print(f"{len(rows)} label(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
