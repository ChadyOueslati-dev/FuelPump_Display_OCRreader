"""Turn the answers people gave on the phone into a labelled test set.

Usage:
    python scripts/export_feedback.py
    python scripts/export_feedback.py --out data/scans/labels.csv

Every stored scan whose true liters is known (the person confirmed the
result, or typed what the pump showed) becomes one row of a labels file,
written next to the scan pictures. Measure the app on real scans with:

    python scripts/evaluate.py data/scans/labels.csv

The labels are only as good as what people typed. Before trusting a
surprising result, open the picture and look.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.db import ScanStore  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, help="default: labels.csv in the scans folder")
    args = parser.parse_args()

    store = ScanStore(args.data_dir)
    out = args.out or store.image_dir / "labels.csv"
    scans = store.recent(limit=1000000)
    labelled = [s for s in scans if s["true_liters"] and (store.image_dir / s["image_file"]).is_file()]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "amount", "volume", "price", "user", "said"])
        for s in reversed(labelled):
            writer.writerow([s["image_file"], "", s["true_liters"], "", s["user"] or "", s["feedback"] or "typed after retake"])

    said = Counter(s["feedback"] or ("typed after retake" if s["true_liters"] else "no answer") for s in scans)
    print(f"{len(scans)} stored scan(s): " + ", ".join(f"{count} {what}" for what, count in said.most_common()))
    print(f"{len(labelled)} labelled scan(s) written to {out}")
    if labelled:
        print(f"Measure with: python scripts/evaluate.py \"{out}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
