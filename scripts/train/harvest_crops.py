"""Cut real number crops, with their true text, out of labelled pump photos.

Usage:
    python scripts/train/harvest_crops.py --photos FOLDER --labels generator_labels.csv --out C:/ocr-train/data/real_ads

The OCR is run on each photo as the app runs it. A detected text box whose
digits equal the digits of the photo's true total, liters or price is that
number, even if the OCR dropped its decimal point ("10841" for "108.41").
The box is saved as a training picture labelled with the true text, decimal
point included. That is exactly the kind of picture the model must learn.

Boxes that match two different labels of the same photo are skipped.
Writes images/*.jpg and list.txt ("path<TAB>text").

Never harvest from photos that are also used to measure accuracy.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ocr.engine import detect  # noqa: E402
from app.ocr.preprocess import load_image, shrink  # noqa: E402

FIELDS = ("amount_sar", "litres", "price_per_l")


def digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="labels.csv written by the generator")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=960, help="read the photo at this size, as the app does")
    args = parser.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    lines = []
    for row in rows:
        image = load_image(args.photos / row["file"])
        if image is None:
            continue
        image = shrink(image, args.side)
        sep = row.get("decimal_sep") or "."
        truth = {digits(row[f]): row[f].replace(".", sep) for f in FIELDS}
        if len(truth) < len(FIELDS):
            truth = {k: v for k, v in truth.items() if sum(digits(row[f]) == k for f in FIELDS) == 1}
        found = 0
        for d in detect(image):
            text = truth.get(digits(d.text))
            if text is None or any(c.isalpha() for c in d.text):
                continue
            x1, y1, x2, y2 = d.box
            pad = max(2, (y2 - y1) // 10)
            crop = image[max(y1 - pad, 0) : y2 + pad, max(x1 - pad, 0) : x2 + pad]
            name = f"images/{Path(row['file']).stem}_{len(lines):05d}.jpg"
            cv2.imwrite(str(args.out / name), crop)
            lines.append(f"{name}\t{text}")
            found += 1
        print(f"{row['file']}: {found} crop(s)", flush=True)

    (args.out / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} crop(s) from {len(rows)} photo(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
