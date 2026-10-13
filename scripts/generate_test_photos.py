"""Generate fresh labelled test photos with the pump image generator.

Usage:
    python -X utf8 scripts/generate_test_photos.py --out data/generated --n 100
    python -X utf8 scripts/generate_test_photos.py --out data/generated --n 100 --seed 500

-X utf8 is needed on Windows: the generator writes Arabic text into its label
file without naming an encoding.

Runs gen_pump.py (the generator that came with the datasets) four times, once
per capture method, each into its own folder:

    close       standing at the pump
    car         through a car window
    close_ads   standing at the pump, with stickers and ads on it
    car_ads     through a car window, with stickers and ads

Each folder gets the photos, the generator's labels.csv, and labels_eval.csv
in the format scripts/evaluate.py reads.

The generator was written for Linux and names fonts that Windows does not
have, so they are swapped for Windows fonts here. The generator file itself
is not changed. Use a seed not used before, or the photos repeat.
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT / "data" / "pump_ads_dataset" / "pump_dataset_ads" / "gen_pump.py"
WINDOWS_FONTS = Path("C:/Windows/Fonts")
FONTS = {  # generator's name -> Windows font files to try, in order
    "FONT_B": ["arialbd.ttf"],
    "FONT_C": ["arialnb.ttf", "arialbd.ttf"],
    "FONT_R": ["arial.ttf"],
    "FONT_OBL": ["arialbi.ttf", "arialbd.ttf"],
    "FONT_SERIF": ["timesbd.ttf", "arialbd.ttf"],
}
METHODS = {  # folder -> generator arguments
    "close": ["--mode", "close"],
    "car": ["--mode", "car"],
    "close_ads": ["--mode", "close", "--clutter"],
    "car_ads": ["--mode", "car", "--clutter"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100, help="photos per capture method")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--generator", type=Path, default=GENERATOR)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("gen_pump", args.generator)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for name, candidates in FONTS.items():
        font = next((WINDOWS_FONTS / f for f in candidates if (WINDOWS_FONTS / f).is_file()), None)
        if font is None:
            print(f"error: none of {candidates} found in {WINDOWS_FONTS}", file=sys.stderr)
            return 2
        setattr(generator, name, str(font))

    for i, (folder, extra) in enumerate(METHODS.items()):
        out = args.out / folder
        sys.argv = ["gen_pump.py", "--n", str(args.n), "--out", str(out), "--seed", str(args.seed + i), *extra]
        generator.main()
        with open(out / "labels.csv", newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        with open(out / "labels_eval.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "amount", "volume", "price"])
            writer.writerows([r["file"], r["amount_sar"], r["litres"], r["price_per_l"]] for r in rows)
    print(f"done: {args.n} photos x {len(METHODS)} capture methods in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
