"""Read the pump display in one or more photos and print the validated values.

Usage:
    python scripts/ocr_one.py path/to/photo.jpg
    python scripts/ocr_one.py data/pump_0000.jpg data/pump_0001.jpg
    python scripts/ocr_one.py path/to/photo.jpg --raw
    python scripts/ocr_one.py path/to/photo.jpg --save-crops
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ocr.engine import detect  # noqa: E402
from app.ocr.preprocess import crop, load_image  # noqa: E402
from app.pipeline import ACCEPTED, read_display  # noqa: E402
from app.report import format_result  # noqa: E402
from app.settings import load_validation_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path, nargs="+", help="pump display photo(s)")
    parser.add_argument("--raw", action="store_true", help="also print every raw OCR detection")
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="save the region the OCR was given as <photo>_screen.png next to each photo",
    )
    args = parser.parse_args()

    config = load_validation_config()
    all_accepted = True
    for path in args.images:
        print(f"\n== {path.name}")
        image = load_image(path) if path.is_file() else None
        if image is None:
            print("error: file not found or not a readable image")
            all_accepted = False
            continue
        result = read_display(image, config, detect)
        print(format_result(result, args.raw))
        if args.save_crops:
            # Exactly the region the OCR was given, so targeting can be checked by eye.
            seen = image if result.screen is None else crop(image, result.screen)
            target = path.with_name(f"{path.stem}_screen.png")
            cv2.imwrite(str(target), seen)
            print(f"Screen crop saved to {target}")
        all_accepted = all_accepted and result.status == ACCEPTED
    return 0 if all_accepted else 1


if __name__ == "__main__":
    sys.exit(main())
