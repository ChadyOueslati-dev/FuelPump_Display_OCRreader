"""List every stored scan and check that each one has both a picture and a value.

Usage:
    python scripts/check_storage.py
    python scripts/check_storage.py --open             (also opens the newest picture)
    python scripts/check_storage.py --delete SCAN_ID   (removes a scan: record and picture)
    python scripts/check_storage.py --clean            (removes records whose picture is gone)

The scan id is the picture's file name without ".jpg". To remove a scan,
use --delete rather than deleting the picture by hand, so its record goes too.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.db import ScanStore  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--open", action="store_true", help="open the newest picture")
    parser.add_argument("--delete", metavar="SCAN_ID", nargs="+", help="remove these scans")
    parser.add_argument("--clean", action="store_true", help="remove records whose picture is gone")
    args = parser.parse_args()

    store = ScanStore(args.data_dir)

    for scan_id in args.delete or []:
        scan_id = scan_id.removesuffix(".jpg")
        print(f"deleted {scan_id}" if store.delete(scan_id) else f"no such scan: {scan_id}")
    if args.clean:
        for image_file in store.check()[0]:
            store.delete(image_file.removesuffix(".jpg"))
            print(f"removed the record of {image_file} (its picture was already gone)")

    scans = store.recent(limit=100000)
    print(f"Pictures folder: {store.image_dir}")
    print(f"Database:        {store.db_path}")
    print(f"\n{len(scans)} scan(s), newest first:\n")
    print(f"{'time (UTC)':<26}{'user':<12}{'status':<10}{'liters':<10}{'picture':<34}on disk")
    for s in scans:
        on_disk = (store.image_dir / s["image_file"]).is_file()
        print(
            f"{s['created_at']:<26}{s['user'] or '-':<12}{s['status']:<10}{s['liters'] or '-':<10}"
            f"{s['image_file']:<34}{'yes' if on_disk else 'MISSING'}"
        )

    missing, orphans = store.check()
    print()
    if not missing and not orphans:
        print("OK: every scan has both its picture and its record.")
    for name in missing:
        print(f"PROBLEM: record without a picture: {name}  (run with --clean to remove the record)")
    for name in orphans:
        print(f"PROBLEM: picture without a record: {name}")

    if args.open and scans:
        os.startfile(store.image_dir / scans[0]["image_file"])  # Windows only
    return 1 if missing or orphans else 0


if __name__ == "__main__":
    sys.exit(main())
