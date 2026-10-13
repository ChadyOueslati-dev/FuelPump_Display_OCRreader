"""Combine the synthetic set and the harvested real crops into the two list
files the training config reads.

Usage:
    python scripts/train/merge_lists.py --data C:/ocr-train/data --real-repeat 20

Writes <data>/train_list.txt and <data>/val_list.txt, with paths relative to
<data>. Real crops are few, so each is listed --real-repeat times; one in
ten real crops is held back for validation instead.
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--real-repeat", type=int, default=20)
    args = parser.parse_args()

    def read(folder: Path, name: str) -> list[str]:
        path = folder / name
        if not path.is_file():
            return []
        return [f"{folder.name}/{line}" for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    synthetic = args.data / "synthetic"
    train, val = read(synthetic, "train_list.txt"), read(synthetic, "val_list.txt")
    rng = random.Random(0)
    real_train = real_val = 0
    for folder in sorted(p for p in args.data.iterdir() if p.is_dir() and p.name.startswith("real_")):
        for line in read(folder, "list.txt"):
            if rng.random() < 0.1:
                val.append(line)
                real_val += 1
            else:
                train.extend([line] * args.real_repeat)
                real_train += 1
    rng.shuffle(train)
    (args.data / "train_list.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.data / "val_list.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"train: {len(train)} lines ({real_train} real crops x {args.real_repeat})")
    print(f"val:   {len(val)} lines ({real_val} real crops)")


if __name__ == "__main__":
    main()
