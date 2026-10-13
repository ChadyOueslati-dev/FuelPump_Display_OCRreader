"""Generate training pictures for the text recognition model.

Each picture is one line of text as the recognition model sees it: a
7-segment number cut out of a pump display, or a short label word. The true
text is known, because the picture is drawn from it.

Usage:
    python scripts/train/make_rec_dataset.py --out C:/ocr-train/data/synthetic --train 20000 --val 1000

Writes images/*.jpg plus train_list.txt and val_list.txt ("path<TAB>text"),
the format PaddleOCR's training reads.

Styles are randomised widely on purpose (segment thickness, slant, spacing,
colours, unlit "ghost" segments, decimal point or comma), and so are the
camera effects (tilt, blur, noise, glare, low resolution, JPEG artefacts),
so the model learns 7-segment digits in general and not one pump design.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#  a
# f b
#  g
# e c
#  d
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}
# Some displays draw these three digits with one segment more or less.
ALTERNATIVES = {"6": "fgedc", "7": "abcf", "9": "abcfg"}

WORDS = [
    "SAR", "SAR/L", "L", "LTR", "LITER", "LITERS", "LITRES", "TOTAL", "SALE", "PRICE", "EURO",
    "GAS 91", "GAS 95", "GAS 98", "DIESEL", "AMOUNT", "VOLUME", "Vol.", "CLR", "OK", "P",
    "PLEASE TURN OFF ENGINE", "LIFT NOZZLE", "THANK YOU", "NO SMOKING", "PUMP", "Gallons", "$", "SR",
]
FONTS = ["arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf", "verdana.ttf", "calibrib.ttf"]


def random_number(rng: random.Random) -> str:
    """Text of a pump readout. The mix follows what pumps show."""
    kind = rng.random()
    if kind < 0.40:  # total
        value = rng.choice([rng.uniform(0, 60), rng.uniform(0, 400), rng.uniform(0, 1000),
                            float(rng.choice([5, 10, 20, 30, 50, 100, 150, 200]))])
        text = f"{value:.2f}"
    elif kind < 0.75:  # volume, sometimes with three decimals
        value = rng.choice([rng.uniform(0, 15), rng.uniform(0, 80), rng.uniform(0, 200)])
        text = f"{value:.3f}" if rng.random() < 0.2 else f"{value:.2f}"
    elif kind < 0.90:  # price per liter; a few in cents with one decimal
        if rng.random() < 0.15:
            text = f"{rng.uniform(90, 250):.1f}"
        else:
            value = rng.uniform(0.5, 5)
            text = f"{value:.3f}" if rng.random() < 0.15 else f"{value:.2f}"
    else:  # no decimals: pump numbers, counters
        text = str(rng.randint(0, 10 ** rng.randint(1, 5) - 1))
    if rng.random() < 0.3:
        text = text.replace(".", ",")
    return text


def colours(rng: random.Random) -> tuple[tuple, tuple, tuple]:
    """(background, lit segment, unlit segment), BGR."""

    def jitter(c, amount):
        return tuple(int(np.clip(v + rng.randint(-amount, amount), 0, 255)) for v in c)

    if rng.random() < 0.55:  # bright digits on a dark panel
        bg = jitter(rng.choice([(15, 15, 15), (25, 30, 20), (10, 10, 10), (40, 25, 20), (30, 20, 10)]), 10)
        on = jitter(rng.choice([(240, 245, 245), (120, 255, 190), (40, 90, 255), (60, 200, 255),
                                (255, 220, 120), (80, 255, 80), (60, 60, 255)]), 15)
        ghost = rng.choice([0, 0, 12, 22, 35])
        off = tuple(min(v + ghost, 255) for v in bg)
    else:  # dark digits on a light panel
        bg = jitter(rng.choice([(200, 210, 210), (185, 215, 200), (230, 235, 235), (170, 200, 205),
                                (225, 215, 190), (150, 170, 160)]), 12)
        on = jitter((35, 35, 35), 25)
        ghost = rng.choice([0, 0, 10, 20, 32])
        off = tuple(max(v - ghost, 0) for v in bg)
    return bg, on, off


def draw_number(text: str, rng: random.Random) -> np.ndarray:
    """A clean 7-segment rendering of the text on its panel."""
    dh = rng.uniform(90, 150)  # digit height; shrunk again later
    dw = dh * rng.uniform(0.46, 0.64)
    t = dh * rng.uniform(0.09, 0.18)  # segment thickness
    gap = dh * rng.uniform(0.12, 0.26)
    skew = rng.uniform(0, 0.16)
    inset = t * rng.uniform(0.0, 0.25)  # space between segments
    pointed = rng.random() < 0.7  # pointed or square segment ends
    bg, on, off = colours(rng)
    ghosts = rng.randint(0, 3) if off != bg else 0  # unlit digit positions on the left
    alternative = rng.random() < 0.3

    digits = sum(c.isdigit() for c in text)
    pad = dh * rng.uniform(0.15, 0.45)
    width = int((digits + ghosts) * (dw + gap) + 2 * pad + skew * dh)
    height = int(dh + 2 * pad)
    image = np.full((height, width, 3), bg, np.uint8)

    def polygon(points, colour):
        pts = np.array([(x + skew * (y0 + dh - y), y) for x, y in points], np.float32)
        cv2.fillPoly(image, [np.round(pts * 16).astype(np.int32)], colour, cv2.LINE_AA, shift=4)

    def horizontal(x, yy):
        e = t / 2 if pointed else 0
        return [(x + t / 2 + inset, yy), (x + t / 2 + inset + e, yy - t / 2), (x + dw - t / 2 - inset - e, yy - t / 2),
                (x + dw - t / 2 - inset, yy), (x + dw - t / 2 - inset - e, yy + t / 2), (x + t / 2 + inset + e, yy + t / 2)]

    def vertical(xx, ya, yb):
        e = t / 2 if pointed else 0
        return [(xx, ya + inset), (xx + t / 2, ya + inset + e), (xx + t / 2, yb - inset - e),
                (xx, yb - inset), (xx - t / 2, yb - inset - e), (xx - t / 2, ya + inset + e)]

    def digit(x, lit):
        mid = y0 + dh / 2
        shapes = {
            "a": horizontal(x, y0 + t / 2), "g": horizontal(x, mid), "d": horizontal(x, y0 + dh - t / 2),
            "f": vertical(x + t / 2, y0 + t / 2, mid), "b": vertical(x + dw - t / 2, y0 + t / 2, mid),
            "e": vertical(x + t / 2, mid, y0 + dh - t / 2), "c": vertical(x + dw - t / 2, mid, y0 + dh - t / 2),
        }
        for name, points in shapes.items():
            if name in lit or off != bg:
                polygon(points, on if name in lit else off)

    x, y0 = pad, pad
    for _ in range(ghosts):
        digit(x, "")
        x += dw + gap
    for ch in text:
        if ch in ".,":
            r = t * rng.uniform(0.45, 0.8)
            cx, cy = x - gap * rng.uniform(0.35, 0.65), y0 + dh - r
            cv2.circle(image, (int(cx * 16), int(cy * 16)), int(r * 16), on, -1, cv2.LINE_AA, shift=4)
            if ch == ",":
                tail = np.array([(cx, cy), (cx + r, cy), (cx - r * 0.8, cy + r * 2.6)], np.float32)
                cv2.fillPoly(image, [np.round(tail * 16).astype(np.int32)], on, cv2.LINE_AA, shift=4)
            continue
        lit = ALTERNATIVES[ch] if alternative and ch in ALTERNATIVES else SEGMENTS[ch]
        digit(x, lit)
        x += dw + gap
    return image


def draw_word(text: str, rng: random.Random) -> np.ndarray | None:
    """A label word in an ordinary font."""
    try:
        font = ImageFont.truetype(rng.choice(FONTS), rng.randint(40, 90))
    except OSError:
        return None
    bg = tuple(rng.randint(0, 255) for _ in range(3))
    fg = (255, 255, 255) if sum(bg) < 380 else (20, 20, 20)
    box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    pad = rng.randint(8, 30)
    canvas = Image.new("RGB", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), bg)
    ImageDraw.Draw(canvas).text((pad - box[0], pad - box[1]), text, font=font, fill=fg)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def camera(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """What a phone does to the picture: tilt, blur, noise, glare, low resolution."""
    h, w = image.shape[:2]
    border = [int(v) for v in image[0, 0]]

    if rng.random() < 0.8:  # perspective: seen from the side, above or below
        j = rng.uniform(0.0, 0.14)
        squeeze = rng.uniform(0.0, 0.25) * h  # one side further from the camera
        left = rng.random() < 0.5
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [w * rng.uniform(0, j), h * rng.uniform(0, j) + (squeeze if left else 0)],
            [w * (1 - rng.uniform(0, j)), h * rng.uniform(0, j) + (0 if left else squeeze)],
            [w * (1 - rng.uniform(0, j)), h * (1 - rng.uniform(0, j)) - (0 if left else squeeze)],
            [w * rng.uniform(0, j), h * (1 - rng.uniform(0, j)) - (squeeze if left else 0)],
        ])
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=border)
    if rng.random() < 0.6:  # the text detector does not always level the line
        turn = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-7, 7), 1)
        image = cv2.warpAffine(image, turn, (w, h), borderValue=border)

    image = image.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    image *= (1 + rng.uniform(-0.3, 0.3) * (xx / w) + rng.uniform(-0.2, 0.2) * (yy / h))[..., None]
    if rng.random() < 0.3:  # glare
        gx, gy, gr = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.2, 0.8) * h
        image += (np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * gr ** 2))) * rng.uniform(40, 140))[..., None]
    image = image * rng.uniform(0.6, 1.15) + rng.uniform(-20, 30)
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Resolution: on a photo of the whole pump the digits are only a few
    # pixels tall. Shrink to that size, which is where the decimal point
    # nearly disappears.
    target = rng.choice([rng.uniform(14, 28), rng.uniform(22, 48), rng.uniform(40, 90)])
    scale = target / h
    image = cv2.resize(image, (max(int(w * scale), 8), max(int(h * scale), 8)), interpolation=cv2.INTER_AREA)

    blur = rng.random()
    if blur < 0.35:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.4))
    elif blur < 0.55:  # motion blur
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1), (k, k))
        image = cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))
    noise = np.random.normal(0, rng.uniform(0, 9), image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:  # the pixel grid of a photographed monitor
        step = rng.choice([2, 3, 4])
        image[::step, :] = (image[::step, :] * rng.uniform(0.8, 0.95)).astype(np.uint8)

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 95)])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--words", type=float, default=0.15, help="fraction that are label words")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    for split, count in (("train", args.train), ("val", args.val)):
        lines = []
        while len(lines) < count:
            if rng.random() < args.words:
                text = rng.choice(WORDS)
                clean = draw_word(text, rng)
                if clean is None:
                    continue
            else:
                text = random_number(rng)
                clean = draw_number(text, rng)
            name = f"images/{split}_{len(lines):06d}.jpg"
            cv2.imwrite(str(args.out / name), camera(clean, rng))
            lines.append(f"{name}\t{text}")
        (args.out / f"{split}_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {count} pictures")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
