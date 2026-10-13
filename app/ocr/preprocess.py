"""OpenCV steps that run before OCR: load the photo and crop it to the display.

Screen detection looks for the largest roughly rectangular area that is
clearly whiter (backlit LCD) or clearly darker (LED display) than its
surroundings. A photo framed so tightly that no screen edge is visible, or a
screen with little contrast against the pump body, has no such area.
"""

from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x_min, y_min, x_max, y_max in pixels

# Limits on what counts as a screen, as fractions of the photo / of its own
# rotated bounding rectangle.
MIN_AREA_FRACTION = 0.05
MAX_AREA_FRACTION = 0.95
MIN_RECTANGULARITY = 0.75
PADDING_FRACTION = 0.02

# Longest side, in pixels, of a photo after loading.
MAX_SIDE = 1600


def load_image(path: Path) -> np.ndarray | None:
    """Read an image file as BGR. Returns None if it is not a readable image."""
    # np.fromfile + imdecode instead of cv2.imread, which fails on non-ASCII
    # Windows paths.
    return decode_image(np.fromfile(str(path), dtype=np.uint8))


def decode_image(data: bytes | np.ndarray) -> np.ndarray | None:
    """Decode an encoded image (JPEG, PNG, ...) as BGR, or None if it is not one.

    Phone photos are shrunk so the longest side is at most MAX_SIDE: OCR on a
    full 12-megapixel photo takes far longer and reads no better.
    """
    buffer = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else data
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # also applies EXIF rotation
    if image is None:
        return None
    return shrink(image, MAX_SIDE)


def shrink(image: np.ndarray, max_side: int) -> np.ndarray:
    """Scale the image down so its longest side is at most max_side."""
    longest = max(image.shape[:2])
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def find_screen(image: np.ndarray) -> Box | None:
    """Locate the display. Returns None if no screen outline is visible."""
    # A backlit LCD is bright and nearly colourless (white, grey, pale
    # green). The smallest of the three colour channels is high only for
    # such pixels, so a vivid blue or red pump body, which is just as bright,
    # is not mistaken for the screen.
    whiteness = cv2.GaussianBlur(image.min(axis=2), (5, 5), 0)
    otsu, white = cv2.threshold(whiteness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    found = _largest_rectangle(white)
    if found is not None:
        return found
    # An LED display is a dark panel on a lighter pump body.
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    brightness = cv2.GaussianBlur(brightness, (5, 5), 0)
    _, dark = cv2.threshold(brightness, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    found = _largest_rectangle(dark)
    if found is not None:
        return found
    # A very bright LCD set in a mid-grey surround merges with the surround
    # at the automatic level. Stricter levels separate the two.
    for fraction in (1 / 3, 2 / 3):
        level = otsu + (255 - otsu) * fraction
        _, mask = cv2.threshold(whiteness, level, 255, cv2.THRESH_BINARY)
        found = _largest_rectangle(mask)
        if found is not None:
            return found
    return None


def _largest_rectangle(mask: np.ndarray) -> Box | None:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < MIN_AREA_FRACTION * width * height:
            break  # sorted by area, so nothing later can qualify
        if area > MAX_AREA_FRACTION * width * height:
            continue
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        if area / (rect_w * rect_h) < MIN_RECTANGULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        pad = round(PADDING_FRACTION * max(w, h))
        return (
            max(x - pad, 0),
            max(y - pad, 0),
            min(x + w + pad, width),
            min(y + h + pad, height),
        )
    return None


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def row_crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Cut out one line of text with some margin, padded so OCR can find it.

    The margin is kept small vertically because display rows sit close
    together; the replicated border gives the text detector room around it.
    """
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    pad_y, pad_x = round(0.35 * (y2 - y1)), round(0.15 * (x2 - x1))
    row = image[max(y1 - pad_y, 0) : min(y2 + pad_y, height), max(x1 - pad_x, 0) : min(x2 + pad_x, width)]
    border = round(0.5 * (y2 - y1))
    return cv2.copyMakeBorder(row, border, border, border, border, cv2.BORDER_REPLICATE)


def ocr_variants(screen: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Versions of the cropped display to run OCR on.

    No single version reads every photo: the original picks up the faint
    unlit segments of an LCD as extra digits ("988.99" for "1.99"), while the
    cleaned versions can erase thin strokes such as a "1". The pipeline reads
    them in this order and only trusts a reading that passes validation.
    """
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Everything brighter than the threshold (background and unlit segments)
    # becomes white; the dark lit segments keep their shading.
    darkest = float(np.percentile(gray, 1))
    scale = 255 / max(threshold - darkest, 1)
    faded = np.clip((gray.astype(np.float32) - darkest) * scale, 0, 255).astype(np.uint8)

    # Thicken the digit strokes. Some displays draw each segment as a thin
    # outline with gaps, which the OCR does not see as digits until the
    # outlines are merged into solid strokes. The digits are the minority
    # colour: dark on an LCD, bright on an LED display.
    size = max(3, max(gray.shape) // 60) | 1  # odd, grows with the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_digits = binary.mean() > 127
    thick = cv2.morphologyEx(binary, cv2.MORPH_OPEN if dark_digits else cv2.MORPH_CLOSE, kernel)

    return [
        ("original", screen),
        ("faded", cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)),
        ("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
        ("thick", cv2.cvtColor(thick, cv2.COLOR_GRAY2BGR)),
    ]
