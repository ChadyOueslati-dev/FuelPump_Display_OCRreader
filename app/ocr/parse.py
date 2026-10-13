"""Turn raw OCR detections into the three pump display fields.

Layout assumption: the display shows three decimal numbers stacked in one
column, top to bottom: amount, volume, unit price. Displays with only the
first two rows are handled by parse_two_rows.

Parsing never guesses. Text is not "corrected" (no O->0, no inserted decimal
point; a decimal comma is read as a decimal point), and anything ambiguous raises ParseError so the caller asks for a
retake.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

from app.ocr.detection import Detection

_DECIMAL_NUMBER = re.compile(r"\d+\.\d+")
_COMMA_NUMBER = re.compile(r"\d+,\d+")


def number_text(text: str) -> str:
    """The text of a detection as a number would be written: no spaces, and a
    decimal comma (European pumps show "25,37") turned into a point."""
    compact = "".join(text.split())
    return compact.replace(",", ".") if _COMMA_NUMBER.fullmatch(compact) else compact


class ParseError(Exception):
    """The detections cannot be read as exactly one pump display."""


@dataclass(frozen=True)
class Field:
    value: Decimal  # keeps the decimal places as displayed, e.g. Decimal("50.00")
    confidence: float
    box: tuple[int, int, int, int] | None = None  # where it was read, in the image given to OCR
    source: str = ""  # which image version it was read from
    inferred: bool = False  # the decimal point was not read but put in by the app


@dataclass(frozen=True)
class DisplayReading:
    amount: Field
    volume: Field
    price: Field

    @property
    def confidence(self) -> float:
        return min(self.amount.confidence, self.volume.confidence, self.price.confidence)


def _x_overlap(a: Detection, b: Detection) -> bool:
    return a.box[0] < b.box[2] and b.box[0] < a.box[2]


def _is_column(group: tuple[Detection, ...]) -> bool:
    return all(_x_overlap(a, b) for a, b in combinations(group, 2))


def _y_center(d: Detection) -> float:
    return (d.box[1] + d.box[3]) / 2


def parse_display(detections: list[Detection], source: str = "") -> DisplayReading:
    """Pick out amount, volume and price. Raises ParseError if not unambiguous."""
    numbers = [d for d in detections if _DECIMAL_NUMBER.fullmatch(number_text(d.text))]
    if len(numbers) < 3:
        raise ParseError(f"found {len(numbers)} decimal number(s), need 3")

    columns = [g for g in combinations(numbers, 3) if _is_column(g)]
    if not columns:
        raise ParseError("no three decimal numbers are stacked in one column")
    if len(columns) > 1:
        raise ParseError("more than one possible display column")

    top, middle, bottom = (_field(d, source) for d in sorted(columns[0], key=_y_center))
    return DisplayReading(amount=top, volume=middle, price=bottom)


def parse_two_rows(detections: list[Detection], source: str = "") -> tuple[Field, Field]:
    """For displays that show only two rows: the total on top, liters below.

    Returns (amount, liters). Stricter than parse_display, because with one
    row missing nothing says which rows the two numbers are: any other digits
    in the same column mean a third row was there but not read as a number,
    and then the reading is refused.
    """
    numbers = [d for d in detections if _DECIMAL_NUMBER.fullmatch(number_text(d.text))]
    if len(numbers) != 2:
        raise ParseError(f"found {len(numbers)} decimal number(s), need 2")
    if not _x_overlap(*numbers):
        raise ParseError("the two decimal numbers are not stacked in one column")
    for d in detections:
        stray_digits = d not in numbers and any(c.isdigit() for c in d.text)
        if stray_digits and all(_x_overlap(d, n) for n in numbers):
            raise ParseError("other digits in the display column")
    top, bottom = (_field(d, source) for d in sorted(numbers, key=_y_center))
    return top, bottom


def _field(d: Detection, source: str) -> Field:
    return Field(Decimal(number_text(d.text)), d.confidence, d.box, source)
