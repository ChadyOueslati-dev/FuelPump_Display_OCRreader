"""Work out which number is the total, the liters and the price per liter.

Three kinds of evidence are combined:

  size      the total and the liters are the biggest numerals on a pump. The
            tallest number and the ones stacked with it at a similar size
            form the main column; small digits elsewhere (keypad, pump
            number) are ignored.
  position  in the main column, top to bottom: total, liters, price per liter.
  labels    words printed next to a number ("SAR", "L", "LITER", "SAR/L")
            say what it is. The word lists are parameters: config/labels.yaml.

A label is only used when it clearly belongs to exactly one number. Then it
wins over position, which is what lets other layouts work: rows in another
order, or the price per liter in a separate panel away from the column.

This matters for correctness, not just convenience: liters x price = total
holds just as well with liters and price swapped, so the arithmetic check
cannot tell which of the two is the liters. Only position and labels can.

Missing decimal points. The OCR often drops the small decimal point of a
7-segment number ("10841" for "108.41"). When allowed, such a number is
given the usual number of decimals and flagged as inferred. The caller must
treat an inferred value as unverified unless the arithmetic confirms it.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from app.ocr.detection import Detection
from app.ocr.parse import Field, ParseError, _x_overlap, _y_center, number_text

AMOUNT, LITERS, PRICE = "amount", "liters", "price"

# A label read with less confidence than this is ignored.
MIN_LABEL_CONFIDENCE = 0.8
# A number belongs to the main column if it is at least this fraction of the
# tallest number's height and stacked with it.
MAIN_HEIGHT_RATIO = 0.6
# A total or liters label belongs to a number if at least this fraction of
# the label's height is on the number's row, and it is no further away than
# this many number-widths.
MIN_ROW_OVERLAP = 0.5
MAX_LABEL_GAP = 1.5
# The total and the price per liter are shown with this many decimals.
MONEY_DECIMALS = 2

_DECIMAL = re.compile(r"\d+\.\d+")
# A number read without its decimal point. Shorter runs are keypad keys, pump
# numbers, noise; longer ones are phone numbers and totalisers ("920012615"
# under an Aramco display), not a value of this sale.
_DIGITS_ONLY = re.compile(r"\d{3,6}")
_EMBEDDED = re.compile(r"(?P<before>[^\d.,]*)(?P<number>\d+[.,]\d+)(?P<after>[^\d.,]*)")


# Arabic is printed in several equivalent spellings: with or without the
# stretching line between letters, vowel marks, and hamza on the alef.
_ARABIC_SAME = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ـ": None})


def _normalise(text: str) -> str:
    """Upper case, with spaces and punctuation other than / $ € removed, and
    Arabic reduced to one spelling."""
    text = "".join(c for c in text if not "\u064b" <= c <= "\u0652")  # vowel marks
    return "".join(c for c in text.upper().translate(_ARABIC_SAME) if c.isalnum() or c in "/$€")


@dataclass(frozen=True)
class Labels:
    """The words that identify each value. Compared after _normalise()."""

    amount: frozenset[str]
    liters: frozenset[str]
    price: frozenset[str]
    # A display in gallons is refused: its number is not liters. Matched as a
    # prefix, so OCR slips like "GALIONS" still count.
    gallons_prefixes: frozenset[str]

    @classmethod
    def from_lists(cls, amount, liters, price, gallons_prefixes) -> "Labels":
        return cls(*(frozenset(_normalise(w) for w in words) for words in (amount, liters, price, gallons_prefixes)))

    def role(self, text: str) -> str | None:
        word = _normalise(text)
        for role, words in ((PRICE, self.price), (LITERS, self.liters), (AMOUNT, self.amount)):
            if word in words:
                return role
        return None

    def is_gallons(self, text: str) -> bool:
        word = _normalise(text)
        return any(word.startswith(prefix) for prefix in self.gallons_prefixes)


DEFAULT_LABELS = Labels.from_lists(
    amount=["SAR", "SR", "RIYAL", "EURO", "EUR", "€", "$", "TOTAL", "SALE", "AMOUNT", "BETRAG", "PURCHASE", "P"],
    liters=["L", "LTR", "LTRS", "LITER", "LITERS", "LITRE", "LITRES", "VOLUME", "ABGABE"],
    price=["SAR/L", "SR/L", "€/L", "$/L", "PRICE", "PRICE/L", "PRICEPERLITER", "PREISJELITER", "UNITPRICE"],
    gallons_prefixes=["GAL"],
)


def parse_layout(
    detections: list[Detection],
    labels: Labels = DEFAULT_LABELS,
    source: str = "",
    infer_decimals: bool = False,
    liters_decimals: int = 2,
    price_candidates: list[Field] | None = None,
) -> tuple[Field, Field, Field | None]:
    """(total, liters, price per liter). Price is None if the display shows none.

    infer_decimals: also accept numbers read without their decimal point, as
    described in the module docstring. liters_decimals is how many decimals
    such a liters value is given.

    price_candidates: if a list is passed and no price could be identified,
    it is filled with the other numbers in the photo. One of them may be the
    price with its label unread; only the arithmetic can tell (the caller's
    job).

    Raises ParseError when the numbers cannot be told apart with certainty.
    """
    confident = [d for d in detections if d.confidence >= MIN_LABEL_CONFIDENCE]
    if any(labels.is_gallons(d.text) for d in confident):
        raise ParseError("the display shows gallons, not liters")

    detections, hints = _split_embedded_labels(detections, labels)
    numbers = [d for d in detections if _is_number(d, infer_decimals)]
    if not numbers:
        raise ParseError("found 0 decimal number(s), need 3")

    main = _main_column(numbers)
    if len(main) < 2:
        raise ParseError(f"found {len(main)} decimal number(s), need 3")
    if len(main) > 3:
        raise ParseError("more than one possible display column")
    others = [n for n in numbers if not any(n is m for m in main)]
    if len(main) == 2:
        # A smaller number stacked directly under the two big ones is the
        # third row: some pumps show the price per liter in smaller digits.
        below = [n for n in others if _in_column(n, main)]
        if len(below) == 1 and _y_center(below[0]) > _y_center(main[-1]):
            main.append(below[0])
            others = [n for n in others if n is not below[0]]

    for label in confident:
        role = labels.role(label.text)
        if role is None:
            continue
        owner = None
        if role == PRICE:
            # The price often sits apart from the column, or in a small box
            # beside the liters with its label under it. So the label goes to
            # the number it is closest to, wherever that is: "on the same row"
            # would hand it to the liters next door.
            owner = _nearest(label, others if len(main) == 2 else numbers)
        if owner is None and not (role == PRICE and len(main) == 2):
            beside = [n for n in main if _beside(label, n)]
            owner = beside[0] if len(beside) == 1 else None  # ambiguous: ignored, never guessed
            # A label that has a number of its own right next to it (a grade
            # list: "GAS 91  2.18  SAR/L") is that number's, however close a
            # main row comes on the same line.
            if owner is not None and any(_gap(label, n) < _gap(label, owner) for n in others):
                owner = None
        if owner is not None and hints.setdefault(id(owner), role) != role:
            raise ParseError("one number carries two different labels")

    for role in (AMOUNT, LITERS, PRICE):
        if sum(hints.get(id(n)) == role for n in numbers) > 1:
            raise ParseError(f"two numbers are labelled as {role}")

    if len(main) == 3:
        amount, liters, price = _assign(main, [AMOUNT, LITERS, PRICE], hints)
    else:
        # With one row missing nothing but the labels says which rows these
        # are, so any other digits in the column (a third row that was not
        # read as a number) make the reading unsafe.
        _refuse_stray_digits(main, detections, numbers)
        amount, liters = _assign(main, [AMOUNT, LITERS], hints)
        price = next((n for n in others if hints.get(id(n)) == PRICE), None)
        if price is None and price_candidates is not None:
            price_candidates.extend(_field(n, source, MONEY_DECIMALS) for n in others)

    return (
        _field(amount, source, MONEY_DECIMALS),
        _field(liters, source, liters_decimals),
        None if price is None else _field(price, source, MONEY_DECIMALS),
    )


def _main_column(numbers: list[Detection]) -> list[Detection]:
    """The stacked numbers that are the display's main rows, top to bottom.

    Each number is tried as the tallest of a column: the column is that
    number plus every number stacked with it that is at least
    MAIN_HEIGHT_RATIO of its height. The column with the tallest lead wins,
    but only columns of two or more count: a single tall token standing
    alone is usually something misread (a grille, a logo), not a readout.
    """
    columns = []
    for lead in numbers:
        column = [
            n for n in numbers
            if n is lead
            or (MAIN_HEIGHT_RATIO * _height(lead) <= _height(n) <= _height(lead) and _x_overlap(n, lead))
        ]
        columns.append((len(column) >= 2, _height(lead), column))
    _, _, best = max(columns, key=lambda c: (c[0], c[1]))
    return sorted(best, key=_y_center)


def _compact(d: Detection) -> str:
    return number_text(d.text)


def _is_number(d: Detection, infer_decimals: bool) -> bool:
    text = _compact(d)
    return bool(_DECIMAL.fullmatch(text) or (infer_decimals and _DIGITS_ONLY.fullmatch(text)))


def _height(d: Detection) -> int:
    return d.box[3] - d.box[1]


def _field(d: Detection, source: str, decimals: int) -> Field:
    text = _compact(d)
    if _DECIMAL.fullmatch(text):
        return Field(Decimal(text), d.confidence, d.box, source)
    # No decimal point was read: put it where this kind of value has it.
    value = Decimal(text).scaleb(-decimals)
    return Field(value, d.confidence, d.box, source, inferred=True)


def _split_embedded_labels(
    detections: list[Detection], labels: Labels
) -> tuple[list[Detection], dict[int, str]]:
    """Turn "SAR179.85" or "4.59L" into the bare number plus a label hint.

    Returns the detections with such tokens replaced, and {id(number): role}.
    A token whose extra characters are not a known label is left untouched,
    so it still does not count as a number.
    """
    result, hints = [], {}
    for d in detections:
        compact = "".join(d.text.split())
        match = _EMBEDDED.fullmatch(compact)
        if match and (match["before"] or match["after"]):
            roles = {labels.role(part) for part in (match["before"], match["after"]) if part}
            if len(roles) == 1 and None not in roles:
                number = Detection(match["number"], d.confidence, d.box)
                hints[id(number)] = roles.pop()
                result.append(number)
                continue
        result.append(d)
    return result, hints


def _gap(a: Detection, b: Detection) -> float:
    """Distance between two boxes; 0 if they touch or overlap."""
    dx = max(a.box[0] - b.box[2], b.box[0] - a.box[2], 0)
    dy = max(a.box[1] - b.box[3], b.box[1] - a.box[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def _beside(label: Detection, number: Detection) -> bool:
    """True if the label sits on the number's row, close to it."""
    overlap = min(label.box[3], number.box[3]) - max(label.box[1], number.box[1])
    if overlap < MIN_ROW_OVERLAP * min(_height(label), _height(number)):
        return False
    gap = max(label.box[0] - number.box[2], number.box[0] - label.box[2], 0)
    return gap <= MAX_LABEL_GAP * (number.box[2] - number.box[0])


def _nearest(label: Detection, numbers: list[Detection]) -> Detection | None:
    """The number a free-standing label belongs to: the closest one, if it is
    close and clearly closer than any other. Works on a tilted photo, where
    "on the same row" does not."""
    ranked = sorted(numbers, key=lambda n: _gap(label, n))
    if not ranked:
        return None
    best = _gap(label, ranked[0])
    reach = 2 * (label.box[2] - label.box[0])
    if best > reach:
        return None
    if len(ranked) > 1 and _gap(label, ranked[1]) < 2 * best + _height(label):
        return None
    return ranked[0]


def _in_column(d: Detection, main: list[Detection]) -> bool:
    """True if d is stacked with the main rows: same horizontal span, and
    no further above or below them than one row height."""
    reach = max(_height(n) for n in main)
    top, bottom = main[0].box[1] - reach, main[-1].box[3] + reach
    return all(_x_overlap(d, n) for n in main) and top <= _y_center(d) <= bottom


def _refuse_stray_digits(
    main: list[Detection], detections: list[Detection], numbers: list[Detection]
) -> None:
    for d in detections:
        if any(d is m for m in main):
            continue
        # A row of digits that was not read as a number ("2 18", "5O.00").
        # A word with one stray digit in it ("TOTAL1" for the label TOTAL)
        # is a label, not a row.
        characters = [c for c in d.text if c.isalnum()]
        mostly_digits = 2 * sum(c.isdigit() for c in characters) >= len(characters) > 0
        if mostly_digits and _in_column(d, main):
            raise ParseError("other digits in the display column")


def _assign(rows: list[Detection], roles: list[str], hints: dict[int, str]) -> list[Detection]:
    """Give each row a role: labelled rows take their label's role, the rest
    take the remaining roles in top-to-bottom order."""
    by_role = {hints[id(n)]: n for n in rows if hints.get(id(n)) in roles}
    unlabelled = iter(n for n in rows if hints.get(id(n)) not in roles)
    return [by_role[role] if role in by_role else next(unlabelled) for role in roles]
