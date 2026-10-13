"""Photo in, liters (or "retake") out.

All three rows (total, liters, price per liter) are read so they can check
each other, but liters is the value the app returns.

Statuses:
  accepted  liters passed the arithmetic check against the other two rows.
  review    the display shows no price per liter, so liters was read and
            re-read but could not be checked by arithmetic.
  retake    nothing trustworthy was read; no value is returned.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product

import numpy as np

from app.ocr.detection import Detection
from app.ocr.layout import parse_layout
from app.ocr.parse import DisplayReading, Field, ParseError, number_text
from app.ocr.preprocess import Box, crop, find_screen, ocr_variants, row_crop, shrink
from app.settings import ValidationConfig
from app.validation.rules import validate

ACCEPTED = "accepted"
REVIEW = "review"
RETAKE = "retake"

PARSE_FAILED = "parse_failed"
CONFLICTING_READINGS = "conflicting_readings"
LOW_CONFIDENCE = "low_confidence"
LITERS_NOT_CONFIRMED = "liters_not_confirmed"
# Flags on a "review" result: why liters is returned but not verified.
NO_PRICE_ROW = "no_price_row"
DECIMAL_POINT_INFERRED = "decimal_point_inferred"

# Name of the extra attempt that re-reads only the liters row.
LITERS_ROW = "liters row"

# Without the liters row re-read, a two-row reading needs this many image
# versions reading the same numbers.
TWO_ROW_MIN_AGREEING = 2
# One decimal number with nothing but letters and symbols around it.
_NUMBER_WITH_WORDS = re.compile(r"[^\d.,]*(?P<number>\d+[.,]\d+)[^\d.,]*")

Detect = Callable[[np.ndarray], list[Detection]]


@dataclass(frozen=True)
class Attempt:
    """What one OCR read produced: a whole-display variant or the liters row."""

    variant: str
    detections: list[Detection]
    reading: DisplayReading | None  # None if parsing failed or not a full read
    failures: list[str]  # empty means this attempt is valid
    detail: str = ""  # what happened, when there is no reading to show
    two_rows: tuple[Field, Field] | None = None  # (total, liters) if only two rows were read


@dataclass(frozen=True)
class Result:
    status: str
    reading: DisplayReading | None  # only set when accepted
    reasons: list[str]
    screen: Box | None  # None means the whole photo was read
    attempts: list[Attempt]
    unverified_liters: Field | None = None  # only set when status is "review"

    @property
    def liters(self) -> Field | None:
        """The value the app returns, or None on a retake."""
        return self.reading.volume if self.reading is not None else self.unverified_liters


def _candidates(fields: Iterable[Field]) -> list[Field]:
    """Distinct values read for one row, each with its best confidence."""
    best: dict[str, Field] = {}
    for field in fields:
        # Keyed by text, so 50.00 and 50.0 count as different readings.
        key = str(field.value)
        # A read that saw the decimal point beats one where it was inferred.
        rank = (not field.inferred, field.confidence)
        if key not in best or rank > (not best[key].inferred, best[key].confidence):
            best[key] = field
    return list(best.values())


def read_display(image: np.ndarray, config: ValidationConfig, detect: Detect) -> Result:
    # With no visible screen outline (photo filled by the display, or a
    # low-contrast panel) the whole photo is read. That is safe: the values
    # still have to pass parsing, validation and the confidence floor.
    box = find_screen(image)
    region = image if box is None else crop(image, box)

    # The same picture is read at more than one size. The text finder loses
    # digits that are very tall in the picture (a close-up cuts each row into
    # a top and a bottom half), and it loses small ones when the picture is
    # shrunk, so no single size suits every photo. The first size that gives
    # liters wins; later sizes only run on what would otherwise be a retake.
    passes = [(region, box, side, "" if i == 0 else f"{side}px ") for i, side in enumerate(config.ocr_sides)]
    if box is not None and config.whole_photo_fallback:
        # The crop may be the wrong area, or the pump may show its values in
        # separate panels, so the whole photo is read too.
        passes += [
            (image, None, side, "whole " if i == 0 else f"whole {side}px ")
            for i, side in enumerate(config.whole_photo_sides)
        ]

    first = None
    attempts: list[Attempt] = []
    done: set[tuple[int, int, int]] = set()
    for number, (source, screen_box, side, prefix) in enumerate(passes):
        picture = shrink(source, side)
        key = (id(source), *picture.shape[:2])
        if key in done:
            continue  # already that small: the same pixels again
        done.add(key)
        variants = ocr_variants(picture)
        if number > 0:
            # Only the two versions that keep the surroundings legible.
            variants = variants[:2]
        result = _read_region(
            picture, screen_box, [(prefix + name, v) for name, v in variants], config, detect, list(attempts)
        )
        attempts += result.attempts
        if first is None:
            first = result
        if result.liters is not None:
            return Result(
                result.status, result.reading, result.reasons, screen_box, attempts, result.unverified_liters
            )
    return Result(RETAKE, None, first.reasons, box, attempts)


def _read_region(
    region: np.ndarray,
    box: Box | None,
    variants: list[tuple[str, np.ndarray]],
    config: ValidationConfig,
    detect: Detect,
    earlier: list[Attempt] = (),
) -> Result:
    """Read one region of the photo through the given image versions.

    earlier: what readings of the same photo at other sizes saw. They cannot
    add support to an unverified result, only contradict it.
    """
    images = dict(variants)

    attempts = []
    result = None
    for name, variant in variants:
        attempts.append(_read(name, detect(variant), config))
        result = _decide(list(attempts), config, box)
        if result.status == ACCEPTED and config.stop_at_first_valid:
            result = _confirmed(result, result.reading.volume, images, config, detect)
            if result.status == ACCEPTED:
                return result
            attempts = list(result.attempts)  # keep the failed check in the report
    if result.status == ACCEPTED:  # only reached when not stopping early
        return _confirmed(result, result.reading.volume, images, config, detect)
    if result.status == REVIEW:  # three rows whose arithmetic holds
        return result

    if config.allow_two_row_displays:
        review = _decide_two_rows(attempts, config, box, earlier)
        if review is not None:
            # Liters must have been read twice. Two image versions agreeing
            # is enough; a single version needs the liters row re-read.
            agreeing = len(_confident_pairs(attempts, config))
            if agreeing >= TWO_ROW_MIN_AGREEING:
                return review
            return _confirmed(review, review.unverified_liters, images, config, detect)
    return result


def _read(name: str, detections: list[Detection], config: ValidationConfig) -> Attempt:
    other_numbers: list[Field] = []
    try:
        amount, liters, price = parse_layout(
            detections,
            config.labels,
            source=name,
            infer_decimals=config.infer_decimal_points,
            liters_decimals=config.inferred_liters_decimals,
            price_candidates=other_numbers,
        )
    except ParseError as e:
        return Attempt(name, detections, None, [PARSE_FAILED], f"could not parse: {e}")
    if price is None:
        # No number carries a price label. If exactly one other number in the
        # photo makes liters x price = total, that number is the price: a
        # stray number satisfying the equation by chance is very unlikely.
        fitting = [
            p for p in other_numbers if not validate(amount.value, liters.value, p.value, config)
        ]
        # The same price shown twice (the grade list and the selected grade)
        # is one price, not two candidates.
        if len({p.value for p in fitting}) == 1:
            price = max(fitting, key=lambda p: p.confidence)
    if price is None:
        detail = f"two rows only: {amount.value} | {liters.value}"
        return Attempt(name, detections, None, [PARSE_FAILED], detail, (amount, liters))
    # liters x price = total holds just as well with the two swapped, so the
    # arithmetic cannot say which is which. A known pump price can: if the
    # "liters" is exactly a price from fuel_prices.yaml and the "price" is
    # not, the rows were taken the wrong way round.
    known = set(config.grades.values())
    if (
        liters.value in known
        and price.value not in known
        and not validate(amount.value, price.value, liters.value, config)
    ):
        liters, price = price, liters
    reading = DisplayReading(amount, liters, price)
    failures = validate(reading.amount.value, reading.volume.value, reading.price.value, config)
    return Attempt(name, detections, reading, failures)


def _confirmed(
    result: Result,
    liters: Field,
    images: dict[str, np.ndarray],
    config: ValidationConfig,
    detect: Detect,
) -> Result:
    """Read the liters row again on its own; it must show the same value.

    The second read sees only that one line, at a different scale and
    without the other rows around it, so it is a partly independent check
    on the one value the app returns. Returns the result unchanged if the
    check is off, with the check appended if it passes, or a retake.
    """
    if not config.confirm_liters_row or liters.box is None:
        return result
    detections = detect(row_crop(images[liters.source], liters.box))
    # The row crop often takes in the word beside the number, and the two are
    # then read as one piece of text ("50,960LITRE"). The number is the part
    # that counts; letters on either side of it are dropped.
    texts = []
    for d in detections:
        glued = _NUMBER_WITH_WORDS.fullmatch("".join(d.text.split()))
        texts.append(number_text(glued.group("number") if glued else d.text))
    wanted = str(liters.value)
    if liters.inferred:
        # The first read had no decimal point, so compare the digits only.
        wanted, texts = wanted.replace(".", ""), [t.replace(".", "") for t in texts]
    confirmed = wanted in texts
    check = Attempt(
        LITERS_ROW,
        detections,
        None,
        [] if confirmed else [LITERS_NOT_CONFIRMED],
        f"liters row re-read: {texts}",
    )
    attempts = result.attempts + [check]
    if confirmed:
        return Result(
            result.status, result.reading, result.reasons, result.screen, attempts,
            result.unverified_liters,
        )
    return Result(RETAKE, None, [LITERS_NOT_CONFIRMED], result.screen, attempts)


def _decide(attempts: list[Attempt], config: ValidationConfig, box: Box | None) -> Result:
    """The outcome given the attempts made so far."""
    # A variant can read one row correctly and another row wrongly, so the
    # rows are combined across variants: every amount read by any variant is
    # tried with every volume and every price.
    readings = [a.reading for a in attempts if a.reading is not None]
    # A version that read only the total and the liters still says what it
    # saw in those two rows. Including it matters for safety: 21.65 and 21.66
    # liters can both satisfy the arithmetic, and only a second version
    # reading the other value reveals that one of them is a misread.
    pairs = [a.two_rows for a in attempts if a.two_rows is not None]
    valid = [
        DisplayReading(amount, volume, price)
        for amount, volume, price in product(
            _candidates([r.amount for r in readings] + [p[0] for p in pairs]),
            _candidates([r.volume for r in readings] + [p[1] for p in pairs]),
            _candidates(r.price for r in readings),
        )
        if not validate(amount.value, volume.value, price.value, config)
    ]
    if not valid:
        reasons = list(dict.fromkeys(f for a in attempts for f in a.failures))
        return Result(RETAKE, None, reasons, box, attempts)
    # Two different combinations that both pass validation means at least one
    # wrong reading slipped through the rules, so trust neither.
    if len(valid) > 1:
        return Result(RETAKE, None, [CONFLICTING_READINGS], box, attempts)
    reading = valid[0]
    # The arithmetic holds. Whatever else is weak about the reading makes
    # it a "review": liters is returned, but not called verified.
    flags = []
    if reading.confidence < config.min_confidence:
        floor = config.review_confidence
        if floor is None or reading.confidence < floor:
            return Result(RETAKE, None, [LOW_CONFIDENCE], box, attempts)
        flags.append(LOW_CONFIDENCE)
    if reading.amount.inferred or reading.volume.inferred or reading.price.inferred:
        # The app put a decimal point into at least one of the numbers.
        flags.append(DECIMAL_POINT_INFERRED)
    if flags:
        return Result(REVIEW, None, flags, box, attempts, reading.volume)
    return Result(ACCEPTED, reading, [], box, attempts)


def _confident_pairs(
    attempts: list[Attempt], config: ValidationConfig
) -> list[tuple[Field, Field]]:
    """The two-row reads whose numbers were both read confidently. A hazy
    read counts neither as support nor as disagreement."""
    return [
        a.two_rows
        for a in attempts
        if a.two_rows is not None
        and min(a.two_rows[0].confidence, a.two_rows[1].confidence) >= config.min_confidence
    ]


def _decide_two_rows(
    attempts: list[Attempt], config: ValidationConfig, box: Box | None, earlier: list[Attempt] = ()
) -> Result | None:
    """A "review" result for a display with only total and liters, or None.

    There is no price row to check liters against, so the bar is different:
    liters must be read twice with the same result, every image version that
    read two numbers confidently must agree, no version may have seen a third number,
    and total / liters must be a plausible price.
    """
    pairs = _confident_pairs(attempts, config)
    seen = {(str(amount.value), str(liters.value)) for amount, liters in pairs}
    # A reading of the same photo at another size counts against this one if
    # it found the same total with a different number under it: at a size
    # where the liters row is missed, an advert's price further down can pass
    # for the second row. Other readings at other sizes say nothing: at the
    # wrong size the digits come out as noise.
    totals = {amount for amount, _ in seen}
    for a in earlier:
        rows = [a.two_rows] if a.two_rows else [(a.reading.amount, a.reading.volume)] if a.reading else []
        for amount, liters in rows:
            if (
                str(amount.value) in totals
                and (str(amount.value), str(liters.value)) not in seen
                and liters.confidence >= config.min_confidence
            ):
                return None
    for a in attempts:
        if a.reading is None:
            continue
        # Some version saw three numbers. If the third was read confidently,
        # this is a three-row display whose arithmetic failed. A hazy third
        # number beside the same total and liters is a stray mark or a price
        # that could not be read: it tells nothing either way.
        same = (str(a.reading.amount.value), str(a.reading.volume.value)) in seen
        if not same or a.reading.price.confidence >= config.min_confidence:
            return None
    # Liters must be read twice. The second read is either another image
    # version or, when it is switched on, the re-read of the liters row that
    # the caller runs on whatever this returns.
    needed = 1 if config.confirm_liters_row else TWO_ROW_MIN_AGREEING
    if len(pairs) < needed:
        return None
    if len(seen) != 1:
        return None  # the versions disagree
    amount = max((p[0] for p in pairs), key=lambda f: f.confidence)
    liters = max((p[1] for p in pairs), key=lambda f: f.confidence)

    low, high = config.two_row_price_range
    plausible = (
        -amount.value.as_tuple().exponent in config.amount_decimals
        and config.min_volume <= liters.value <= config.max_volume
        and -liters.value.as_tuple().exponent in config.volume_decimals
        and low <= amount.value / liters.value <= high
        and min(amount.confidence, liters.confidence) >= config.min_confidence
    )
    if not plausible:
        return None
    flags = [NO_PRICE_ROW]
    if amount.inferred or liters.inferred:
        flags.append(DECIMAL_POINT_INFERRED)
    return Result(REVIEW, None, flags, box, attempts, liters)
