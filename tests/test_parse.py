from decimal import Decimal

import pytest

from app.ocr.detection import Detection
from app.ocr.parse import ParseError, parse_display, parse_two_rows

# Actual PaddleOCR output for the first real pump photo (display 50.00 / 22.93 / 2.18).
REAL_PHOTO = [
    Detection("50.00", 0.991, (83, 0, 160, 45)),
    Detection("P", 0.993, (33, 23, 57, 55)),
    Detection("", 0.000, (0, 29, 37, 63)),
    Detection("22.93", 0.987, (81, 23, 163, 78)),
    Detection("L", 0.978, (35, 58, 55, 87)),
    Detection("P", 0.211, (0, 61, 39, 95)),
    Detection("2.18", 0.976, (82, 59, 150, 109)),
    Detection("Pr", 0.738, (0, 93, 33, 121)),
    Detection("jioo", 0.817, (42, 145, 110, 185)),
    Detection("91", 1.000, (42, 166, 114, 220)),
]


def column(*texts, x=(80, 160), confidence=0.99):
    """Detections stacked top to bottom in one column."""
    return [
        Detection(text, confidence, (x[0], i * 50, x[1], i * 50 + 40))
        for i, text in enumerate(texts)
    ]


def test_real_photo():
    reading = parse_display(REAL_PHOTO)
    assert reading.amount.value == Decimal("50.00")
    assert reading.volume.value == Decimal("22.93")
    assert reading.price.value == Decimal("2.18")


def test_confidence_is_lowest_field():
    reading = parse_display(REAL_PHOTO)
    assert reading.price.confidence == 0.976
    assert reading.confidence == 0.976


def test_order_comes_from_position_not_list_order():
    reading = parse_display(list(reversed(column("50.00", "22.93", "2.18"))))
    assert reading.amount.value == Decimal("50.00")
    assert reading.price.value == Decimal("2.18")


def test_decimal_places_are_preserved():
    reading = parse_display(column("116.50", "50.000", "2.33"))
    assert str(reading.amount.value) == "116.50"
    assert str(reading.volume.value) == "50.000"


def test_spaces_inside_a_number_are_ignored():
    reading = parse_display(column("50.00", "22. 93", "2.18"))
    assert reading.volume.value == Decimal("22.93")


def test_number_without_decimal_point_is_not_used():
    with pytest.raises(ParseError, match="found 2"):
        parse_display(column("5000", "22.93", "2.18"))


@pytest.mark.parametrize("bad", ["5O.00", "50,00,1", "1,234.50", "50.00.1", "50.", ".50", ",50", "-50.00", "SAR50.00"])
def test_malformed_number_is_not_corrected(bad):
    with pytest.raises(ParseError):
        parse_display(column(bad, "22.93", "2.18"))


def test_no_detections():
    with pytest.raises(ParseError, match="found 0"):
        parse_display([])


def test_stray_decimal_outside_the_column_is_ignored():
    stray = Detection("2.33", 0.99, (300, 60, 360, 100))
    reading = parse_display(column("50.00", "22.93", "2.18") + [stray])
    assert reading.price.value == Decimal("2.18")


def test_numbers_side_by_side_are_not_a_display():
    row = [Detection(t, 0.99, (i * 100, 0, i * 100 + 80, 40)) for i, t in enumerate(["50.00", "22.93", "2.18"])]
    with pytest.raises(ParseError, match="column"):
        parse_display(row)


def test_fourth_number_in_the_column_is_ambiguous():
    with pytest.raises(ParseError, match="more than one"):
        parse_display(column("50.00", "22.93", "2.18", "0.00"))


def test_fields_remember_where_they_were_read():
    reading = parse_display(REAL_PHOTO, source="faded")
    assert reading.volume.box == (81, 23, 163, 78)
    assert reading.volume.source == "faded"


def test_two_rows_are_total_then_liters():
    amount, liters = parse_two_rows(list(reversed(column("57.62", "33.899"))))
    assert (amount.value, liters.value) == (Decimal("57.62"), Decimal("33.899"))


def test_two_rows_ignore_labels_beside_the_numbers():
    labels = [Detection("Sale", 1.0, (0, 0, 60, 40)), Detection("Litres", 1.0, (0, 50, 60, 90))]
    amount, liters = parse_two_rows(column("57.62", "33.899") + labels)
    assert liters.value == Decimal("33.899")


@pytest.mark.parametrize("texts", [("57.62",), ("57.62", "33.899", "1.70"), ("5762", "33.899")])
def test_two_rows_need_exactly_two_decimal_numbers(texts):
    with pytest.raises(ParseError):
        parse_two_rows(column(*texts))


def test_two_rows_refuse_other_digits_in_the_column():
    with pytest.raises(ParseError, match="other digits"):
        parse_two_rows(column("5000", "22.93", "2.18"))


def test_two_rows_side_by_side_are_refused():
    row = [Detection(t, 0.99, (i * 100, 0, i * 100 + 80, 40)) for i, t in enumerate(["57.62", "33.899"])]
    with pytest.raises(ParseError, match="column"):
        parse_two_rows(row)


def test_decimal_comma_is_read_as_a_decimal_point():
    reading = parse_display(column("37,01", "25,37", "1,459"))
    assert reading.amount.value == Decimal("37.01")
    assert str(reading.volume.value) == "25.37"
    amount, liters = parse_two_rows(column("20,53", "11,63"))
    assert liters.value == Decimal("11.63")
