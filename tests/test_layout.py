from decimal import Decimal as D

import pytest

from app.ocr.detection import Detection
from app.ocr.layout import DEFAULT_LABELS, Labels, parse_layout
from app.ocr.parse import ParseError
from app.settings import load_validation_config


def det(text, x1, y1, x2, y2, confidence=0.99):
    return Detection(text, confidence, (x1, y1, x2, y2))


def values(result):
    return tuple(None if f is None else str(f.value) for f in result)


# What PaddleOCR read on a real Aramco pump photo: total and liters stacked
# on the right, the price per liter in a separate panel on the left.
ARAMCO = [
    det("10.00", 604, 459, 857, 592, 0.970),
    det("GAS 91", 170, 494, 300, 541),
    det("SAR", 846, 515, 931, 557),
    det("218", 142, 570, 222, 619, 0.973),  # the price, read without its decimal point
    det("SAR/L", 216, 573, 330, 622),
    det("4.59", 684, 593, 841, 696, 0.991),
    det("aramco", 576, 799, 729, 853),
    det("2", 0, 814, 60, 911),
]


def test_real_aramco_layout_gives_total_and_liters():
    assert values(parse_layout(ARAMCO)) == ("10.00", "4.59", None)


def test_real_aramco_price_without_its_decimal_point_can_be_restored():
    amount, liters, price = parse_layout(ARAMCO, infer_decimals=True)
    assert (str(amount.value), str(liters.value), str(price.value)) == ("10.00", "4.59", "2.18")
    assert price.inferred and not liters.inferred


def test_price_in_a_separate_panel_is_found_by_its_label():
    readable_price = [d if d.text != "218" else det("2.18", 142, 570, 222, 619) for d in ARAMCO]
    assert values(parse_layout(readable_price)) == ("10.00", "4.59", "2.18")


def test_no_labels_falls_back_to_row_order():
    rows = [det("50.00", 80, 0, 160, 40), det("22.93", 80, 50, 160, 90), det("2.18", 80, 100, 160, 140)]
    assert values(parse_layout(rows)) == ("50.00", "22.93", "2.18")
    assert values(parse_layout(rows[:2])) == ("50.00", "22.93", None)


def test_labels_decide_when_the_rows_are_in_another_order():
    # Liters on top, total below, price last: position alone would get the
    # first two the wrong way round.
    rows = [
        det("22.93", 80, 0, 160, 40), det("L", 170, 5, 190, 35),
        det("50.00", 80, 50, 160, 90), det("SAR", 170, 55, 210, 85),
        det("2.18", 80, 100, 160, 140),
    ]
    assert values(parse_layout(rows)) == ("50.00", "22.93", "2.18")


def test_one_label_is_enough_to_reorder():
    # Total, price, liters: only the liters row is labelled. Liters and price
    # satisfy the arithmetic either way round, so the label is what counts.
    rows = [
        det("50.00", 80, 0, 160, 40),
        det("2.18", 80, 50, 160, 90),
        det("22.93", 80, 100, 160, 140), det("LITERS", 170, 105, 240, 135),
    ]
    assert values(parse_layout(rows)) == ("50.00", "22.93", "2.18")


def test_label_attached_to_the_number_is_split_off():
    rows = [det("SAR179.85", 40, 0, 200, 40), det("LTR77.19", 40, 50, 200, 90), det("2.33", 90, 100, 200, 140)]
    assert values(parse_layout(rows)) == ("179.85", "77.19", "2.33")
    assert values(parse_layout([det("10.00", 80, 0, 160, 40), det("4.59L", 80, 50, 175, 90)])) == ("10.00", "4.59", None)


def test_unknown_letters_attached_to_a_number_are_not_stripped():
    rows = [det("PPP30.00", 40, 0, 200, 40), det("12.87", 80, 50, 200, 90), det("2.33", 80, 100, 200, 140)]
    with pytest.raises(ParseError):
        parse_layout(rows)


def test_label_between_two_rows_is_ignored_not_guessed():
    # Seen on a tiny real photo: the OCR box of "L" overlaps both the liters
    # row and the price row. An ambiguous label must not reorder anything.
    rows = [
        det("50.00", 83, 0, 160, 45), det("22.93", 81, 23, 163, 78), det("2.18", 82, 59, 150, 109),
        det("P", 33, 23, 57, 55), det("L", 35, 58, 55, 87),
    ]
    assert values(parse_layout(rows)) == ("50.00", "22.93", "2.18")


def test_label_far_from_any_number_is_ignored():
    rows = [det("10.00", 600, 0, 700, 40), det("4.59", 600, 50, 700, 90), det("L", 0, 55, 20, 85)]
    assert values(parse_layout(rows)) == ("10.00", "4.59", None)


def test_low_confidence_label_is_ignored():
    rows = [det("22.93", 80, 0, 160, 40), det("50.00", 80, 50, 160, 90), det("L", 170, 55, 190, 85, 0.4)]
    assert values(parse_layout(rows)) == ("22.93", "50.00", None)


def test_two_numbers_with_the_same_label_are_refused():
    rows = [det("22.93", 80, 0, 160, 40), det("L", 170, 5, 190, 35), det("50.00", 80, 50, 160, 90), det("L", 170, 55, 190, 85)]
    with pytest.raises(ParseError, match="labelled as liters"):
        parse_layout(rows)


@pytest.mark.parametrize("word", ["Gallons", "Galions", "GAL"])
def test_gallons_display_is_refused(word):
    rows = [det("25.00", 80, 0, 160, 40), det("6.099", 80, 50, 160, 90), det(word, 20, 100, 70, 120)]
    with pytest.raises(ParseError, match="gallons"):
        parse_layout(rows)


def test_small_numbers_elsewhere_in_the_photo_are_ignored():
    # The total and liters are found by size; a small decimal number
    # somewhere else (a sticker, a second price) does not disturb them.
    rows = ARAMCO + [det("7.77", 400, 700, 480, 740)]
    rows = [d if d.text != "218" else det("2.18", 142, 570, 222, 619) for d in rows]
    assert values(parse_layout(rows)) == ("10.00", "4.59", "2.18")


# What PaddleOCR read on a rendered pump photo taken from a distance: the
# decimal points of the two big numbers are missing, the keypad is read as
# digits, and the key labelled SAR sits next to the 9 key.
WHOLE_PUMP = [
    det("SA", 1290, 300, 1330, 330), det("10841", 900, 280, 1280, 407),
    det("GAS 95", 520, 330, 700, 380), det("4653", 980, 420, 1280, 512), det("L", 1290, 450, 1310, 480),
    det("2.33", 480, 400, 580, 449), det("SAR/L", 590, 405, 720, 450),
    det("3", 420, 560, 480, 640), det("PLEASE TURN OFF ENGINE", 620, 600, 1100, 640),
    det("123", 1340, 600, 1420, 630), det("456", 1340, 640, 1420, 670), det("789", 1340, 680, 1420, 710),
    det("SAR", 1430, 680, 1470, 710), det("CLR", 1340, 720, 1370, 750), det("Vol.", 1430, 720, 1470, 750),
]


def test_missing_decimal_points_are_put_back_when_allowed():
    amount, liters, price = parse_layout(WHOLE_PUMP, infer_decimals=True)
    assert (str(amount.value), str(liters.value), str(price.value)) == ("108.41", "46.53", "2.33")
    assert (amount.inferred, liters.inferred, price.inferred) == (True, True, False)


def test_missing_decimal_points_are_not_put_back_by_default():
    with pytest.raises(ParseError):
        parse_layout(WHOLE_PUMP)


def test_inferred_liters_can_have_three_decimals():
    _, liters, _ = parse_layout(WHOLE_PUMP, infer_decimals=True, liters_decimals=3)
    assert str(liters.value) == "4.653"


def test_keypad_digits_and_keys_do_not_count():
    # "123", "456", "789" and the SAR key are small and off to the side.
    amount, liters, price = parse_layout(WHOLE_PUMP, infer_decimals=True)
    assert amount.box == (900, 280, 1280, 407) and liters.box == (980, 420, 1280, 512)


def test_price_label_is_matched_on_a_tilted_photo():
    # The label sits lower than its number, as on a pump seen at an angle.
    rows = [
        det("29.14", 1140, 200, 1290, 310), det("12.45", 1180, 310, 1290, 380),
        det("2.34", 860, 180, 950, 240), det("SAR/L", 960, 215, 1050, 270),
        det("6", 815, 340, 865, 410),
    ]
    assert values(parse_layout(rows)) == ("29.14", "12.45", "2.34")


def test_price_label_between_two_small_numbers_is_ignored():
    rows = [
        det("29.14", 1140, 200, 1290, 310), det("12.45", 1180, 310, 1290, 380),
        det("2.34", 800, 200, 890, 250), det("SAR/L", 900, 205, 990, 250), det("1.66", 1000, 200, 1090, 250),
    ]
    assert values(parse_layout(rows)) == ("29.14", "12.45", None)


def test_smaller_price_row_under_the_two_big_numbers_is_the_third_row():
    rows = [det("50.00", 80, 0, 200, 60), det("22.93", 80, 70, 200, 130), det("2.18", 120, 140, 200, 170)]
    assert values(parse_layout(rows)) == ("50.00", "22.93", "2.18")


def test_unreadable_third_row_makes_a_two_row_reading_unsafe():
    rows = [det("50.00", 80, 0, 200, 60), det("22.93", 80, 70, 200, 130), det("2 18", 120, 140, 200, 170)]
    with pytest.raises(ParseError, match="other digits"):
        parse_layout(rows)


def test_label_words_are_parameters():
    custom = Labels.from_lists(amount=["DH"], liters=["Vol."], price=["DH/L"], gallons_prefixes=[])
    rows = [det("22.93", 80, 0, 160, 40), det("VOL", 170, 5, 210, 35), det("50.00", 80, 50, 160, 90), det("dh", 170, 55, 200, 85)]
    assert values(parse_layout(rows, custom)) == ("50.00", "22.93", None)
    assert values(parse_layout(rows, DEFAULT_LABELS)) == ("22.93", "50.00", None)  # unknown words: row order


def test_labels_file_loads():
    labels = load_validation_config().labels
    assert labels.role("Liters") == "liters" and labels.role("L") == "liters"
    assert labels.role("SAR") == "amount" and labels.role("sar/l") == "price"
    assert labels.role("Preis je Liter") == "price"
    assert labels.role("TOTAL") == "amount" and labels.role("TOTALS") is None
    assert labels.is_gallons("Galions") and not labels.is_gallons("Liter")
    assert values(parse_layout(ARAMCO, labels)) == ("10.00", "4.59", None)
    assert parse_layout(ARAMCO, labels)[1].value == D("4.59")


def test_a_tall_misread_token_standing_alone_is_not_the_display():
    # Seen on a rendered pump: the speaker grille read as "11111", taller
    # than the real numbers and off to the side.
    rows = [det("11111", 200, 69, 293, 147, 0.85), det("13,24", 130, 131, 188, 175), det("4,90", 151, 150, 189, 181)]
    assert values(parse_layout(rows, infer_decimals=True)) == ("13.24", "4.90", None)


def test_other_numbers_are_offered_as_price_candidates():
    rows = [det("22.33", 372, 185, 418, 212), det("10.29", 390, 205, 418, 222), det("2,17", 287, 184, 324, 202)]
    others = []
    assert values(parse_layout(rows, price_candidates=others)) == ("22.33", "10.29", None)
    assert [str(f.value) for f in others] == ["2.17"]


def test_label_with_one_stray_digit_is_not_a_third_row():
    # Seen with the trained model: the label TOTAL read as "TOTAL1".
    rows = [
        det("TOTAL1", 99, 0, 238, 59, 0.89), det("10.14", 129, 49, 472, 242),
        det("7.050", 130, 212, 375, 350), det("LITRES", 101, 332, 221, 399),
    ]
    assert values(parse_layout(rows)) == ("10.14", "7.050", None)


@pytest.mark.parametrize("unread_row", ["2 18", "5O.00", "PPP30.00", "2293", "21B"])
def test_mostly_digit_token_in_the_column_is_still_a_third_row(unread_row):
    rows = [det("50.00", 80, 0, 200, 60), det("22.93", 80, 70, 200, 130), det(unread_row, 80, 140, 200, 200)]
    with pytest.raises(ParseError):
        parse_layout(rows)


def test_arabic_labels_match_whatever_the_spelling():
    from app.settings import load_validation_config

    labels = load_validation_config().labels
    assert labels.role("الكمية باللتر") == "liters"
    assert labels.role("الكـمـيـة بـالـلـتـر") == "liters"  # stretched letters
    assert labels.role("الإجمالي") == labels.role("الاجمالي") == "amount"
    assert labels.role("سِعر اللتر") == "price"  # with a vowel mark
    assert labels.role("اللتر") is None  # half of "price per liter": says nothing
    assert labels.role("ريال") is None
