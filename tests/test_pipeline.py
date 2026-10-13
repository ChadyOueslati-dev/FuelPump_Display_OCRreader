from dataclasses import replace
from decimal import Decimal as D

import cv2
import numpy as np

from app.ocr.detection import Detection
from app.pipeline import (
    ACCEPTED,
    CONFLICTING_READINGS,
    DECIMAL_POINT_INFERRED,
    LITERS_NOT_CONFIRMED,
    LITERS_ROW,
    LOW_CONFIDENCE,
    NO_PRICE_ROW,
    PARSE_FAILED,
    RETAKE,
    REVIEW,
    read_display,
)
from app.settings import ValidationConfig
from app.validation.rules import AMOUNT_MISMATCH, PRICE_NOT_A_FUEL_GRADE

CONFIG = ValidationConfig(
    grades={"91": D("2.18"), "95": D("2.33"), "diesel": D("1.79")},
    amount_tolerance=D("0.02"),
    min_volume=D("1"),
    max_volume=D("150"),
    min_confidence=0.95,
)


def photo_with_screen():
    image = np.full((300, 400, 3), 90, np.uint8)
    cv2.rectangle(image, (100, 50), (300, 250), (255, 240, 230), -1)
    return image


def column(*texts, confidence=0.99):
    return [
        Detection(text, confidence, (80, i * 50, 160, i * 50 + 40))
        for i, text in enumerate(texts)
    ]


def fake_ocr(*outputs):
    """A detect() that returns the given outputs in order: original, faded,
    binary, thick, then the liters row re-read. Reads nothing once they run out."""
    remaining = list(outputs)
    return lambda image: remaining.pop(0) if remaining else []


GOOD = column("50.00", "22.93", "2.18")
GHOST = column("50.00", "22.93", "888.18")  # unlit segments read as digits
GARBLED = column("5000", "22.93", "2.18")


def test_all_variants_agree():
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(GOOD, GOOD, GOOD))
    assert result.status == ACCEPTED
    assert result.reading.amount.value == D("50.00")
    assert result.reasons == []


def test_one_valid_variant_is_enough():
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(GHOST, GARBLED, GOOD))
    assert result.status == ACCEPTED
    assert result.reading.price.value == D("2.18")


def test_rows_are_combined_across_variants():
    # Seen on a real photo: the original misreads the price row (ghost
    # segments), the cleaned variants lose the thin leading 1 of the volume.
    original = column("30.00", "12.87", "8882.33")
    cleaned = column("30.00", "2.87", "2.33")
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(original, cleaned, cleaned))
    assert result.status == ACCEPTED
    assert result.reading.volume.value == D("12.87")
    assert result.reading.price.value == D("2.33")


def test_highest_confidence_valid_reading_is_returned():
    weak = column("50.00", "22.93", "2.18", confidence=0.80)
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(weak, GOOD, GARBLED))
    assert result.reading.confidence == 0.99


def test_valid_reading_below_min_confidence_is_a_retake():
    weak = column("50.00", "22.93", "2.18", confidence=0.949)
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(weak, weak, weak))
    assert result.status == RETAKE
    assert result.reading is None
    assert result.reasons == [LOW_CONFIDENCE]


def test_one_weak_row_is_enough_for_a_retake():
    reading = column("50.00", "22.93", "2.18")
    reading[1] = Detection("22.93", 0.90, reading[1].box)
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(reading, reading, reading))
    assert result.reasons == [LOW_CONFIDENCE]


def test_confidence_exactly_at_the_minimum_is_accepted():
    edge = column("50.00", "22.93", "2.18", confidence=0.95)
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(edge, edge, edge))
    assert result.status == ACCEPTED


def test_no_valid_variant_is_a_retake_with_no_values():
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(GHOST, GARBLED, GHOST))
    assert result.status == RETAKE
    assert result.reading is None
    assert result.reasons == [PRICE_NOT_A_FUEL_GRADE, AMOUNT_MISMATCH, PARSE_FAILED]


def test_two_different_valid_readings_are_a_retake():
    other = column("21.80", "10.00", "2.18")
    result = read_display(photo_with_screen(), CONFIG, fake_ocr(GOOD, other, GOOD))
    assert result.status == RETAKE
    assert result.reading is None
    assert result.reasons == [CONFLICTING_READINGS]


def test_stop_at_first_valid_skips_the_remaining_reads():
    fast = replace(CONFIG, stop_at_first_valid=True)
    calls = []

    def detect(image):
        calls.append(1)
        return GOOD

    result = read_display(photo_with_screen(), fast, detect)
    assert result.status == ACCEPTED
    assert len(calls) == 1
    assert [a.variant for a in result.attempts] == ["original"]


def test_stop_at_first_valid_keeps_reading_until_one_validates():
    fast = replace(CONFIG, stop_at_first_valid=True)
    result = read_display(photo_with_screen(), fast, fake_ocr(GHOST, GOOD, GARBLED))
    assert result.status == ACCEPTED
    assert [a.variant for a in result.attempts] == ["original", "faded"]


def test_stop_at_first_valid_does_not_stop_on_low_confidence():
    fast = replace(CONFIG, stop_at_first_valid=True)
    weak = column("50.00", "22.93", "2.18", confidence=0.5)
    result = read_display(photo_with_screen(), fast, fake_ocr(weak, weak, GOOD))
    assert result.status == ACCEPTED
    assert [a.variant for a in result.attempts] == ["original", "faded", "binary"]


def test_large_screen_is_shrunk_before_ocr():
    image = np.full((1500, 1200, 3), 90, np.uint8)
    cv2.rectangle(image, (100, 100), (1100, 1400), (255, 240, 230), -1)
    seen = []

    def detect(variant):
        seen.append(max(variant.shape[:2]))
        return GOOD

    read_display(image, CONFIG, detect)
    assert seen == [CONFIG.ocr_max_side] * 4


def test_without_a_screen_outline_the_whole_photo_is_read():
    blank = np.full((300, 400, 3), 90, np.uint8)
    seen = []

    def detect(image):
        seen.append(image.shape)
        return GOOD

    result = read_display(blank, CONFIG, detect)
    assert result.status == ACCEPTED
    assert result.screen is None
    assert seen == [(300, 400, 3)] * 4


def test_whole_photo_reading_still_has_to_validate():
    blank = np.full((300, 400, 3), 90, np.uint8)
    result = read_display(blank, CONFIG, fake_ocr(GHOST, GHOST, GHOST))
    assert result.status == RETAKE
    assert result.reading is None


CONFIRM = replace(CONFIG, confirm_liters_row=True)


def test_liters_row_reread_confirms_the_value():
    seen = []

    def detect(image):
        seen.append(image.shape)
        # The row re-read sees one line: only the liters value.
        return GOOD if len(seen) <= 4 else [Detection("22.93", 0.99, (10, 10, 90, 40))]

    result = read_display(photo_with_screen(), CONFIRM, detect)
    assert result.status == ACCEPTED
    assert result.reading.volume.value == D("22.93")
    assert result.attempts[-1].variant == LITERS_ROW
    assert result.attempts[-1].failures == []
    assert len(seen) == 5  # four variants + the row


def test_liters_row_reread_disagreeing_is_a_retake():
    row = [Detection("22.98", 0.99, (10, 10, 90, 40))]
    result = read_display(photo_with_screen(), CONFIRM, fake_ocr(GOOD, GOOD, GOOD, GOOD, row))
    assert result.status == RETAKE
    assert result.reading is None
    assert result.reasons == [LITERS_NOT_CONFIRMED]


def test_liters_row_reread_reading_nothing_is_a_retake():
    result = read_display(photo_with_screen(), CONFIRM, fake_ocr(GOOD, GOOD, GOOD, GOOD, []))
    assert result.reasons == [LITERS_NOT_CONFIRMED]


def test_with_early_stop_a_failed_confirmation_tries_the_next_variant():
    fast = replace(CONFIRM, stop_at_first_valid=True)
    wrong_row = [Detection("2.93", 0.99, (10, 10, 90, 40))]
    good_row = [Detection("22.93", 0.99, (10, 10, 90, 40))]
    # original validates, its row check fails; faded validates, its row check passes
    result = read_display(photo_with_screen(), fast, fake_ocr(GOOD, wrong_row, GOOD, good_row))
    assert result.status == ACCEPTED
    assert [a.variant for a in result.attempts] == ["original", LITERS_ROW, "faded", LITERS_ROW]


def test_no_row_reread_on_a_retake():
    calls = []

    def detect(image):
        calls.append(1)
        return GHOST

    read_display(photo_with_screen(), CONFIRM, detect)
    assert len(calls) == 4


# Displays that show only the total and the liters (no price per liter).
TWO_ROW = replace(CONFIG, allow_two_row_displays=True)
PAIR = column("57.62", "33.899")


def test_two_row_display_read_the_same_by_two_versions_is_a_review():
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr([], [], PAIR, PAIR))
    assert result.status == REVIEW
    assert result.reasons == [NO_PRICE_ROW]
    assert result.liters.value == D("33.899")
    assert result.reading is None  # nothing was verified by arithmetic


def test_two_row_display_is_a_retake_when_the_feature_is_off():
    result = read_display(photo_with_screen(), CONFIG, fake_ocr([], [], PAIR, PAIR))
    assert result.status == RETAKE
    assert result.liters is None


def test_two_row_display_read_by_only_one_version_is_a_retake():
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr([], [], [], PAIR))
    assert result.status == RETAKE


def test_one_version_plus_a_confirming_liters_row_reread_is_a_review():
    # Seen on a real scan: only the thickened version could read the display.
    config = replace(TWO_ROW, confirm_liters_row=True)
    row = [Detection("33.899", 0.99, (10, 10, 90, 40))]
    result = read_display(photo_with_screen(), config, fake_ocr([], [], [], PAIR, row))
    assert result.status == REVIEW
    assert result.liters.value == D("33.899")

    no_confirmation = read_display(photo_with_screen(), config, fake_ocr([], [], [], PAIR, []))
    assert no_confirmation.status == RETAKE


def test_two_row_versions_that_disagree_are_a_retake():
    other = column("57.62", "33.890")
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr([], [], PAIR, other))
    assert result.status == RETAKE


def test_two_numbers_from_a_three_row_display_are_not_a_review():
    # One version saw three numbers, so this is a three-row display whose
    # reading failed validation, not a two-row display.
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr(GHOST, PAIR, PAIR, PAIR))
    assert result.status == RETAKE


def test_two_numbers_with_other_digits_in_the_column_are_not_a_review():
    # The third row was detected but not as a number: which rows the two
    # numbers are is unknown.
    partly = column("5000", "22.93", "2.18")
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr(partly, partly, partly, partly))
    assert result.status == RETAKE


def test_two_rows_that_are_really_liters_and_price_are_refused():
    # 22.93 / 2.18 = 10.5 per liter: not a plausible price, so these are the
    # wrong two rows (the total was not detected at all).
    wrong_rows = column("22.93", "2.18")
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr(wrong_rows, wrong_rows))
    assert result.status == RETAKE


def test_two_row_liters_below_min_confidence_is_a_retake():
    weak = column("57.62", "33.899", confidence=0.5)
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr(weak, weak, weak, weak))
    assert result.status == RETAKE


def test_two_agreeing_versions_do_not_need_the_liters_row_reread():
    config = replace(TWO_ROW, confirm_liters_row=True)
    calls = []

    def detect(image):
        calls.append(1)
        return PAIR if len(calls) in (3, 4) else []

    result = read_display(photo_with_screen(), config, detect)
    assert result.status == REVIEW
    assert len(calls) == 4  # the four versions, no extra read


def test_single_version_with_a_disagreeing_reread_says_why():
    config = replace(TWO_ROW, confirm_liters_row=True)
    bad_row = [Detection("33.890", 0.99, (10, 10, 90, 40))]
    result = read_display(photo_with_screen(), config, fake_ocr([], [], [], PAIR, bad_row))
    assert result.status == RETAKE and result.liters is None
    assert result.reasons == [LITERS_NOT_CONFIRMED]


def test_liters_row_reread_accepts_a_decimal_comma():
    config = replace(TWO_ROW, confirm_liters_row=True)
    pair = column("20,53", "11,63")
    row = [Detection("11,63", 0.99, (10, 10, 90, 40))]
    result = read_display(photo_with_screen(), config, fake_ocr([], [], pair, pair, row))
    assert result.status == REVIEW
    assert str(result.liters.value) == "11.63"


# When the cropped display gives nothing, the whole photo is read as well.
FALLBACK = replace(CONFIG, whole_photo_fallback=True)


def test_whole_photo_is_read_when_the_crop_gives_nothing():
    shapes = []

    def detect(image):
        shapes.append(image.shape[:2])
        return GOOD if image.shape[:2] == (300, 400) else []

    result = read_display(photo_with_screen(), FALLBACK, detect)
    assert result.status == ACCEPTED
    assert result.screen is None  # the value came from the whole photo
    assert [a.variant for a in result.attempts] == [
        "original", "faded", "binary", "thick", "whole original", "whole faded"
    ]
    assert shapes[-1] == (300, 400)


def test_whole_photo_is_not_read_when_the_crop_succeeds():
    calls = []

    def detect(image):
        calls.append(1)
        return GOOD

    result = read_display(photo_with_screen(), replace(FALLBACK, stop_at_first_valid=True), detect)
    assert result.status == ACCEPTED and len(calls) == 1


def test_fallback_that_also_fails_is_a_retake_with_all_attempts():
    result = read_display(photo_with_screen(), FALLBACK, lambda image: GHOST)
    assert result.status == RETAKE and result.liters is None
    assert len(result.attempts) == 6  # four on the crop, two on the whole photo
    assert result.screen is not None


def test_whole_photo_fallback_is_off_by_default():
    calls = []

    def detect(image):
        calls.append(1)
        return []

    read_display(photo_with_screen(), CONFIG, detect)
    assert len(calls) == 4


def test_labels_reorder_rows_in_the_pipeline():
    # Total, price, liters, with the liters row labelled. Without the label
    # the app would return 2.18 as the liters, and the arithmetic would pass.
    rows = column("50.00", "2.18", "22.93") + [Detection("LITERS", 0.99, (170, 105, 240, 135))]
    result = read_display(photo_with_screen(), CONFIG, lambda image: rows)
    assert result.status == ACCEPTED
    assert result.liters.value == D("22.93")


def test_gallons_display_is_a_retake():
    rows = column("25.00", "6.099") + [Detection("Gallons", 0.99, (20, 110, 70, 130))]
    result = read_display(photo_with_screen(), TWO_ROW, lambda image: rows)
    assert result.status == RETAKE and result.liters is None


# Numbers read without their decimal point.
INFER = replace(CONFIG, infer_decimal_points=True)
NO_POINTS = column("5000", "2293", "2.18")


def test_restored_decimal_points_that_fit_the_arithmetic_are_a_review():
    result = read_display(photo_with_screen(), INFER, lambda image: NO_POINTS)
    assert result.status == REVIEW
    assert result.reasons == [DECIMAL_POINT_INFERRED]
    assert str(result.liters.value) == "22.93"
    assert result.reading is None  # never "accepted"


def test_restored_decimal_points_that_break_the_arithmetic_are_a_retake():
    # The last digit of the total was not read: 500 -> 5.00, which no longer
    # equals 22.93 x 2.18.
    truncated = column("500", "2293", "2.18")
    result = read_display(photo_with_screen(), INFER, lambda image: truncated)
    assert result.status == RETAKE and result.liters is None


def test_a_version_that_read_the_points_makes_it_accepted():
    result = read_display(photo_with_screen(), INFER, fake_ocr(NO_POINTS, GOOD, NO_POINTS, NO_POINTS))
    assert result.status == ACCEPTED
    assert result.liters.inferred is False


def test_numbers_without_points_are_a_retake_when_restoring_is_off():
    result = read_display(photo_with_screen(), CONFIG, lambda image: NO_POINTS)
    assert result.status == RETAKE


def test_two_row_display_with_restored_points_carries_both_flags():
    config = replace(INFER, allow_two_row_displays=True)
    pair = column("5000", "3012")  # 50.00 / 30.12 = 1.66 per liter: plausible
    result = read_display(photo_with_screen(), config, lambda image: pair)
    assert result.status == REVIEW
    assert result.reasons == [NO_PRICE_ROW, DECIMAL_POINT_INFERRED]
    assert str(result.liters.value) == "30.12"


def test_liters_row_reread_compares_digits_when_the_point_was_restored():
    config = replace(INFER, allow_two_row_displays=True, confirm_liters_row=True)
    pair = column("5000", "3012")
    row = [Detection("30.12", 0.99, (10, 10, 90, 40))]
    result = read_display(photo_with_screen(), config, fake_ocr([], [], [], pair, row))
    assert result.status == REVIEW and str(result.liters.value) == "30.12"


def test_low_confidence_reading_whose_arithmetic_holds_can_be_a_review():
    lenient = replace(CONFIG, review_confidence=0.6)
    hazy = column("50.00", "22.93", "2.18", confidence=0.80)
    result = read_display(photo_with_screen(), lenient, lambda image: hazy)
    assert result.status == REVIEW
    assert result.reasons == [LOW_CONFIDENCE]
    assert result.liters.value == D("22.93") and result.reading is None

    too_hazy = column("50.00", "22.93", "2.18", confidence=0.55)
    assert read_display(photo_with_screen(), lenient, lambda image: too_hazy).status == RETAKE


def test_review_flags_combine():
    lenient = replace(INFER, review_confidence=0.6)
    hazy = column("5000", "2293", "2.18", confidence=0.80)
    result = read_display(photo_with_screen(), lenient, lambda image: hazy)
    assert result.reasons == [LOW_CONFIDENCE, DECIMAL_POINT_INFERRED]


def test_two_row_display_gets_no_low_confidence_allowance():
    lenient = replace(TWO_ROW, review_confidence=0.6)
    hazy = column("57.62", "33.899", confidence=0.80)
    assert read_display(photo_with_screen(), lenient, lambda image: hazy).status == RETAKE


def test_unlabelled_price_is_found_by_the_arithmetic():
    # The price sits apart from the two big numbers and its SAR/L label was
    # not read. 10.29 x 2.17 = 22.33, so 2.17 is the price.
    rows = [
        Detection("22.33", 0.96, (372, 185, 418, 212)), Detection("10.29", 0.98, (390, 205, 418, 222)),
        Detection("2,17", 0.97, (287, 184, 324, 202)), Detection("7.50", 0.97, (100, 400, 140, 420)),
    ]
    any_price = replace(CONFIG, check_fuel_price=False)
    result = read_display(photo_with_screen(), any_price, lambda image: rows)
    assert result.status == ACCEPTED
    assert result.liters.value == D("10.29")
    assert result.reading.price.value == D("2.17")


def test_two_different_numbers_fitting_the_arithmetic_is_not_a_price():
    # 1.00 L: both 2.18 and 2.19 are within the tolerance of the total
    rows = [
        Detection("2.18", 0.99, (372, 185, 418, 212)), Detection("1.00", 0.99, (390, 205, 418, 222)),
        Detection("2.18", 0.99, (287, 184, 324, 202)), Detection("2.19", 0.99, (100, 400, 140, 420)),
    ]
    any_price = replace(CONFIG, check_fuel_price=False)
    result = read_display(photo_with_screen(), any_price, lambda image: rows)
    assert result.status == RETAKE


def test_the_same_price_shown_twice_is_one_price():
    # Aramco shows the price in the grade list and again beside the liters.
    rows = [
        Detection("21.80", 0.99, (372, 185, 418, 212)), Detection("10.00", 0.99, (390, 205, 418, 222)),
        Detection("2.18", 0.99, (287, 184, 324, 202)), Detection("2.18", 0.99, (100, 400, 140, 420)),
    ]
    result = read_display(photo_with_screen(), CONFIG, lambda image: rows)
    assert result.status == ACCEPTED
    assert result.liters.value == D("10.00")


def test_rows_taken_the_wrong_way_round_are_put_right_by_a_known_price():
    # Read top to bottom this is total 30.00, liters 2.18, price 13.76, and
    # the arithmetic holds either way. 2.18 is a pump price; 13.76 is not.
    rows = column("30.00", "2.18", "13.76")
    any_price = replace(CONFIG, check_fuel_price=False)
    result = read_display(photo_with_screen(), any_price, lambda image: rows)
    assert result.status == ACCEPTED
    assert result.liters.value == D("13.76")
    assert result.reading.price.value == D("2.18")


def test_hazy_two_row_read_does_not_block_two_confident_ones():
    # Seen on a rendered pump: two versions read 100.00 | 36.90 clearly, a
    # third read 35.90 with low confidence.
    hazy = column("100.00", "35.90", confidence=0.81)
    clear = column("100.00", "36.90")
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr(clear, clear, hazy, []))
    assert result.status == REVIEW
    assert result.liters.value == D("36.90")


def test_confident_two_row_reads_that_disagree_are_still_a_retake():
    other = column("100.00", "35.90")
    clear = column("100.00", "36.90")
    result = read_display(photo_with_screen(), TWO_ROW, fake_ocr(clear, clear, other, []))
    assert result.status == RETAKE


def test_two_row_read_that_contradicts_the_liters_makes_it_a_retake():
    # Seen on a rendered pump: one version read 50.24 | 21.65 | 2.32, which
    # satisfies the arithmetic, while another read the liters as 21.66 (the
    # true value) but missed the price. Both fit, so neither is trusted.
    any_price = replace(CONFIG, check_fuel_price=False)
    three = column("50.24", "21.65", "2.32")
    two = column("50.24", "21.66")
    result = read_display(photo_with_screen(), any_price, fake_ocr(three, two, [], []))
    assert result.status == RETAKE
    assert result.reasons == [CONFLICTING_READINGS]
    assert result.liters is None


def test_two_row_read_that_agrees_with_the_liters_changes_nothing():
    any_price = replace(CONFIG, check_fuel_price=False)
    three = column("50.24", "21.66", "2.32")
    two = column("50.24", "21.66")
    result = read_display(photo_with_screen(), any_price, fake_ocr(three, two, [], []))
    assert result.status == ACCEPTED


def test_settings_file_has_the_low_confidence_allowance_off():
    from app.settings import load_validation_config

    assert load_validation_config().review_confidence is None


# Close-ups: digits that are very tall in the picture are not found, so a
# picture that gave nothing is read again at a smaller size.
SIZES = replace(CONFIG, extra_ocr_sides=(280,), allow_two_row_displays=True, confirm_liters_row=True)


def close_up():
    image = np.full((1500, 1200, 3), 90, np.uint8)
    cv2.rectangle(image, (100, 100), (1100, 1400), (255, 240, 230), -1)
    return image


def test_a_picture_that_gives_nothing_is_read_again_smaller():
    def detect(image):
        if image.shape[1] > image.shape[0]:  # wider than tall: the liters row on its own
            return column("22.93")
        return GOOD if max(image.shape[:2]) == 280 else []

    result = read_display(close_up(), SIZES, detect)
    assert result.status == ACCEPTED
    assert result.reading.volume.value == D("22.93")
    assert [a.variant for a in result.attempts][:5] == ["original", "faded", "binary", "thick", "280px original"]


def test_no_smaller_read_when_the_first_size_succeeds():
    sizes = []

    def detect(image):
        sizes.append(max(image.shape[:2]))
        return GOOD

    read_display(close_up(), SIZES, detect)
    assert 280 not in sizes


def test_another_size_that_saw_a_different_second_row_blocks_a_two_row_result():
    # At full size: total, liters and an advert's price, which fail the
    # arithmetic. Smaller, the liters row is missed and the advert's price is
    # left under the total. That must not come back as the liters.
    def detect(image):
        if max(image.shape[:2]) == 280:
            return column("244.38", "5.00")
        return column("244.38", "112.10", "5.00") if image.shape[0] > 100 else column("5.00")

    result = read_display(close_up(), SIZES, detect)
    assert result.status == RETAKE
    assert result.liters is None
