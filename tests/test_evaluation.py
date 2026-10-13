from decimal import Decimal as D

from app.evaluation import Report, digit_confusions
from app.ocr.parse import DisplayReading, Field
from app.pipeline import ACCEPTED, RETAKE, Attempt, Result

TRUTH = {"amount": D("50.00"), "volume": D("22.93"), "price": D("2.18")}


def reading(amount, volume, price):
    return DisplayReading(Field(D(amount), 0.99), Field(D(volume), 0.99), Field(D(price), 0.99))


def attempt(variant, amount, volume, price, failures=()):
    return Attempt(variant, [], reading(amount, volume, price), list(failures))


def accepted(a):
    return Result(ACCEPTED, a.reading, [], None, [a])


def test_digit_confusions():
    assert digit_confusions("89.34", "99.34") == [("8", "9")]
    assert digit_confusions("50.00", "50.00") == []
    assert digit_confusions("1.99", "988.99") == []  # different length


def test_correct_liters():
    report = Report()
    report.add("a.jpg", accepted(attempt("original", "50.00", "22.93", "2.18")), TRUTH)
    assert report.statuses[ACCEPTED] == 1
    assert report.liters_correct == 1
    assert report.wrong_liters == []
    assert report.variant_correct[("original", "volume")] == 1


def test_wrong_liters_returned_is_counted():
    report = Report()
    report.add("a.jpg", accepted(attempt("original", "50.00", "22.98", "2.18")), TRUTH)
    assert report.wrong_liters == ["a.jpg"]
    assert report.liters_correct == 0
    assert report.confusions[("3", "8")] == 1


def test_wrong_total_with_right_liters_is_not_a_wrong_return():
    # Only liters is returned, so a misread total alone does not count.
    report = Report()
    report.add("a.jpg", accepted(attempt("original", "49.99", "22.93", "2.18")), TRUTH)
    assert report.wrong_liters == []
    assert report.liters_correct == 1
    assert report.confusions == {}


def test_label_without_trailing_zeros_matches():
    report = Report()
    truth = {"amount": D("50"), "volume": D("22.930"), "price": D("2.18")}
    report.add("a.jpg", accepted(attempt("original", "50.00", "22.93", "2.18")), truth)
    assert report.wrong_liters == []


def test_retake_counts_reasons_and_liters_read_errors():
    report = Report()
    lost = attempt("original", "50.00", "2.93", "2.18", ["amount_mismatch"])
    swap = attempt("faded", "50.00", "22.98", "2.18", ["amount_mismatch"])
    row = Attempt("liters row", [], None, ["liters_not_confirmed"], "liters row re-read: []")
    result = Result(RETAKE, None, ["amount_mismatch"], None, [lost, swap, row])
    report.add("a.jpg", result, TRUTH)
    assert report.statuses[RETAKE] == 1
    assert report.retake_reasons["amount_mismatch"] == 1
    assert report.liters_correct == 0
    assert report.length_errors == {"missing": 1}
    assert report.confusions[("3", "8")] == 1
    assert report.variant_reads == {"original": 1, "faded": 1}


def test_review_liters_count_as_returned():
    right = Result("review", None, ["no_price_row"], None, [], Field(D("22.93"), 0.99))
    wrong = Result("review", None, ["no_price_row"], None, [], Field(D("22.98"), 0.99))
    report = Report()
    report.add("a.jpg", right, TRUTH)
    report.add("b.jpg", wrong, TRUTH)
    assert report.statuses["review"] == 2
    assert report.liters_correct == 1
    assert report.wrong_liters == ["b.jpg"]


def test_format_runs_on_empty_and_filled_reports():
    assert "No photos" in Report().format()
    report = Report()
    report.add("a.jpg", accepted(attempt("original", "50.00", "22.93", "2.18")), TRUTH)
    text = report.format()
    assert "WRONG liters returned: 0" in text
    assert "returned correctly:    1/1 (100%)" in text
    assert "faded     liters 0/0" in text
