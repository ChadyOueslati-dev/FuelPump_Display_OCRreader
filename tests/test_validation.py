from dataclasses import replace
from decimal import Decimal as D

import pytest

from app.settings import ValidationConfig, load_validation_config
from app.validation.rules import (
    AMOUNT_MISMATCH,
    AMOUNT_NOT_2_DECIMALS,
    PRICE_NOT_A_FUEL_GRADE,
    VOLUME_OUT_OF_RANGE,
    VOLUME_WRONG_DECIMALS,
    validate,
)

CONFIG = ValidationConfig(
    grades={"91": D("2.18"), "95": D("2.33"), "diesel": D("1.79")},
    amount_tolerance=D("0.02"),
    min_volume=D("1"),
    max_volume=D("150"),
    min_confidence=0.95,
)


def check(amount, volume, price):
    return validate(D(amount), D(volume), D(price), CONFIG)


@pytest.mark.parametrize(
    "amount, volume, price",
    [
        ("50.00", "22.93", "2.18"),  # first real photo
        ("30.00", "12.87", "2.33"),
        ("89.34", "49.91", "1.79"),
        ("116.50", "50.000", "2.33"),  # 3-decimal volume
    ],
)
def test_valid_readings(amount, volume, price):
    assert check(amount, volume, price) == []


@pytest.mark.parametrize("amount", ["50", "50.0", "50.000"])
def test_amount_must_have_exactly_2_decimals(amount):
    assert AMOUNT_NOT_2_DECIMALS in check(amount, "22.93", "2.18")


def test_three_decimal_amount_passes_only_where_the_config_allows_it():
    # Kuwaiti dinar: 12.975 for 49.91 L at 0.260 per liter
    loose = replace(CONFIG, check_fuel_price=False, amount_decimals=(2, 3))
    assert validate(D("12.975"), D("49.91"), D("0.260"), loose) == []
    strict = replace(CONFIG, check_fuel_price=False)
    assert validate(D("12.975"), D("49.91"), D("0.260"), strict) == [AMOUNT_NOT_2_DECIMALS]


@pytest.mark.parametrize("price", ["1.99", "988.99", "2.19"])
def test_price_must_be_a_configured_grade(price):
    assert PRICE_NOT_A_FUEL_GRADE in check("50.00", "22.93", price)


def test_price_rule_can_be_switched_off():
    relaxed = replace(CONFIG, check_fuel_price=False)
    assert validate(D("64.04"), D("32.18"), D("1.99"), relaxed) == []
    # The arithmetic check still applies, so a misread price is still caught.
    assert validate(D("64.04"), D("32.18"), D("988.99"), relaxed) == [AMOUNT_MISMATCH]


def test_amount_tolerance_boundary():
    # 10.00 L * 2.18 = 21.80 exactly
    assert check("21.82", "10.00", "2.18") == []
    assert check("21.78", "10.00", "2.18") == []
    assert check("21.83", "10.00", "2.18") == [AMOUNT_MISMATCH]
    assert check("21.77", "10.00", "2.18") == [AMOUNT_MISMATCH]


def test_misread_digit_breaks_the_arithmetic():
    assert check("30.00", "2.87", "2.33") == [AMOUNT_MISMATCH]  # lost the leading 1
    assert check("99.34", "49.91", "1.79") == [AMOUNT_MISMATCH]  # 8 read as 9


def test_volume_range_boundary():
    assert check("2.18", "1.00", "2.18") == []
    assert check("327.00", "150.00", "2.18") == []
    assert check("2.16", "0.99", "2.18") == [VOLUME_OUT_OF_RANGE]
    assert check("327.02", "150.01", "2.18") == [VOLUME_OUT_OF_RANGE]


@pytest.mark.parametrize("volume", ["22.9", "23", "22.9300"])
def test_liters_must_have_2_or_3_decimals(volume):
    assert VOLUME_WRONG_DECIMALS in check("50.00", volume, "2.18")


def test_liters_with_3_decimals_is_allowed():
    assert check("50.00", "22.936", "2.18") == []


def test_all_failures_are_reported():
    assert check("50.0", "500.0", "9.99") == [
        AMOUNT_NOT_2_DECIMALS,
        PRICE_NOT_A_FUEL_GRADE,
        AMOUNT_MISMATCH,
        VOLUME_OUT_OF_RANGE,
        VOLUME_WRONG_DECIMALS,
    ]


def test_config_files_load():
    config = load_validation_config()
    assert config.grades["91"] == D("2.18")
    assert config.grades["95"] == D("2.33")
    assert config.grades["diesel"] == D("1.79")
    assert config.amount_tolerance == D("0.02")
    assert (config.min_volume, config.max_volume) == (D("1"), D("150"))
    assert config.volume_decimals == (2, 3)
    assert isinstance(config.check_fuel_price, bool)
    # The threshold is meant to be tuned, so only check it is a sane value.
    assert 0 < config.min_confidence <= 1


def test_ocr_settings_load():
    from app.settings import load_ocr_settings

    settings = load_ocr_settings()
    assert settings.device in ("cpu", "gpu")
    assert settings.recognition_model_name
