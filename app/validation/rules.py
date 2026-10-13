"""Validation rules for one display reading. Any failure means "retake"."""

from decimal import Decimal

from app.settings import ValidationConfig

AMOUNT_NOT_2_DECIMALS = "amount_not_2_decimals"
PRICE_NOT_A_FUEL_GRADE = "price_not_a_fuel_grade"
AMOUNT_MISMATCH = "amount_mismatch"
VOLUME_OUT_OF_RANGE = "volume_out_of_range"
VOLUME_WRONG_DECIMALS = "volume_wrong_decimals"


def validate(
    amount: Decimal, volume: Decimal, price: Decimal, config: ValidationConfig
) -> list[str]:
    """Return the rules this reading breaks. An empty list means it is valid.

    The Decimals must keep the decimal places as displayed (Decimal("50.00"),
    not Decimal("50")), because the decimal-count rules check them.
    """
    failures = []
    if -amount.as_tuple().exponent not in config.amount_decimals:
        failures.append(AMOUNT_NOT_2_DECIMALS)
    if config.check_fuel_price and price not in config.grades.values():
        failures.append(PRICE_NOT_A_FUEL_GRADE)
    if abs(volume * price - amount) > config.amount_tolerance:
        failures.append(AMOUNT_MISMATCH)
    if not config.min_volume <= volume <= config.max_volume:
        failures.append(VOLUME_OUT_OF_RANGE)
    if -volume.as_tuple().exponent not in config.volume_decimals:
        failures.append(VOLUME_WRONG_DECIMALS)
    return failures
