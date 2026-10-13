"""Loads the YAML files in config/."""

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml

from app.ocr.layout import DEFAULT_LABELS, Labels

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@dataclass(frozen=True)
class ValidationConfig:
    grades: dict[str, Decimal]  # grade name -> SAR per liter
    amount_tolerance: Decimal
    min_volume: Decimal
    max_volume: Decimal
    min_confidence: float
    # A reading whose arithmetic holds but whose confidence is between this
    # and min_confidence is a "review" instead of a retake. None: always retake.
    review_confidence: float | None = None
    ocr_max_side: int = 480
    # Sizes (longest side, pixels) the display is read at, in order. Empty: ocr_max_side only.
    extra_ocr_sides: tuple[int, ...] = ()
    stop_at_first_valid: bool = False
    confirm_liters_row: bool = False
    volume_decimals: tuple[int, ...] = (2, 3)
    amount_decimals: tuple[int, ...] = (2,)  # SAR has 2; some currencies show 3
    check_fuel_price: bool = True
    allow_two_row_displays: bool = False
    # Plausible total / liters on a two-row display, where no price is shown.
    two_row_price_range: tuple[Decimal, Decimal] = (Decimal("0.5"), Decimal("5"))
    labels: Labels = DEFAULT_LABELS
    whole_photo_fallback: bool = False
    whole_photo_max_side: int = 960
    extra_whole_photo_sides: tuple[int, ...] = ()
    infer_decimal_points: bool = False
    inferred_liters_decimals: int = 2

    @property
    def ocr_sides(self) -> tuple[int, ...]:
        return (self.ocr_max_side, *self.extra_ocr_sides)

    @property
    def whole_photo_sides(self) -> tuple[int, ...]:
        return (self.whole_photo_max_side, *self.extra_whole_photo_sides)


@dataclass(frozen=True)
class OcrSettings:
    device: str = "cpu"  # "cpu", or "gpu" with the GPU build of PaddlePaddle
    recognition_model_dir: str = ""  # empty: PaddleOCR's stock model
    recognition_model_name: str = "PP-OCRv6_medium_rec"


def load_ocr_settings(config_dir: Path = CONFIG_DIR) -> OcrSettings:
    ocr = _load(config_dir / "settings.yaml").get("ocr") or {}
    # The two environment variables override the file for one run, which is
    # how a newly trained model is compared with the current one.
    model_dir = os.environ.get("PUMP_OCR_REC_MODEL_DIR", str(ocr.get("recognition_model_dir") or ""))
    if model_dir and not Path(model_dir).is_absolute():
        model_dir = str(config_dir.parent / model_dir)
    return OcrSettings(
        device=os.environ.get("PUMP_OCR_DEVICE") or str(ocr.get("device") or "cpu"),
        recognition_model_dir=model_dir,
        recognition_model_name=str(ocr.get("recognition_model_name") or "PP-OCRv6_medium_rec"),
    )


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_validation_config(config_dir: Path = CONFIG_DIR) -> ValidationConfig:
    prices = _load(config_dir / "fuel_prices.yaml")
    settings = _load(config_dir / "settings.yaml")
    validation, pipeline = settings["validation"], settings["pipeline"]
    words = _load(config_dir / "labels.yaml")
    return ValidationConfig(
        labels=Labels.from_lists(
            words["amount"], words["liters"], words["price"], words["gallons_prefixes"]
        ),
        whole_photo_fallback=bool(pipeline["whole_photo_fallback"]),
        whole_photo_max_side=int(pipeline["whole_photo_max_side"]),
        infer_decimal_points=bool(pipeline["infer_decimal_points"]),
        inferred_liters_decimals=int(pipeline["inferred_liters_decimals"]),
        ocr_max_side=int(pipeline["ocr_max_side"]),
        extra_ocr_sides=tuple(int(n) for n in pipeline.get("extra_ocr_sides") or ()),
        extra_whole_photo_sides=tuple(int(n) for n in pipeline.get("extra_whole_photo_sides") or ()),
        stop_at_first_valid=bool(pipeline["stop_at_first_valid"]),
        confirm_liters_row=bool(pipeline["confirm_liters_row"]),
        volume_decimals=tuple(int(n) for n in validation["volume_decimals"]),
        amount_decimals=tuple(int(n) for n in validation["amount_decimals"]),
        check_fuel_price=bool(validation["check_fuel_price"]),
        allow_two_row_displays=bool(pipeline["allow_two_row_displays"]),
        two_row_price_range=(
            Decimal(str(pipeline["two_row_min_price"])),
            Decimal(str(pipeline["two_row_max_price"])),
        ),
        grades={str(name): Decimal(str(price)) for name, price in prices["grades"].items()},
        amount_tolerance=Decimal(str(validation["amount_tolerance_sar"])),
        min_volume=Decimal(str(validation["min_volume_l"])),
        max_volume=Decimal(str(validation["max_volume_l"])),
        min_confidence=float(validation["min_confidence"]),
        review_confidence=(
            None if validation.get("review_confidence") is None else float(validation["review_confidence"])
        ),
    )
