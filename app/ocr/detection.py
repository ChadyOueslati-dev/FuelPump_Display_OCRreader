from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One text region returned by the OCR engine."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max in pixels
