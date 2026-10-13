"""Human-readable text for a pipeline result (terminal output)."""

from app.pipeline import ACCEPTED, REVIEW, Result


def format_result(result: Result, raw: bool = False) -> str:
    lines = []
    if result.screen is None:
        lines.append("Screen: no outline found, read the whole photo")
    else:
        x1, y1, x2, y2 = result.screen
        lines.append(f"Screen: ({x1},{y1})-({x2},{y2})")

    if result.status == ACCEPTED:
        # Only liters is returned. The total and price per liter were read
        # too, and the liters value passed the arithmetic check against them.
        liters = result.reading.volume
        lines.append("Status: accepted")
        lines.append(f"  Liters  {liters.value}  (confidence {liters.confidence:.3f})")
        if raw:
            r = result.reading
            lines.append(f"  checked against total {r.amount.value} and price per liter {r.price.value}")
    elif result.status == REVIEW:
        # Liters is returned but not verified: the display has no price row to
        # check it against, or a decimal point had to be put in by the app.
        liters = result.liters
        lines.append(f"Status: review  ({', '.join(result.reasons)}: liters not verified)")
        lines.append(f"  Liters  {liters.value}  (confidence {liters.confidence:.3f})")
    else:
        lines.append(f"Status: retake  ({', '.join(result.reasons)})")

    # On a retake, show what each attempt read so the cause is visible. These
    # values failed validation and must not be used.
    if raw or result.status != ACCEPTED:
        for attempt in result.attempts:
            if attempt.reading is None:
                read = attempt.detail
            else:
                r = attempt.reading
                read = f"{r.amount.value} | {r.volume.value} | {r.price.value}"
            verdict = ", ".join(attempt.failures) or "valid"
            if attempt.reading is not None:
                verdict += f"  (lowest confidence {attempt.reading.confidence:.3f})"
            lines.append(f"  [{attempt.variant:<14}] {read}  -> {verdict}")
            if raw:
                for d in attempt.detections:
                    lines.append(f"      {d.confidence:.3f}  {d.text!r:<12} box={d.box}")
    return "\n".join(lines)
