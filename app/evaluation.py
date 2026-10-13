"""Scoring of pipeline results against hand-labelled values.

Liters is the value the app returns, so it is the headline: how often it is
returned correctly and how often a wrong one is returned. The total and price
per liter are scored too, as diagnostics, because they are what checks liters.
"""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from app.pipeline import Result

FIELDS = ("amount", "volume", "price")
VARIANTS = ("original", "faded", "binary", "thick")


def digit_confusions(truth: str, read: str) -> list[tuple[str, str]]:
    """(true character, read character) for each position that differs.

    Only comparable when both strings have the same length. A different
    length means characters were added or dropped, which is counted
    separately by the caller.
    """
    if len(truth) != len(read):
        return []
    return [(t, r) for t, r in zip(truth, read) if t != r]


@dataclass
class Report:
    total: int = 0
    statuses: Counter = field(default_factory=Counter)
    retake_reasons: Counter = field(default_factory=Counter)
    liters_correct: int = 0  # liters was returned and equals the label
    # Photos whose returned liters (accepted or review) differs from the label.
    wrong_liters: list[str] = field(default_factory=list)
    # (variant, field) -> photos where that read alone got the label right
    variant_correct: Counter = field(default_factory=Counter)
    variant_reads: Counter = field(default_factory=Counter)  # variant -> reads that ran
    # Liters misreads in individual reads, before validation.
    confusions: Counter = field(default_factory=Counter)
    length_errors: Counter = field(default_factory=Counter)  # "extra" / "missing"

    def add(self, name: str, result: Result, truth: dict[str, Decimal]) -> None:
        self.total += 1
        self.statuses[result.status] += 1
        if result.liters is not None:  # accepted, or review (returned unverified)
            if result.liters.value == truth["volume"]:
                self.liters_correct += 1
            else:
                self.wrong_liters.append(name)
        else:
            self.retake_reasons.update(result.reasons)

        for attempt in result.attempts:
            if attempt.reading is None:
                continue
            self.variant_reads[attempt.variant] += 1
            for f in FIELDS:
                # Labels from phone feedback have only the liters.
                if truth.get(f) is not None and getattr(attempt.reading, f).value == truth[f]:
                    self.variant_correct[(attempt.variant, f)] += 1

            read = attempt.reading.volume.value
            if read == truth["volume"]:
                continue
            # Compare as displayed text, with the label formatted to the same
            # number of decimals as the reading so "22.9" and "22.90" line up.
            read_text = str(read)
            truth_text = f"{truth['volume']:.{-read.as_tuple().exponent}f}"
            if len(read_text) == len(truth_text):
                self.confusions.update(digit_confusions(truth_text, read_text))
            else:
                kind = "extra" if len(read_text) > len(truth_text) else "missing"
                self.length_errors[kind] += 1

    def format(self) -> str:
        n = self.total
        if n == 0:
            return "No photos evaluated."

        def pct(count: int, of: int = n) -> str:
            return f"{count}/{of} ({count / of:.0%})" if of else "0/0"

        lines = [f"Photos evaluated: {n}", "", "LITERS"]
        lines.append(f"  returned correctly:    {pct(self.liters_correct)}")
        lines.append(f"  WRONG liters returned: {len(self.wrong_liters)}  (target: 0)")
        lines += [f"    {name}" for name in self.wrong_liters]

        lines += ["", "Status:"]
        for status in ("accepted", "retake", "review"):
            lines.append(f"  {status:<9} {pct(self.statuses[status])}")
        if self.retake_reasons:
            lines += ["", "Retake reasons:"]
            lines += [f"  {reason:<24} {count}" for reason, count in self.retake_reasons.most_common()]

        lines += ["", "Liters misreads in individual reads, before validation (true -> read):"]
        if self.confusions:
            lines += [f"  {t!r} -> {r!r}   {count}" for (t, r), count in self.confusions.most_common(10)]
        else:
            lines.append("  no digit swaps")
        if self.length_errors:
            lines.append(
                f"  reads with extra characters: {self.length_errors['extra']}, "
                f"with missing characters: {self.length_errors['missing']}"
            )

        lines += [
            "",
            "Accuracy of each read that ran, before validation (diagnostic; reads",
            "after the first are skipped when stop_at_first_valid is on):",
        ]
        for variant in VARIANTS:
            reads = self.variant_reads[variant]
            row = "  ".join(
                f"{label} {pct(self.variant_correct[(variant, f)], reads)}"
                for label, f in (("liters", "volume"), ("total", "amount"), ("price", "price"))
            )
            lines.append(f"  {variant:<9} {row}")
        return "\n".join(lines)
