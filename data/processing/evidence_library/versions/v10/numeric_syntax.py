"""One numeric grammar for deterministic measurement extraction in this release."""

import math
import re
from decimal import Decimal
from typing import Any

# Whitespace belongs around commas, never between digits of a group.
UNSIGNED_NUMBER = (
    r"(?:(?:[0-9]{1,3}(?:\s*,\s*[0-9]{3})+|[0-9]+)"
    r"(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
NUMBER = rf"[+-]?{UNSIGNED_NUMBER}"
_NUMBER = re.compile(NUMBER)
_FINITE_POINT = re.compile(
    rf"(?P<value>{NUMBER})(?:\s*(?:±|\+/-)\s*(?P<variation>{NUMBER}))?"
)


def number_text(value: Any) -> str | None:
    """Validate a whole token before removing grouping commas and whitespace."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if _NUMBER.fullmatch(text) is None:
        return None
    text = re.sub(r"\s*,\s*", "", text)
    if not Decimal(text).is_finite() or not math.isfinite(float(text)):
        return None
    return text


def finite_point_text(value: Any) -> tuple[str, str | None] | None:
    """Parse one finite number with an optional finite uncertainty."""
    if value is None or isinstance(value, bool):
        return None
    match = _FINITE_POINT.fullmatch(str(value).strip())
    if match is None:
        return None
    central = number_text(match.group("value"))
    variation = (
        number_text(match.group("variation"))
        if match.group("variation") is not None
        else None
    )
    if central is None or (match.group("variation") is not None and variation is None):
        return None
    return central, variation
