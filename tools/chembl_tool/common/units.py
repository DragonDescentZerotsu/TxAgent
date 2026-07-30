"""Robust, self-contained unit normalization for evidence-library ingestion.

Public API:

* ``clean_unit`` -- a *meaning-preserving* display cleaner that tidies raw unit strings
  (spacing, superscripts, micro-sign, SI letter casing) while preserving scientific typography
  (``µ``, ``·``, ``×``). Keeps ``µM`` as ``µM`` rather than collapsing it to ``um`` (micrometres).

* ``canonicalize_unit`` -- a small dimensional engine (no ``pint`` dependency) returning a
  ``CanonicalUnit`` (``canonical`` equivalence key, ``scale``, ``dimension``, ``transform``,
  ``unknown_tokens``, ...). It folds ``×10^N`` scale factors, parses composite units into a
  base-dimension signature, reorders factors so ``ng·h/mL`` and ``h·ng/mL`` share one key, and
  flags log/ln ``transform`` quantities. ``canonical_unit`` / ``unit_dimension`` are thin accessors.

* ``canonicalize_measurement(value, unit, ...)`` (+ ``canonicalized_unit`` / ``canonicalized_value``)
  -- **fold-only by default**: keeps the source's own unit/prefix and only folds ``×10^N`` into the
  number, so magnitudes never shift unexpectedly. Standardizing to one domain unit per measurement
  is *opt-in* via ``measurement_class`` (endpoint-derived; guarded so a mislabeled unit returns
  ``None``) or an explicit ``targets`` override. ``units_compatible`` answers whether a unit fits a
  named endpoint quantity-kind.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any


UNIT_NORMALIZER_VERSION = "unit_normalizer.v3"

_NULL_VALUES = {"", "nan", "none", "null", "na", "n/a", "-", "unspecified"}

# Fold the several unicode dashes/minus signs onto ASCII hyphen-minus.
_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", "‒": "-", "−": "-"})
# Superscript characters -> their plain equivalents (used with a leading ``^``).
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SUPERSCRIPT_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")

# Per-token SI casing / spelling canonicalization.  Keyed on the exact token
# where the metre-vs-molar (``mm`` vs ``mM``) distinction matters, and on the
# lowercase form otherwise.  ASCII ``u`` is treated as the micro prefix in unit
# context (``ug`` -> ``µg``).
_TOKEN_CASE = {
    # time
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "d": "d", "day": "d", "days": "d",
    # volume (litre): capital L, lowercase prefix
    "l": "L", "ml": "mL", "µl": "µL", "ul": "µL", "dl": "dL", "cl": "cL",
    "nl": "nL", "pl": "pL", "kl": "kL",
    "liter": "L", "litre": "L", "liters": "L", "litres": "L",
    # amount-of-substance concentration (molar): capital M
    "M": "M", "mM": "mM", "µM": "µM", "uM": "µM", "nM": "nM", "pM": "pM",
    # mass
    "g": "g", "mg": "mg", "µg": "µg", "ug": "µg", "mcg": "µg", "ng": "ng", "pg": "pg", "kg": "kg",
    # length
    "m": "m", "cm": "cm", "mm": "mm", "nm": "nm", "µm": "µm", "um": "µm", "pm": "pm",
    # amount
    "mol": "mol", "mmol": "mmol", "µmol": "µmol", "umol": "µmol",
    "nmol": "nmol", "pmol": "pmol",
    "mole": "mol", "moles": "mol", "µmoles": "µmol", "umoles": "µmol",
    "nmoles": "nmol", "pmoles": "pmol", "mmoles": "mmol",
    # dimensionless / qualifiers
    "%": "%", "fraction": "fraction", "fold": "fold", "ratio": "ratio", "protein": "protein", "cells": "cells",
    "ppm": "ppm", "ppb": "ppb", "dimensionless": "dimensionless", "unitless": "dimensionless",
}
# Runs of letters / micro-sign / percent that a token-casing pass rewrites.
_TOKEN_RE = re.compile(r"[A-Za-zµ%]+")


def _as_nonempty_text(value: Any) -> str | None:
    """Return a stripped string, or ``None`` for null-like / empty inputs."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _NULL_VALUES:
        return None
    return text


def _case_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if token in _TOKEN_CASE:
        return _TOKEN_CASE[token]
    lowered = token.lower()
    return _TOKEN_CASE.get(lowered, token)


@lru_cache(maxsize=100_000)
def clean_unit(value: Any) -> str | None:
    """Return a tidy, meaning-preserving rendering of a raw unit string.

    Trims whitespace, converts superscripts to ``^`` notation, folds dashes and
    the Greek/micro mu onto the display-canonical micro sign, removes spaces
    around ``/`` ``·`` ``*`` operators, and canonicalizes SI letter casing.
    Case and scientific typography are preserved (``µM`` stays ``µM``).
    """
    text = _as_nonempty_text(value)
    if text is None:
        return None
    # Superscripts -> ``^...`` *before* NFKC so ``10⁻⁶`` becomes ``10^-6``, not ``10-6``.
    text = _SUPERSCRIPT_RE.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), text)
    # NFKC folds fullwidth forms (``％`` -> ``%``, fullwidth digits); it also maps
    # the micro sign onto Greek mu, which the next step reverses.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASHES)
    text = text.replace("μ", "µ")  # Greek small mu (U+03BC) -> micro sign (U+00B5)
    text = re.sub(r"(?i)\bper\b", "/", text)  # "ng per mL" -> "ng/mL"
    text = re.sub(r"(?i)\bx(?=\s*10)", "×", text)  # ascii "x10^-6" -> "×10^-6"
    text = re.sub(r"\^\{(-?\d+)\}", r"^\1", text)  # LaTeX braces "10^{-6}" -> "10^-6"
    text = re.sub(r"(?<=[A-Za-zµ])\.(?=[A-Za-zµ])", "·", text)  # "ng.h" -> "ng·h"
    # Bare unit exponents ("mL-1", "s-1", "cm2") -> caret form; the negative-lookahead
    # and letter-lookbehind keep ``10-6`` / ``x10-6`` scale factors untouched.
    text = re.sub(r"(?<=[A-Za-zµ%])(-\d+|[23])(?![\d^])", r"^\1", text)
    text = re.sub(r"\s*([/·*])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _TOKEN_RE.sub(_case_token, text)
    return text or None


# --------------------------------------------------------------------------- #
# Level 2: dimensional engine
# --------------------------------------------------------------------------- #

# Base units (no SI prefix) -> (scale relative to the base unit of that dimension, dimension).
# Base units: gram, litre, metre, mole, second. Time units carry real scales (min=60 s, h=3600 s)
# so ``.scale`` is a true magnitude in base units and can convert across h/min/s.
_BASE_UNITS: dict[str, tuple[float, dict[str, int]]] = {
    "g": (1.0, {"mass": 1}),
    "L": (1.0, {"volume": 1}),
    "l": (1.0, {"volume": 1}),
    "m": (1.0, {"length": 1}),
    "mol": (1.0, {"amount": 1}),
    "s": (1.0, {"time": 1}),
    "min": (60.0, {"time": 1}),
    "h": (3600.0, {"time": 1}),
    "d": (86400.0, {"time": 1}),
    "M": (1.0, {"amount": 1, "volume": -1}),  # molar == mol/L
    "%": (1.0, {}),
    "fraction": (1.0, {}),
    "fold": (1.0, {}),
    "ratio": (1.0, {}),
    "ppm": (1.0, {}),
    "ppb": (1.0, {}),
    "dimensionless": (1.0, {}),
    "cells": (1.0, {"count": 1}),
}
# Qualifier tokens carry no dimension but are preserved in the canonical string. (``cells`` is
# NOT here -- it is a genuine count basis, e.g. ``µL/min/10^6 cells``.)
_QUALIFIERS = {"protein", "tissue", "skin"}
_SI_PREFIX = {"p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1, "k": 1e3}

_SCALE_RE = re.compile(r"^(?:×|x)?10(?:\^([-+]?\d+)|([-+]\d+))$")
# An explicit ``×10^N`` / ``x10^N`` factor anywhere in the string (multiplicative, any position).
_NOTATION_FACTOR_RE = re.compile(r"(?:×|x)\s*10(?:\^([-+]?\d+)|([-+]\d+))")
_ANY_NOTATION_RE = re.compile(r"(?:×|x)?\s*10(?:\^([-+]?\d+)|([-+]\d+))")
# OCR-compressed forms such as ``×106 cm/s`` have lost the sign/caret boundary:
# interpreting them as 10^6 is mechanically possible but scientifically unsafe.
_AMBIGUOUS_COMPRESSED_NOTATION_RE = re.compile(
    r"(?:(?:×|x)\s*10[4-9]\b|(?<![\d^])10[4-9]\s*(?=(?:cm|nm|mm|µm|um)/))",
    re.IGNORECASE,
)
# A whole expression wrapped in a log/ln transform is dimensionless (log of a unit).
_FUNC_RE = re.compile(r"^(-?log10|-?log2|-?log|-?ln|logit)\((.+)\)$", re.I)
_PREFIX_FUNC_RE = re.compile(
    r"^(-?log10|-?log2|-?log|-?ln)\s+(.+)$", re.I
)
# A bare log readout (logP, logD, logBB, logPapp, log10P, lnKp, ...): a log function directly
# followed by a quantity name.
_BARE_LOG_RE = re.compile(r"^(-?)(log10|log2|log|ln)([A-Za-z]\w*)$", re.I)
# A p-prefixed readout (pIC50, pEC50, pEC3, pKa, pKi, pA2, ...) == -log10 of a molar quantity.
# Explicit set so SI pico-units (pM, pmol, pg, pL, ppm, ppb) are NOT caught.
_PLOG_RE = re.compile(r"^p(?:(?:ic|ec|gi|cc|tc|lc)\d+|ka|ki|kd|kb|a2|d2)$", re.I)
# A single factor that is a parenthesised sub-expression, optionally raised to a power.
_PAREN_RE = re.compile(r"^\((.*)\)(?:\^(-?\d+))?$")


def _format_canonical(token_exponents: dict[str, int]) -> str:
    """Render net token exponents as ``num/den`` with negative powers in the denominator."""
    def render(token: str, exponent: int) -> str:
        magnitude = abs(exponent)
        return token if magnitude == 1 else f"{token}^{magnitude}"

    numerator = sorted(render(t, e) for t, e in token_exponents.items() if e > 0)
    denominator = sorted(render(t, e) for t, e in token_exponents.items() if e < 0)
    canonical = "·".join(numerator)
    if denominator:
        canonical = f"{canonical}/{'·'.join(denominator)}"
    return canonical


def _split_top_level(expr: str, separators: str) -> list[str]:
    """Split ``expr`` on any char in ``separators`` that sits outside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in expr:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif depth == 0 and char in separators:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _evaluate(expr: str) -> tuple[float, dict[str, int]]:
    """Evaluate a (possibly parenthesised) unit expression to (scale, token exponents).

    Conservative precedence: the first ``/``-segment is the numerator and every later segment
    is the denominator (``ng/mL·h`` == ``ng/(mL·h)``, ``µL/min/mg protein`` keeps ``protein``
    in the denominator, ``µg/cm²·h`` is flux). This is deliberately the *safe* reading of the
    ``/X·Y`` shape, which is genuinely endpoint-dependent (AUC vs flux vs clearance); callers
    that know the endpoint override it via ``measurement_class`` on the ``canonicalized_*``
    functions. Parentheses recurse, so ``ng/mL/(mg/kg)`` keeps ``kg`` in the numerator.
    Leading/suffix ``×10^N`` factors are pulled out as scale before this runs (see
    :func:`canonicalize_unit`), so only structural ``/10^N`` divisors reach here.
    """
    scale = 1.0
    exponents: dict[str, int] = {}
    for segment_index, segment in enumerate(_split_top_level(expr, "/")):
        sign = 1 if segment_index == 0 else -1
        for factor in _split_top_level(segment, "·*× \t"):
            factor = factor.strip()
            if not factor:
                continue
            # Preserve the v1 interpretation of denominator qualifiers such as
            # ``g/100 mL`` and ``cells/100 LCs``. This is not scientific
            # exponent notation and is intentionally outside the v2 change.
            if factor == "100":
                continue
            scale_match = _SCALE_RE.match(factor)
            if scale_match:
                exponent = scale_match.group(1) or scale_match.group(2)
                scale *= 10.0 ** (int(exponent) * sign)
                continue
            paren_match = _PAREN_RE.match(factor)
            if paren_match:
                inner_scale, inner_exponents = _evaluate(paren_match.group(1))
                power = int(paren_match.group(2)) if paren_match.group(2) else 1
                multiplier = power * sign
                if inner_scale != 1.0:
                    scale *= inner_scale ** multiplier
                for token, exponent in inner_exponents.items():
                    exponents[token] = exponents.get(token, 0) + exponent * multiplier
                continue
            base, _, exp_text = factor.partition("^")
            token = base or factor
            exponent = int(exp_text) if exp_text and re.fullmatch(r"-?\d+", exp_text) else 1
            exponents[token] = exponents.get(token, 0) + exponent * sign
    return scale, exponents


def _resolve_token(token: str) -> tuple[float, dict[str, int] | None]:
    """Return ``(scale, dimension)`` for a single unit token.

    ``dimension`` is ``None`` when the token cannot be resolved (an unknown unit).
    ``{}`` denotes a resolved but dimensionless token.
    """
    if token in _QUALIFIERS:
        return 1.0, {}
    if token in _BASE_UNITS:  # check full token first (``min``, ``mol``, bare ``m``/``d``)
        base_scale, dimension = _BASE_UNITS[token]
        return base_scale, dict(dimension)
    if len(token) > 1 and token[0] in _SI_PREFIX and token[1:] in _BASE_UNITS:
        base_scale, dimension = _BASE_UNITS[token[1:]]
        return _SI_PREFIX[token[0]] * base_scale, dict(dimension)
    return 1.0, None


@dataclass(frozen=True)
class CanonicalUnit:
    """Dimensional canonicalization of one unit string."""

    raw: str
    cleaned: str = ""
    canonical: str = ""
    scale: float = 1.0
    dimension: tuple[tuple[str, int], ...] = ()
    is_dimensionless: bool = False
    unknown_tokens: tuple[str, ...] = ()
    # Non-empty (e.g. "log10", "ln") when the quantity is a transform of an inner unit.
    # Such a value is a pure number in magnitude but is NOT interchangeable with a plain
    # ratio, nor with a transform of a different inner unit -- hence not is_dimensionless.
    transform: str = ""
    # Scientific-notation provenance is separate from the physical canonical key.
    notation_status: str = "none"
    notation_factor: float | None = None
    normalizer_version: str = UNIT_NORMALIZER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=100_000)
def canonicalize_unit(value: Any) -> CanonicalUnit:
    """Parse a unit into a dimension signature, scale factor, and canonical key.

    Scientific-notation scale factors (``×10⁻⁶``, ``x10^-6``, ``10-6``) are folded
    into ``scale``; multiplicative factors are reordered so that dimensionally
    equivalent forms (``ng·h/mL`` and ``h·ng/mL``) share one ``canonical`` string
    and one ``dimension``.
    """
    raw = "" if value is None else str(value)
    cleaned = clean_unit(value)
    if cleaned is None:
        return CanonicalUnit(raw=raw)
    notation_status, notation_factor = scientific_notation_metadata(raw, cleaned)

    # A log/ln transform carries no base dimension in magnitude (you cannot take the log of
    # a dimensioned quantity), but it is a distinct KIND of quantity: a log(cm/s) value is
    # tied to its inner reference unit and is not interchangeable with a plain ratio or with
    # a log of a different unit. Mark it as a transform rather than plain dimensionless.
    func_match = _FUNC_RE.match(cleaned) or _PREFIX_FUNC_RE.match(cleaned)
    if func_match:
        inner = canonicalize_unit(func_match.group(2))
        return CanonicalUnit(
            raw=raw,
            cleaned=cleaned,
            canonical=f"{func_match.group(1).lower()}({inner.canonical})",
            scale=1.0,
            dimension=(),
            is_dimensionless=False,
            unknown_tokens=inner.unknown_tokens,
            transform=func_match.group(1).lower(),
            notation_status=notation_status,
            notation_factor=notation_factor,
        )

    # Bare (``logP``, ``logD``, ``logBB``) and p-prefixed (``pIC50``, ``pKa``, ``pEC3``, ``pA2``)
    # log readouts. Detected on the RAW stripped text because clean_unit's exponent-insertion would
    # mangle ``pEC3`` -> ``pEC^3`` / ``pA2`` -> ``pA^2``; the p-log matcher is an explicit set so SI
    # pico-units (``pM``, ``pmol``, ``pg``, ``pL``, ``ppm``) are left to resolve normally.
    stripped = _as_nonempty_text(value)
    if stripped is not None:
        plog = _PLOG_RE.match(stripped)
        bare = _BARE_LOG_RE.match(stripped)
        if plog or bare:
            transform = "-log10" if plog else (bare.group(1) + bare.group(2)).lower()
            return CanonicalUnit(
                raw=raw,
                cleaned=stripped,
                canonical=stripped,
                scale=1.0,
                dimension=(),
                is_dimensionless=False,
                transform=transform,
                notation_status=notation_status,
                notation_factor=notation_factor,
            )

    # Pull out ``×10^N`` / ``x10^N`` factors anywhere in the string -- these always multiply the
    # magnitude regardless of position (``cm/s ×10^-6`` == ``×10^-6 cm/s``). A bare ``/10^N``
    # divisor is left in place so the structural parser divides by it.
    notation = 1.0

    def _pull_notation(match: re.Match[str]) -> str:
        nonlocal notation
        exponent = match.group(1) or match.group(2)
        notation *= 10.0 ** int(exponent)
        return " "

    body = (
        cleaned
        if notation_status == "ambiguous_scientific_notation"
        else _NOTATION_FACTOR_RE.sub(_pull_notation, cleaned)
    )
    scale, token_exponents = _evaluate(body)
    scale *= notation
    token_exponents = {token: net for token, net in token_exponents.items() if net != 0}

    dimension: dict[str, int] = {}
    unknown: list[str] = []
    for token, net in token_exponents.items():
        token_scale, token_dim = _resolve_token(token)
        if token_dim is None:
            unknown.append(token)
            continue
        if token_scale != 1.0:
            scale *= token_scale ** net
        for name, power in token_dim.items():
            dimension[name] = dimension.get(name, 0) + power * net

    dimension = {name: power for name, power in dimension.items() if power != 0}
    dim_signature = tuple(sorted(dimension.items()))
    # Build the canonical key from net token exponents so that ``cm/s``, ``cm·s^-1`` and
    # ``cm s-1`` collapse (negative exponents move to the denominator) and factor order
    # never matters.
    canonical = _format_canonical(token_exponents)

    return CanonicalUnit(
        raw=raw,
        cleaned=cleaned,
        canonical=canonical,
        scale=scale,
        dimension=dim_signature,
        is_dimensionless=not dim_signature and not unknown,
        unknown_tokens=tuple(unknown),
        notation_status=notation_status,
        notation_factor=notation_factor,
    )


def scientific_notation_metadata(raw: Any, cleaned: str | None = None) -> tuple[str, float | None]:
    """Classify unit scale notation without guessing OCR-compressed exponents."""
    raw_text = _as_nonempty_text(raw) or ""
    if _AMBIGUOUS_COMPRESSED_NOTATION_RE.search(raw_text):
        return "ambiguous_scientific_notation", None
    normalized = cleaned if cleaned is not None else clean_unit(raw)
    if not normalized:
        return "none", None
    matches = list(_ANY_NOTATION_RE.finditer(normalized))
    if not matches:
        return "none", None
    factor = 1.0
    for match in matches:
        exponent = match.group(1) or match.group(2)
        factor *= 10.0 ** int(exponent)
    return "unambiguous_scientific_notation", factor


def unit_dimension(value: Any) -> tuple[tuple[str, int], ...]:
    """Return the sorted base-dimension signature of a unit string."""
    return canonicalize_unit(value).dimension


def canonical_unit(value: Any) -> str | None:
    """Return the canonical display key of a unit string, or ``None`` for null-like input.

    This is the order-normalized, dimensionally-canonical rendering (e.g. ``ng/mL``,
    ``cm/s``, ``h·ng/mL``, ``µg/cm^2·h``, ``log10(cm/s)``) suitable for grouping, joining,
    or displaying "the same unit" downstream. Forms that differ only by typography, factor
    order, or numeric scale share one key; ``clean_unit`` gives the faithful source-style
    rendering instead.
    """
    result = canonicalize_unit(value)
    return None if result.cleaned == "" else result.canonical


def _to_float(value: Any) -> float | None:
    """Parse a scalar numeric value, or ``None`` when it is not a finite number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    text = _as_nonempty_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


# Named measurement classes -> the domain-standard target unit. Standardization is OPT-IN only:
# it happens when the caller passes ``measurement_class`` (which it derives from the endpoint,
# since the unit dimension alone cannot tell potency from solubility, or AUC from a rate). Targets
# are chosen by domain convention to keep magnitudes sensible (e.g. rate constants in h⁻¹, not s⁻¹;
# molar concentrations in µM, not nM). Easily editable -- tune per your endpoint conventions.
_MEASUREMENT_CLASSES: dict[str, str] = {
    "auc": "ng·h/mL",
    "auc_mass": "ng·h/mL",
    "auc_molar": "µmol·h/L",
    "concentration": "ng/mL",                       # plasma concentration / Cmax (mass basis)
    "molar_concentration": "µM",                    # molar concentration / molar solubility
    "potency": "nM",                                # IC50 / EC50 / Ki / Kd
    "mass_solubility": "mg/mL",                     # mass-basis solubility
    "permeability": "cm/s",
    "clearance": "mL/min",
    "weight_normalized_clearance": "mL/min/kg",     # in-vivo CL per kg body weight
    "intrinsic_clearance": "µL/min/mg protein",     # CLint per mg microsomal protein
    "flux": "µg/cm^2/h",
    "areic_dose": "µg/cm^2",
    "duration": "h",
    "rate_constant": "h^-1",
}


def _resolve_target(result: CanonicalUnit, targets: Iterable[str] | None) -> str:
    """Return an explicit per-call target unit for ``result``, or ``""`` for no standardization.

    ``targets`` is an iterable of unit strings; each supplies the target for its own dimension
    (e.g. ``targets=["cm/h"]`` standardizes permeability to cm/h). With no ``targets`` there is
    NO automatic standardization -- the value/unit stay in the source's own (folded) form.
    """
    if targets:
        for unit in targets:
            override = canonicalize_unit(unit)
            if override.dimension == result.dimension and not override.unknown_tokens:
                return override.cleaned
    return ""


def _class_target(measurement_class: str) -> str:
    if measurement_class not in _MEASUREMENT_CLASSES:
        raise KeyError(f"unknown measurement_class: {measurement_class!r}")
    return _MEASUREMENT_CLASSES[measurement_class]


def _class_compatible(result: CanonicalUnit, target_unit: str) -> bool:
    """Whether ``result``'s unit could plausibly belong to the class whose unit is ``target_unit``.

    Compatible iff the unit resolves cleanly (no unknown tokens, not a transform) and carries the
    same SET of base dimensions as the target (exponent signs ignored, since the ``/X·Y`` shape is
    what the class disambiguates). So ``ng/mL·h`` is compatible with AUC (``ng·h/mL``), but ``%``,
    ``fold``, and a plain ``ng/mL`` are not -- a mislabeled record is flagged rather than coerced.
    """
    if result.unknown_tokens or result.transform:
        return False
    target = canonicalize_unit(target_unit)
    if {name for name, _ in result.dimension} != {name for name, _ in target.dimension}:
        return False
    # Endpoint classes may reinterpret ambiguous slash/product placement, but an
    # explicit exponent is not ambiguous. For example, ``h nmol^-1 mL^-1`` must
    # not be coerced to molar AUC, whose amount exponent is positive.
    target_dimension = dict(target.dimension)
    for match in re.finditer(r"([A-Za-zµ%]+)\^(-?\d+)", result.cleaned):
        token, exponent_text = match.groups()
        _, token_dimension = _resolve_token(token)
        if not token_dimension:
            continue
        structural_sign = -1 if "/" in result.cleaned[: match.start()] else 1
        explicit_exponent = int(exponent_text) * structural_sign
        for dimension, base_exponent in token_dimension.items():
            expected = target_dimension.get(dimension)
            actual = explicit_exponent * base_exponent
            if expected is None or actual != expected:
                return False
    return True


def _magnitude_under(source_cleaned: str, target_unit: str) -> float:
    """Scale of ``source_cleaned``'s magnitude when its tokens take the target's numer/denom signs.

    Lets a caller reinterpret an ambiguous form under a known class: e.g. ``ng/mL·h`` parses
    conservatively as a rate, but under target ``ng·h/mL`` (AUC) its ``h`` moves to the
    numerator so the value scales correctly.
    """
    notation = 1.0

    def pull(match: re.Match[str]) -> str:
        nonlocal notation
        exponent = match.group(1) or match.group(2)
        notation *= 10.0 ** int(exponent)
        return " "

    struct_scale, src_exponents = _evaluate(_NOTATION_FACTOR_RE.sub(pull, source_cleaned))
    _, target_exponents = _evaluate(
        _NOTATION_FACTOR_RE.sub(lambda m: " ", canonicalize_unit(target_unit).cleaned)
    )
    scale = notation * struct_scale
    for token, net in src_exponents.items():
        token_scale, dimension = _resolve_token(token)
        if dimension is None or token_scale == 1.0 or net == 0:
            continue
        target_net = target_exponents.get(token, net)  # keep source's own side if not in target
        scale *= token_scale ** (abs(net) if target_net > 0 else -abs(net))
    return scale


def canonicalized_unit(
    value: Any, targets: Iterable[str] | None = None, measurement_class: str | None = None
) -> str | None:
    """Return the unit for ``value``, or ``None`` for null-like input.

    By default this is the ``.canonical`` equivalence key -- the source's own unit/prefix with
    typography normalized and any ``×10^N`` factor folded out (``×10⁻⁶ cm/s`` -> ``cm/s``,
    ``µM`` -> ``µM``, ``day`` -> ``d``). There is NO automatic standardization across prefixes or
    time units, so magnitudes never shift unexpectedly. Standardization to one domain unit per
    measurement is opt-in: pass ``measurement_class`` (a known endpoint class, e.g. ``"auc"`` or
    ``"potency"``) or ``targets`` (explicit unit strings). Transforms return the ``log(...)`` form
    unchanged.
    """
    result = canonicalize_unit(value)
    if result.cleaned == "":
        return None
    if measurement_class is not None:
        target = _class_target(measurement_class)
        # Guard: flag a unit that does not fit the declared class instead of coercing it.
        return target if _class_compatible(result, target) else None
    if result.transform:
        return result.canonical
    target = _resolve_target(result, targets)
    return target or result.canonical


def canonicalized_value(
    value: Any,
    unit: Any,
    targets: Iterable[str] | None = None,
    measurement_class: str | None = None,
) -> float | None:
    """Return ``value`` expressed in :func:`canonicalized_unit`'s unit.

    By default the value stays in the source's own unit with only the ``×10^N`` notation folded
    into it (``2.5`` + ``×10⁻⁶ cm/s`` -> ``2.5e-6`` in ``cm/s``; ``2`` + ``µM`` -> ``2`` in
    ``µM``) -- no prefix/time conversion, so no magnitude shifts. Returns ``None`` for non-numeric
    values, and the value unchanged for transforms and null-like units. Opt into standardization
    with ``measurement_class`` (converts to that class's target and reinterprets an ambiguous form,
    e.g. ``ng/mL·h`` under ``"auc"`` -> ``ng·h/mL``) or ``targets``; a unit that does not match a
    declared conversion class (e.g. ``"%"`` under ``"auc"``) returns ``None`` so callers can retain
    the unconverted record.
    """
    number = _to_float(value)
    if number is None:
        return None
    result = canonicalize_unit(unit)
    if result.cleaned == "":
        return number
    if measurement_class is not None:
        target = _class_target(measurement_class)
        # Guard: return None (flag) when the unit is not compatible with the declared class.
        if not _class_compatible(result, target):
            return None
        target_scale = canonicalize_unit(target).scale
        if target_scale == 0:
            return number
        return number * (_magnitude_under(result.cleaned, target) / target_scale)
    if result.transform:
        return number
    target = _resolve_target(result, targets) or result.canonical
    target_scale = canonicalize_unit(target).scale
    if target_scale == 0:
        return number
    return number * (result.scale / target_scale)


def canonicalize_measurement(
    value: Any,
    unit: Any,
    targets: Iterable[str] | None = None,
    measurement_class: str | None = None,
) -> tuple[float | None, str | None]:
    """Return ``(canonicalized_value, canonicalized_unit)`` as a consistent (value, unit) pair."""
    return (
        canonicalized_value(value, unit, targets, measurement_class),
        canonicalized_unit(unit, targets, measurement_class),
    )


# Expected dimension signature per named endpoint quantity-kind.
_QUANTITY_DIMENSIONS: dict[str, tuple[tuple[str, int], ...]] = {
    "concentration": (("amount", 1), ("volume", -1)),        # molar, e.g. µM
    "mass_concentration": (("mass", 1), ("volume", -1)),     # e.g. ng/mL
    "permeability": (("length", 1), ("time", -1)),           # e.g. cm/s
    "clearance": (("time", -1), ("volume", 1)),              # e.g. mL/min
    "time": (("time", 1),),
    "fraction_percent": (),                                   # dimensionless
    "auc": (("mass", 1), ("time", 1), ("volume", -1)),        # e.g. ng·h/mL
}


def _normalize_signature(dimension: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(dimension))


def units_compatible(value: Any, expected: str) -> bool | None:
    """Return whether ``value`` is dimensionally compatible with an endpoint kind.

    ``expected`` is a key of :data:`_QUANTITY_DIMENSIONS`.  Returns ``None`` when
    the unit contains unresolved tokens (compatibility cannot be decided) so
    callers can flag rather than silently mis-merge.
    """
    if expected not in _QUANTITY_DIMENSIONS:
        raise KeyError(f"unknown endpoint quantity-kind: {expected!r}")
    canonical = canonicalize_unit(value)
    if canonical.unknown_tokens:
        return None
    if canonical.cleaned == "":
        return None
    if canonical.transform:
        # A transformed quantity (e.g. log10(cm/s)) is not the raw quantity an endpoint
        # expects; base-dimension compatibility does not apply -- let the caller decide.
        return None
    return _normalize_signature(canonical.dimension) == _normalize_signature(_QUANTITY_DIMENSIONS[expected])
