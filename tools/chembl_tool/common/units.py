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

* Providing ``task`` and a dynamic ``assay`` mapping activates the central exact-match
  contextual policy. Matching reviewed rules always replace the canonical prefix; the
  shared parser does not hard-code an assay schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any


UNIT_NORMALIZER_VERSION = "unit_normalizer.v6"
CONTEXTUAL_UNIT_POLICY_SCHEMA_VERSION = "contextual_canonical_unit_policy.schema.v1"
DEFAULT_CONTEXTUAL_UNIT_POLICY_PATH = Path(__file__).with_name(
    "contextual_unit_policy.json"
)
QUALIFIER_VOCABULARY_SCHEMA_VERSION = "qualifier_vocabulary.schema.v1"
DEFAULT_QUALIFIER_VOCABULARY_PATH = Path(__file__).with_name(
    "qualifier_vocabulary_policy.json"
)

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
    "wk": "wk", "week": "wk", "weeks": "wk",
    # volume (litre): capital L, lowercase prefix
    "l": "L", "ml": "mL", "µl": "µL", "ul": "µL", "dl": "dL", "cl": "cL",
    "nl": "nL", "pl": "pL", "kl": "kL",
    "liter": "L", "litre": "L", "liters": "L", "litres": "L",
    "milliliter": "mL", "millilitre": "mL",
    "milliliters": "mL", "millilitres": "mL",
    # amount-of-substance concentration (molar): capital M
    "M": "M", "mM": "mM", "µM": "µM", "uM": "µM", "nM": "nM", "pM": "pM",
    # mass
    "g": "g", "gm": "g", "gram": "g", "grams": "g",
    "mg": "mg", "µg": "µg", "ug": "µg", "mcg": "µg", "ng": "ng", "pg": "pg", "kg": "kg",
    "da": "Da", "Da": "Da", "kda": "kDa", "kDa": "kDa", "dalton": "Da", "daltons": "Da",
    # length
    "m": "m", "cm": "cm", "mm": "mm", "nm": "nm", "µm": "µm", "um": "µm", "pm": "pm",
    # amount
    "mol": "mol", "mmol": "mmol", "µmol": "µmol", "umol": "µmol",
    "nmol": "nmol", "pmol": "pmol",
    "mole": "mol", "moles": "mol", "µmoles": "µmol", "umoles": "µmol",
    "nmoles": "nmol", "pmoles": "pmol", "mmoles": "mmol",
    "mmole": "mmol", "µmole": "µmol", "umole": "µmol",
    "nmole": "nmol", "pmole": "pmol",
    # dimensionless / qualifiers
    "%": "%", "%ID": "%ID", "percent": "%", "pct": "%",
    "times": "fold", "fold": "fold", "folds": "fold", "fraction": "fraction", "fold": "fold", "ratio": "ratio", "protein": "protein", "cells": "cells",
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
    # A letter run touching a digit is part of an alphanumeric designator
    # (``CYP2D6``), not a unit token: real unit exponents were already split
    # onto ``^`` before this pass, so ``cm^2`` never reaches here as ``cm2``.
    # Casing such a run corrupts the name (``CYP2D6`` -> ``CYP2d6``).
    text = match.string
    before = text[match.start() - 1] if match.start() else ""
    after = text[match.end()] if match.end() < len(text) else ""
    if before.isdigit() or after.isdigit():
        return token
    if token in _TOKEN_CASE:
        return _TOKEN_CASE[token]
    lowered = token.lower()
    return _TOKEN_CASE.get(lowered, token)


def _is_known_unit_token(token: str) -> bool:
    """True when ``token`` names a token the structural parser can resolve.

    Consults every declared qualifier, not just the calling task's: ``mg protein^-1`` and
    ``g brain^-1`` are real bases whose ``-1`` is an exponent no matter who is parsing.
    Deciding that is tokenization, so the union keeps ``clean_unit`` context-free; whether
    the token resolves stays task-scoped later. Tokens outside the vocabulary are left
    alone, which is what protects hyphenated designators (``P-450``, ``CYP-1``) from being
    read as exponents.
    """
    cased = _TOKEN_CASE.get(token, _TOKEN_CASE.get(token.lower(), token))
    if cased in _BASE_UNITS or cased in load_qualifier_vocabulary().all_tokens:
        return True
    return len(cased) > 1 and cased[0] in _SI_PREFIX and cased[1:] in _BASE_UNITS


def _bare_exponent(match: re.Match[str]) -> str:
    """Promote a trailing digit to an exponent only after a real unit token.

    ``cm2`` and ``mL-1`` are unit exponents, but the same shape occurs inside
    enzyme and protein designators (``CYP2B6``, ``CYP3A4``) where the digit is
    part of the name. Rewriting those produced ``CYP^2·B6`` -- a fake squared
    dimension on an unresolvable token. Gating on a resolvable base keeps the
    unit forms and leaves designators intact for the fail-closed path.
    """
    base, exponent = match.group("base"), match.group("exponent")
    if not _is_known_unit_token(base):
        return match.group(0)
    # The base may be a prefix of a longer designator that is itself declared: once ``CYP``
    # is vocabulary, ``CYP2D6`` would otherwise split into ``CYP^2·D6``. Whenever the run
    # continues past the digits, prefer the longest declared token.
    run = re.match(r"[A-Za-zµ%]+[\w-]*", match.string[match.start():])
    if run and run.group(0) != f"{base}{exponent}" and _is_known_unit_token(run.group(0)):
        return match.group(0)
    return f"{base}^{exponent}"


# Numeric words are basis multipliers, not units: ``million cells`` is the same basis as
# ``10^6 cells``. They must fold into scale -- treating them as dimensionless qualifiers
# would leave the value wrong by the factor while making the unit look resolved.
_NUMBER_WORDS = {
    "hundred": 100.0,
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}

_PER_RE = re.compile(r"(?i)\bper\b")


def _apply_per_division(text: str) -> str:
    """Rewrite the word ``per`` to ``/``, grouping composite divisors.

    ``per`` is a binary operator: everything up to the next ``per`` is a single
    divisor. When that divisor is itself a ratio (``ng/mL per mg/kg``, the usual
    dose-normalized exposure shape), the grouping is load-bearing -- a bare ``/``
    would flatten it to ``ng/mL/mg/kg`` and put ``kg`` in the denominator, which
    inverts its dimension. Parenthesising the divisor hands it to the recursive
    reading :func:`_evaluate` already documents. Divisors with no internal ``/``
    (``µL/min per mg protein``) render as a plain ``/`` so the conventional flat
    chain is untouched.
    """
    head, *divisors = _PER_RE.split(text)
    if not divisors:
        return text
    rendered = head.strip()
    for divisor in divisors:
        divisor = divisor.strip()
        if not divisor:
            continue
        rendered += f"/({divisor})" if "/" in divisor else f"/{divisor}"
    return rendered


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
    # ``per cent`` is one word split by a space, not a division: join it before the ``per``
    # operator runs, or the rewrite yields ``/cent``.
    text = re.sub(r"(?i)\bper\s+cent\b", "percent", text)
    # ``sq.cm`` / ``sq cm`` is cm^2 and ``cu.cm`` is cm^3. Left alone these parse one power
    # short (``µg/sq.cm`` reads as length:-1), so the dimension is wrong rather than merely
    # unresolved -- it only fails closed today because ``sq`` itself is unknown.
    text = re.sub(r"(?i)\bsq\.?\s*(cm|mm|m|µm|nm|in|ft)\.?", r"\1^2", text)
    text = re.sub(r"(?i)\bcu\.?\s*(cm|mm|m|µm|nm)\.?", r"\1^3", text)
    text = _apply_per_division(text)  # "ng per mL" -> "ng/mL"
    # A leading multiplication marker before ordinary E notation is redundant
    # but common in extracted table cells (``x 1e-6 cm/s``).
    text = re.sub(
        r"(?i)^(?:x|×)\s+(?=(?:\d+(?:\.\d*)?|\.\d+)\s*e[+-]?\d+)",
        "",
        text,
    )
    text = re.sub(r"(?i)\bx(?=\s*10)", "×", text)  # ascii "x10^-6" -> "×10^-6"
    text = re.sub(r"\^\{(-?\d+)\}", r"^\1", text)  # LaTeX braces "10^{-6}" -> "10^-6"
    text = re.sub(r"\^\(\s*([+-]?\d+)\s*\)", r"^\1", text)
    text = re.sub(r"(?<=[A-Za-zµ])\.(?=[A-Za-zµ])", "·", text)  # "ng.h" -> "ng·h"
    text = re.sub(r"(?<=[0-9)])\.(?=[A-Za-zµ%])", "·", text)
    # Publishers sometimes retain abbreviation punctuation around division
    # (``µg./mL.``). A period immediately before ``/`` or at the end carries
    # no dimensional meaning; internal periods remain multiplication marks.
    text = re.sub(r"(?<=[A-Za-zµ])\.(?=/|$)", "", text)
    # Bare unit exponents ("mL-1", "s-1", "cm2") -> caret form; the negative-lookahead
    # and letter-lookbehind keep ``10-6`` / ``x10-6`` scale factors untouched.
    text = re.sub(
        r"(?<![0-9][eE])(?P<base>[A-Za-zµ%]+)(?P<exponent>-\d+|[23])(?![\d^])",
        _bare_exponent,
        text,
    )
    # Compact products commonly omit the multiplication mark after an
    # exponent (``cm2h``, ``cm^-2h^-1``).  The exponent boundary is explicit,
    # so inserting the operator preserves rather than infers the unit.
    text = re.sub(r"(\^-?\d+)(?=[A-Za-zµ%])", r"\1·", text)
    # A basis count written flush against its unit (``100g^-1``, ``mL/100g/min``) is two
    # factors, not one token. Split only when what follows is a real unit token: that keeps
    # ``1e-6 cm/s`` intact (one number) and leaves designators such as ``1a`` alone, the
    # same way ``_bare_exponent`` protects ``CYP2B6``.
    text = re.sub(
        r"(?<![A-Za-zµ%^\d.])(\d+)([A-Za-zµ]+)",
        lambda m: f"{m.group(1)} {m.group(2)}"
        if _is_known_unit_token(m.group(2))
        else m.group(0),
        text,
    )
    text = re.sub(r"\s*([/·*])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _TOKEN_RE.sub(_case_token, text)
    # Exact compact inverse-time/permeability spellings.  Restrict this to
    # whole tokens so ordinary words containing ``cms`` are never rewritten.
    text = re.sub(r"(?i)(?<![A-Za-zµ])cms\^?-?1(?![A-Za-zµ\d])", "cm·s^-1", text)
    text = re.sub(
        r"(?i)(?<![A-Za-zµ])cm(?:sec|second)\^?-?1(?![A-Za-zµ\d])",
        "cm·s^-1",
        text,
    )
    # A signed exponent immediately followed by a unit is still explicit
    # scientific notation (``10-6cm/s``); add only the missing boundary.
    text = re.sub(
        r"(?i)(10(?:\^[+-]?\d+|[+-]\d+))(?=[A-Za-zµ%])",
        r"\1 ",
        text,
    )
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
    "Da": (1.66053906660e-27, {"mass": 1}),   # unified atomic mass unit
    "kDa": (1.66053906660e-24, {"mass": 1}),
    "wk": (604800.0, {"time": 1}),  # 7 d
    "M": (1.0, {"amount": 1, "volume": -1}),  # molar == mol/L
    "%": (1.0, {}),
    "%ID": (1.0, {}),
    "fraction": (1.0, {}),
    "fold": (1.0, {}),
    "ratio": (1.0, {}),
    "ppm": (1.0, {}),
    "ppb": (1.0, {}),
    "dimensionless": (1.0, {}),
    "cells": (1.0, {"count": 1}),
}
# Qualifier tokens carry no dimension but are preserved in the canonical string. (``cells`` is
# NOT here -- it is a genuine count basis, e.g. ``µL/min/10^6 cells``.) The vocabulary itself is
# task-scoped and lives in ``qualifier_vocabulary_policy.json``: one task's vocabulary must never
# silently change another task's parsing. The structural parser stays context-free and reports
# every unresolved token in ``unknown_tokens``; :func:`canonicalize_unit` then clears the tokens
# the requested task declares. See :func:`load_qualifier_vocabulary`.
_SI_PREFIX = {"p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1, "k": 1e3}

_SCALE_RE = re.compile(r"^(?:×|x)?10(?:\^([-+]?\d+)|([-+]\d+))$")
# An explicit ``×10^N`` / ``x10^N`` factor anywhere in the string (multiplicative, any position).
_NOTATION_FACTOR_RE = re.compile(r"(?:×|x)\s*10(?:\^([-+]?\d+)|([-+]\d+))")
_ANY_NOTATION_RE = re.compile(r"(?:×|x)?\s*10(?:\^([-+]?\d+)|([-+]\d+))")
_LEADING_DECIMAL_SCIENTIFIC_RE = re.compile(
    r"^\s*(?P<coefficient>(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:[eE]\s*(?P<e_exponent>[+-]?\d+)|"
    r"(?:×|x)\s*10(?:\^(?P<caret>[+-]?\d+)|(?P<signed>[+-]\d+)))"
    r"(?=\s|[A-Za-zµ%])",
)
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


def _extract_unambiguous_notation(
    cleaned: str,
    notation_status: str,
) -> tuple[float, str]:
    """Return the explicit magnitude factor and notation-free unit body.

    Both ordinary unit parsing and endpoint-class conversion must use the same
    extraction.  Otherwise a leading decimal form such as ``2e-6 cm/s`` can be
    recognized while building the unit, then lose its coefficient when the
    class-aware magnitude is recomputed.
    """
    if notation_status == "ambiguous_scientific_notation":
        return 1.0, cleaned

    notation = 1.0
    decimal_scientific = _LEADING_DECIMAL_SCIENTIFIC_RE.match(cleaned)
    if decimal_scientific:
        exponent = (
            decimal_scientific.group("e_exponent")
            or decimal_scientific.group("caret")
            or decimal_scientific.group("signed")
        )
        notation *= float(decimal_scientific.group("coefficient")) * 10.0 ** int(
            exponent
        )
        body = cleaned[decimal_scientific.end() :]
    else:
        body = cleaned

    def pull(match: re.Match[str]) -> str:
        nonlocal notation
        exponent = match.group(1) or match.group(2)
        notation *= 10.0 ** int(exponent)
        return " "

    body = _NOTATION_FACTOR_RE.sub(pull, body)
    return notation, body


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


def _evaluate(
    expr: str, *, fold_numeric_basis: bool = True
) -> tuple[float, dict[str, int]]:
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
        # Plain tokens are collected before accumulating so a trailing basis exponent
        # can reach the unit it qualifies (see _bind_compound_basis).
        plain: list[list[Any]] = []
        for factor in _split_top_level(segment, "·*× \t"):
            factor = factor.strip()
            if not factor:
                continue
            scale_match = _SCALE_RE.match(factor)
            if scale_match:
                exponent = scale_match.group(1) or scale_match.group(2)
                scale *= 10.0 ** (int(exponent) * sign)
                continue
            paren_match = _PAREN_RE.match(factor)
            if paren_match:
                inner_scale, inner_exponents = _evaluate(
                    paren_match.group(1), fold_numeric_basis=fold_numeric_basis
                )
                power = int(paren_match.group(2)) if paren_match.group(2) else 1
                multiplier = power * sign
                if inner_scale != 1.0:
                    scale *= inner_scale ** multiplier
                for token, exponent in inner_exponents.items():
                    exponents[token] = exponents.get(token, 0) + exponent * multiplier
                continue
            base, separator, exp_text = factor.partition("^")
            if separator and not re.fullmatch(r"-?\d+", exp_text):
                # Fail closed.  Previously ``cm^2h`` was treated as ``cm`` by
                # silently discarding the malformed exponent suffix.
                token = factor
                exponent = 1
            else:
                token = base or factor
                exponent = int(exp_text) if separator else 1
            plain.append([token, exponent, bool(separator)])
        if fold_numeric_basis:
            scale *= _fold_numeric_basis(plain, sign)
        _bind_compound_basis(plain)
        for token, exponent, _ in plain:
            exponents[token] = exponents.get(token, 0) + exponent * sign
    return scale, exponents


def _fold_numeric_basis(factors: list[list[Any]], sign: int) -> float:
    """Fold bare-integer basis multipliers into scale; return the multiplier.

    A bare integer is a basis count, not a unit: ``mL/100 g`` is per *hundred* grams,
    ``mg/24 h`` per *24* hours. It inherits the orientation of the unit it qualifies -- the
    next dimensioned token to its right -- which is what separates ``mL/100 g`` from
    ``mL·100 g^-1·min^-1``: the ``100`` sits in the numerator there, but the basis it counts
    is inverted, so both divide. Only the *sign* is inherited, never the magnitude, so the
    ``10`` in ``µg/10 cm^2`` divides once rather than squaring.

    An integer with no unit to its right is left in place as an unresolved token. A trailing
    ``mL/g/min ×100`` could equally be a per-100 basis or a reported-value multiplier, and
    guessing either way silently rescales real measurements -- so it fails closed instead.
    Consumed integers are removed from ``factors``; ``10^N`` never reaches here.
    """
    multiplier = 1.0
    consumed: list[int] = []
    for index, (token, _, explicit) in enumerate(factors):
        if explicit:
            continue
        word_value = _NUMBER_WORDS.get(token.casefold())
        if word_value is None and not token.isdigit():
            continue
        if token.strip("0") == "":
            # A zero basis is meaningless and cannot be inverted; leave it unresolved so
            # the unit fails closed rather than raising or folding to infinity.
            continue
        if token == "1":
            # Multiplicative identity: ``1/min`` is min^-1 and ``mg·h/L/1 mg/kg`` is
            # per one mg/kg. It carries no magnitude wherever it sits, so it needs no
            # basis to attach to and can always be dropped.
            consumed.append(index)
            continue
        for following in range(index + 1, len(factors)):
            next_token, next_exponent, _ = factors[following]
            dimension = _resolve_token(next_token)[1]
            if not dimension:
                continue  # unresolved or dimensionless: keep looking right
            orientation = sign * (-1 if next_exponent < 0 else 1)
            multiplier *= (word_value if word_value is not None else float(token)) ** orientation
            consumed.append(index)
            break
    for index in reversed(consumed):
        del factors[index]
    return multiplier


def _bind_compound_basis(factors: list[list[Any]]) -> None:
    """Attach a trailing negative exponent to the basis unit it qualifies, in place.

    ``µL·min^-1·mg protein^-1`` and ``µL/min/mg protein`` are the same quantity, but the
    first parsed as ``µL·mg/(min·protein)`` -- mass positive -- because the ``^-1`` bound to
    ``protein`` alone. Scientifically the exponent applies to the whole ``mg protein``
    basis, so the unit it qualifies inherits it.

    Fires only for an explicit *negative* exponent on a token that does not resolve as a
    unit, and only onto the nearest preceding dimensioned unit that carries no exponent of
    its own. Dimensionless bases (``%``, ``fold``) are skipped: ``% dose^-1`` has no basis
    to invert, so rewriting it would churn the canonical key for no dimensional gain.
    Callers pass one ``/``-segment at a time, so this can never cross a division boundary.
    """
    for index, (token, exponent, explicit) in enumerate(factors):
        if not explicit or exponent >= 0 or _resolve_token(token)[1] is not None:
            continue
        for previous in range(index - 1, -1, -1):
            candidate, _, candidate_explicit = factors[previous]
            if not _resolve_token(candidate)[1]:
                continue  # unresolved or dimensionless: keep walking left
            if not candidate_explicit:
                factors[previous][1] = exponent
            break


def _resolve_token(token: str) -> tuple[float, dict[str, int] | None]:
    """Return ``(scale, dimension)`` for a single unit token.

    ``dimension`` is ``None`` when the token cannot be resolved (an unknown unit).
    ``{}`` denotes a resolved but dimensionless token. Task-declared qualifiers are
    deliberately *not* resolved here -- this function is context-free, so they surface as
    unknown tokens and :func:`canonicalize_unit` clears them for the requested task.
    """
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
    contextual_policy_status: str = "not_requested"
    contextual_rule_id: str | None = None
    contextual_policy_version: str | None = None
    contextual_conversion_factor: float | None = None
    normalizer_version: str = UNIT_NORMALIZER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


JSONScalar = str | int | float | bool | None


@dataclass(frozen=True)
class ContextualCanonicalUnitRule:
    rule_id: str
    task: str
    match: tuple[tuple[str, JSONScalar], ...]
    accepted_units: tuple[str, ...]
    canonical_unit: str


@dataclass(frozen=True)
class ContextualCanonicalUnitPolicy:
    schema_version: str
    policy_version: str
    rules: tuple[ContextualCanonicalUnitRule, ...]
    path: str
    sha256: str


@dataclass(frozen=True)
class QualifierVocabularyPolicy:
    """Which unresolved tokens each task declares dimensionless.

    ``shared`` applies to every call; ``by_task`` adds the requested task's own tokens.
    A task that declares nothing therefore still gets the shared vocabulary, and an
    unknown task id resolves to the shared vocabulary alone rather than raising.
    """

    schema_version: str
    policy_version: str
    shared: tuple[str, ...]
    by_task: tuple[tuple[str, tuple[str, ...]], ...]
    path: str
    sha256: str

    def tokens_for(self, task: str | None) -> frozenset[str]:
        tokens = set(self.shared)
        if task is not None:
            tokens.update(dict(self.by_task).get(task, ()))
        return frozenset(tokens)

    @property
    def all_tokens(self) -> frozenset[str]:
        """Every declared token, across all tasks.

        Used only for *tokenization* -- deciding that the ``-1`` in ``g brain-1`` is an
        exponent rather than part of a name. That question has one answer regardless of who
        is asking, so consulting the union keeps ``clean_unit`` context-free. Whether a
        token then *resolves* stays task-scoped in :func:`_apply_qualifier_vocabulary`.
        """
        tokens = set(self.shared)
        for _, task_tokens in self.by_task:
            tokens.update(task_tokens)
        return frozenset(tokens)


@lru_cache(maxsize=100_000)
def _canonicalize_unit_basic(value: Any) -> CanonicalUnit:
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
        inner = _canonicalize_unit_basic(func_match.group(2))
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
    notation, body = _extract_unambiguous_notation(cleaned, notation_status)
    # Under OCR-compressed notation (``×106 cm/s``) the digits may be a lost
    # ``10^6`` rather than a basis count. Folding them would resolve the unit and
    # silently undo the quarantine, so leave them as unresolved tokens.
    scale, token_exponents = _evaluate(
        body,
        fold_numeric_basis=notation_status != "ambiguous_scientific_notation",
    )
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


@lru_cache(maxsize=8)
def load_contextual_unit_policy(
    path: str | Path = DEFAULT_CONTEXTUAL_UNIT_POLICY_PATH,
) -> ContextualCanonicalUnitPolicy:
    """Load and fully validate the frozen dynamic-assay unit policy."""
    policy_path = Path(path)
    raw_bytes = policy_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "policy_version",
        "rules",
    }:
        raise ValueError("invalid contextual unit policy root")
    if payload["schema_version"] != CONTEXTUAL_UNIT_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported contextual unit policy schema: "
            f"{payload['schema_version']!r}"
        )
    policy_version = payload["policy_version"]
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("contextual unit policy_version must be nonempty")
    raw_rules = payload["rules"]
    if not isinstance(raw_rules, list):
        raise ValueError("contextual unit policy rules must be a list")

    rules: list[ContextualCanonicalUnitRule] = []
    seen_rule_ids: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict) or set(raw_rule) != {
            "rule_id",
            "task",
            "match",
            "accepted_units",
            "canonical_unit",
            "review",
        }:
            raise ValueError("invalid contextual unit rule shape")
        rule_id = raw_rule["rule_id"]
        task = raw_rule["task"]
        match = raw_rule["match"]
        accepted_units = raw_rule["accepted_units"]
        target_unit = raw_rule["canonical_unit"]
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen_rule_ids:
            raise ValueError(f"duplicate or invalid contextual rule_id: {rule_id!r}")
        seen_rule_ids.add(rule_id)
        if not isinstance(task, str) or not task:
            raise ValueError(f"invalid task in contextual rule {rule_id!r}")
        if not isinstance(match, dict) or not match:
            raise ValueError(f"contextual rule {rule_id!r} requires a match object")
        for field, expected in match.items():
            if not isinstance(field, str) or not field or not _is_json_scalar(expected):
                raise ValueError(f"invalid dynamic match in contextual rule {rule_id!r}")
        if (
            not isinstance(accepted_units, list)
            or len(accepted_units) < 2
            or not all(isinstance(unit, str) and unit for unit in accepted_units)
            or len(accepted_units) != len(set(accepted_units))
        ):
            raise ValueError(f"invalid accepted_units in contextual rule {rule_id!r}")
        if not isinstance(target_unit, str) or target_unit not in accepted_units:
            raise ValueError(
                f"canonical_unit must occur in accepted_units for {rule_id!r}"
            )
        if not isinstance(raw_rule["review"], dict):
            raise ValueError(f"invalid review metadata in contextual rule {rule_id!r}")

        target = _canonicalize_unit_basic(target_unit)
        if target.canonical != target_unit or target.unknown_tokens or target.transform:
            raise ValueError(f"noncanonical target unit in {rule_id!r}: {target_unit!r}")
        for unit in accepted_units:
            parsed = _canonicalize_unit_basic(unit)
            if parsed.canonical != unit or parsed.unknown_tokens or parsed.transform:
                raise ValueError(f"noncanonical accepted unit in {rule_id!r}: {unit!r}")
            if parsed.dimension != target.dimension:
                raise ValueError(f"dimension mismatch in contextual rule {rule_id!r}")
        rules.append(
            ContextualCanonicalUnitRule(
                rule_id=rule_id,
                task=task,
                match=tuple(sorted(match.items())),
                accepted_units=tuple(accepted_units),
                canonical_unit=target_unit,
            )
        )

    for index, left in enumerate(rules):
        for right in rules[index + 1 :]:
            if _rules_can_overlap(left, right):
                raise ValueError(
                    "ambiguous contextual unit rules can match the same assay: "
                    f"{left.rule_id!r}, {right.rule_id!r}"
                )
    return ContextualCanonicalUnitPolicy(
        schema_version=payload["schema_version"],
        policy_version=policy_version,
        rules=tuple(rules),
        path=str(policy_path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def contextual_unit_policy_manifest() -> dict[str, Any]:
    policy = load_contextual_unit_policy()
    return {
        "schema_version": policy.schema_version,
        "policy_version": policy.policy_version,
        "path": policy.path,
        "sha256": policy.sha256,
        "rule_ids": [rule.rule_id for rule in policy.rules],
        "matching": "exact_all_declared_fields_fail_closed",
    }


@lru_cache(maxsize=8)
def load_qualifier_vocabulary(
    path: str | Path = DEFAULT_QUALIFIER_VOCABULARY_PATH,
) -> QualifierVocabularyPolicy:
    """Load and fully validate the frozen task-scoped qualifier vocabulary."""
    policy_path = Path(path)
    raw_bytes = policy_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "policy_version",
        "vocabularies",
    }:
        raise ValueError("invalid qualifier vocabulary root")
    if payload["schema_version"] != QUALIFIER_VOCABULARY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported qualifier vocabulary schema: {payload['schema_version']!r}"
        )
    policy_version = payload["policy_version"]
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("qualifier vocabulary policy_version must be nonempty")
    raw_entries = payload["vocabularies"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("qualifier vocabulary vocabularies must be a nonempty list")

    shared: tuple[str, ...] | None = None
    by_task: dict[str, tuple[str, ...]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "scope",
            "task",
            "tokens",
            "review",
        }:
            raise ValueError("invalid qualifier vocabulary entry shape")
        scope = raw_entry["scope"]
        task = raw_entry["task"]
        tokens = raw_entry["tokens"]
        if scope not in {"shared", "task"}:
            raise ValueError(f"invalid qualifier vocabulary scope: {scope!r}")
        if scope == "shared":
            if task is not None:
                raise ValueError("shared qualifier vocabulary must not declare a task")
            if shared is not None:
                raise ValueError("duplicate shared qualifier vocabulary entry")
        else:
            if not isinstance(task, str) or not task:
                raise ValueError("task qualifier vocabulary requires a task id")
            if task in by_task:
                raise ValueError(f"duplicate qualifier vocabulary for task {task!r}")
        if not isinstance(tokens, list) or any(
            not isinstance(token, str) or not token for token in tokens
        ):
            raise ValueError(f"invalid qualifier tokens for scope {scope!r}")
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"duplicate qualifier tokens for scope {scope!r}")
        if tokens != sorted(tokens):
            raise ValueError(f"qualifier tokens must be sorted for scope {scope!r}")
        for token in tokens:
            # A qualifier is dimensionless by declaration. Letting a real unit be declared
            # one would silently erase its dimension everywhere it appears.
            if _resolve_token(token)[1] is not None:
                raise ValueError(f"qualifier token resolves as a unit: {token!r}")
        if not isinstance(raw_entry["review"], dict):
            raise ValueError(f"invalid qualifier review metadata for scope {scope!r}")
        if scope == "shared":
            shared = tuple(tokens)
        else:
            by_task[task] = tuple(tokens)

    if shared is None:
        raise ValueError("qualifier vocabulary requires a shared entry")
    for task, tokens in by_task.items():
        overlap = sorted(set(tokens) & set(shared))
        if overlap:
            raise ValueError(
                f"task {task!r} redeclares shared qualifier tokens: {overlap}"
            )
    return QualifierVocabularyPolicy(
        schema_version=payload["schema_version"],
        policy_version=policy_version,
        shared=shared,
        by_task=tuple(sorted(by_task.items())),
        path=str(policy_path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def qualifier_vocabulary_manifest() -> dict[str, Any]:
    policy = load_qualifier_vocabulary()
    return {
        "schema_version": policy.schema_version,
        "policy_version": policy.policy_version,
        "path": policy.path,
        "sha256": policy.sha256,
        "shared_tokens": list(policy.shared),
        "tasks": {task: list(tokens) for task, tokens in policy.by_task},
        "scoping": "shared_plus_declared_task_fail_closed",
    }


def _apply_qualifier_vocabulary(
    basic: CanonicalUnit, task: str | None
) -> CanonicalUnit:
    """Clear the tokens ``task`` declares dimensionless from ``unknown_tokens``.

    A qualifier and an unknown token differ in exactly this one field: ``_resolve_token``
    returns ``(1.0, {})`` for a resolved dimensionless token, which contributes nothing to
    ``scale`` or ``dimension``, and ``canonical`` is built from the token exponents without
    consulting it at all. So the vocabulary can be applied after the cached parse instead of
    inside it, which keeps ``_canonicalize_unit_basic`` context-free and its cache keyed on
    the unit string alone.
    """
    if not basic.unknown_tokens:
        return basic
    declared = load_qualifier_vocabulary().tokens_for(task)
    remaining = tuple(
        token for token in basic.unknown_tokens if token not in declared
    )
    if remaining == basic.unknown_tokens:
        return basic
    return replace(
        basic,
        unknown_tokens=remaining,
        is_dimensionless=not basic.dimension and not remaining,
    )


def canonicalize_unit(
    value: Any,
    *,
    task: str | None = None,
    assay: Mapping[str, JSONScalar] | None = None,
) -> CanonicalUnit:
    """Parse a unit, then apply the task's qualifier vocabulary and assay unit rule.

    Two independent layers of task context, in order:

    1. Qualifier vocabulary -- always applied. The shared tokens plus whatever ``task``
       declares are cleared from ``unknown_tokens``. Passing no task yields the shared
       vocabulary only, so a task-scoped token stays unresolved rather than leaking.
    2. Contextual unit rules -- require ``task`` *and* ``assay`` together. Every field
       declared by a rule must be present and exactly equal; missing differs from
       explicit ``None``. Extra assay fields are ignored.
    """
    basic = _apply_qualifier_vocabulary(_canonicalize_unit_basic(value), task)
    if assay is None:
        if task is not None and not (isinstance(task, str) and task):
            raise ValueError("contextual canonicalization requires a nonempty task")
        return basic
    if not isinstance(task, str) or not task or not isinstance(assay, Mapping):
        raise ValueError("contextual canonicalization requires task and assay together")
    for field, actual in assay.items():
        if not isinstance(field, str) or not field or not _is_json_scalar(actual):
            raise ValueError("assay context must map nonempty strings to JSON scalars")
    policy = load_contextual_unit_policy()
    matches = [
        rule
        for rule in policy.rules
        if rule.task == task
        and basic.canonical in rule.accepted_units
        and all(field in assay and assay[field] == expected for field, expected in rule.match)
    ]
    if not matches:
        return replace(
            basic,
            contextual_policy_status="no_matching_rule",
            contextual_policy_version=policy.policy_version,
        )
    if len(matches) != 1:
        raise ValueError(
            "multiple contextual unit rules matched: "
            + ", ".join(rule.rule_id for rule in matches)
        )
    rule = matches[0]
    target = _canonicalize_unit_basic(rule.canonical_unit)
    source_canonical = _canonicalize_unit_basic(basic.canonical)
    factor = source_canonical.scale / target.scale
    return replace(
        basic,
        canonical=rule.canonical_unit,
        contextual_policy_status=(
            "matched_target_unit"
            if basic.canonical == rule.canonical_unit
            else "converted"
        ),
        contextual_rule_id=rule.rule_id,
        contextual_policy_version=policy.policy_version,
        contextual_conversion_factor=factor,
    )


def _is_json_scalar(value: Any) -> bool:
    return value is None or (
        isinstance(value, (str, int, float, bool))
        and not (isinstance(value, float) and not math.isfinite(value))
    )


def _rules_can_overlap(
    left: ContextualCanonicalUnitRule,
    right: ContextualCanonicalUnitRule,
) -> bool:
    if left.task != right.task or not set(left.accepted_units) & set(right.accepted_units):
        return False
    left_match = dict(left.match)
    right_match = dict(right.match)
    return all(
        left_match[field] == right_match[field]
        for field in set(left_match) & set(right_match)
    )


def scientific_notation_metadata(raw: Any, cleaned: str | None = None) -> tuple[str, float | None]:
    """Classify unit scale notation without guessing OCR-compressed exponents."""
    raw_text = _as_nonempty_text(raw) or ""
    if _AMBIGUOUS_COMPRESSED_NOTATION_RE.search(raw_text):
        return "ambiguous_scientific_notation", None
    normalized = cleaned if cleaned is not None else clean_unit(raw)
    if not normalized:
        return "none", None
    decimal_scientific = _LEADING_DECIMAL_SCIENTIFIC_RE.match(normalized)
    if decimal_scientific:
        exponent = (
            decimal_scientific.group("e_exponent")
            or decimal_scientific.group("caret")
            or decimal_scientific.group("signed")
        )
        return (
            "unambiguous_scientific_notation",
            float(decimal_scientific.group("coefficient")) * 10.0 ** int(exponent),
        )
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
    "diffusivity": "cm^2/s",
    "areic_dose": "µg/cm^2",
    "duration": "h",
    "rate_constant": "h^-1",
}


def _resolve_target(
    result: CanonicalUnit, targets: Iterable[str] | None, task: str | None = None
) -> str:
    """Return an explicit per-call target unit for ``result``, or ``""`` for no standardization.

    ``targets`` is an iterable of unit strings; each supplies the target for its own dimension
    (e.g. ``targets=["cm/h"]`` standardizes permeability to cm/h). With no ``targets`` there is
    NO automatic standardization -- the value/unit stay in the source's own (folded) form.
    """
    if targets:
        for unit in targets:
            override = canonicalize_unit(unit, task=task)
            if override.dimension == result.dimension and not override.unknown_tokens:
                return override.cleaned
    return ""


def _class_target(measurement_class: str) -> str:
    if measurement_class not in _MEASUREMENT_CLASSES:
        raise KeyError(f"unknown measurement_class: {measurement_class!r}")
    return _MEASUREMENT_CLASSES[measurement_class]


def _class_compatible(
    result: CanonicalUnit, target_unit: str, task: str | None = None
) -> bool:
    """Whether ``result``'s unit could plausibly belong to the class whose unit is ``target_unit``.

    Compatible iff the unit resolves cleanly (no unknown tokens, not a transform) and carries the
    same SET of base dimensions as the target (exponent signs ignored, since the ``/X·Y`` shape is
    what the class disambiguates). So ``ng/mL·h`` is compatible with AUC (``ng·h/mL``), but ``%``,
    ``fold``, and a plain ``ng/mL`` are not -- a mislabeled record is flagged rather than coerced.
    """
    if result.unknown_tokens or result.transform:
        return False
    target = canonicalize_unit(target_unit, task=task)
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


def _magnitude_under(
    source_cleaned: str, target_unit: str, task: str | None = None
) -> float:
    """Scale of ``source_cleaned``'s magnitude when its tokens take the target's numer/denom signs.

    Lets a caller reinterpret an ambiguous form under a known class: e.g. ``ng/mL·h`` parses
    conservatively as a rate, but under target ``ng·h/mL`` (AUC) its ``h`` moves to the
    numerator so the value scales correctly.
    """
    notation_status, _ = scientific_notation_metadata(source_cleaned)
    notation, source_body = _extract_unambiguous_notation(
        source_cleaned, notation_status
    )
    struct_scale, src_exponents = _evaluate(
        source_body,
        fold_numeric_basis=notation_status != "ambiguous_scientific_notation",
    )
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
    value: Any,
    targets: Iterable[str] | None = None,
    measurement_class: str | None = None,
    *,
    task: str | None = None,
    assay: Mapping[str, JSONScalar] | None = None,
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
    result = canonicalize_unit(value, task=task, assay=assay)
    if result.cleaned == "":
        return None
    if measurement_class is not None:
        target = _class_target(measurement_class)
        # Guard: flag a unit that does not fit the declared class instead of coercing it.
        return target if _class_compatible(result, target, task) else None
    if result.transform:
        return result.canonical
    target = _resolve_target(result, targets, task)
    return target or result.canonical


def canonicalized_value(
    value: Any,
    unit: Any,
    targets: Iterable[str] | None = None,
    measurement_class: str | None = None,
    *,
    task: str | None = None,
    assay: Mapping[str, JSONScalar] | None = None,
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
    result = canonicalize_unit(unit, task=task, assay=assay)
    if result.cleaned == "":
        return number
    if measurement_class is not None:
        target = _class_target(measurement_class)
        # Guard: return None (flag) when the unit is not compatible with the declared class.
        if not _class_compatible(result, target, task):
            return None
        target_scale = canonicalize_unit(target, task=task).scale
        if target_scale == 0:
            return number
        return number * (_magnitude_under(result.cleaned, target, task) / target_scale)
    if result.transform:
        return number
    target = _resolve_target(result, targets, task) or result.canonical
    target_scale = canonicalize_unit(target, task=task).scale
    if target_scale == 0:
        return number
    return number * (result.scale / target_scale)


def canonicalize_measurement(
    value: Any,
    unit: Any,
    targets: Iterable[str] | None = None,
    measurement_class: str | None = None,
    *,
    task: str | None = None,
    assay: Mapping[str, JSONScalar] | None = None,
) -> tuple[float | None, str | None]:
    """Return ``(canonicalized_value, canonicalized_unit)`` as a consistent (value, unit) pair."""
    return (
        canonicalized_value(
            value,
            unit,
            targets,
            measurement_class,
            task=task,
            assay=assay,
        ),
        canonicalized_unit(
            unit,
            targets,
            measurement_class,
            task=task,
            assay=assay,
        ),
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


def units_compatible(
    value: Any, expected: str, *, task: str | None = None
) -> bool | None:
    """Return whether ``value`` is dimensionally compatible with an endpoint kind.

    ``expected`` is a key of :data:`_QUANTITY_DIMENSIONS`.  Returns ``None`` when
    the unit contains unresolved tokens (compatibility cannot be decided) so
    callers can flag rather than silently mis-merge.  Pass the same ``task`` the
    records were normalized under, or a unit resting on that task's vocabulary
    (``µL/min/g brain``) reads as undecidable here while resolving everywhere else.
    """
    if expected not in _QUANTITY_DIMENSIONS:
        raise KeyError(f"unknown endpoint quantity-kind: {expected!r}")
    canonical = canonicalize_unit(value, task=task)
    if canonical.unknown_tokens:
        return None
    if canonical.cleaned == "":
        return None
    if canonical.transform:
        # A transformed quantity (e.g. log10(cm/s)) is not the raw quantity an endpoint
        # expects; base-dimension compatibility does not apply -- let the caller decide.
        return None
    return _normalize_signature(canonical.dimension) == _normalize_signature(_QUANTITY_DIMENSIONS[expected])
