"""Skin_Reaction categorical response encoders.

The two qualitative sources carry no measurement, so they are excluded from
every pair bucket.  These frozen, reviewed encoders give the informative
subset of those records a continuous value; everything downstream is unchanged.

Four encoders on three latent scales.  The scales are separate ``canonical_unit``
values, and each encoder additionally contributes its ``categorical_encoder_id``
to the pair-bucket key, so no two of them are ever compared to each other:

``count_logit`` (``logit_response``)
    ``direct_skin_reaction`` rows with ``positive_count`` / ``total_tested`` and
    ``n >= 2``.  Genuine incidence data.  ``η = logit((k + ½)/(n + 1))``.

``single_subject_logit`` (``logit_response``)
    The same computation for ``n == 1``.  Held apart deliberately: 42.7% of the
    count rows are a single subject, and "this one person reacted" is a much
    weaker claim than "45 of 50 reacted".  Pooling them would let a 1/1 set the
    scale for a 45/50.

``ordinal_severity_grade`` (``ordinal_severity_grade``)
    The Draize-style ``+`` / ``++`` / ``+++`` / ``++++`` ladder in
    ``effect_metric``.

    This is **not** an incidence rate and must not be mapped onto one.  Of the
    graded rows that also carry counts, 94.7% have ``n == 1`` — the grade
    describes how strongly one subject reacted, not what fraction of a group
    did.  Encoding ``++`` as "50% positive" would conflate severity with
    incidence.  It therefore gets its own ordinal scale, and because the
    transfer policy standardises by within-bucket SD, the absolute spacing of
    the ladder is irrelevant — only the ratios between grades matter.

``signed_direction`` (``signed_effect_direction``)
    ``phototoxicity_irritation_local_damage`` ``result_label``.  ``protective``
    is the signed opposite of ``positive``, not a weaker positive, so a single
    ordinal ladder would place it on the wrong side of ``negative``.

``inconclusive``, ``not_classified`` and ``mixed_or_inconclusive`` receive no
value at all.  They are absence of information, not evidence of no effect, and
mapping them to the midpoint would fabricate a measurement.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.categorical_response import (
    LOGIT_RESPONSE_UNIT,
    ORDINAL_SEVERITY_UNIT,
    SIGNED_DIRECTION_UNIT,
    CategoricalEncoding,
    CategoricalResponsePolicy,
    count_logit,
    logit,
    render_measurement,
    shrunk_rate,
)


CATEGORICAL_RESPONSE_VERSION = "skin_reaction_categorical_response.v1"

DIRECT_SOURCE = "direct_skin_reaction"
PHOTOTOXICITY_SOURCE = "phototoxicity_irritation_local_damage"

# A bare percentage carries a rate but no denominator.  It is used directly as
# the probability; this nominal size only resolves the reported 0% and 100%
# boundaries, where the logit would otherwise be infinite.  It is a reviewed
# constant, not an estimate of the study's real sample size.
NOMINAL_PERCENT_SAMPLE_SIZE = 20.0

# Signed hazard direction.  Equal spacing asserts that protective and positive
# are equally far from no effect; SD standardisation makes the absolute scale
# irrelevant, so this ratio is the only reviewed choice here.
SIGNED_DIRECTION_ANCHORS: dict[str, float] = {
    "protective": -1.0,
    "negative": 0.0,
    "positive": 1.0,
}

# Outcomes that must never receive a value.
UNINFORMATIVE_LABELS = frozenset(
    {"inconclusive", "not_classified", "mixed_or_inconclusive", "equivocal"}
)

_GRADE_TEXT = re.compile(r"^(\+{1,4})(?![\w+])")
_NUMERIC_GRADE_TEXT = re.compile(r"^([1-4])\s*\+(?![\w+])")
_NEGATIVE_GRADE_TEXT = frozenset(
    {"-", "--", "—", "–", "0", "negative", "no reaction", "none", "nil", "no response"}
)
_PERCENT_TEXT = re.compile(
    r"^(\d+(?:\.\d+)?)\s*%\s*(?:positive|of\s+\w+|reacted|response|responders)?\.?$",
    re.IGNORECASE,
)
_FRACTION_TEXT = re.compile(r"^(\d+)\s*/\s*(\d+)\b")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _effect_text(record: Mapping[str, Any]) -> str:
    """The cleaned source measurement for a direct record.

    ``effect_metric`` is this source's ``measurement_field``, so the cleaning
    stage promotes it to ``measurement_text`` rather than keeping it as a
    context column.  Reading the raw name alone would silently never match.
    """
    return _text(record.get("measurement_text")) or _text(record.get("effect_metric"))


def _counts(record: Mapping[str, Any]) -> tuple[float, float] | None:
    """Parse the count pair, tolerating ``total_tested`` stored as a string."""
    raw_k = _text(record.get("positive_count"))
    raw_n = _text(record.get("total_tested"))
    if not raw_k or not raw_n:
        # A ``k/n`` fraction written into effect_metric is real count data too.
        match = _FRACTION_TEXT.match(_effect_text(record))
        if match is None:
            return None
        raw_k, raw_n = match.group(1), match.group(2)
    try:
        k, n = float(raw_k), float(raw_n)
    except ValueError:
        return None
    if not math.isfinite(k) or not math.isfinite(n):
        return None
    if n <= 0 or k < 0 or k > n:
        return None
    return k, n


def encode_counts(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    """Incidence from an explicit numerator and denominator, ``n >= 2``."""
    if str(record.get("source_id") or "") != DIRECT_SOURCE:
        return None
    parsed = _counts(record)
    if parsed is None:
        return None
    k, n = parsed
    if n < 2:
        return None
    value = count_logit(k, n)
    return CategoricalEncoding(
        encoder_id="count_logit",
        value=value,
        unit=LOGIT_RESPONSE_UNIT,
        measurement_text=render_measurement(value),
        inputs={"positive_count": k, "total_tested": n, "shrunk_rate": shrunk_rate(k, n)},
        sample_size=int(n),
    )


def encode_single_subject(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    """The same rate computation for a single-subject report, kept separate."""
    if str(record.get("source_id") or "") != DIRECT_SOURCE:
        return None
    parsed = _counts(record)
    if parsed is None:
        return None
    k, n = parsed
    if n != 1:
        return None
    value = count_logit(k, n)
    return CategoricalEncoding(
        encoder_id="single_subject_logit",
        value=value,
        unit=LOGIT_RESPONSE_UNIT,
        measurement_text=render_measurement(value),
        inputs={"positive_count": k, "total_tested": n},
        sample_size=1,
    )


def parse_severity_grade(text: str) -> int | None:
    """Return a 0-4 Draize-style grade, or None when the text is not a grade."""
    cleaned = text.strip().casefold()
    if not cleaned:
        return None
    if cleaned in _NEGATIVE_GRADE_TEXT:
        return 0
    match = _GRADE_TEXT.match(cleaned)
    if match is not None:
        return len(match.group(1))
    numeric = _NUMERIC_GRADE_TEXT.match(cleaned)
    if numeric is not None:
        return int(numeric.group(1))
    return None


def encode_severity_grade(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    """Ordinal reaction severity; never converted to an incidence rate."""
    if str(record.get("source_id") or "") != DIRECT_SOURCE:
        return None
    raw_text = _effect_text(record)
    grade = parse_severity_grade(raw_text)
    grade_source = "effect_metric"
    if grade is None:
        reconciled = _text(record.get("global_severity_grade"))
        # The reconciler's output contract is exactly one canonical digit 0-4.
        # Accept that canonical form without broadening the raw effect parser,
        # where a bare number can be an assay value rather than a grade.
        grade = int(reconciled) if re.fullmatch(r"[0-4]", reconciled) else None
        grade_source = "global_severity_grade"
    if grade is None:
        return None
    value = float(grade)
    return CategoricalEncoding(
        encoder_id="ordinal_severity_grade",
        value=value,
        unit=ORDINAL_SEVERITY_UNIT,
        measurement_text=render_measurement(value),
        inputs={
            "effect_metric": raw_text,
            "grade": grade,
            "grade_source": grade_source,
        },
    )


def encode_percent_positive(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    """A reported percentage with no denominator, used directly as the rate."""
    if str(record.get("source_id") or "") != DIRECT_SOURCE:
        return None
    match = _PERCENT_TEXT.match(_effect_text(record))
    if match is None:
        return None
    percent = float(match.group(1))
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        return None
    rate = percent / 100.0
    if 0.0 < rate < 1.0:
        value = logit(rate)
        shrunk = rate
    else:
        # Resolve only the reported boundaries, using the declared nominal size.
        shrunk = shrunk_rate(rate * NOMINAL_PERCENT_SAMPLE_SIZE, NOMINAL_PERCENT_SAMPLE_SIZE)
        value = logit(shrunk)
    return CategoricalEncoding(
        encoder_id="percent_positive_logit",
        value=value,
        unit=LOGIT_RESPONSE_UNIT,
        measurement_text=render_measurement(value),
        inputs={
            "effect_metric": _effect_text(record),
            "reported_percent": percent,
            "rate": shrunk,
            "nominal_sample_size_used": not 0.0 < rate < 1.0,
        },
    )


def encode_signed_direction(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    """Signed hazard direction for phototoxicity / irritation / local damage."""
    if str(record.get("source_id") or "") != PHOTOTOXICITY_SOURCE:
        return None
    label = _text(record.get("result_label")).casefold()
    if label in UNINFORMATIVE_LABELS or label not in SIGNED_DIRECTION_ANCHORS:
        return None
    value = SIGNED_DIRECTION_ANCHORS[label]
    return CategoricalEncoding(
        encoder_id="signed_direction",
        value=value,
        unit=SIGNED_DIRECTION_UNIT,
        measurement_text=render_measurement(value),
        inputs={"result_label": label},
    )


POLICY = CategoricalResponsePolicy(
    version=CATEGORICAL_RESPONSE_VERSION,
    encoders=(
        encode_counts,
        encode_severity_grade,
        encode_percent_positive,
        encode_single_subject,
        encode_signed_direction,
    ),
)


def encoding_policy_manifest() -> dict[str, Any]:
    """Self-describing metadata for the stage-02 artifact."""
    return {
        "version": CATEGORICAL_RESPONSE_VERSION,
        "encoders": [
            {
                "encoder_id": "count_logit",
                "source_id": DIRECT_SOURCE,
                "unit": LOGIT_RESPONSE_UNIT,
                "definition": "logit((k + 0.5) / (n + 1)) for n >= 2",
                "scale": "log-odds of incidence",
            },
            {
                "encoder_id": "ordinal_severity_grade",
                "source_id": DIRECT_SOURCE,
                "unit": ORDINAL_SEVERITY_UNIT,
                "definition": "Draize-style + ladder, 0-4",
                "scale": "ordinal severity of one subject's reaction",
                "note": (
                    "deliberately not an incidence rate; 94.7% of graded rows "
                    "carrying counts have n == 1"
                ),
            },
            {
                "encoder_id": "percent_positive_logit",
                "source_id": DIRECT_SOURCE,
                "unit": LOGIT_RESPONSE_UNIT,
                "definition": "logit(reported percent), boundaries shrunk",
                "nominal_sample_size": NOMINAL_PERCENT_SAMPLE_SIZE,
                "scale": "log-odds of incidence, no denominator reported",
            },
            {
                "encoder_id": "single_subject_logit",
                "source_id": DIRECT_SOURCE,
                "unit": LOGIT_RESPONSE_UNIT,
                "definition": "logit((k + 0.5) / (n + 1)) for n == 1",
                "scale": "log-odds of incidence, single subject",
            },
            {
                "encoder_id": "signed_direction",
                "source_id": PHOTOTOXICITY_SOURCE,
                "unit": SIGNED_DIRECTION_UNIT,
                "definition": "protective -1, negative 0, positive +1",
                "anchors": dict(SIGNED_DIRECTION_ANCHORS),
                "scale": "signed hazard direction",
            },
        ],
        "uninformative_labels_receive_no_value": sorted(UNINFORMATIVE_LABELS),
        "encoders_never_overwrite_a_real_measurement": True,
        "encoder_id_is_part_of_the_pair_bucket_key": True,
    }


__all__ = [
    "CATEGORICAL_RESPONSE_VERSION",
    "NOMINAL_PERCENT_SAMPLE_SIZE",
    "ORDINAL_SEVERITY_UNIT",
    "POLICY",
    "SIGNED_DIRECTION_ANCHORS",
    "UNINFORMATIVE_LABELS",
    "encode_counts",
    "encode_percent_positive",
    "encode_severity_grade",
    "encode_signed_direction",
    "encode_single_subject",
    "encoding_policy_manifest",
    "parse_severity_grade",
]
