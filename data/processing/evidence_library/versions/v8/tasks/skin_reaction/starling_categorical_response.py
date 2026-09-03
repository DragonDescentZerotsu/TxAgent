"""Controlled Skin_Reaction response encoders.

Incidence is retained on its source-faithful fraction scale. Explicit structured
counts are source-exact; percentages and fractions embedded in text reach these
encoders only after measurement resolution or as a non-scalar fallback. A
single-subject result remains a separate scale from group incidence.

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

from data.processing.evidence_library.shared.v1.categorical_response import (
    CanonicalCategory,
    ORDINAL_SEVERITY_UNIT,
    SIGNED_DIRECTION_UNIT,
    CategoricalEncoding,
    CategoricalResponsePolicy,
    ControlledMeasurementSpec,
    render_measurement,
)


CATEGORICAL_RESPONSE_VERSION = "skin_reaction_categorical_response.v2"

DIRECT_SOURCE = "direct_skin_reaction"
PHOTOTOXICITY_SOURCE = "phototoxicity_irritation_local_damage"

INCIDENCE_FRACTION_UNIT = "fraction"

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
_INCIDENCE_PERCENT_TEXT = re.compile(
    r"^\d+(?:\.\d+)?\s*%\s*(?:positive|of\s+\w+|reacted|response|responders)?\.?$",
    re.IGNORECASE,
)


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
        return None
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
    value = k / n
    return CategoricalEncoding(
        encoder_id="incidence_fraction.v1",
        value=value,
        unit=INCIDENCE_FRACTION_UNIT,
        measurement_text=render_measurement(value),
        inputs={"positive_count": k, "total_tested": n},
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
    value = k / n
    return CategoricalEncoding(
        encoder_id="single_subject_fraction.v1",
        value=value,
        unit=INCIDENCE_FRACTION_UNIT,
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


def encode_incidence_fraction(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    """Use structured counts or the frozen resolver's incidence quantity."""
    counted = encode_counts(record)
    if counted is not None:
        return counted
    if str(record.get("source_id") or "") != DIRECT_SOURCE:
        return None
    if record.get("resolved_scalar_value") is None:
        return None
    value = float(record["resolved_scalar_value"])
    resolved_unit = str(record.get("resolved_unit_text") or "")
    if resolved_unit == "%" and _INCIDENCE_PERCENT_TEXT.match(_effect_text(record)):
        value /= 100.0
    elif resolved_unit != INCIDENCE_FRACTION_UNIT:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return CategoricalEncoding(
        encoder_id="incidence_fraction.v1",
        value=value,
        unit=INCIDENCE_FRACTION_UNIT,
        measurement_text=render_measurement(value),
        inputs={"measurement_resolution_origin": record.get("measurement_resolution_origin")},
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


CONTROLLED_MEASUREMENTS = (
    ControlledMeasurementSpec(
        scale_id="incidence_fraction.v1",
        source_id=DIRECT_SOURCE,
        input_fields=("positive_count", "total_tested", "measurement_text"),
        encoder=encode_incidence_fraction,
        kind="continuous",
        parser_id="skin.incidence_fraction.v1",
        definition="source-exact or resolved positive fraction for n >= 2",
    ),
    ControlledMeasurementSpec(
        scale_id="single_subject_fraction.v1",
        source_id=DIRECT_SOURCE,
        input_fields=("positive_count", "total_tested", "measurement_text"),
        encoder=encode_single_subject,
        kind="binary",
        parser_id="skin.single_subject_fraction.v1",
        definition="source-exact response fraction for n == 1",
        categories=(
            CanonicalCategory("no_response", 0, 0.0),
            CanonicalCategory("response", 1, 1.0),
        ),
    ),
    ControlledMeasurementSpec(
        scale_id="ordinal_severity_grade",
        source_id=DIRECT_SOURCE,
        input_fields=("measurement_text", "global_severity_grade"),
        encoder=encode_severity_grade,
        kind="ordinal",
        parser_id="skin.ordinal_severity_grade.v1",
        definition="Draize-style negative and + through ++++ ladder",
        categories=tuple(
            CanonicalCategory(f"grade_{grade}", grade, float(grade))
            for grade in range(5)
        ),
    ),
    ControlledMeasurementSpec(
        scale_id="signed_direction",
        source_id=PHOTOTOXICITY_SOURCE,
        input_fields=("result_label",),
        encoder=encode_signed_direction,
        kind="ordinal",
        parser_id="skin.signed_direction.v1",
        definition="protective < negative/no-effect < positive hazard direction",
        categories=(
            CanonicalCategory("protective", 0, -1.0),
            CanonicalCategory("negative", 1, 0.0),
            CanonicalCategory("positive", 2, 1.0),
        ),
    ),
)

MEASUREMENT_SCALES = {
    item.scale_id: item for item in CONTROLLED_MEASUREMENTS
}

POLICY = CategoricalResponsePolicy(
    version=CATEGORICAL_RESPONSE_VERSION,
    controlled_measurements=CONTROLLED_MEASUREMENTS,
)


def encoding_policy_manifest() -> dict[str, Any]:
    """Self-describing metadata for the stage-02 artifact."""
    return {
        **POLICY.manifest(),
        "version": CATEGORICAL_RESPONSE_VERSION,
        "encoders": [
            {
                "encoder_id": "incidence_fraction.v1",
                "source_id": DIRECT_SOURCE,
                "unit": INCIDENCE_FRACTION_UNIT,
                "definition": "positive_count / total_tested or resolved percent",
                "scale": "raw incidence fraction",
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
                "encoder_id": "single_subject_fraction.v1",
                "source_id": DIRECT_SOURCE,
                "unit": INCIDENCE_FRACTION_UNIT,
                "definition": "positive_count / total_tested for n == 1",
                "scale": "raw incidence fraction, single subject",
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
        "numeric_incidence_is_normalized_after_measurement_resolution": True,
        "encoder_id_is_part_of_the_pair_bucket_key": True,
    }


__all__ = [
    "CATEGORICAL_RESPONSE_VERSION",
    "CONTROLLED_MEASUREMENTS",
    "MEASUREMENT_SCALES",
    "INCIDENCE_FRACTION_UNIT",
    "ORDINAL_SEVERITY_UNIT",
    "POLICY",
    "SIGNED_DIRECTION_ANCHORS",
    "UNINFORMATIVE_LABELS",
    "encode_counts",
    "encode_incidence_fraction",
    "encode_severity_grade",
    "encode_signed_direction",
    "encode_single_subject",
    "encoding_policy_manifest",
    "parse_severity_grade",
]
