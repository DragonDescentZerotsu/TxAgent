"""Controlled qualitative measurements for Ames Stage 1 routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.shared.v2.categorical_response import (
    BINARY_OUTCOME_UNIT,
    ORDINAL_OUTCOME_UNIT,
    SIGNED_DIRECTION_UNIT,
    CanonicalCategory,
    CategoricalEncoding,
    CategoricalResponsePolicy,
    ControlledMeasurementSpec,
    render_measurement,
)

CATEGORICAL_RESPONSE_VERSION = "ames_categorical_response.v1"


def _encode(
    record: Mapping[str, Any],
    *,
    source_id: str,
    field: str,
    scale_id: str,
    values: Mapping[str, float],
    unit: str,
) -> CategoricalEncoding | None:
    if record.get("source_id") != source_id:
        return None
    raw = record.get(field)
    value = values.get(str(raw or "").strip())
    if value is None:
        return None
    return CategoricalEncoding(
        encoder_id=scale_id,
        value=value,
        unit=unit,
        measurement_text=render_measurement(value),
        inputs={field: raw},
    )


def encode_base(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    return _encode(
        record,
        source_id="mutagenicity_outcomes",
        field="measurement_text",
        scale_id="ames_mutagenicity_ordinal.v1",
        values={
            "negative": 0.0,
            "weak_positive": 1.0,
            "positive": 2.0,
            "strong_positive": 3.0,
        },
        unit=ORDINAL_OUTCOME_UNIT,
    )


def encode_fixed_mutation(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    return _encode(
        record,
        source_id="fixed_mutation",
        field="result_call",
        scale_id="ames_fixed_mutation_ordinal.v1",
        values={"negative": 0.0, "weak_positive": 1.0, "positive": 2.0},
        unit=ORDINAL_OUTCOME_UNIT,
    )


def encode_damage_direction(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    return _encode(
        record,
        source_id="premutagenic_damage",
        field="result_status",
        scale_id="ames_damage_direction.v1",
        values={
            "damage_or_response_decreased": -1.0,
            "no_damage_or_response_change": 0.0,
            "damage_or_response_increased": 1.0,
        },
        unit=SIGNED_DIRECTION_UNIT,
    )


def encode_mechanism_detection(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    return _encode(
        record,
        source_id="mutagenicity_mechanism",
        field="result_direction",
        scale_id="ames_mechanism_detection.v1",
        values={"not_detected": -1.0, "detected": 1.0},
        unit=BINARY_OUTCOME_UNIT,
    )


def encode_mechanism_direction(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    return _encode(
        record,
        source_id="mutagenicity_mechanism",
        field="result_direction",
        scale_id="ames_mechanism_direction.v1",
        values={"decreased": -1.0, "no_effect": 0.0, "increased": 1.0},
        unit=SIGNED_DIRECTION_UNIT,
    )


CONTROLLED_MEASUREMENTS = (
    ControlledMeasurementSpec(
        "ames_mutagenicity_ordinal.v1",
        "mutagenicity_outcomes",
        ("measurement_text",),
        encode_base,
        "ordinal",
        "ames.mutagenicity_ordinal.v1",
        "negative, weak, ordinary, and strong positive mutagenicity calls",
        (
            CanonicalCategory("negative", 0, 0.0),
            CanonicalCategory("weak_positive", 1, 1.0),
            CanonicalCategory("positive", 2, 2.0),
            CanonicalCategory("strong_positive", 3, 3.0),
        ),
    ),
    ControlledMeasurementSpec(
        "ames_fixed_mutation_ordinal.v1",
        "fixed_mutation",
        ("result_call",),
        encode_fixed_mutation,
        "ordinal",
        "ames.fixed_mutation_ordinal.v1",
        "negative, weak-positive, and positive fixed-mutation calls",
        (
            CanonicalCategory("negative", 0, 0.0),
            CanonicalCategory("weak_positive", 1, 1.0),
            CanonicalCategory("positive", 2, 2.0),
        ),
    ),
    ControlledMeasurementSpec(
        "ames_damage_direction.v1",
        "premutagenic_damage",
        ("result_status",),
        encode_damage_direction,
        "ordinal",
        "ames.damage_direction.v1",
        "decreased, unchanged, or increased premutagenic damage or response",
        (
            CanonicalCategory("decreased", 0, -1.0),
            CanonicalCategory("no_change", 1, 0.0),
            CanonicalCategory("increased", 2, 1.0),
        ),
    ),
    ControlledMeasurementSpec(
        "ames_mechanism_detection.v1",
        "mutagenicity_mechanism",
        ("result_direction",),
        encode_mechanism_detection,
        "binary",
        "ames.mechanism_detection.v1",
        "mechanistic endpoint detected versus not detected",
        (
            CanonicalCategory("not_detected", 0, -1.0),
            CanonicalCategory("detected", 1, 1.0),
        ),
    ),
    ControlledMeasurementSpec(
        "ames_mechanism_direction.v1",
        "mutagenicity_mechanism",
        ("result_direction",),
        encode_mechanism_direction,
        "ordinal",
        "ames.mechanism_direction.v1",
        "decreased, unchanged, or increased mechanistic endpoint",
        (
            CanonicalCategory("decreased", 0, -1.0),
            CanonicalCategory("no_effect", 1, 0.0),
            CanonicalCategory("increased", 2, 1.0),
        ),
    ),
)

MEASUREMENT_SCALES = {item.scale_id: item for item in CONTROLLED_MEASUREMENTS}
POLICY = CategoricalResponsePolicy(
    version=CATEGORICAL_RESPONSE_VERSION,
    controlled_measurements=CONTROLLED_MEASUREMENTS,
)


__all__ = [
    "CATEGORICAL_RESPONSE_VERSION",
    "CONTROLLED_MEASUREMENTS",
    "MEASUREMENT_SCALES",
    "POLICY",
]
