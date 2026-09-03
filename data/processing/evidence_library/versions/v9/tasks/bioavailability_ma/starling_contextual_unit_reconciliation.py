"""Apply the central contextual-unit policy to Bioavailability canonical fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.contracts import MeasurementPair
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    has_non_atomic_directional_context,
    parse_point_measurement,
    render_point_measurement,
)
from tools.chembl_tool.common.units import (
    canonicalize_measurement,
    canonicalize_unit,
)


TASK_ID = "bioavailability_ma"
ASSAY_CONTEXT_FIELDS = (
    "source_id",
    "canonical_endpoint",
    "global_context",
    "global_species_context",
)
CONTEXTUAL_STANDARDIZATION_STATUS = "contextual_unit_standardized"

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "bioavailability_ma"



def bioavailability_assay_context(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete dynamic assay mapping passed to the shared parser."""
    return {field: record.get(field) for field in ASSAY_CONTEXT_FIELDS}


def contextual_standardization_of_unit(
    record: Mapping[str, Any],
    pair: MeasurementPair,
) -> MeasurementPair:
    """Always apply a matching central policy rule to the atomic pair."""
    if pair.canonical_measurement is None or pair.canonical_unit is None:
        return pair
    assay = bioavailability_assay_context(record)
    unit_result = canonicalize_unit(pair.canonical_unit, task=TASK_ID, assay=assay)
    if unit_result.contextual_policy_status == "no_matching_rule":
        return pair
    parsed = parse_point_measurement(pair.canonical_measurement)
    if parsed.value is None or parsed.kind in {"point_with_interval", "mean_with_context"}:
        return pair
    value, unit = canonicalize_measurement(
        parsed.value,
        pair.canonical_unit,
        task=TASK_ID,
        assay=assay,
    )
    if value is None or unit is None:
        raise ValueError("matched contextual unit rule failed to convert a scalar pair")
    variation = None
    if parsed.variation is not None:
        variation, variation_unit = canonicalize_measurement(
            parsed.variation,
            pair.canonical_unit,
            task=TASK_ID,
            assay=assay,
        )
        if variation is None or variation_unit != unit:
            raise ValueError("contextual unit rule failed to convert pair variation")
    rendered = render_point_measurement(
        pair.canonical_measurement, value, variation
    )
    if unit == pair.canonical_unit and rendered == pair.canonical_measurement:
        return pair
    return MeasurementPair(
        rendered,
        unit,
        CONTEXTUAL_STANDARDIZATION_STATUS,
        pair.unit_notation_status,
        pair.unit_notation_factor,
    )


def contextual_canonical_record_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every canonical scalar field after contextual standardization."""
    source_pair = MeasurementPair(
        record.get("canonical_measurement"),
        record.get("canonical_unit"),
        str(record.get("measurement_unit_status") or ""),
        str(record.get("unit_notation_status") or "none"),
        record.get("unit_notation_factor"),
    )
    assay = bioavailability_assay_context(record)
    source_unit_result = canonicalize_unit(
        source_pair.canonical_unit,
        task=TASK_ID,
        assay=assay,
    )
    pair = contextual_standardization_of_unit(record, source_pair)
    parsed = parse_point_measurement(pair.canonical_measurement)
    canonical_unit_result = canonicalize_unit(pair.canonical_unit, task=_TASK_VOCAB)
    non_atomic_directional_context = (
        parsed.value is not None
        and has_non_atomic_directional_context(
            pair.canonical_measurement, pair.canonical_unit
        )
    )
    finite_scalar = (
        parsed.value
        if parsed.value is not None
        and bool(canonical_unit_result.cleaned)
        and not canonical_unit_result.unknown_tokens
        and pair.status != "ambiguous_scientific_notation"
        and not non_atomic_directional_context
        else None
    )
    return {
        "canonical_measurement": pair.canonical_measurement,
        "canonical_unit": pair.canonical_unit,
        "measurement_parse_kind": parsed.kind,
        "measurement_unit_status": (
            "non_atomic_directional_context"
            if non_atomic_directional_context
            else pair.status
        ),
        "unit_notation_status": pair.unit_notation_status,
        "unit_notation_factor": pair.unit_notation_factor,
        "unit_dimension_json": json.dumps(
            canonical_unit_result.dimension, ensure_ascii=False
        ),
        "finite_scalar_value": finite_scalar,
        "is_absolute_and_continuous": finite_scalar is not None,
        "absolute_and_continuous_value": finite_scalar,
        "variation_value": parsed.variation,
        "canonical_unit_policy_status": source_unit_result.contextual_policy_status,
        "canonical_unit_rule_id": source_unit_result.contextual_rule_id,
        "canonical_unit_policy_version": (
            source_unit_result.contextual_policy_version
        ),
        "canonical_unit_conversion_factor": (
            source_unit_result.contextual_conversion_factor
        ),
    }


__all__ = [
    "ASSAY_CONTEXT_FIELDS",
    "CONTEXTUAL_STANDARDIZATION_STATUS",
    "TASK_ID",
    "bioavailability_assay_context",
    "contextual_canonical_record_fields",
    "contextual_standardization_of_unit",
]
