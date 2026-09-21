"""Prompt-safe semantic display fields for Ames categorical records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.versions.v10.tasks.ames.starling_categorical_response import (
    MEASUREMENT_SCALES,
)


SEMANTIC_DISPLAY_VERSION = "ames_semantic_display.v1"
_SCALE_UNITS = {
    "ames_mutagenicity_ordinal.v1": "mutagenicity outcome",
    "ames_fixed_mutation_ordinal.v1": "fixed-mutation outcome",
    "ames_damage_direction.v1": "premutagenic damage or response direction",
    "ames_mechanism_detection.v1": "mechanistic endpoint detection status",
    "ames_mechanism_direction.v1": "mechanistic endpoint direction",
}


def semantic_display_fields(record: Mapping[str, Any]) -> dict[str, str | None]:
    """Return semantic categorical text without changing numeric geometry."""
    measurement = record.get("canonical_measurement_text") or record.get("measurement_text")
    unit = record.get("canonical_unit_text") or record.get("unit_text")
    if record.get("measurement_kind") not in {"binary", "ordinal"}:
        return {
            "measurement_text": None if measurement is None else str(measurement),
            "unit_text": None if unit is None else str(unit),
        }
    scale = str(record.get("canonical_measurement_scale_id") or "")
    spec = MEASUREMENT_SCALES.get(scale)
    if spec is None or scale not in _SCALE_UNITS:
        raise ValueError(f"unsupported Ames display scale: {scale!r}")
    category = str(record.get("canonical_category_id") or "")
    allowed = {item.category_id for item in spec.categories}
    if category not in allowed:
        raise ValueError(f"invalid Ames category for {scale}: {category!r}")
    return {
        "measurement_text": category.replace("_", " "),
        "unit_text": _SCALE_UNITS[scale],
    }


def semantic_prompt_payload(record: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Replace encoder inputs with their model-facing semantic category."""
    display = semantic_display_fields({**payload, **record})
    scale = str(record.get("canonical_measurement_scale_id") or "")
    consumed = MEASUREMENT_SCALES[scale].input_fields if scale in MEASUREMENT_SCALES else ()
    projected = {key: value for key, value in payload.items() if key not in consumed}
    projected.update(display)
    return projected


__all__ = ["SEMANTIC_DISPLAY_VERSION", "semantic_display_fields", "semantic_prompt_payload"]
