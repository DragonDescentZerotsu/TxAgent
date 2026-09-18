"""Prompt-safe semantic display fields for BBB evidence records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_categorical_response import MEASUREMENT_SCALES

from data.processing.evidence_library.versions.v9.tasks.bbb_martins.transporter_identifiers import (
    canonical_transporter_identifier,
)


SEMANTIC_DISPLAY_VERSION = "bbb_semantic_display.v2"
_BINARY_LABELS = {
    "bbb_permeability_binary.v1": (
        "impermeable",
        "permeable",
        "BBB permeability",
    ),
    "passive_bbb_interpretation_binary.v1": (
        "low",
        "high",
        "passive BBB permeability",
    ),
    "efflux_substrate_binary.v1": (
        "non-substrate",
        "substrate",
        "efflux-transporter substrate status",
    ),
    "efflux_inhibitor_binary.v1": (
        "non-inhibitor",
        "inhibitor",
        "efflux-transporter inhibition status",
    ),
}


def semantic_display_fields(record: Mapping[str, Any]) -> dict[str, str | None]:
    """Return display-only fields without changing canonical numeric geometry."""
    measurement = record.get("canonical_measurement_text") or record.get("measurement_text")
    unit = record.get("canonical_unit_text") or record.get("unit_text")
    scale = str(
        record.get("canonical_measurement_scale_id")
        or record.get("categorical_encoder_id")
        or ""
    )
    labels = _BINARY_LABELS.get(scale)
    if (not labels and record.get("measurement_kind") in {"binary", "ordinal", "categorical"}
            and record.get("source_id", "direct_bbb") in {
                "direct_bbb", "efflux_transport", "influx_transport", "passive_permeability"}):
        raise ValueError(f"unsupported BBB display scale: {scale!r}; refresh legacy cache metadata")
    if labels:
        category = record.get("canonical_category_id")
        if category in (None, ""):
            encoded = MEASUREMENT_SCALES[scale].category_for_value(record.get("finite_scalar_value"))
            category = encoded.category_id if encoded else None
        if category not in {"negative", "positive"}:
            raise ValueError(f"invalid BBB category for {scale}: {category!r}")
        measurement = labels[category == "positive"]
        unit = labels[2]
    transporter = canonical_transporter_identifier(
        record.get("canonical_transporter_identifier")
        if "canonical_transporter_identifier" in record
        else record.get("transporter_identifier")
    )
    return {
        "measurement_text": None if measurement is None else str(measurement),
        "unit_text": None if unit is None else str(unit),
        "transporter_identifier": transporter,
    }


def semantic_prompt_payload(record: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project source fields for display, removing the active encoder's inputs."""
    display = semantic_display_fields({**payload, **record})
    scale = record.get("canonical_measurement_scale_id") or record.get("categorical_encoder_id")
    consumed = MEASUREMENT_SCALES[scale].input_fields if scale in _BINARY_LABELS else ()
    projected = {key: value for key, value in payload.items() if key not in consumed}
    projected.update(measurement_text=display["measurement_text"], unit_text=display["unit_text"])
    if "transporter_identifier" in projected:
        if display["transporter_identifier"] is None:
            projected.pop("transporter_identifier")
        else:
            projected["transporter_identifier"] = display["transporter_identifier"]
    return projected


__all__ = ["SEMANTIC_DISPLAY_VERSION", "semantic_display_fields", "semantic_prompt_payload"]
