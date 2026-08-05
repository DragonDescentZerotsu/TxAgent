"""Controlled binary outcome encoders for BBB Martins normalized evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.starling.categorical_response import (
    BINARY_OUTCOME_UNIT,
    CanonicalCategory,
    CategoricalEncoding,
    CategoricalResponsePolicy,
    ControlledMeasurementSpec,
    render_measurement,
)


CATEGORICAL_RESPONSE_VERSION = "bbb_martins_categorical_response.v2"

DIRECT_POSITIVE = frozenset(
    {"permeable", "good_penetration", "increased_permeability", "high_permeability"}
)
DIRECT_NEGATIVE = frozenset(
    {"poor_penetration", "impermeable", "low_permeability", "restricted"}
)
PASSIVE_POSITIVE = frozenset({"high", "permeable_or_high"})
PASSIVE_NEGATIVE = frozenset({"low", "impermeable_or_low"})


def _token(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().casefold()).strip("_")


def _binary(
    *, encoder_id: str, value: float, field: str, raw: Any
) -> CategoricalEncoding:
    return CategoricalEncoding(
        encoder_id=encoder_id,
        value=value,
        unit=BINARY_OUTCOME_UNIT,
        measurement_text=render_measurement(value),
        inputs={field: raw},
    )


def encode_direct_permeability(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    if record.get("source_id") != "direct_bbb":
        return None
    raw = record.get("bbb_permeability_label")
    label = _token(raw)
    if label in DIRECT_POSITIVE:
        return _binary(
            encoder_id="bbb_permeability_binary.v1",
            value=1.0,
            field="bbb_permeability_label",
            raw=raw,
        )
    if label in DIRECT_NEGATIVE:
        return _binary(
            encoder_id="bbb_permeability_binary.v1",
            value=-1.0,
            field="bbb_permeability_label",
            raw=raw,
        )
    return None


def encode_passive_interpretation(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    if record.get("source_id") != "passive_permeability":
        return None
    raw = record.get("passive_bbb_interpretation")
    label = _token(raw)
    if label in PASSIVE_POSITIVE:
        return _binary(
            encoder_id="passive_bbb_interpretation_binary.v1",
            value=1.0,
            field="passive_bbb_interpretation",
            raw=raw,
        )
    if label in PASSIVE_NEGATIVE:
        return _binary(
            encoder_id="passive_bbb_interpretation_binary.v1",
            value=-1.0,
            field="passive_bbb_interpretation",
            raw=raw,
        )
    return None


def encode_efflux_substrate(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    if record.get("source_id") != "efflux_transport":
        return None
    raw = record.get("interaction_conclusion")
    label = _token(raw)
    if label == "substrate":
        value = 1.0
    elif label == "non_substrate":
        value = -1.0
    else:
        return None
    return _binary(
        encoder_id="efflux_substrate_binary.v1",
        value=value,
        field="interaction_conclusion",
        raw=raw,
    )


def encode_efflux_inhibition(record: Mapping[str, Any]) -> CategoricalEncoding | None:
    if record.get("source_id") != "efflux_transport":
        return None
    raw = record.get("interaction_conclusion")
    label = _token(raw)
    if label == "inhibitor":
        value = 1.0
    elif label == "non_inhibitor":
        value = -1.0
    else:
        return None
    return _binary(
        encoder_id="efflux_inhibitor_binary.v1",
        value=value,
        field="interaction_conclusion",
        raw=raw,
    )


_BINARY_DOMAIN = (
    CanonicalCategory("negative", 0, -1.0),
    CanonicalCategory("positive", 1, 1.0),
)

CONTROLLED_MEASUREMENTS = (
    ControlledMeasurementSpec(
        scale_id="bbb_permeability_binary.v1",
        source_id="direct_bbb",
        input_fields=("bbb_permeability_label",),
        encoder=encode_direct_permeability,
        kind="binary",
        parser_id="bbb.permeability_binary.v1",
        definition="reviewed BBB permeability outcome aliases",
        categories=_BINARY_DOMAIN,
    ),
    ControlledMeasurementSpec(
        scale_id="passive_bbb_interpretation_binary.v1",
        source_id="passive_permeability",
        input_fields=("passive_bbb_interpretation",),
        encoder=encode_passive_interpretation,
        kind="binary",
        parser_id="bbb.passive_interpretation_binary.v1",
        definition="reviewed passive-permeability interpretation aliases",
        categories=_BINARY_DOMAIN,
    ),
    ControlledMeasurementSpec(
        scale_id="efflux_substrate_binary.v1",
        source_id="efflux_transport",
        input_fields=("interaction_conclusion",),
        encoder=encode_efflux_substrate,
        kind="binary",
        parser_id="bbb.efflux_substrate_binary.v1",
        definition="substrate versus non-substrate",
        categories=_BINARY_DOMAIN,
    ),
    ControlledMeasurementSpec(
        scale_id="efflux_inhibitor_binary.v1",
        source_id="efflux_transport",
        input_fields=("interaction_conclusion",),
        encoder=encode_efflux_inhibition,
        kind="binary",
        parser_id="bbb.efflux_inhibitor_binary.v1",
        definition="inhibitor versus non-inhibitor",
        categories=_BINARY_DOMAIN,
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
    return {
        **POLICY.manifest(),
        "version": CATEGORICAL_RESPONSE_VERSION,
        "unit": BINARY_OUTCOME_UNIT,
        "anchors": {"negative": -1.0, "positive": 1.0},
        "encoders": [
            "bbb_permeability_binary.v1",
            "passive_bbb_interpretation_binary.v1",
            "efflux_substrate_binary.v1",
            "efflux_inhibitor_binary.v1",
        ],
        "semantic_endpoint_by_encoder": {
            "bbb_permeability_binary.v1": "bbb_permeability_outcome",
            "passive_bbb_interpretation_binary.v1": "passive_bbb_permeability_outcome",
            "efflux_substrate_binary.v1": "efflux_substrate_outcome",
            "efflux_inhibitor_binary.v1": "efflux_inhibition_outcome",
        },
        "abstentions": {
            "direct_transport_labels": True,
            "passive_intermediate_or_not_stated": True,
            "efflux_other_conclusions": True,
            "influx_positive_only_prose": True,
        },
        "real_scalar_precedence": True,
        "parseable_unresolved_unit_measurement_precedence": True,
        "encoder_id_is_part_of_the_pair_bucket_key": True,
    }


__all__ = [
    "CATEGORICAL_RESPONSE_VERSION",
    "CONTROLLED_MEASUREMENTS",
    "DIRECT_NEGATIVE",
    "DIRECT_POSITIVE",
    "POLICY",
    "MEASUREMENT_SCALES",
    "encoding_policy_manifest",
]
