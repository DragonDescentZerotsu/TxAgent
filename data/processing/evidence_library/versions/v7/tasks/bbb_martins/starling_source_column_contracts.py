"""Fail-closed source-column visibility contracts for BBB Starling rows."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


# This is the shared wire/schema version consumed by evidence_contract.  BBB's
# source-specific alias truth table has its own policy version below; changing
# the wire version would make otherwise compatible compact evidence unreadable.
SOURCE_COLUMN_CONTRACT_VERSION = "source_column_contract.v1"
SOURCE_COLUMN_POLICY_VERSION = "bbb_martins_source_column_policy.v2"

SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "direct_bbb": (
        "source_index", "pmid", "support_text", "bbb_permeability_label",
        "bbb_transport_label", "quant_metric", "quant_value", "quant_units",
        "assay_model", "species", "qualifying_conditions", "extra_details", "smiles",
    ),
    "passive_permeability": (
        "paragraph_idx", "support_text", "global_identifier", "assay_type",
        "biological_system", "metric_name", "metric_value", "metric_uncertainty",
        "metric_units", "passive_bbb_interpretation", "qualifying_conditions",
        "extra_details", "confidence", "needs_more_context", "pmid", "extraction_id",
        "SMILES",
    ),
    "efflux_transport": (
        "paragraph_idx", "support_text", "global_identifier", "transporter_identifier",
        "evidence_type", "interaction_conclusion", "assay_system", "quantitative_metric",
        "quantitative_value", "perturbation", "qualifying_conditions", "extra_details",
        "confidence", "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
    "influx_transport": (
        "paragraph_idx", "support_text", "global_identifier", "mediator_name",
        "mediator_identifier", "transport_mechanism", "transport_endpoint",
        "evidence_basis", "assay_model", "reported_result", "qualifying_conditions",
        "extra_details", "confidence", "needs_more_context", "pmid", "extraction_id",
        "SMILES",
    ),
}

# A normalized alias is source-derived only when that source actually supplies
# the corresponding field.  In particular, influx has no scalar/unit column,
# efflux has no unit column, and Direct has no confidence column.
_ALIAS_SOURCE_FIELD: dict[str, dict[str, str]] = {
    "direct_bbb": {
        "source_record_id": "source_index",
        "endpoint_name": "quant_metric",
        "measurement_text": "quant_value",
        "unit_text": "quant_units",
        "source_smiles": "smiles",
        "support_text": "support_text",
    },
    "passive_permeability": {
        "source_record_id": "extraction_id",
        "endpoint_name": "metric_name",
        "measurement_text": "metric_value",
        "unit_text": "metric_units",
        "source_smiles": "SMILES",
        "confidence": "confidence",
        "support_text": "support_text",
    },
    "efflux_transport": {
        "source_record_id": "extraction_id",
        "endpoint_name": "quantitative_metric",
        "measurement_text": "quantitative_value",
        "source_smiles": "SMILES",
        "confidence": "confidence",
        "support_text": "support_text",
    },
    "influx_transport": {
        "source_record_id": "extraction_id",
        "endpoint_name": "transport_endpoint",
        "source_smiles": "SMILES",
        "confidence": "confidence",
        "support_text": "support_text",
    },
}


def source_fields_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    declared = SOURCE_COLUMNS.get(source_id)
    if declared is None:
        raise ValueError(f"no source-column contract for source_id={source_id!r}")
    if all(column in record for column in declared):
        return {column: _json_value(record.get(column)) for column in declared}
    try:
        payload = json.loads(str(record.get("source_payload_json") or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source_payload_json for {source_id!r}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"source_payload_json is not an object for {source_id!r}")
    return {column: _json_value(payload.get(column)) for column in declared}


def llm_source_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    return {
        "contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "source_id": source_id,
        "source_or_simply_cleaned": {
            column: True for column in SOURCE_COLUMNS[source_id]
        },
        "source_fields": source_fields_from_record(record),
    }


def normalized_column_contract(
    source_id: str, normalized_columns: Sequence[str]
) -> dict[str, bool]:
    if source_id not in SOURCE_COLUMNS:
        raise ValueError(f"no source-column contract for source_id={source_id!r}")
    allowed = set(_ALIAS_SOURCE_FIELD[source_id]) | set(SOURCE_COLUMNS[source_id])
    return {str(column): str(column) in allowed for column in normalized_columns}


def source_column_contract_manifest(
    normalized_columns: Sequence[str],
) -> dict[str, Any]:
    columns = tuple(str(column) for column in normalized_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("normalized artifact has duplicate column names")
    return {
        "contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "policy_version": SOURCE_COLUMN_POLICY_VERSION,
        "semantics": {
            "true": "genuine source value or only whitespace/null cleaned",
            "false": "derived, canonical, inferred, policy, scoring, audit, or helper value",
            "llm_projection": "only declared source-schema fields; no canonical fallback",
        },
        "sources": {
            source_id: {
                "source_schema": {
                    column: {"source_or_simply_cleaned": True}
                    for column in source_columns
                },
                "normalized_artifact_columns": {
                    column: {"source_or_simply_cleaned": allowed}
                    for column, allowed in normalized_column_contract(source_id, columns).items()
                },
            }
            for source_id, source_columns in SOURCE_COLUMNS.items()
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return str(value)


__all__ = [
    "SOURCE_COLUMN_CONTRACT_VERSION",
    "SOURCE_COLUMN_POLICY_VERSION",
    "SOURCE_COLUMNS",
    "llm_source_projection",
    "normalized_column_contract",
    "source_column_contract_manifest",
    "source_fields_from_record",
]
