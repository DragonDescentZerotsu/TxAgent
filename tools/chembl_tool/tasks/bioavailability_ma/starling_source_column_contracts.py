"""Versioned source-column visibility contracts for Bioavailability Starling rows.

The normalized artifacts retain every working column.  This module separately
defines which columns are genuine source values (or only whitespace/null cleaned)
and may therefore be projected into reasoning prompts.  Canonical, inferred,
policy, scoring, and audit columns remain stored but are never a prompt fallback.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SOURCE_COLUMN_CONTRACT_VERSION = "source_column_contract.v2"

_HF_BIOAVAILABILITY_COLUMNS = (
    "bioavailability_report_type", "comparator", "dose", "extra_details",
    "molecule_name", "oral_bioavailability_value", "oral_exposure_mode", "pmid",
    "qualifying_conditions", "smiles", "species_or_population", "support_text",
)

SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "oral_exposure": (
        "pmid", "extraction_id", "global_identifier", "confidence",
        "paragraph_idx", "support_text", "exposure_measure", "parameter_value",
        "parameter_units", "statistic_type", "oral_dose", "study_context",
        "comparator_exposure", "qualifying_conditions", "extra_details", "smiles",
    ),
    "fa": (
        "pmid", "extraction_id", "global_identifier", "confidence",
        "paragraph_idx", "support_text", "endpoint_category", "assay_system",
        "reported_value", "reported_units", "condition_medium", "biological_context",
        "formulation_or_solid_form", "qualifying_conditions", "extra_details", "smiles",
    ),
    "fg": (
        "pmid", "extraction_id", "global_identifier", "confidence",
        "paragraph_idx", "support_text", "molecule_name", "gut_wall_process",
        "transporter_or_enzyme", "substrate_status", "assay_system", "intestinal_site",
        "measured_value", "qualifying_conditions", "extra_details", "smiles",
    ),
    "fh": (
        "pmid", "extraction_id", "global_identifier", "confidence",
        "paragraph_idx", "support_text", "metric_type", "assay_system", "species",
        "molecular_form", "reported_value", "reported_units", "enzyme_or_pathway",
        "qualifying_conditions", "extra_details", "smiles",
    ),
    "hf_bioavailability": _HF_BIOAVAILABILITY_COLUMNS,
}

# Top-level normalized columns that are still direct or simply-cleaned aliases.
# This is deliberately source-specific: constants and mapped/canonical structures
# are not source columns merely because they are convenient top-level fields.
_SOURCE_ALIASES: dict[str, set[str]] = {
    "oral_exposure": {
        "source_record_id", "endpoint_name", "measurement_text", "unit_text",
        "confidence", "support_text", "statistic_type", "oral_dose", "study_context",
        "comparator_exposure", "qualifying_conditions", "extra_details",
    },
    "fa": {
        "source_record_id", "endpoint_name", "measurement_text", "unit_text",
        "confidence", "support_text", "assay_system", "condition_medium",
        "biological_context", "formulation_or_solid_form", "qualifying_conditions",
        "extra_details",
    },
    "fg": {
        "source_record_id", "endpoint_name", "measurement_text", "molecule_name",
        "confidence", "support_text", "transporter_or_enzyme", "substrate_status",
        "assay_system", "intestinal_site", "qualifying_conditions", "extra_details",
    },
    "fh": {
        "source_record_id", "endpoint_name", "measurement_text", "unit_text",
        "confidence", "support_text", "assay_system", "species", "molecular_form",
        "enzyme_or_pathway", "qualifying_conditions", "extra_details",
    },
    "hf_bioavailability": {
        "measurement_text", "source_smiles", "molecule_name", "support_text",
        "bioavailability_report_type", "species_or_population", "dose",
        "oral_exposure_mode", "qualifying_conditions", "comparator", "extra_details",
    },
}


def source_fields_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the declared source fields, never normalization fallbacks."""
    source_id = str(record.get("source_id") or "")
    declared = SOURCE_COLUMNS.get(source_id)
    if declared is None:
        raise ValueError(f"no source-column contract for source_id={source_id!r}")
    # Compact v6 records persist the source schema as sparse Parquet columns.
    # Selecting only declared fields remains fail-closed while avoiding a repeated
    # per-record JSON copy.
    if all(column in record for column in declared):
        return {column: _json_value(record.get(column)) for column in declared}
    payload = record.get("source_payload_json")
    try:
        raw = json.loads(str(payload or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source_payload_json for source_id={source_id!r}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"source_payload_json is not an object for source_id={source_id!r}")
    # Keep declared missing columns explicit.  This makes the projection schema
    # stable and prevents a canonical alias from silently filling a source null.
    return {column: _json_value(raw.get(column)) for column in declared}


def llm_source_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete fail-closed source record made available to an LLM."""
    source_id = str(record.get("source_id") or "")
    return {
        "contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "source_id": source_id,
        "source_or_simply_cleaned": {column: True for column in SOURCE_COLUMNS[source_id]},
        "source_fields": source_fields_from_record(record),
    }


def llm_source_projection_from_mapping(
    source_id: str,
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an unwrapped source row using the same exhaustive contract."""
    if source_id not in SOURCE_COLUMNS:
        raise ValueError(f"no source-column contract for source_id={source_id!r}")
    fields = {
        column: _json_value(source_row.get(column))
        for column in SOURCE_COLUMNS[source_id]
    }
    return {
        "contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "source_id": source_id,
        "source_or_simply_cleaned": {column: True for column in SOURCE_COLUMNS[source_id]},
        "source_fields": fields,
    }


def normalized_column_contract(
    source_id: str,
    normalized_columns: Sequence[str],
) -> dict[str, bool]:
    """Classify every persisted normalized column with the requested Boolean."""
    if source_id not in SOURCE_COLUMNS:
        raise ValueError(f"no source-column contract for source_id={source_id!r}")
    aliases = _SOURCE_ALIASES[source_id] | set(SOURCE_COLUMNS[source_id])
    return {str(column): str(column) in aliases for column in normalized_columns}


def source_column_contract_manifest(
    normalized_columns: Sequence[str],
) -> dict[str, Any]:
    """Return an exhaustive, per-source artifact and raw-schema contract."""
    columns = tuple(str(column) for column in normalized_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("normalized artifact has duplicate column names")
    return {
        "contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "semantics": {
            "true": "genuine source value or only whitespace/null cleaned",
            "false": "derived, canonical, inferred, policy, scoring, audit, or helper value",
            "llm_projection": "only true source-schema fields; no canonical fallback",
        },
        "sources": {
            source_id: {
                "source_schema": {
                    column: {"source_or_simply_cleaned": True}
                    for column in source_columns
                },
                "normalized_artifact_columns": {
                    column: {"source_or_simply_cleaned": allowed}
                    for column, allowed in normalized_column_contract(
                        source_id, columns
                    ).items()
                },
            }
            for source_id, source_columns in SOURCE_COLUMNS.items()
        },
    }


def llm_visibility_map() -> dict[str, dict[str, bool]]:
    """Single non-repeated map of source fields exposed to reasoning prompts."""
    return {
        source_id: {column: True for column in columns}
        for source_id, columns in SOURCE_COLUMNS.items()
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
    "SOURCE_COLUMNS",
    "llm_source_projection",
    "llm_visibility_map",
    "llm_source_projection_from_mapping",
    "normalized_column_contract",
    "source_column_contract_manifest",
    "source_fields_from_record",
]
