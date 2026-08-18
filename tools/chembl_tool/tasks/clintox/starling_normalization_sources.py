"""Pinned seven-source input and source-visibility contracts for ClinTox v7."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.contracts import (
    NormalizedSourceProfile,
)
from tools.chembl_tool.tasks.clintox.starling_source import (
    DIRECT_SOURCE_ID,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    SOURCE_COLUMNS,
)

def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    base = data_dir / DIRECT_SOURCE_ID / "extractions.parquet"
    return [
        NormalizedSourceProfile(
            source_id=DIRECT_SOURCE_ID,
            source_name="ClinTox send_v2 human clinical toxicity",
            source_path=str(base),
            # The shared inventory reader requires a real source column. The
            # policy replaces this inventory with the single declared constant
            # endpoint; cleaning still uses ``endpoint_constant`` below.
            endpoint_field="toxicity_category",
            endpoint_constant="human_clinical_toxicity",
            measurement_field="toxicity_category",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("molecule_name",),
            context_fields=(
                "toxicity_outcome",
                "outcome_measure",
                "clinical_context",
                "dose_or_exposure",
                "fda_approval_status",
                "approved_indication",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=("toxicity_category", "fda_approval_status"),
        ),
        NormalizedSourceProfile(
            source_id="nonclinical_in_vivo_toxicity",
            source_name="ClinTox nonclinical in vivo toxicity",
            source_path=str(data_dir / "nonclinical_in_vivo_toxicity/extractions.parquet"),
            endpoint_field="evidence_type",
            measurement_field="endpoint_value",
            unit_field="endpoint_unit",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "administered_dose",
                "animal_context",
                "exposure_context",
                "observed_effect",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=("evidence_type",),
        ),
        NormalizedSourceProfile(
            source_id="organ_specific_toxicity",
            source_name="ClinTox organ-specific toxicity",
            source_path=str(data_dir / "organ_specific_toxicity/extractions.parquet"),
            endpoint_field="toxicity_endpoint",
            measurement_field="effect_status",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "organ_system",
                "evidence_context",
                "biological_system",
                "exposure_regimen",
                "quantitative_result",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=(
                "organ_system",
                "effect_status",
                "evidence_context",
            ),
        ),
        NormalizedSourceProfile(
            source_id="genotoxicity_carcinogenicity",
            source_name="ClinTox genotoxicity and carcinogenicity",
            source_path=str(data_dir / "genotoxicity_carcinogenicity/extractions.parquet"),
            endpoint_field="endpoint",
            measurement_field="result_direction",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "evidence_category",
                "assay_type",
                "study_context",
                "biological_system",
                "exposure_conditions",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=(
                "evidence_category",
                "assay_type",
                "study_context",
                "result_direction",
            ),
        ),
        NormalizedSourceProfile(
            source_id="cellular_stress",
            source_name="ClinTox cellular stress",
            source_path=str(data_dir / "cellular_stress/extractions.parquet"),
            endpoint_field="stress_endpoint",
            measurement_field="effect_direction",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "evidence_basis",
                "mechanistic_effect",
                "target_or_pathway",
                "biological_model",
                "dose_and_duration",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=(
                "stress_endpoint",
                "effect_direction",
                "evidence_basis",
            ),
        ),
        NormalizedSourceProfile(
            source_id="general_cytotoxicity",
            source_name="ClinTox general cytotoxicity",
            source_path=str(data_dir / "general_cytotoxicity/extractions.parquet"),
            endpoint_field="endpoint_type",
            measurement_field="result_value",
            unit_field="result_unit",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "cell_model",
                "test_concentration",
                "exposure_time_h",
                "assay_method",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=("endpoint_type",),
        ),
        NormalizedSourceProfile(
            source_id="off_target_ddi_exposure",
            source_name="ClinTox off-target, DDI, and exposure",
            source_path=str(data_dir / "off_target_ddi_exposure/extractions.parquet"),
            endpoint_field="target_or_endpoint",
            measurement_field="result_value",
            unit_field="result_unit",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "evidence_type",
                "target_identifier",
                "result_metric",
                "assay_context",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
            literal_text_fields=("evidence_type",),
        ),
    ]


def validate_source_digest(source_id: str, source_parquet: Path) -> str:
    expected = EXPECTED_SOURCE_SHA256.get(source_id)
    if expected is None:
        raise ValueError(f"no pinned digest for source {source_id!r}")
    actual = file_sha256(source_parquet)
    if actual != expected:
        raise ValueError(
            f"source digest drift for {source_id}: expected {expected}, found {actual}"
        )
    return actual


def source_fields_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    declared = SOURCE_COLUMNS.get(source_id)
    if declared is None:
        raise ValueError(f"unknown ClinTox source_id={source_id!r}")
    if all(field in record for field in declared):
        return {field: _json_value(record.get(field)) for field in declared}
    try:
        raw = json.loads(str(record.get("source_payload_json") or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source_payload_json for {source_id!r}") from exc
    if not isinstance(raw, Mapping):
        raise TypeError(f"source_payload_json is not an object for {source_id!r}")
    return {field: _json_value(raw.get(field)) for field in declared}


def llm_source_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    fields = source_fields_from_record(record)
    return {
        "contract_version": "clintox_source_column_contract.v2",
        "source_id": source_id,
        "source_or_simply_cleaned": {field: True for field in fields},
        "source_fields": fields,
    }


def source_column_contract_manifest(
    persisted_columns: Sequence[str],
) -> dict[str, Any]:
    columns = tuple(str(column) for column in persisted_columns)
    return {
        "contract_version": "clintox_source_column_contract.v2",
        "global_identifier_policy": "absent_from_send_v2_and_all_derived_artifacts",
        "qualifying_conditions": {
            DIRECT_SOURCE_ID: "unavailable_in_source_schema",
            "other_sources": "source_visible_when_reported",
        },
        "persisted_columns": list(columns),
        "sources": {
            source_id: {field: True for field in fields}
            for source_id, fields in sorted(SOURCE_COLUMNS.items())
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
    "EXPECTED_SOURCE_ROWS",
    "EXPECTED_SOURCE_SHA256",
    "SOURCE_COLUMNS",
    "llm_source_projection",
    "source_column_contract_manifest",
    "source_fields_from_record",
    "source_profiles",
    "validate_source_digest",
]
