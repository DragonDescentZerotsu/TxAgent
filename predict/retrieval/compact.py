"""Load compact normalized-evidence indices for inference."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from data.processing.evidence_library.compact_artifacts import (
    CompactArtifactProfile,
    load_compact_neighbor_index,
)
EVIDENCE_SOURCE_LABELS = {
    "bbb_martins": "Starling normalized BBB evidence",
    "bioavailability_ma": "Starling normalized oral bioavailability",
    "skin_reaction": "Starling normalized skin reaction",
}

SOURCE_COLUMNS_BY_TASK: dict[str, dict[str, tuple[str, ...]]] = {
    "bbb_martins": {
        "direct_bbb": (
            "source_index", "pmid", "support_text", "bbb_permeability_label",
            "bbb_transport_label", "assay_model", "species",
            "qualifying_conditions", "extra_details", "endpoint_name",
            "measurement_text", "unit_text", "smiles",
        ),
        "passive_permeability": (
            "paragraph_idx", "support_text", "global_identifier", "assay_type",
            "biological_system", "metric_uncertainty", "passive_bbb_interpretation",
            "qualifying_conditions", "extra_details", "confidence",
            "needs_more_context", "pmid", "extraction_id", "endpoint_name",
            "measurement_text", "unit_text", "smiles",
        ),
        "efflux_transport": (
            "paragraph_idx", "support_text", "global_identifier",
            "transporter_identifier", "evidence_type", "interaction_conclusion",
            "assay_system", "perturbation", "qualifying_conditions",
            "extra_details", "confidence", "needs_more_context", "pmid",
            "extraction_id", "endpoint_name", "measurement_text", "unit_text", "smiles",
        ),
        "influx_transport": (
            "paragraph_idx", "support_text", "global_identifier", "mediator_name",
            "mediator_identifier", "transport_mechanism", "evidence_basis",
            "assay_model", "qualifying_conditions", "extra_details", "confidence",
            "needs_more_context", "pmid", "extraction_id", "endpoint_name",
            "measurement_text", "unit_text", "smiles",
        ),
    },
    "bioavailability_ma": {
        "oral_exposure": (
            "pmid", "extraction_id", "global_identifier", "confidence",
            "paragraph_idx", "support_text", "statistic_type", "oral_dose",
            "study_context", "comparator_exposure", "qualifying_conditions",
            "extra_details", "endpoint_name", "measurement_text", "unit_text", "smiles",
        ),
        "fa": (
            "pmid", "extraction_id", "global_identifier", "confidence",
            "paragraph_idx", "support_text", "assay_system", "condition_medium",
            "biological_context", "formulation_or_solid_form",
            "qualifying_conditions", "extra_details", "endpoint_name",
            "measurement_text", "unit_text", "smiles",
        ),
        "fg": (
            "pmid", "extraction_id", "global_identifier", "confidence",
            "paragraph_idx", "support_text", "molecule_name", "transporter_or_enzyme",
            "substrate_status", "assay_system", "intestinal_site",
            "qualifying_conditions", "extra_details", "endpoint_name",
            "measurement_text", "unit_text", "smiles",
        ),
        "fh": (
            "pmid", "extraction_id", "global_identifier", "confidence",
            "paragraph_idx", "support_text", "assay_system", "species",
            "molecular_form", "enzyme_or_pathway", "qualifying_conditions",
            "extra_details", "endpoint_name", "measurement_text", "unit_text", "smiles",
        ),
        "hf_bioavailability": (
            "bioavailability_report_type", "comparator", "dose", "extra_details",
            "molecule_name", "oral_exposure_mode", "pmid", "qualifying_conditions",
            "species_or_population", "support_text", "measurement_text", "unit_text", "smiles",
        ),
    },
    "skin_reaction": {
        "direct_skin_reaction": (
            "pmid", "extraction_id", "confidence", "paragraph_idx", "support_text",
            "outcome_label", "assay_or_test", "species_or_population",
            "dose_or_concentration", "positive_count", "total_tested", "extra_details",
            "endpoint_name", "measurement_text", "unit_text", "smiles",
        ),
        "sensitization_aop": (
            "pmid", "extraction_id", "global_identifier", "confidence",
            "paragraph_idx", "support_text", "assay_type", "aop_event", "result_label",
            "experimental_conditions", "qualifying_conditions", "extra_details",
            "needs_more_context", "endpoint_name", "measurement_text", "unit_text", "smiles",
        ),
        "phototoxicity_irritation_local_damage": (
            "pmid", "extraction_id", "global_identifier", "confidence", "paragraph_idx",
            "support_text", "evidence_system", "assay_method", "result_label",
            "experimental_conditions", "light_conditions", "qualifying_conditions",
            "extra_details", "needs_more_context", "endpoint_name", "measurement_text",
            "unit_text", "smiles",
        ),
        "skin_exposure": (
            "pmid", "extraction_id", "global_identifier", "confidence", "paragraph_idx",
            "support_text", "study_design", "skin_source", "formulation_vehicle",
            "exposure_time", "qualifying_conditions", "extra_details", "needs_more_context",
            "endpoint_name", "measurement_text", "unit_text", "smiles",
        ),
    },
}


def project_source_record(
    record: Mapping[str, Any], *, source_columns: Mapping[str, tuple[str, ...]]
) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    if source_id not in source_columns:
        raise ValueError(f"no compact source contract for {source_id!r}")
    columns = source_columns[source_id]
    return {
        "contract_version": "source_column_contract.v1",
        "source_id": source_id,
        "source_or_simply_cleaned": {column: True for column in columns},
        "source_fields": {column: _json_value(record.get(column)) for column in columns},
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


def load_compact_index(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Hydrate one compact index using its task's prompt-visible source fields."""
    task_id = _task_id(manifest)
    try:
        label = EVIDENCE_SOURCE_LABELS[task_id]
    except KeyError as exc:
        raise ValueError(f"unsupported compact inference task: {task_id!r}") from exc
    source_columns = SOURCE_COLUMNS_BY_TASK[task_id]
    profile = CompactArtifactProfile(
        task_id=task_id,
        artifact_version=str(manifest.get("artifact_version") or ""),
        index_version=str(manifest.get("index_version") or ""),
        evidence_source_label=label,
        source_columns=source_columns,
        llm_source_projection=partial(
            project_source_record,
            source_columns=source_columns,
        ),
        record_contract_version=str(manifest.get("record_contract_version") or ""),
        source_contract_version=str(manifest.get("source_contract_version") or ""),
    )
    return load_compact_neighbor_index(path, profile=profile)


def read_compact_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"index directory lacks manifest.json: {path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _task_id(manifest: dict[str, Any]) -> str:
    task_id = str(manifest.get("task_id") or "").strip()
    if task_id:
        return task_id
    version = str(manifest.get("index_version") or "")
    prefix, separator, suffix = version.partition(".")
    return prefix if separator and suffix == "compact_neighbor_index.v1" else ""
