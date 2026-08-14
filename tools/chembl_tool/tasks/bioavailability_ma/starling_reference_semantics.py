"""Oral-Bioavailability row-level reference-semantics policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.reference_semantics import (
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_COMPARATOR,
    REFERENCE_SCOPE_NOT_APPLICABLE,
    REFERENCE_SCOPE_UNKNOWN,
    ReferenceAssignment,
    ReferenceSemanticsConfig,
    ReferenceSourcePromptSpec,
    deterministic_default_assignment,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    normalize_bioavailability_report_type,
)


TASK_ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "bioavailability_reference_semantics_prompt.v2"
MAPPING_VERSION = "bioavailability_reference_semantics.v2"
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/reference_semantics_v2/reference_semantics.parquet"
)
DEFAULT_RECORDS_PATH = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7/02_canonicalized/records.parquet"
)


def deterministic_assignment(record: Mapping[str, Any]) -> ReferenceAssignment | None:
    common = deterministic_default_assignment(record, output_basis=False)
    if common is not None:
        return common
    if str(record.get("source_id") or "") != "hf_bioavailability":
        return None
    report_type = normalize_bioavailability_report_type(
        record.get("bioavailability_report_type")
    )
    if report_type in {"absolute", "systemic_availability", "extent_f"}:
        return ReferenceAssignment(
            REFERENCE_SCOPE_ABSOLUTE, None, "authoritative_hf_report_type"
        )
    if report_type in {"relative_comparison", "apparent"}:
        return ReferenceAssignment(
            REFERENCE_SCOPE_COMPARATOR, None, "authoritative_hf_report_type"
        )
    return ReferenceAssignment(
        REFERENCE_SCOPE_UNKNOWN, None, "unresolved_hf_report_type"
    )


REFERENCE_SEMANTICS_CONFIG = ReferenceSemanticsConfig(
    task_id="bioavailability_ma",
    canonical_records_path=DEFAULT_RECORDS_PATH,
    mapping_path=DEFAULT_MAPPING_PATH,
    prompt_registry_path=TASK_ROOT / "data_processing/reference_semantics_prompts.json",
    prompt_version=PROMPT_VERSION,
    output_basis=False,
    source_specs={
        "oral_exposure": ReferenceSourcePromptSpec(
            "oral_exposure",
            (
                "statistic_type",
                "oral_dose",
                "study_context",
                "comparator_exposure",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        "fa": ReferenceSourcePromptSpec(
            "fa",
            (
                "assay_system",
                "condition_medium",
                "biological_context",
                "formulation_or_solid_form",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        "fg": ReferenceSourcePromptSpec(
            "fg",
            (
                "transporter_or_enzyme",
                "substrate_status",
                "assay_system",
                "intestinal_site",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        "fh": ReferenceSourcePromptSpec(
            "fh",
            (
                "assay_system",
                "species",
                "molecular_form",
                "enzyme_or_pathway",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        # Deterministic HF handling means this spec is a defensive invariant.
        "hf_bioavailability": ReferenceSourcePromptSpec(
            "hf_bioavailability", ("bioavailability_report_type", "comparator")
        ),
    },
    deterministic_assignment=deterministic_assignment,
    eligible_scopes=(REFERENCE_SCOPE_ABSOLUTE, REFERENCE_SCOPE_NOT_APPLICABLE),
    batch_size=50,
    prompt_fields=("measurement_text", "support_text"),
    labels_only_output=True,
    batch_across_sources=False,
)


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "REFERENCE_SEMANTICS_CONFIG",
    "deterministic_assignment",
]
