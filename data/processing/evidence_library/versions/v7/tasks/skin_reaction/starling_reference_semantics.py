"""Skin-Reaction row-level reference-semantics generation and policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v1.reference_semantics import (
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_ENDPOINT_RATIO,
    REFERENCE_SCOPE_NOT_APPLICABLE,
    REFERENCE_SCOPE_STANDARD_CONTROL,
    ReferenceAssignment,
    ReferenceSemanticsConfig,
    ReferenceSourcePromptSpec,
    deterministic_default_assignment,
)
from data.processing.evidence_library.versions.v7.prompts import PROMPT_ROOT


TASK_ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "skin_reaction_reference_semantics_prompt.v2"
MAPPING_VERSION = "skin_reaction_reference_semantics.v3"
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/reference_semantics_v3/reference_semantics.parquet"
)
DEFAULT_RECORDS_PATH = Path(
    "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
    "starling_normalized_v7/02_canonicalized/records.parquet"
)


def deterministic_assignment(record: Mapping[str, Any]) -> ReferenceAssignment | None:
    return deterministic_default_assignment(record, output_basis=True)


REFERENCE_SEMANTICS_CONFIG = ReferenceSemanticsConfig(
    task_id="skin_reaction",
    canonical_records_path=DEFAULT_RECORDS_PATH,
    mapping_path=DEFAULT_MAPPING_PATH,
    prompt_registry_path=PROMPT_ROOT / "reference_semantics/skin_reaction.json",
    prompt_version=PROMPT_VERSION,
    output_basis=True,
    source_specs={
        "direct_skin_reaction": ReferenceSourcePromptSpec(
            "direct_skin_reaction",
            (
                "outcome_label",
                "assay_or_test",
                "species_or_population",
                "dose_or_concentration",
                "positive_count",
                "total_tested",
                "extra_details",
            ),
        ),
        "sensitization_aop": ReferenceSourcePromptSpec(
            "sensitization_aop",
            (
                "assay_type",
                "aop_event",
                "result_label",
                "experimental_conditions",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
            ),
        ),
        "phototoxicity_irritation_local_damage": ReferenceSourcePromptSpec(
            "phototoxicity_irritation_local_damage",
            (
                "evidence_system",
                "assay_method",
                "result_label",
                "experimental_conditions",
                "light_conditions",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
            ),
        ),
        "skin_exposure": ReferenceSourcePromptSpec(
            "skin_exposure",
            (
                "study_design",
                "skin_source",
                "formulation_vehicle",
                "exposure_time",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
            ),
        ),
    },
    deterministic_assignment=deterministic_assignment,
    eligible_scopes=(
        REFERENCE_SCOPE_ABSOLUTE,
        REFERENCE_SCOPE_ENDPOINT_RATIO,
        REFERENCE_SCOPE_STANDARD_CONTROL,
        REFERENCE_SCOPE_NOT_APPLICABLE,
    ),
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
