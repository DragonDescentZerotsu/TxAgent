"""Shared reference-semantics policy for new V10 evidence tasks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from data.processing.evidence_library.shared.v2.reference_semantics import (
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_ENDPOINT_RATIO,
    REFERENCE_SCOPE_NOT_APPLICABLE,
    REFERENCE_SCOPE_STANDARD_CONTROL,
    ReferenceAssignment,
    ReferenceSemanticsConfig,
    ReferenceSourcePromptSpec,
    deterministic_default_assignment,
)
from data.processing.evidence_library.versions.v10.prompts import PROMPT_ROOT


PROMPT_VERSION = "general_assay_transfer_reference_semantics_prompt.v1"


def _deterministic_assignment(
    record: Mapping[str, Any],
) -> ReferenceAssignment | None:
    return deterministic_default_assignment(record, output_basis=True)


def standard_reference_semantics_config(
    *, task_id: str, source_ids: Sequence[str], mapping_path: Path
) -> ReferenceSemanticsConfig:
    """Return the common fail-closed policy used by newly normalized tasks."""
    return ReferenceSemanticsConfig(
        task_id=task_id,
        canonical_records_path=Path(
            f"data/evidence_libraries/{task_id}/v10/02_canonicalized/records.parquet"
        ),
        mapping_path=mapping_path,
        prompt_registry_path=(
            PROMPT_ROOT / "reference_semantics/general_assay_transfer.json"
        ),
        prompt_version=PROMPT_VERSION,
        output_basis=True,
        source_specs={
            source_id: ReferenceSourcePromptSpec(source_id, ())
            for source_id in source_ids
        },
        deterministic_assignment=_deterministic_assignment,
        eligible_scopes=(
            REFERENCE_SCOPE_ABSOLUTE,
            REFERENCE_SCOPE_ENDPOINT_RATIO,
            REFERENCE_SCOPE_STANDARD_CONTROL,
            REFERENCE_SCOPE_NOT_APPLICABLE,
        ),
        batch_size=50,
        prompt_fields=(
            "canonical_endpoint_name",
            "measurement_text",
            "support_text",
        ),
        labels_only_output=True,
        batch_across_sources=False,
    )


__all__ = ["PROMPT_VERSION", "standard_reference_semantics_config"]
