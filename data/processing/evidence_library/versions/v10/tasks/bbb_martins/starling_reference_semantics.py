"""BBB row-level reference-semantics generation and runtime policy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.shared.v2.reference_semantics import (
    REFERENCE_BASIS_NONE,
    REFERENCE_BASIS_UNKNOWN,
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_COMPARATOR,
    REFERENCE_SCOPE_ENDPOINT_RATIO,
    REFERENCE_SCOPE_NOT_APPLICABLE,
    REFERENCE_SCOPE_STANDARD_CONTROL,
    REFERENCE_SCOPE_UNKNOWN,
    ReferenceAssignment,
    ReferenceSemanticsConfig,
    ReferenceSourcePromptSpec,
    deterministic_default_assignment,
)
from data.processing.evidence_library.versions.v10.prompts import PROMPT_ROOT


TASK_ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "bbb_reference_semantics_prompt.v2"
MAPPING_VERSION = "bbb_reference_semantics.v4"
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/reference_semantics_v4/reference_semantics.parquet"
)
DEFAULT_RECORDS_PATH = Path(
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
    "starling_normalized_v7/02_canonicalized/records.parquet"
)


def deterministic_assignment(record: Mapping[str, Any]) -> ReferenceAssignment | None:
    default = deterministic_default_assignment(record, output_basis=True)
    return default if default is not None else generation_no_call_assignment(record)


_PHYSICAL_ABSOLUTE_QUANTITY_KINDS = frozenset(
    {
        "auc",
        "concentration",
        "influx_clearance_rate",
        "linear_permeability",
        "log_permeability",
        "negative_log_permeability",
    }
)
_EXPLICIT_DENOMINATOR_RE = re.compile(
    r"_to_(unbound_plasma|plasma|serum|blood|csf|brain|tissue)(?:_|$)"
)


def _unit_has_reference_basis(value: Any) -> bool:
    """Return whether a canonical unit embeds dose/fraction reference semantics."""
    parsed = canonicalize_unit(value, task="bbb_martins")
    canonical = parsed.canonical.casefold()
    if not canonical or parsed.unknown_tokens:
        return True
    segments = set(re.split(r"[\s·/*]+", canonical))
    return "%" in canonical or bool({"dose", "injected", "id"} & segments)


def generation_no_call_assignment(
    record: Mapping[str, Any],
) -> ReferenceAssignment | None:
    """Resolve only BBB scalar reference semantics fixed by reviewed definitions."""
    status = str(record.get("canonical_semantics_status") or "")
    quantity_kind = str(record.get("canonical_quantity_kind") or "")
    if (
        status == "approved"
        and quantity_kind in _PHYSICAL_ABSOLUTE_QUANTITY_KINDS
        and not _unit_has_reference_basis(
            record.get("canonical_unit_text") or record.get("canonical_unit")
        )
    ):
        return ReferenceAssignment(
            REFERENCE_SCOPE_ABSOLUTE,
            REFERENCE_BASIS_NONE,
            "bbb_no_call_approved_physical_scalar.v1",
        )
    if status == "approved" and quantity_kind == "fold_change":
        return ReferenceAssignment(
            REFERENCE_SCOPE_COMPARATOR,
            REFERENCE_BASIS_UNKNOWN,
            "bbb_no_call_fold_change_ineligible.v1",
        )
    if status == "approved" and quantity_kind == "ratio":
        endpoint = str(record.get("canonical_endpoint_name") or "")
        match = _EXPLICIT_DENOMINATOR_RE.search(endpoint)
        if match is not None:
            return ReferenceAssignment(
                REFERENCE_SCOPE_ENDPOINT_RATIO,
                match.group(1),
                "bbb_no_call_explicit_tissue_fluid_ratio.v1",
            )
    return None


REFERENCE_SEMANTICS_CONFIG = ReferenceSemanticsConfig(
    task_id="bbb_martins",
    canonical_records_path=DEFAULT_RECORDS_PATH,
    mapping_path=DEFAULT_MAPPING_PATH,
    prompt_registry_path=PROMPT_ROOT / "reference_semantics/bbb.json",
    prompt_version=PROMPT_VERSION,
    output_basis=True,
    source_specs={
        "direct_bbb": ReferenceSourcePromptSpec(
            "direct_bbb",
            ("assay_model", "species", "qualifying_conditions", "extra_details"),
        ),
        "passive_permeability": ReferenceSourcePromptSpec(
            "passive_permeability",
            (
                "assay_type",
                "biological_system",
                "metric_uncertainty",
                "passive_bbb_interpretation",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
            ),
        ),
        "efflux_transport": ReferenceSourcePromptSpec(
            "efflux_transport",
            (
                "transporter_identifier",
                "evidence_type",
                "interaction_conclusion",
                "assay_system",
                "perturbation",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
            ),
        ),
        "influx_transport": ReferenceSourcePromptSpec(
            "influx_transport",
            (
                "transport_mechanism",
                "evidence_basis",
                "assay_model",
                # ``reported_result`` is the measurement role now, so it reaches
                # the prompt through CORE_FIELDS' ``measurement_text``.
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
    batch_across_sources=True,
    generation_policy_fields=(
        "canonical_endpoint_name",
        "canonical_unit_text",
        "canonical_quantity_kind",
        "canonical_semantics_status",
    ),
    generation_no_call_assignment=generation_no_call_assignment,
)


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "REFERENCE_SEMANTICS_CONFIG",
    "deterministic_assignment",
    "generation_no_call_assignment",
]
