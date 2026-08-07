"""Frozen row-level reference semantics for Starling scalar evidence.

The LLM is an offline classifier.  Runtime normalization only joins its frozen
mapping and applies deterministic precedence for authoritative source fields,
controlled scales, and non-scalars.  Pair-bucket eligibility remains a Stage-04
decision and is deliberately separate from factual measurement validity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


REFERENCE_SEMANTICS_VERSION = "starling_reference_semantics.v1"
REFERENCE_SCOPE_ABSOLUTE = "absolute"
REFERENCE_SCOPE_ENDPOINT_RATIO = "endpoint_defined_ratio"
REFERENCE_SCOPE_STANDARD_CONTROL = "standardized_control_ratio"
REFERENCE_SCOPE_COMPARATOR = "comparator_relative"
REFERENCE_SCOPE_NOT_APPLICABLE = "not_applicable"
REFERENCE_SCOPE_UNKNOWN = "unknown"

REFERENCE_SCOPES = frozenset(
    {
        REFERENCE_SCOPE_ABSOLUTE,
        REFERENCE_SCOPE_ENDPOINT_RATIO,
        REFERENCE_SCOPE_STANDARD_CONTROL,
        REFERENCE_SCOPE_COMPARATOR,
        REFERENCE_SCOPE_NOT_APPLICABLE,
        REFERENCE_SCOPE_UNKNOWN,
    }
)

REFERENCE_BASIS_NONE = "none"
REFERENCE_BASIS_UNKNOWN = "unknown"
REFERENCE_BASES = frozenset(
    {
        REFERENCE_BASIS_NONE,
        "administered_dose",
        "applied_dose",
        "oral_dose",
        "recovered_material",
        "total_absorbed_amount",
        "plasma",
        "unbound_plasma",
        "blood",
        "serum",
        "csf",
        "brain",
        "tissue",
        "a_to_b_transport",
        "b_to_a_transport",
        "assay_control",
        "vehicle_control",
        "untreated_control",
        "baseline",
        "wild_type",
        "comparator_drug",
        "comparator_formulation",
        "comparator_treatment",
        "other_explicit",
        REFERENCE_BASIS_UNKNOWN,
    }
)

ENDPOINT_RATIO_BASES = frozenset(
    {
        "administered_dose",
        "applied_dose",
        "oral_dose",
        "recovered_material",
        "total_absorbed_amount",
        "plasma",
        "unbound_plasma",
        "blood",
        "serum",
        "csf",
        "brain",
        "tissue",
        "a_to_b_transport",
        "b_to_a_transport",
        "other_explicit",
    }
)
STANDARD_CONTROL_BASES = frozenset(
    {"assay_control", "vehicle_control", "untreated_control"}
)


@dataclass(frozen=True)
class ReferenceAssignment:
    scope: str
    basis: str | None
    method: str

    def __post_init__(self) -> None:
        if self.scope not in REFERENCE_SCOPES:
            raise ValueError(f"unsupported reference scope: {self.scope!r}")
        if self.basis is not None and self.basis not in REFERENCE_BASES:
            raise ValueError(f"unsupported reference basis: {self.basis!r}")


@dataclass(frozen=True)
class ReferenceSourcePromptSpec:
    source_id: str
    extra_fields: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceSemanticsConfig:
    task_id: str
    canonical_records_path: Path
    mapping_path: Path
    prompt_registry_path: Path
    prompt_version: str
    output_basis: bool
    source_specs: Mapping[str, ReferenceSourcePromptSpec]
    deterministic_assignment: Callable[[Mapping[str, Any]], ReferenceAssignment | None]
    eligible_scopes: tuple[str, ...]
    batch_size: int = 25
    prompt_fields: tuple[str, ...] = ()
    labels_only_output: bool = False
    batch_across_sources: bool = False
    generation_policy_fields: tuple[str, ...] = ()
    generation_no_call_assignment: (
        Callable[[Mapping[str, Any]], ReferenceAssignment | None] | None
    ) = None


@dataclass(frozen=True)
class ReferenceEligibilitySpec:
    eligible_scopes: tuple[str, ...]
    basis_required: bool


class ReferenceSemanticsAttacher:
    """Attach one frozen row mapping with deterministic no-call precedence."""

    def __init__(self, config: ReferenceSemanticsConfig, *, allow_missing: bool = False):
        self.config = config
        self.path = Path(config.mapping_path)
        self.allow_missing = allow_missing
        self._mapping: dict[str, ReferenceAssignment] = {}
        if self.path.exists():
            frame = pd.read_parquet(self.path)
            required = {"cleaned_record_id", "reference_scope", "assignment_method"}
            if config.output_basis:
                required.add("reference_basis")
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"reference mapping {self.path} lacks columns {sorted(missing)}"
                )
            if frame["cleaned_record_id"].astype(str).duplicated().any():
                raise ValueError("reference mapping cleaned_record_id values are not unique")
            for row in frame.to_dict("records"):
                record_id = str(row.get("cleaned_record_id") or "")
                basis = row.get("reference_basis") if config.output_basis else None
                if basis is not None and pd.isna(basis):
                    basis = None
                self._mapping[record_id] = ReferenceAssignment(
                    str(row.get("reference_scope") or ""),
                    str(basis) if basis is not None else None,
                    str(row.get("assignment_method") or "frozen_mapping"),
                )
        elif not allow_missing:
            raise FileNotFoundError(f"reference semantics mapping not found: {self.path}")

    @property
    def ready(self) -> bool:
        return self.path.exists()

    def assignment(self, record: Mapping[str, Any]) -> ReferenceAssignment:
        deterministic = self.config.deterministic_assignment(record)
        if deterministic is not None:
            return deterministic
        record_id = str(record.get("cleaned_record_id") or "")
        mapped = self._mapping.get(record_id)
        if mapped is not None:
            return mapped
        if self.allow_missing:
            return ReferenceAssignment(
                REFERENCE_SCOPE_UNKNOWN,
                REFERENCE_BASIS_UNKNOWN if self.config.output_basis else None,
                "mapping_not_available",
            )
        raise ValueError(f"missing reference-semantics assignment for {record_id!r}")

    def attach(self, record: Mapping[str, Any]) -> dict[str, Any]:
        assignment = self.assignment(record)
        output = {
            "canonical_reference_scope": assignment.scope,
            "reference_semantics_assignment_method": assignment.method,
        }
        if self.config.output_basis:
            output["canonical_reference_basis"] = assignment.basis
        return output

    def coverage_audit(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        missing: list[str] = []
        for record in records:
            deterministic = self.config.deterministic_assignment(record)
            if deterministic is not None:
                counts[deterministic.method] += 1
                continue
            record_id = str(record.get("cleaned_record_id") or "")
            if record_id in self._mapping:
                counts[self._mapping[record_id].method] += 1
            else:
                missing.append(record_id)
        return {
            "records": len(records),
            "assignment_method_counts": dict(sorted(counts.items())),
            "missing_mapping_records": len(missing),
            "missing_mapping_record_examples": missing[:20],
            "validations": {
                "mapping_available": self.ready,
                "all_applicable_records_mapped": not missing,
            },
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "reference_semantics_version": REFERENCE_SEMANTICS_VERSION,
            "prompt_version": self.config.prompt_version,
            "mapping_path": str(self.path),
            "mapping_sha256": file_sha256(self.path) if self.path.exists() else None,
            "mapping_rows": len(self._mapping),
            "output_basis": self.config.output_basis,
        }


def deterministic_default_assignment(
    record: Mapping[str, Any], *, output_basis: bool
) -> ReferenceAssignment | None:
    """Return no-call assignments shared by every task."""
    scale_value = record.get("canonical_measurement_scale_id")
    if scale_value is None or pd.isna(scale_value):
        scale_value = record.get("categorical_encoder_id")
    scale_id = "" if scale_value is None or pd.isna(scale_value) else str(scale_value)
    if scale_id:
        return ReferenceAssignment(
            REFERENCE_SCOPE_NOT_APPLICABLE,
            REFERENCE_BASIS_NONE if output_basis else None,
            "controlled_measurement_scale",
        )
    finite_scalar = record.get("finite_scalar_value")
    if finite_scalar is None or pd.isna(finite_scalar):
        return ReferenceAssignment(
            REFERENCE_SCOPE_UNKNOWN,
            REFERENCE_BASIS_UNKNOWN if output_basis else None,
            "non_scalar",
        )
    validity_status = str(
        record.get("canonicalization_status")
        or record.get("normalization_validity_status")
        or ""
    )
    if validity_status and validity_status != "valid":
        return ReferenceAssignment(
            REFERENCE_SCOPE_UNKNOWN,
            REFERENCE_BASIS_UNKNOWN if output_basis else None,
            "ineligible_before_reference_classification",
        )
    return None


def reference_exclusion_reason(
    record: Mapping[str, Any], spec: ReferenceEligibilitySpec
) -> str | None:
    """Return a Stage-04 exclusion reason, never a factual validity status."""
    scope = str(record.get("canonical_reference_scope") or "")
    if not scope:
        return "missing_canonical_reference_scope"
    if scope not in REFERENCE_SCOPES:
        return "unsupported_canonical_reference_scope"
    if scope not in spec.eligible_scopes:
        return f"reference_scope_{scope}"
    basis = str(record.get("canonical_reference_basis") or "") if spec.basis_required else ""
    if not spec.basis_required:
        return None
    if basis not in REFERENCE_BASES:
        return "unsupported_canonical_reference_basis"
    if scope == REFERENCE_SCOPE_ABSOLUTE:
        return None if basis == REFERENCE_BASIS_NONE else "absolute_reference_has_basis"
    if scope == REFERENCE_SCOPE_ENDPOINT_RATIO:
        return None if basis in ENDPOINT_RATIO_BASES else "unsupported_endpoint_ratio_basis"
    if scope == REFERENCE_SCOPE_STANDARD_CONTROL:
        return None if basis in STANDARD_CONTROL_BASES else "unsupported_standard_control_basis"
    if scope == REFERENCE_SCOPE_NOT_APPLICABLE:
        scale_id = str(record.get("canonical_measurement_scale_id") or "")
        if not scale_id:
            return "not_applicable_without_controlled_scale"
        return None if basis == REFERENCE_BASIS_NONE else "not_applicable_reference_has_basis"
    return f"reference_scope_{scope}"


__all__ = [
    "ENDPOINT_RATIO_BASES",
    "REFERENCE_BASES",
    "REFERENCE_SCOPES",
    "REFERENCE_SCOPE_ABSOLUTE",
    "REFERENCE_SCOPE_COMPARATOR",
    "REFERENCE_SCOPE_ENDPOINT_RATIO",
    "REFERENCE_SCOPE_NOT_APPLICABLE",
    "REFERENCE_SCOPE_STANDARD_CONTROL",
    "REFERENCE_SCOPE_UNKNOWN",
    "ReferenceAssignment",
    "ReferenceEligibilitySpec",
    "ReferenceSemanticsAttacher",
    "ReferenceSemanticsConfig",
    "ReferenceSourcePromptSpec",
    "deterministic_default_assignment",
    "reference_exclusion_reason",
]
