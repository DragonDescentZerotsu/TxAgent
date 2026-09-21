"""Shared Stage-2 behavior for local-source pair-dimension mappings."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.auxiliary_metadata import (
    AuxiliaryMetadataAttacher,
)
from data.processing.evidence_library.shared.v2.clustered_auxiliary_mapping import (
    DEFAULT_NULL_LIKE,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import (
    MeasurementPair,
)
from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
)
from data.processing.evidence_library.shared.v2.reference_semantics import (
    ReferenceSemanticsAttacher,
    ReferenceSemanticsConfig,
)


def build_local_source_attacher(
    args: argparse.Namespace,
    *,
    contract: Any,
    mapping_version: str,
    output_fields_by_source: dict[str, tuple[str, ...]],
) -> AuxiliaryMetadataAttacher:
    return AuxiliaryMetadataAttacher(
        args.auxiliary_mapping,
        mapping_version=mapping_version,
        applicable_sources=tuple(contract.sources),
        null_like=tuple(DEFAULT_NULL_LIKE),
        output_fields=output_fields_by_source,
    )


def standard_validity_status(record: dict[str, Any]) -> str:
    if record.get("structure_status") != "resolved" or not record.get(
        "canonical_smiles"
    ):
        return "unresolved_structure"
    endpoint = str(record.get("canonical_endpoint_name") or "").casefold()
    if endpoint in {"", "missing_endpoint", "unknown", "__unknown__"}:
        return "missing_canonical_endpoint"
    resolution = str(record.get("measurement_resolution_status") or "")
    if resolution == "relative":
        return "relative_measurement"
    if resolution == "unsure":
        return "measurement_resolution_unsure"
    if resolution in {"unavailable", "not_extracted"}:
        return "non_scalar_measurement"
    value = record.get("finite_scalar_value")
    try:
        finite = value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        return "non_scalar_measurement"
    if not record.get("canonical_unit"):
        return "missing_canonical_unit"
    if record.get("measurement_numeric_domain_status") == "outside_declared_domain":
        return "outside_reviewed_numeric_domain"
    return "valid"


def no_family(*args: Any) -> None:
    del args
    return None


def build_standard_hooks(
    args: argparse.Namespace,
    *,
    task_id: str,
    contract: Any,
    mapping_version: str,
    output_fields_by_source: dict[str, tuple[str, ...]],
    reference_config: ReferenceSemanticsConfig,
) -> NormalizationHooks:
    attacher = build_local_source_attacher(
        args,
        contract=contract,
        mapping_version=mapping_version,
        output_fields_by_source=output_fields_by_source,
    )
    reference = ReferenceSemanticsAttacher(
        replace(
            reference_config,
            mapping_path=Path(args.reference_semantics_mapping),
        ),
        allow_missing=bool(args.allow_missing_reference_semantics),
        fail_closed_unmapped=True,
    )

    def enrich(record: dict[str, Any]) -> dict[str, Any]:
        auxiliary = attacher.attach(record)
        validity = {
            "normalization_validity_status": standard_validity_status(
                {**record, **auxiliary}
            ),
            "normalization_domain_rules_version": (
                f"{task_id}_normalization_domains.v1"
            ),
        }
        return {
            **auxiliary,
            **validity,
            **reference.attach({**record, **auxiliary, **validity}),
        }

    return NormalizationHooks(
        endpoint_normalizer=lambda source, endpoint: endpoint,
        endpoint_standardizer=lambda endpoint, pair: pair,
        family_resolver=no_family,
        record_enricher=enrich,
        run_state={"auxiliary": attacher, "reference": reference},
    )


def build_standard_stage_documents(
    *,
    task_id: str,
    contract: Any,
    hooks: NormalizationHooks,
    normalized: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    unit_policy_manifest: dict[str, Any],
) -> StageDocuments:
    coverage = hooks.run_state["auxiliary"].coverage_audit(normalized)
    reference = hooks.run_state["reference"]
    reference_coverage = reference.coverage_audit(normalized)
    validations = {
        "canonical_endpoint_present": all(
            row.get("canonical_endpoint_name") for row in persisted
        ),
        # A reviewed mapping may intentionally assign null.  Completeness is
        # exact tuple coverage; this check only requires the projected field.
        "canonical_endpoint_concept_present": all(
            "canonical_endpoint_concept" in row for row in persisted
        ),
        "policy_independent_validity_present": all(
            row.get("canonicalization_status") for row in persisted
        ),
        "globally_reconciled_auxiliary_coverage": coverage["validations"][
            "all_applicable_records_mapped"
        ],
        "local_source_tuple_coverage": coverage["validations"][
            "all_applicable_records_mapped"
        ],
        "contextual_unit_policy_loaded": True,
        "source_column_contract_complete": True,
    }
    if not all(validations.values()):
        raise ValueError(f"{task_id} Stage 2 validation failure: {validations}")
    validations.update(
        {
            "reference_semantics_mapping_complete": bool(
                reference_coverage["validations"]["all_applicable_records_mapped"]
            ),
            "reference_semantics_assignment_complete": bool(
                reference_coverage["validations"]["all_applicable_records_assigned"]
            ),
        }
    )
    return StageDocuments(
        validity_policy={
            "normalization_domain_rules_version": (
                f"{task_id}_normalization_domains.v1"
            ),
            "non_scalar_records_retained_as_evidence": True,
            "transfer_semantics": {
                "eligible_reference_scopes": list(reference.config.eligible_scopes),
                "reference_basis_required": reference.config.output_basis,
            },
        },
        auxiliary_mapping_manifest={
            **hooks.run_state["auxiliary"].manifest(),
            "coverage": coverage,
        },
        source_column_contract=contract.manifest(),
        reference_semantics_manifest={
            **reference.manifest(),
            "coverage": reference_coverage,
        },
        validations={
            **validations,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
        },
    )


def validate_registered_mapping(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    task_name: str,
    selected: Path | None,
    selected_reference: Path | None,
) -> None:
    supplied = Path(str(args.auxiliary_mapping or ""))
    if selected is None or not selected.is_file():
        parser.error(
            f"{task_name} Stage 2 requires a registered reviewed auxiliary mapping"
        )
    if supplied.resolve() != selected.resolve():
        parser.error(
            f"{task_name} Stage 2 requires the registry-selected auxiliary mapping"
        )
    if args.validation_level != "full":
        parser.error(f"{task_name} Stage 2 requires --validation-level full")
    supplied_reference = Path(str(args.reference_semantics_mapping or ""))
    if selected_reference is None or not selected_reference.is_file():
        if not args.allow_missing_reference_semantics:
            parser.error(
                f"{task_name} Stage 2 requires a registered reference-semantics mapping"
            )
        return
    if supplied_reference.resolve() != selected_reference.resolve():
        parser.error(
            f"{task_name} Stage 2 requires the registry-selected reference mapping"
        )


__all__ = [
    "build_local_source_attacher",
    "build_standard_hooks",
    "build_standard_stage_documents",
    "no_family",
    "validate_registered_mapping",
]
