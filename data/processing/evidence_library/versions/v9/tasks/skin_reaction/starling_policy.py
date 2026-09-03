"""Skin_Reaction plug-in for the shared normalized-Starling builder.

Wiring only: every rule lives in the module that owns it.  Two differences from
Bioavailability_Ma are structural rather than incidental:

* all four sources carry their own ``SMILES`` column, so there is no shared
  identifier-to-SMILES mapping and no separately pinned direct snapshot; and
* only ``sensitization_aop`` and ``skin_exposure`` carry scalars, so only they
  receive reconciled auxiliary context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from data.processing.paths import evidence_library_root, raw_starling_task_root
from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    RESOLUTION_APPLY_VERSION,
)
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    clean_source_values,
)
from data.processing.evidence_library.versions.v9.measurement_routing import (
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v9.prompts import PROMPT_ROOT
from data.processing.evidence_library.shared.v2.normalization.contracts import MeasurementPair
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    canonicalize_endpoint,
    normalize_measurement_and_unit,
    parse_point_measurement,
)
from data.processing.evidence_library.shared.v2.reference_semantics import (
    ReferenceSemanticsAttacher,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_schema import (
    RECORD_CONTRACT,
    SOURCE_EXTRACTION_PAIR_PRODUCER_IDS,
    SOURCE_PAIR_PRODUCER_IDS,
    SOURCE_RULE_PAIR_PRODUCER_IDS,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    encoding_policy_manifest,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_categorical_response import (
    POLICY as CATEGORICAL_RESPONSE_POLICY,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    DEFAULT_MAPPING_PATH,
    AuxiliaryMetadataAttacher,
    PendingAuxiliaryAttacher,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    endpoint_specific_standardization_of_unit,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_measurement_semantics import (
    DEFAULT_REGISTRY_PATH,
    MEASUREMENT_SEMANTICS_VERSION,
    apply_measurement_semantics,
    default_policy as measurement_semantics_policy,
    measurement_semantics_manifest,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    source_profiles,
    validate_source_digest,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_record_canonicalization import (
    NORMALIZATION_DOMAIN_RULES_VERSION,
    enrich_skin_reaction_validity,
    validity_policy_manifest,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.canonical_starling_source import (
    PARTITION_AUDIT_PATH,
    REJECT_PARTITION,
    partition_for_record,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_reference_semantics import (
    DEFAULT_MAPPING_PATH as DEFAULT_REFERENCE_SEMANTICS_MAPPING,
    REFERENCE_SEMANTICS_CONFIG,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
)


AUXILIARY_PROMPT_REGISTRY_PATH = (
    PROMPT_ROOT / "auxiliary_canonicalization/skin_reaction.json"
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_PATHS,
    ENDPOINT_CONCEPT_VERSION,
    SPACING_AND_SPELLING_VERSION,
    endpoint_concept,
    family_assignment,
    spacing_and_spelling_decision,
    spacing_and_spelling_endpoint,
    validate_endpoint_inventory,
)


TASK_ID = "skin_reaction"
DATASET_NAME = "starling-labs/Skin_Reaction"
DEFAULT_STARLING_DATA_DIR = str(raw_starling_task_root(TASK_ID))
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v9"))
DEFAULT_SOURCE_VALUE_REPAIRS = (
    Path(__file__).resolve().parent
    / "data_processing/source_value_cleaning_v1/reviewed_repairs.jsonl"
)
DEFAULT_REVIEWED_SOURCE_DROPS = (
    Path(__file__).resolve().parent
    / "data_processing/source_value_cleaning_v1/reviewed_drops.jsonl"
)
DEFAULT_SMILES_IDENTITY_AUDIT = (
    Path(__file__).resolve().parent
    / "data_processing/source_value_cleaning_v1/smiles_identity_audit.jsonl"
)
DEFAULT_SMILES_SAMPLE_AUDIT = (
    Path(__file__).resolve().parent
    / "data_processing/source_value_cleaning_v1/smiles_identity_sample_audit_20260829.json"
)
DEFAULT_REVIEWED_NAME_SMILES_CONFLICTS = (
    Path(__file__).resolve().parent
    / "data_processing/source_value_cleaning_v1/reviewed_name_smiles_conflicts.v2.parquet"
)


def _clean_source_values(records: list[dict[str, Any]], args: argparse.Namespace):
    require_all = not int(getattr(args, "max_rows_per_source", 0) or 0)
    result = clean_source_values(
        records,
        task_id=TASK_ID,
        reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS,
        reviewed_drops_path=DEFAULT_REVIEWED_SOURCE_DROPS,
        smiles_identity_audit_path=DEFAULT_SMILES_IDENTITY_AUDIT,
        reviewed_smiles_conflicts_path=DEFAULT_REVIEWED_NAME_SMILES_CONFLICTS,
        require_all_reviewed_repairs=require_all,
        require_all_reviewed_drops=require_all,
        require_all_reviewed_smiles_overrides=require_all,
    )
    auxiliary_mapping = Path(args.auxiliary_mapping)
    attacher = (
        PendingAuxiliaryAttacher(auxiliary_mapping)
        if args.allow_missing_auxiliary_mapping
        else AuxiliaryMetadataAttacher(auxiliary_mapping)
    )

    def endpoint(record: dict[str, Any]) -> str:
        source_id = str(record.get("source_id") or "")
        initial = canonicalize_endpoint(
            spacing_and_spelling_endpoint(
                source_id, str(record.get("endpoint_name") or "")
            )
        )
        if source_id not in {"sensitization_aop", "skin_exposure"}:
            return initial
        pair = normalize_measurement_and_unit(
            record.get("measurement_text"),
            record.get("unit_text"),
            task=TASK_ID,
        )
        pair = endpoint_specific_standardization_of_unit(initial, pair)
        parsed = parse_point_measurement(pair.canonical_measurement)
        semantic = apply_measurement_semantics(
            {
                **record,
                **attacher.attach(record),
                "canonical_endpoint": initial,
                "canonical_measurement": pair.canonical_measurement,
                "canonical_unit": pair.canonical_unit,
                "measurement_unit_status": pair.status,
                "unit_notation_status": pair.unit_notation_status,
                "unit_notation_factor": pair.unit_notation_factor,
                "finite_scalar_value": parsed.value,
                "variation_value": parsed.variation,
            }
        )
        return str(semantic.get("canonical_endpoint") or initial)

    return replace(
        result,
        records=attach_stage1_routes(
            result.records,
            task=TASK_ID,
            endpoint_resolver=endpoint,
        ),
        input_paths=(
            *result.input_paths,
            *((auxiliary_mapping,) if auxiliary_mapping.is_file() else ()),
            DEFAULT_REGISTRY_PATH,
        ),
    )


DEFAULT_BENCHMARK_SPLIT_ROOT = "data/gold_labels/legacy/processed_starling/Skin_Reaction"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--auxiliary-mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument(
        "--allow-missing-auxiliary-mapping",
        action="store_true",
        help=(
            "Build stages 01-05 before the reconciliation pass has run. "
            "Records get auxiliary_mapping_status=not_available and the build "
            "fails its auxiliary-coverage validation on purpose."
        ),
    )
    parser.add_argument(
        "--reference-semantics-mapping",
        default=str(DEFAULT_REFERENCE_SEMANTICS_MAPPING),
    )
    parser.add_argument(
        "--allow-missing-reference-semantics",
        action="store_true",
        help="Allow an explicitly incomplete pre-generation smoke build.",
    )


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    needs_normalization = args.through_stage not in {"source", "clean"}
    if (
        not Path(args.auxiliary_mapping).exists()
        and not args.allow_missing_auxiliary_mapping
    ):
        parser.error(
            f"reconciled auxiliary mapping not found: {args.auxiliary_mapping}. "
            "Build it with tasks.skin_reaction.data_processing, or pass "
            "--allow-missing-auxiliary-mapping for an explicitly incomplete "
            "stage 01-05 build."
        )
    reference_mapping = Path(args.reference_semantics_mapping)
    if (
        needs_normalization
        and not reference_mapping.exists()
        and not args.allow_missing_reference_semantics
    ):
        parser.error(
            f"reference-semantics mapping not found: {reference_mapping}; build it "
            "with common.starling.build_reference_semantics_mapping"
        )


def endpoint_inventory(
    source_id: str, endpoints: list[str], *, strict: bool
) -> dict[str, Any]:
    if strict:
        return validate_endpoint_inventory(source_id, endpoints)
    unique = sorted(set(endpoints))
    registry = [
        spacing_and_spelling_decision(source_id, endpoint).to_dict()
        for endpoint in unique
    ]
    return {
        "source_id": source_id,
        "count": len(unique),
        "n_reviewed_corrections": sum(
            item["status"] == "reviewed_correction" for item in registry
        ),
        "coverage": 1.0,
        "strict_frozen_validation": False,
        "endpoints": registry,
    }


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    mapping_path = Path(args.auxiliary_mapping)
    attacher = (
        PendingAuxiliaryAttacher(mapping_path)
        if args.allow_missing_auxiliary_mapping
        else AuxiliaryMetadataAttacher(mapping_path)
    )
    semantic_results: dict[str, dict[str, Any]] = {}
    reference_attacher = ReferenceSemanticsAttacher(
        replace(
            REFERENCE_SEMANTICS_CONFIG,
            mapping_path=Path(args.reference_semantics_mapping),
        ),
        allow_missing=args.allow_missing_reference_semantics,
        fail_closed_unmapped=True,
    )

    def resolve_measurement_semantics(
        record: dict[str, Any],
        canonical_endpoint: str,
        pair: MeasurementPair,
    ) -> MeasurementPair:
        # The invariant checker calls the same hook on the completed row.  At
        # that point the frozen rule ID/status is the authoritative semantic
        # decision; return the persisted atomic pair rather than attempting a
        # second contextual classification against an already-refined endpoint.
        if record.get("measurement_semantics_status"):
            return MeasurementPair(
                record.get("canonical_measurement"),
                record.get("canonical_unit"),
                str(record.get("measurement_unit_status") or pair.status),
                str(record.get("unit_notation_status") or pair.unit_notation_status),
                record.get("unit_notation_factor"),
            )
        parsed = parse_point_measurement(pair.canonical_measurement)
        auxiliary = attacher.attach(record)
        semantic_endpoint = str(
            record.get("pre_refinement_canonical_endpoint")
            or canonical_endpoint
            or ""
        )
        semantic = apply_measurement_semantics(
            {
                **record,
                **auxiliary,
                "canonical_endpoint": semantic_endpoint,
                "canonical_measurement": pair.canonical_measurement,
                "canonical_unit": pair.canonical_unit,
                "measurement_unit_status": pair.status,
                "unit_notation_status": pair.unit_notation_status,
                "unit_notation_factor": pair.unit_notation_factor,
                "finite_scalar_value": parsed.value,
                "variation_value": parsed.variation,
            }
        )
        semantic_results[str(record.get("cleaned_record_id") or "")] = semantic
        return MeasurementPair(
            semantic.get("canonical_measurement"),
            semantic.get("canonical_unit"),
            str(semantic.get("measurement_unit_status") or pair.status),
            pair.unit_notation_status,
            pair.unit_notation_factor,
        )

    return NormalizationHooks(
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        family_resolver=family_assignment,
        source_measurement_resolver=resolve_measurement_semantics,
        record_enricher=lambda record: _enrich_record(
            record, attacher, semantic_results, reference_attacher
        ),
        assay_transfer_revalidator=lambda record: _revalidate_assay_transfer_record(
            record, reference_attacher
        ),
        run_state={"auxiliary": attacher, "reference": reference_attacher},
    )


def _revalidate_assay_transfer_record(
    record: dict[str, Any], reference_attacher: ReferenceSemanticsAttacher
) -> dict[str, Any]:
    validity = enrich_skin_reaction_validity(record)
    reference = reference_attacher.attach_post_scale_fail_closed(
        {**record, **validity}
    )
    return {**validity, **reference}


def attach_source_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Columnize the raw source contract before removing duplicate JSON."""
    for row in rows:
        for field, value in source_fields_from_record(row).items():
            row.setdefault(field, value)
    return rows


def stage_documents(
    *,
    args: argparse.Namespace,
    hooks: NormalizationHooks,
    normalized: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    unit_policy_manifest: dict[str, Any],
) -> StageDocuments:
    del args
    attacher = hooks.run_state["auxiliary"]
    reference_attacher = hooks.run_state["reference"]
    coverage = attacher.coverage_audit(normalized)
    reference_coverage = reference_attacher.coverage_audit(normalized)
    semantics_audit = measurement_semantics_policy().audit(normalized)
    return StageDocuments(
        validity_policy={
            **validity_policy_manifest(),
            "categorical_response": encoding_policy_manifest(),
            "measurement_semantics": measurement_semantics_manifest(),
            "measurement_semantics_audit": semantics_audit,
        },
        auxiliary_mapping_manifest={**attacher.manifest(), "coverage": coverage},
        source_column_contract=source_column_contract_manifest(
            sorted({column for row in persisted for column in row})
        ),
        validations={
            "one_to_one_measurement_inputs_to_normalized_ids": True,
            "endpoint_orthography_provenance": True,
            "canonical_endpoint_present": True,
            "policy_independent_validity_present": True,
            "globally_reconciled_auxiliary_coverage": bool(
                coverage["validations"]["all_applicable_records_mapped"]
            ),
            "contextual_unit_policy_loaded": True,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
            "source_column_contract_complete": True,
            "llm_source_projection_fail_closed": True,
            "measurement_semantics_coverage": bool(
                semantics_audit["validations"][
                    "all_numeric_records_have_explicit_status"
                ]
            ),
            "reference_semantics_mapping_complete": bool(
                reference_coverage["validations"]["all_applicable_records_mapped"]
            ),
            "reference_semantics_assignment_complete": bool(
                reference_coverage["validations"]["all_applicable_records_assigned"]
            ),
        },
        reference_semantics_manifest={
            **reference_attacher.manifest(),
            "coverage": reference_coverage,
        },
    )


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_concept_version": ENDPOINT_CONCEPT_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "measurement_semantics_version": MEASUREMENT_SEMANTICS_VERSION,
    }
    if complete:
        versions["source_column_contract_version"] = SOURCE_COLUMN_CONTRACT_VERSION
    return versions


def census_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report scalar yield per source, and how much of it is categorically encoded."""
    counts: dict[str, int] = {source_id: 0 for source_id in EXPECTED_SOURCE_ROWS}
    encoded: dict[str, int] = {}
    for record in records:
        if record.get("finite_scalar_value") is None:
            continue
        source_id = str(record.get("source_id") or "")
        counts[source_id] = counts.get(source_id, 0) + 1
        encoder_id = str(record.get("categorical_encoder_id") or "")
        if encoder_id:
            encoded[encoder_id] = encoded.get(encoder_id, 0) + 1
    return {
        "n_finite_scalars_by_source": dict(sorted(counts.items())),
        "n_categorically_encoded_by_encoder": dict(sorted(encoded.items())),
    }


def _enrich_record(
    record: dict[str, Any],
    attacher: Any,
    semantic_results: dict[str, dict[str, Any]],
    reference_attacher: ReferenceSemanticsAttacher,
) -> dict[str, Any]:
    source_projection = llm_source_projection(record)
    auxiliary = attacher.attach(record)
    exact = str(record.get("measurement_resolution_status") or "") in {
        "ok",
        "relative",
        "unsure",
        "unavailable",
    }
    resolution_route = str(record.get("measurement_resolution_route") or "")
    routed = exact or (
        bool(record.get("measurement_resolution_active")) and bool(resolution_route)
    )
    mapped = exact and record.get("measurement_unit_mapping_status") == "mapped"
    encoded = CATEGORICAL_RESPONSE_POLICY.apply({**record, **auxiliary})
    semantic = semantic_results.get(str(record.get("cleaned_record_id") or ""))
    if semantic is None and not routed:
        raise ValueError("measurement semantics were not resolved atomically")
    semantic = semantic or (
        {
            "measurement_semantics_status": "exact_measurement_unit_map",
            "measurement_semantics_rule_id": EXACT_UNIT_MAPPING_VERSION,
            "measurement_semantics_policy_version": EXACT_UNIT_MAPPING_VERSION,
            "measurement_numeric_domain": record.get("measurement_numeric_domain"),
        }
        if mapped
        else {
            "measurement_semantics_status": "categorical_or_non_scalar_route",
            "measurement_semantics_rule_id": str(encoded["categorical_encoder_id"]),
            "measurement_semantics_policy_version": CATEGORICAL_RESPONSE_VERSION,
            "measurement_numeric_domain": "controlled",
        }
        if encoded
        else {
            "measurement_semantics_status": "frozen_measurement_resolution",
            "measurement_semantics_rule_id": RESOLUTION_APPLY_VERSION,
            "measurement_semantics_policy_version": RESOLUTION_APPLY_VERSION,
            "measurement_numeric_domain": None,
        }
        if exact
        else {
            "measurement_semantics_status": "categorical_or_non_scalar_route",
            "measurement_semantics_rule_id": "measurement_resolution_route.v1",
            "measurement_semantics_policy_version": "measurement_resolution_route.v1",
            "measurement_numeric_domain": None,
        }
    )
    source_id = str(record.get("source_id") or "")
    reviewed_endpoint_concept = endpoint_concept(
        source_id,
        str(record.get("endpoint_name") or ""),
        str(record.get("canonical_endpoint") or ""),
    )
    pair_producer_id = str(encoded.get("categorical_encoder_id") or "") or (
        SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id]
        if routed and resolution_route == "extract"
        else SOURCE_RULE_PAIR_PRODUCER_IDS[source_id]
        if routed and resolution_route == "accept"
        else SOURCE_PAIR_PRODUCER_IDS[source_id]
    )
    if encoded:
        validity = {
            "normalization_validity_status": (
                "unresolved_structure"
                if str(record.get("structure_status") or "") != "resolved"
                or not record.get("canonical_smiles")
                else "valid"
            ),
            "normalization_domain_rules_version": CATEGORICAL_RESPONSE_VERSION,
        }
    elif exact:
        validity = {
            "normalization_validity_status": (
                "unresolved_structure"
                if str(record.get("structure_status") or "") != "resolved"
                or not record.get("canonical_smiles")
                else "valid" if mapped else "exact_measurement_excluded"
            ),
            "normalization_domain_rules_version": (
                EXACT_UNIT_MAPPING_VERSION
                if mapped
                else RESOLUTION_APPLY_VERSION
            ),
        }
    elif routed:
        validity = {
            "normalization_validity_status": (
                "unresolved_structure"
                if str(record.get("structure_status") or "") != "resolved"
                or not record.get("canonical_smiles")
                else "valid" if encoded else "non_scalar_measurement"
            ),
            "normalization_domain_rules_version": "measurement_resolution_route.v1",
        }
    else:
        validity = enrich_skin_reaction_validity(
            {**record, **auxiliary, **semantic, **encoded}
        )
    reference = reference_attacher.attach(
        {**record, **auxiliary, **semantic, **encoded, **validity}
    )
    partition = partition_for_record(record)
    partition_fields = (
        {
            "canonical_sensitization_partition": partition.partition,
            "canonical_sensitization_partition_reason": partition.reason,
            "canonical_sensitization_aop_event": partition.aop_event or None,
            "retrieval_exclusion_reason": (
                f"canonical_skin_partition_reject:{partition.reason}"
                if partition.partition == REJECT_PARTITION
                else None
            ),
        }
        if partition is not None
        else {}
    )
    return {
        "source_column_contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "canonical_pair_producer_id": pair_producer_id,
        "canonical_endpoint_concept": reviewed_endpoint_concept,
        **encoded,
        "llm_source_contract_json": json.dumps(
            {
                key: value
                for key, value in source_projection.items()
                if key != "source_fields"
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "llm_source_fields_json": json.dumps(
            source_projection["source_fields"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        **auxiliary,
        **semantic,
        **validity,
        **reference,
        **partition_fields,
    }


POLICY = StarlingTaskPolicy(
    task_id=TASK_ID,
    dataset_name=DATASET_NAME,
    default_data_dir=DEFAULT_STARLING_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    compact=COMPACT_PROFILE,
    expected_source_rows=EXPECTED_SOURCE_ROWS,
    record_contract=RECORD_CONTRACT,
    source_value_cleaner=_clean_source_values,
    source_profiles=source_profiles,
    endpoint_inventory=endpoint_inventory,
    family_resolver=family_assignment,
    build_hooks=build_hooks,
    attach_source_columns=attach_source_columns,
    stage_documents=stage_documents,
    manifest_versions=manifest_versions,
    add_cli_arguments=add_cli_arguments,
    validate_arguments=validate_arguments,
    census_extras=census_extras,
    verify_source_digest=lambda source_id, path: validate_source_digest(source_id, path),
    scientific_assets=(
        DEFAULT_SOURCE_VALUE_REPAIRS,
        DEFAULT_REVIEWED_SOURCE_DROPS,
        DEFAULT_SMILES_IDENTITY_AUDIT,
        DEFAULT_REVIEWED_NAME_SMILES_CONFLICTS,
        DEFAULT_SMILES_SAMPLE_AUDIT,
        DEFAULT_REGISTRY_PATH,
        *ENDPOINT_CONCEPT_PATHS,
        REFERENCE_SEMANTICS_CONFIG.prompt_registry_path,
        DEFAULT_MAPPING_PATH,
        PARTITION_AUDIT_PATH,
        AUXILIARY_PROMPT_REGISTRY_PATH,
        Path(__file__).parent
        / "data_processing/assay_transfer_measurements_v2/policy.json",
    ),
    assay_transfer_measurement_policy=(
        Path(__file__).parent
        / "data_processing/assay_transfer_measurements_v2/policy.json"
    ),
    family_resolver_input_fields=("canonical_sensitization_partition",),
    reference_semantics_enabled=True,
    measurement_resolution_enabled=True,
    exact_unit_mapping_path=(
        Path(__file__).parent
        / "data_processing/canonicalization_v7/exact_measurement_unit_map.v2.json"
    ),
    endpoint_identity_required_sources=(
        "direct_skin_reaction",
        "sensitization_aop",
        "phototoxicity_irritation_local_damage",
        "skin_exposure",
    ),
)


__all__ = ["POLICY", "TASK_ID"]
