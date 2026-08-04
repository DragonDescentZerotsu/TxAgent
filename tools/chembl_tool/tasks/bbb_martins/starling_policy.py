"""BBB Martins plug-in for the shared normalized-Starling v6 builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from tools.chembl_tool.tasks.bbb_martins.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    DEFAULT_MAPPING_PATH,
    AuxiliaryMetadataAttacher,
    PendingAuxiliaryAttacher,
)
from tools.chembl_tool.tasks.bbb_martins.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    POLICY as CATEGORICAL_RESPONSE_POLICY,
    encoding_policy_manifest,
)
from tools.chembl_tool.tasks.bbb_martins.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from tools.chembl_tool.tasks.bbb_martins.starling_endpoint_normalization import (
    DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING,
    ENDPOINT_NORMALIZATION_VERSION,
    EndpointNormalizer,
    context_fields as endpoint_context_fields,
)
from tools.chembl_tool.tasks.bbb_martins.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    SOURCE_MEASUREMENT_RESOLVER_VERSION,
    endpoint_specific_standardization_of_unit,
    policy_manifest as normalization_policy_manifest,
    source_measurement_resolver,
    unit_provenance_fields,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    parse_point_measurement,
)
from tools.chembl_tool.tasks.bbb_martins.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    source_profiles,
    validate_source_digest,
)
from tools.chembl_tool.tasks.bbb_martins.starling_record_canonicalization import (
    NORMALIZATION_DOMAIN_RULES_VERSION,
    enrich_bbb_validity,
    validity_policy_manifest,
)
from tools.chembl_tool.tasks.bbb_martins.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    SOURCE_COLUMN_POLICY_VERSION,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
)
from tools.chembl_tool.tasks.bbb_martins.starling_spacing_and_spelling import (
    SPACING_AND_SPELLING_VERSION,
    family_assignment,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)


TASK_ID = "bbb_martins"
DATASET_NAME = "starling-labs/BBB plus BBB mechanism-family acquisitions"
DEFAULT_STARLING_DATA_DIR = "data/starling_data/bbb_martins"
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v6"
)
DEFAULT_BENCHMARK_SPLIT_ROOT = "data/processed_starling/BBB_Martins"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--auxiliary-mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument(
        "--allow-missing-auxiliary-mapping",
        action="store_true",
        help="Allow an explicitly incomplete clean/normalize/organize smoke build.",
    )
    parser.add_argument(
        "--benchmark-split-root", default=DEFAULT_BENCHMARK_SPLIT_ROOT
    )
    parser.add_argument(
        "--direct-endpoint-mapping",
        default=str(DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING),
    )
    parser.add_argument(
        "--allow-missing-direct-endpoint-mapping",
        action="store_true",
        help="Allow an explicitly incomplete pre-approval smoke build.",
    )


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not Path(args.auxiliary_mapping).exists() and not args.allow_missing_auxiliary_mapping:
        parser.error(
            f"reconciled auxiliary mapping not found: {args.auxiliary_mapping}; "
            "build it with bbb_martins.data_processing.build_embedding_bucket_mapping"
        )
    direct_mapping = Path(args.direct_endpoint_mapping)
    needs_normalization = args.through_stage != "clean"
    if (
        needs_normalization
        and not direct_mapping.exists()
        and not args.allow_missing_direct_endpoint_mapping
    ):
        parser.error(
            f"human-approved Direct endpoint mapping not found: {direct_mapping}; "
            "build and review the proposal with "
            "bbb_martins.data_processing.build_direct_endpoint_mapping"
        )


def endpoint_inventory(source_id: str, endpoints: list[str], *, strict: bool) -> dict[str, Any]:
    if strict:
        inventory = validate_endpoint_inventory(source_id, endpoints)
        return {
            **inventory,
            "inventory_semantics": "raw_source_inventory_before_run_specific_endpoint_mapping",
            "decision_semantics": "default_policy_preview_not_runtime_registry",
            "runtime_decision_registry": "02_normalized/endpoint_registry.json",
        }
    unique = sorted(set(endpoints))
    return {
        "source_id": source_id,
        "inventory_semantics": "raw_source_inventory_before_run_specific_endpoint_mapping",
        "decision_semantics": "default_policy_preview_not_runtime_registry",
        "runtime_decision_registry": "02_normalized/endpoint_registry.json",
        "count": len(unique),
        "n_reviewed_corrections": 0,
        "coverage": 1.0,
        "strict_frozen_validation": False,
        "endpoints": [
            spacing_and_spelling_decision(source_id, value).to_dict()
            for value in unique
        ],
    }


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    mapping_path = Path(args.auxiliary_mapping)
    attacher = (
        AuxiliaryMetadataAttacher(mapping_path)
        if mapping_path.exists()
        else PendingAuxiliaryAttacher(mapping_path)
    )
    endpoint_normalizer = EndpointNormalizer(args.direct_endpoint_mapping)
    return NormalizationHooks(
        endpoint_normalizer=endpoint_normalizer.decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        source_measurement_resolver=source_measurement_resolver,
        family_resolver=family_assignment,
        record_enricher=lambda record: _enrich_record(record, attacher),
        run_state={"auxiliary": attacher, "endpoint": endpoint_normalizer},
    )


def attach_source_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        for field, value in source_fields_from_record(row).items():
            enriched.setdefault(field, value)
        output.append(enriched)
    return output


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
    endpoint_normalizer = hooks.run_state["endpoint"]
    coverage = attacher.coverage_audit(normalized)
    return StageDocuments(
        validity_policy={
            **validity_policy_manifest(),
            "normalization": normalization_policy_manifest(),
            "categorical_response": encoding_policy_manifest(),
            "endpoint_normalization": endpoint_normalizer.manifest(),
        },
        auxiliary_mapping_manifest={**attacher.manifest(), "coverage": coverage},
        source_column_contract=source_column_contract_manifest(
            sorted({column for row in persisted for column in row})
        ),
        validations={
            "one_to_one_cleaned_to_normalized_ids": True,
            "measurement_unit_pair_errors": 0,
            "endpoint_orthography_provenance": True,
            "canonical_endpoint_present": True,
            "direct_endpoint_mapping_human_approved": endpoint_normalizer.direct_ready,
            "policy_independent_validity_present": True,
            "globally_reconciled_auxiliary_coverage": bool(
                coverage["validations"]["all_applicable_records_mapped"]
            ),
            "contextual_unit_policy_loaded": True,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
            "source_column_contract_complete": True,
            "llm_source_projection_fail_closed": True,
        },
        endpoint_registry=_endpoint_registry(normalized),
    )


def _endpoint_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist the decisions actually applied by this run's endpoint hook."""
    registry: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        source_id = str(record.get("source_id") or "")
        endpoint_name = str(record.get("endpoint_name") or "missing_endpoint")
        encoder_id = str(record.get("categorical_encoder_id") or "")
        decision = {
            "endpoint_name": endpoint_name,
            "decision_variant": encoder_id or "source_measurement",
            "categorical_encoder_id": encoder_id or None,
            "spacing_and_spelling_endpoint": record.get("spacing_and_spelling_endpoint"),
            "canonical_endpoint": record.get("canonical_endpoint"),
            "status": record.get("spacing_and_spelling_status"),
            "reason": record.get("spacing_and_spelling_reason"),
            "version": record.get("spacing_and_spelling_version"),
        }
        decision_key = f"{endpoint_name}\u0000{encoder_id or 'source_measurement'}"
        previous = registry.setdefault(source_id, {}).setdefault(decision_key, decision)
        if previous != decision:
            raise ValueError(
                f"endpoint normalization is not deterministic for {source_id}/{endpoint_name}"
            )
    return {
        source_id: [mapping[key] for key in sorted(mapping)]
        for source_id, mapping in sorted(registry.items())
    }


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "source_measurement_resolver_version": SOURCE_MEASUREMENT_RESOLVER_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "endpoint_normalization_version": ENDPOINT_NORMALIZATION_VERSION,
    }
    if complete:
        versions["source_column_contract_version"] = SOURCE_COLUMN_CONTRACT_VERSION
        versions["source_column_policy_version"] = SOURCE_COLUMN_POLICY_VERSION
    return versions


def census_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalars = {source: 0 for source in EXPECTED_SOURCE_ROWS}
    encoded: dict[str, int] = {}
    for record in records:
        if record.get("finite_scalar_value") is None:
            continue
        source = str(record.get("source_id") or "")
        scalars[source] = scalars.get(source, 0) + 1
        encoder = str(record.get("categorical_encoder_id") or "")
        if encoder:
            encoded[encoder] = encoded.get(encoder, 0) + 1
    return {
        "n_finite_scalars_by_source": dict(sorted(scalars.items())),
        "n_categorically_encoded_by_encoder": dict(sorted(encoded.items())),
    }


def _enrich_record(record: dict[str, Any], attacher: Any) -> dict[str, Any]:
    source_projection = llm_source_projection(record)
    auxiliary = attacher.attach(record)
    # A parseable numeric value with an unresolved unit remains numeric evidence
    # and must not be overwritten by a categorical anchor.
    parsed_source = parse_point_measurement(record.get("canonical_measurement"))
    encoded = (
        {}
        if parsed_source.value is not None
        else CATEGORICAL_RESPONSE_POLICY.apply({**record, **auxiliary})
    )
    if encoded:
        encoder_id = str(encoded["categorical_encoder_id"])
        semantic_endpoint = {
            "bbb_permeability_binary.v1": "bbb_permeability_outcome",
            "passive_bbb_interpretation_binary.v1": "passive_bbb_permeability_outcome",
            "efflux_substrate_binary.v1": "efflux_substrate_outcome",
            "efflux_inhibitor_binary.v1": "efflux_inhibition_outcome",
        }[encoder_id]
        encoded = {**encoded, "canonical_endpoint": semantic_endpoint}
    endpoint_fields = endpoint_context_fields({**record, **encoded})
    if encoded:
        endpoint_fields.update(
            {
                "canonical_endpoint_source_field": {
                    "bbb_permeability_binary.v1": "bbb_permeability_label",
                    "passive_bbb_interpretation_binary.v1": "passive_bbb_interpretation",
                    "efflux_substrate_binary.v1": "interaction_conclusion",
                    "efflux_inhibitor_binary.v1": "interaction_conclusion",
                }[encoder_id],
                "canonical_endpoint_policy_status": "categorical_encoded",
                "canonical_endpoint_rule_id": encoder_id,
                "canonical_endpoint_policy_version": CATEGORICAL_RESPONSE_VERSION,
            }
        )
    unit_fields = unit_provenance_fields(record, encoded=encoded)
    validity = enrich_bbb_validity(
        {**record, **auxiliary, **encoded, **endpoint_fields, **unit_fields}
    )
    return {
        "source_column_contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "source_column_policy_version": SOURCE_COLUMN_POLICY_VERSION,
        **encoded,
        "llm_source_contract_json": json.dumps(
            {key: value for key, value in source_projection.items() if key != "source_fields"},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "llm_source_fields_json": json.dumps(
            source_projection["source_fields"], ensure_ascii=False, sort_keys=True
        ),
        **auxiliary,
        **endpoint_fields,
        **unit_fields,
        **validity,
    }


POLICY = StarlingTaskPolicy(
    task_id=TASK_ID,
    dataset_name=DATASET_NAME,
    default_data_dir=DEFAULT_STARLING_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    compact=COMPACT_PROFILE,
    expected_source_rows=EXPECTED_SOURCE_ROWS,
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
)


__all__ = [
    "DEFAULT_BENCHMARK_SPLIT_ROOT",
    "DEFAULT_OUT_DIR",
    "POLICY",
    "TASK_ID",
]
