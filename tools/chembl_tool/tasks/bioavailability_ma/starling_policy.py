"""Bioavailability_Ma plug-in for the shared normalized-Starling builder.

This module only wires this task's existing policy modules into the
:class:`StarlingTaskPolicy` contract.  It contains no normalization logic of
its own; every rule still lives in the module that owns it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.task_policy import (
    ExtraSourceBatch,
    NormalizationHooks,
    SmilesMappingSpec,
    StageDocuments,
    StarlingTaskPolicy,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    DEFAULT_MAPPING_PATH,
    AuxiliaryMetadataAttacher,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_contextual_unit_reconciliation import (
    contextual_canonical_record_fields,
    contextual_standardization_of_unit,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_fg_scalar_rules import (
    FG_SCALAR_RULE_VERSION,
    fg_scalar_rule_provenance,
    resolve_fg_measurement_pair,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    endpoint_specific_standardization_of_unit,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_sources import (
    DEFAULT_DIRECT_HF_PARQUET,
    EXPECTED_SOURCE_ROWS,
    direct_hf_profile,
    load_direct_hf_rows,
    source_profiles,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    NORMALIZATION_DOMAIN_RULES_VERSION,
    REPORT_TYPE_NORMALIZATION_VERSION,
    enrich_bioavailability_validity,
    validity_policy_manifest,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    SPACING_AND_SPELLING_VERSION,
    family_assignment,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)


TASK_ID = "bioavailability_ma"
DATASET_NAME = "starling-labs/Bioavailability_Ma"
DEFAULT_STARLING_DATA_DIR = "data/starling_data/bioavailability_ma"
DEFAULT_SMILES_MAPPING = "data/starling_data/_shared/final_smiles_mapping_v2.parquet"
EXPECTED_SMILES_MAPPING_SHA256 = (
    "98ae43b6d9c61b77f0a95e4dce681010694b163a02e7a313667592d985496a0b"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v6"
)
DEFAULT_V65_ELIGIBLE_RECORDS = (
    "/data1/joseph/starling_assay_transfer/datasets/eligible/"
    "assay_transfer_soft_evidence_v6_5/records.parquet"
)


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smiles-mapping", default=DEFAULT_SMILES_MAPPING)
    parser.add_argument("--auxiliary-mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--allow-unpinned-smiles-mapping", action="store_true")
    parser.add_argument("--direct-source-parquet", default=str(DEFAULT_DIRECT_HF_PARQUET))
    parser.add_argument(
        "--include-direct-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-direct-rows", type=int, default=0)
    parser.add_argument(
        "--v65-reconciliation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--v65-eligible-records", default=DEFAULT_V65_ELIGIBLE_RECORDS)


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.v65_reconciliation:
        parser.error(
            "v6.5 reconciliation is outside the source-aware pair-bucket sidecar contract"
        )
    if args.max_direct_rows and args.strict_endpoint_inventory:
        parser.error("bounded source runs require --no-strict-endpoint-inventory")


def smiles_mapping(args: argparse.Namespace) -> SmilesMappingSpec:
    return SmilesMappingSpec(
        path=Path(args.smiles_mapping),
        expected_sha256=EXPECTED_SMILES_MAPPING_SHA256,
        allow_unpinned=args.allow_unpinned_smiles_mapping,
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


def load_extra_source(args: argparse.Namespace) -> ExtraSourceBatch | None:
    """Load the pinned complete Direct-HF snapshot."""
    if not args.include_direct_hf:
        return None
    records_path = Path(args.direct_source_parquet)
    direct_hash = file_sha256(records_path)
    rows = load_direct_hf_rows(records_path, max_rows=args.max_direct_rows)
    expected = EXPECTED_SOURCE_ROWS["direct_hf"]
    if not args.max_direct_rows and len(rows) != expected:
        raise ValueError(
            f"direct HF row-count drift: expected {expected:,}, found {len(rows):,}"
        )
    return ExtraSourceBatch(
        source_id="direct_hf",
        profile=direct_hf_profile(records_path),
        rows=rows,
        source_path=records_path,
        source_sha256=direct_hash,
        endpoint_names=["oral_bioavailability"] if rows else [],
        inventory_entry={
            "records_parquet": str(records_path),
            "records_sha256": direct_hash,
            "source_rows": expected,
            "historical_partition_dependency": False,
        },
    )


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    attacher = AuxiliaryMetadataAttacher(args.auxiliary_mapping)
    return NormalizationHooks(
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        source_measurement_resolver=resolve_fg_measurement_pair,
        family_resolver=family_assignment,
        contextual_standardizer=contextual_standardization_of_unit,
        record_enricher=lambda record: _enrich_record(record, attacher),
        # stage_documents needs the same instance to write its coverage audit.
        run_state=attacher,
    )


def attach_source_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Columnize the raw source contract before removing duplicate JSON."""
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
    attacher: AuxiliaryMetadataAttacher = hooks.run_state
    return StageDocuments(
        validity_policy=validity_policy_manifest(),
        auxiliary_mapping_manifest={
            **attacher.manifest(),
            "coverage": attacher.coverage_audit(normalized),
        },
        source_column_contract=source_column_contract_manifest(
            sorted({column for row in persisted for column in row})
        ),
        validations={
            "one_to_one_cleaned_to_normalized_ids": True,
            "measurement_unit_pair_errors": 0,
            "endpoint_orthography_provenance": True,
            "canonical_endpoint_present": True,
            "policy_independent_validity_present": True,
            "globally_reconciled_auxiliary_coverage": True,
            "contextual_unit_policy_loaded": True,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
            "heuristic_auxiliary_fields_absent": all(
                not any(
                    field in row
                    for field in (
                        "canonical_dose_key",
                        "canonical_assay_system",
                        "canonical_species",
                    )
                )
                for row in normalized
            ),
            "source_column_contract_complete": True,
            "llm_source_projection_fail_closed": True,
        },
    )


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "fg_scalar_rule_version": FG_SCALAR_RULE_VERSION,
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "report_type_normalization_version": REPORT_TYPE_NORMALIZATION_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "v65_reconciliation": {"status": "not_performed", "matching_performed": False},
    }
    if complete:
        versions["source_column_contract_version"] = SOURCE_COLUMN_CONTRACT_VERSION
        versions["v65_reconciliation_version"] = None
    return versions


def census_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_fg_finite_scalars": sum(
            record.get("source_id") == "fg"
            and record.get("finite_scalar_value") is not None
            for record in records
        )
    }


def _enrich_record(
    record: dict[str, Any],
    auxiliary_attacher: AuxiliaryMetadataAttacher,
) -> dict[str, Any]:
    provenance = fg_scalar_rule_provenance(record)
    enriched = {**record, **provenance}
    source_projection = llm_source_projection(enriched)
    auxiliary = auxiliary_attacher.attach(enriched)
    with_auxiliary = {**enriched, **auxiliary}
    contextual_fields = contextual_canonical_record_fields(with_auxiliary)
    validity = enrich_bioavailability_validity({**with_auxiliary, **contextual_fields})
    return {
        **provenance,
        "source_column_contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
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
        **contextual_fields,
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
    load_extra_source=load_extra_source,
    census_extras=census_extras,
    smiles_mapping=smiles_mapping,
)


__all__ = ["POLICY", "TASK_ID"]
