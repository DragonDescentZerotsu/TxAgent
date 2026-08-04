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
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from tools.chembl_tool.tasks.skin_reaction.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    encoding_policy_manifest,
)
from tools.chembl_tool.tasks.skin_reaction.starling_categorical_response import (
    POLICY as CATEGORICAL_RESPONSE_POLICY,
)
from tools.chembl_tool.tasks.skin_reaction.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    DEFAULT_MAPPING_PATH,
    AuxiliaryMetadataAttacher,
    PendingAuxiliaryAttacher,
)
from tools.chembl_tool.tasks.skin_reaction.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from tools.chembl_tool.tasks.skin_reaction.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    endpoint_specific_standardization_of_unit,
)
from tools.chembl_tool.tasks.skin_reaction.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    source_profiles,
    validate_source_digest,
)
from tools.chembl_tool.tasks.skin_reaction.starling_record_canonicalization import (
    NORMALIZATION_DOMAIN_RULES_VERSION,
    enrich_skin_reaction_validity,
    validity_policy_manifest,
)
from tools.chembl_tool.tasks.skin_reaction.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
)
from tools.chembl_tool.tasks.skin_reaction.starling_spacing_and_spelling import (
    SPACING_AND_SPELLING_VERSION,
    family_assignment,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)


TASK_ID = "skin_reaction"
DATASET_NAME = "starling-labs/Skin_Reaction"
DEFAULT_STARLING_DATA_DIR = "data/starling_data/skin_reaction"
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v6"
)
DEFAULT_BENCHMARK_SPLIT_ROOT = "data/processed_starling/Skin_Reaction"


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
        "--benchmark-split-root",
        default=DEFAULT_BENCHMARK_SPLIT_ROOT,
        help=(
            "Directory containing random/scaffold train and heldout molecule "
            "label mappings used by the post-record filtering stage."
        ),
    )


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not Path(args.auxiliary_mapping).exists() and not args.allow_missing_auxiliary_mapping:
        parser.error(
            f"reconciled auxiliary mapping not found: {args.auxiliary_mapping}. "
            "Build it with tasks.skin_reaction.data_processing, or pass "
            "--allow-missing-auxiliary-mapping for an explicitly incomplete "
            "stage 01-05 build."
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
    return NormalizationHooks(
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        family_resolver=family_assignment,
        record_enricher=lambda record: _enrich_record(record, attacher),
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
    attacher = hooks.run_state
    coverage = attacher.coverage_audit(normalized)
    return StageDocuments(
        validity_policy={
            **validity_policy_manifest(),
            "categorical_response": encoding_policy_manifest(),
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
            "policy_independent_validity_present": True,
            "globally_reconciled_auxiliary_coverage": bool(
                coverage["validations"]["all_applicable_records_mapped"]
            ),
            "contextual_unit_policy_loaded": True,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
            "source_column_contract_complete": True,
            "llm_source_projection_fail_closed": True,
        },
    )


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
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


def _enrich_record(record: dict[str, Any], attacher: Any) -> dict[str, Any]:
    source_projection = llm_source_projection(record)
    auxiliary = attacher.attach(record)
    # Categorical sources report an outcome, not a measurement, so they carry no
    # scalar and never reach a pair bucket.  The encoder places the informative
    # subset on a named latent scale; it only ever fills a record that has no
    # scalar of its own, so a real measurement is never overwritten.
    encoded = CATEGORICAL_RESPONSE_POLICY.apply({**record, **auxiliary})
    validity = enrich_skin_reaction_validity({**record, **auxiliary, **encoded})
    return {
        "source_column_contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
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


__all__ = ["POLICY", "TASK_ID"]
