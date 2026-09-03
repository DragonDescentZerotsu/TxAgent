"""Staged builder for the first two normalized-Starling core stages.

Stages are ``source -> clean -> normalize``. Pair buckets and transfer
calibration are built separately as the split-independent Stage 3. Each stage
writes into a temporary directory and is published only after its outputs exist
and its validations pass, so a failed computation leaves the previous coherent
build in place.  Resuming a stage re-verifies both the stage artifact hash and
the recorded immediate-upstream input hash, so a self-consistent but stale
downstream stage cannot be resumed.

Everything task-specific arrives through a
:class:`~data.processing.evidence_library.shared.v1.normalization.task_policy.StarlingTaskPolicy`.
Run it for a task with ``--task``, or through the task's own thin wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import pandas as pd

from data.processing.paths import assert_canonical_library_output
from tools.chembl_tool.common.export import ensure_dir
from data.processing.evidence_library.compact_artifacts import (
    assert_compact_schema,
    build_relational_evidence_catalog,
    compact_persisted_record,
    compact_persisted_records,
    write_compact_neighbor_index,
)
from data.processing.evidence_library.shared.v1.canonicalization_v7 import (
    CANONICAL_ARTIFACT_VERSION,
    CANONICAL_RECORD_VERSION,
)
from data.processing.evidence_library.shared.v1.build_runtime import (
    assert_unpublished_build_root,
    FileDigestCache,
    build_cache_metadata,
    cache_metadata_matches,
    clean_sources_ordered,
    normalize_and_project_records_ordered,
    normalize_records_ordered,
    starling_build_session,
)
from data.processing.evidence_library.shared.v1.normalization.audit import (
    read_parquet_records,
    scalar_distribution_audit,
    stage_manifest,
    validate_cleaned_normalized_identity_ids,
    validate_measurement_pairs,
    validate_stage_schema,
    write_parquet,
)
from data.processing.evidence_library.shared.v1.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    RESOLUTION_APPLY_VERSION,
    apply_measurement_resolution,
)
from data.processing.evidence_library.versions.v8.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    file_sha256,
    normalize_endpoint_name,
)
from data.processing.evidence_library.shared.v1.normalization.contracts import (
    CLEANING_STAGE_VERSION,
    NORMALIZATION_STAGE_VERSION,
    NORMALIZED_ARTIFACT_VERSION,
    NORMALIZED_RECORD_VERSION,
    ORGANIZATION_STAGE_VERSION,
    SCALAR_PARSER_VERSION,
    SOURCE_STAGE_VERSION,
)
from data.processing.evidence_library.shared.v1.normalization.organization import (
    is_absolute_continuous,
    organize_normalized_records,
)
from data.processing.evidence_library.shared.v1.normalization.task_policy import (
    ExtraSourceBatch,
    StarlingTaskPolicy,
)
from data.processing.evidence_library.versions.v8.task_registry import import_task_module
from tools.chembl_tool.common.task_workflows.evidence_library import fingerprint_metadata
from tools.chembl_tool.common.units import (
    UNIT_NORMALIZER_VERSION,
    contextual_unit_policy_manifest,
    qualifier_vocabulary_manifest,
)


SOURCE_INVENTORY_FILENAME = "00_source/source_inventory.json"
ENDPOINT_INVENTORY_FILENAME = "00_source/endpoint_inventory.json"
CLEANED_FILENAME = "01_cleaned/records.parquet"
NORMALIZED_RECORDS_FILENAME = "02_normalized/records.parquet"
CANONICALIZED_RECORDS_FILENAME = "02_canonicalized/records.parquet"
RECORDS_FILENAME = "03_records/records.parquet"
REJECTIONS_FILENAME = "03_records/exclusions.parquet"
DUPLICATES_FILENAME = "03_records/duplicates.parquet"
DISTRIBUTION_AUDIT_FILENAME = "03_records/scalar_distribution.parquet"
ENDPOINT_REGISTRY_FILENAME = "02_normalized/endpoint_registry.json"
ENDPOINT_UNIT_PROFILE_FILENAME = "01_cleaned/endpoint_unit_profile.json"
SOURCE_VALUE_CLEANING_AUDIT_FILENAME = (
    "01_cleaned/source_value_cleaning_audit.parquet"
)
SOURCE_VALUE_CLEANING_MANIFEST_FILENAME = (
    "01_cleaned/source_value_cleaning_manifest.json"
)
STRUCTURE_REJECTIONS_FILENAME = "01_cleaned/structure_rejections.parquet"
MANIFEST_FILENAME = "manifest.json"
EVIDENCE_FAMILIES_FILENAME = "04_evidence_catalog/molecule_families.parquet"
EVIDENCE_BRIDGE_FILENAME = "04_evidence_catalog/molecule_family_records.parquet"
EVIDENCE_MANIFEST_FILENAME = "04_evidence_catalog/manifest.json"
INDEX_MOLECULES_FILENAME = "05_neighbor_index/molecules.parquet"
INDEX_FINGERPRINTS_FILENAME = "05_neighbor_index/fingerprints.npz"
INDEX_MEMBERSHIP_FILENAME = "05_neighbor_index/group_membership.parquet"
INDEX_META_FILENAME = "05_neighbor_index/manifest.json"
VALIDITY_POLICY_FILENAME = "02_normalized/record_validity_policy.json"
AUXILIARY_MAPPING_MANIFEST_FILENAME = "02_normalized/auxiliary_mapping_manifest.json"
SOURCE_COLUMN_CONTRACT_FILENAME = "02_normalized/source_contract.json"


def _measurement_resolution_mapping(args: argparse.Namespace) -> Path | None:
    """The frozen extraction to apply, or None for a pre-generation build."""
    path = getattr(args, "measurement_resolution_mapping", None)
    return Path(path) if path else None


def _exact_unit_mapping(policy: StarlingTaskPolicy) -> Path | None:
    path = policy.exact_unit_mapping_path
    if policy.measurement_resolution_enabled and path is None:
        raise ValueError(
            f"{policy.task_id} enables measurement resolution without an "
            "exact unit mapping"
        )
    return Path(path) if path is not None else None


def _measurement_resolution_manifest(
    policy: StarlingTaskPolicy, args: argparse.Namespace
) -> dict[str, Any]:
    mapping = _measurement_resolution_mapping(args)
    enabled = policy.measurement_resolution_enabled
    routing_enabled = enabled or policy.stage1_measurement_routing_enabled
    unit_mapping = _exact_unit_mapping(policy) if enabled else None
    return {
        "enabled": enabled,
        "active": enabled,
        "stage1_routing_enabled": routing_enabled,
        "frozen_extraction_loaded": mapping is not None,
        "apply_version": RESOLUTION_APPLY_VERSION,
        "exact_unit_mapping_version": EXACT_UNIT_MAPPING_VERSION,
        "exact_unit_mapping_path": str(unit_mapping) if unit_mapping else None,
        "exact_unit_mapping_sha256": (
            file_sha256(unit_mapping) if enabled and unit_mapping else None
        ),
        "legacy_parser_active": not enabled,
    }


def _scientific_assets(
    policy: StarlingTaskPolicy, args: argparse.Namespace
) -> tuple[Path, ...]:
    assets = [Path(path) for path in policy.scientific_assets]
    if policy.measurement_resolution_enabled:
        unit_mapping = _exact_unit_mapping(policy)
        if unit_mapping not in assets:
            assets.append(unit_mapping)
        mapping = _measurement_resolution_mapping(args)
        if mapping is not None:
            assets.append(mapping)
    return tuple(assets)


def _stage_paths(policy: StarlingTaskPolicy | None) -> dict[str, str]:
    """Return the frozen v6 or strict v7 persisted stage paths."""
    canonical_dir = (
        "02_canonicalized"
        if policy is not None and policy.record_contract
        else "02_normalized"
    )
    return {
        "normalized_records": f"{canonical_dir}/records.parquet",
        "endpoint_registry": f"{canonical_dir}/endpoint_registry.json",
        "validity_policy": f"{canonical_dir}/record_validity_policy.json",
        "auxiliary_manifest": f"{canonical_dir}/auxiliary_mapping_manifest.json",
        "reference_semantics_manifest": (
            f"{canonical_dir}/reference_semantics_manifest.json"
        ),
        "source_contract": f"{canonical_dir}/source_contract.json",
        "normalize_manifest": f"{canonical_dir}/manifest.json",
    }


def _stage_artifacts(
    policy: StarlingTaskPolicy | None,
) -> dict[str, tuple[str, str, str]]:
    paths = _stage_paths(policy)
    return {
        **STAGE_ARTIFACTS,
        "normalize": (
            paths["normalized_records"],
            paths["normalize_manifest"],
            NORMALIZATION_STAGE_VERSION,
        ),
    }


def _stage_output_filenames(
    policy: StarlingTaskPolicy | None,
) -> dict[str, tuple[str, ...]]:
    paths = _stage_paths(policy)
    clean_outputs = STAGE_OUTPUT_FILENAMES["clean"] + (
        (
            SOURCE_VALUE_CLEANING_AUDIT_FILENAME,
            SOURCE_VALUE_CLEANING_MANIFEST_FILENAME,
        )
        if policy is not None and policy.source_value_cleaner is not None
        else ()
    ) + (
        (ENDPOINT_UNIT_PROFILE_FILENAME,)
        if policy is not None
        and (
            policy.measurement_resolution_enabled
            or policy.stage1_measurement_routing_enabled
        )
        else ()
    ) + (
        (STRUCTURE_REJECTIONS_FILENAME,)
        if policy is not None and policy.record_contract is not None
        else ()
    )
    return {
        **STAGE_OUTPUT_FILENAMES,
        "clean": clean_outputs,
        "normalize": (
            paths["normalized_records"],
            paths["normalize_manifest"],
            paths["validity_policy"],
            paths["auxiliary_manifest"],
            paths["source_contract"],
            paths["endpoint_registry"],
        )
        + (
            (paths["reference_semantics_manifest"],)
            if policy is not None and policy.reference_semantics_enabled
            else ()
        ),
    }

CORE_STAGES = ("source", "clean", "normalize", "pair-buckets")
# Frozen callers still replay the historical organize/index stages. The task
# wrappers use CORE_STAGES and hand ``pair-buckets`` to the Stage-3 builder;
# this shared record builder itself never executes that stage.
STAGES = ("source", "clean", "normalize", "organize", "index")
STAGE_ARTIFACTS = {
    "source": (
        SOURCE_INVENTORY_FILENAME,
        "00_source/manifest.json",
        SOURCE_STAGE_VERSION,
    ),
    "clean": (CLEANED_FILENAME, "01_cleaned/manifest.json", CLEANING_STAGE_VERSION),
    "normalize": (
        NORMALIZED_RECORDS_FILENAME,
        "02_normalized/manifest.json",
        NORMALIZATION_STAGE_VERSION,
    ),
    "organize": (RECORDS_FILENAME, "03_records/manifest.json", ORGANIZATION_STAGE_VERSION),
}
STAGE_OUTPUT_FILENAMES = {
    "source": (
        SOURCE_INVENTORY_FILENAME,
        ENDPOINT_INVENTORY_FILENAME,
        "00_source/manifest.json",
    ),
    "clean": (
        CLEANED_FILENAME,
        "01_cleaned/manifest.json",
    ),
    "normalize": (
        NORMALIZED_RECORDS_FILENAME,
        "02_normalized/manifest.json",
        VALIDITY_POLICY_FILENAME,
        AUXILIARY_MAPPING_MANIFEST_FILENAME,
        SOURCE_COLUMN_CONTRACT_FILENAME,
        ENDPOINT_REGISTRY_FILENAME,
    ),
    "organize": (
        RECORDS_FILENAME,
        "03_records/manifest.json",
        DUPLICATES_FILENAME,
        REJECTIONS_FILENAME,
        DISTRIBUTION_AUDIT_FILENAME,
    ),
    "index": (
        EVIDENCE_FAMILIES_FILENAME,
        EVIDENCE_BRIDGE_FILENAME,
        EVIDENCE_MANIFEST_FILENAME,
        INDEX_MOLECULES_FILENAME,
        INDEX_FINGERPRINTS_FILENAME,
        INDEX_MEMBERSHIP_FILENAME,
        INDEX_META_FILENAME,
        MANIFEST_FILENAME,
    ),
}
STAGE_UPSTREAM_INPUTS = {
    "clean": ("source_inventory", SOURCE_INVENTORY_FILENAME),
    "normalize": ("cleaned_records", CLEANED_FILENAME),
    "organize": ("normalized_records", NORMALIZED_RECORDS_FILENAME),
}
RECORD_DEPENDENT_DIRECTORIES = (
    "03_pair_buckets",
    "04_pair_buckets",
    "05_deduplicated_records",
    "05_collapsed_records",
    "06_collapsed_records",
    "05_distance_calibration",
    "06_distance_calibration",
    "07_distance_calibration",
    "05_assay_transfer_policy",
    "06_remove_heldout_overlap",
    "07_molecule_evidence",
    "08_neighbor_index",
    "09_audits",
    "06_pair_buckets",
    "07_assay_transfer_policy",
    "07_endpoint_policies",
    "08_audits",
    "pair_buckets",
    "endpoint_policies",
    "analysis",
)
RECORD_DEPENDENT_FILES = (
    "v65_reconciliation.parquet",
    "v65_reconciliation_summary.json",
    "source_inventory.json",
    "endpoint_inventory.json",
    "01_cleaned/source_inventory.json",
    "01_cleaned/endpoint_inventory.json",
    "03_records/scalar_distribution_audit.parquet",
)
LEGACY_FLAT_ARTIFACTS = (
    "01_cleaned_records.parquet",
    "01_cleaning.manifest.json",
    "02_normalized_records.parquet",
    "02_normalization.manifest.json",
    "records.parquet",
    "03_organization.manifest.json",
    "duplicates.parquet",
    "organization_exclusions.parquet",
    "scalar_distribution_audit.parquet",
    "molecule_family_evidence.jsonl",
    "neighbor_index.pkl",
    "neighbor_index.meta.json",
    "endpoint_metric_registry.json",
)


def load_task_policy(task_id: str) -> StarlingTaskPolicy:
    """Resolve the policy owned by this construction release."""
    module = import_task_module(task_id, "starling_policy")
    policy = getattr(module, "POLICY", None)
    if not isinstance(policy, StarlingTaskPolicy):
        raise TypeError(
            f"tasks.{task_id}.starling_policy must publish POLICY as a StarlingTaskPolicy"
        )
    return policy


def build(policy: StarlingTaskPolicy, argv: list[str] | None = None) -> int:
    return run_with_args(policy, parse_args(policy, argv))


def run_with_args(policy: StarlingTaskPolicy, args: argparse.Namespace) -> int:
    """Run an already-validated argument namespace through the shared stages."""
    assert_canonical_library_output(args.out_dir)
    assert_unpublished_build_root(args.out_dir)
    with starling_build_session(args.out_dir):
        return _run(policy, args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--task", required=True)
    known, rest = parser.parse_known_args(argv)
    return build(load_task_policy(known.task), rest)


def _run(policy: StarlingTaskPolicy, args: argparse.Namespace) -> int:
    started = time.monotonic()
    unit_policy_manifest = contextual_unit_policy_manifest()
    qualifier_manifest = qualifier_vocabulary_manifest()
    out_dir = Path(ensure_dir(args.out_dir))
    first_stage = STAGES.index(args.from_stage)
    final_stage = STAGES.index(args.through_stage)
    invalidated_artifacts: list[str] = []
    stage_paths = _stage_paths(policy)
    normalized_records_filename = stage_paths["normalized_records"]
    cleaned_persisted: list[dict[str, Any]] = []
    normalized_persisted: list[dict[str, Any]] = []
    structure_rejection_ids: set[str] = set()

    digests = FileDigestCache()
    active_manifest = _read_optional_json(out_dir / MANIFEST_FILENAME)
    cached = active_manifest.get("record_build_cache")
    cached_stage = str((cached or {}).get("completed_stage") or "")
    if (
        args.cache_mode == "auto"
        and args.validation_level == "strict"
        and first_stage == STAGES.index("source")
        and cached_stage in STAGES
        and cached_stage == args.through_stage
        and cache_metadata_matches(
            cached,
            task_id=policy.task_id,
            completed_stage=cached_stage,
            args=args,
            digests=digests,
            scientific_assets=_scientific_assets(policy, args),
        )
    ):
        _log(
            f"cache hit through {cached_stage}: content_key={cached['content_key']}"
        )
        return 0

    source_inventory = _read_optional_json(out_dir / SOURCE_INVENTORY_FILENAME)
    endpoint_inventories = _read_optional_json(out_dir / ENDPOINT_INVENTORY_FILENAME)

    prepared_sources = None
    if first_stage <= STAGES.index("source"):
        prepared_sources = _prepare_source_inputs(policy, args)
        _, source_inventory, endpoint_inventories, source_inputs = prepared_sources
        with _temporary_stage_directory(out_dir, "source") as stage_dir:
            _write_json(stage_dir / SOURCE_INVENTORY_FILENAME, source_inventory)
            _write_json(
                stage_dir / ENDPOINT_INVENTORY_FILENAME, endpoint_inventories
            )
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "source",
                policy=policy,
                inputs=source_inputs,
                output_filename=SOURCE_INVENTORY_FILENAME,
                row_counts={
                    "source_records": sum(
                        source_inventory["source_row_counts"].values()
                    )
                },
                validations={
                    "source_hashes_recorded": True,
                    "source_row_counts_verified": True,
                    "source_rows_immutable": True,
                },
                sidecars={
                    "endpoint_inventory": stage_dir / ENDPOINT_INVENTORY_FILENAME
                },
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "source", stage_dir, policy)
            )
        _log(
            "source: records="
            f"{sum(source_inventory['source_row_counts'].values()):,}"
        )
    else:
        _verify_stage_artifact(out_dir, "source", policy)

    if final_stage == STAGES.index("source"):
        return _finish_partial(
            policy,
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= STAGES.index("clean"):
        if prepared_sources is None:
            prepared_sources = _prepare_source_inputs(policy, args)
            _, current_inventory, current_endpoints, _ = prepared_sources
            if (
                current_inventory != source_inventory
                or current_endpoints != endpoint_inventories
            ):
                raise ValueError("current sources differ from the frozen Stage-00 snapshot")
        (
            cleaned,
            cleaning_inputs,
            source_value_cleaning_audit,
            source_value_cleaning_manifest,
        ) = _clean_prepared_sources(policy, args, prepared_sources)
        structure_rejections: list[dict[str, Any]] = []
        if policy.record_contract:
            cleaned, structure_rejections = _retain_valid_structures(cleaned)
            structure_rejection_ids = {
                str(row.get("cleaned_record_id") or "")
                for row in structure_rejections
            }
        _require_valid_schema(cleaned, "clean")
        if (
            policy.measurement_resolution_enabled
            or policy.stage1_measurement_routing_enabled
        ):
            from data.processing.evidence_library.versions.v8.measurement_routing import (
                STAGE1_ROUTE_BUCKETS,
            )

            invalid_routes = [
                row.get("measurement_resolution_route")
                for row in cleaned
                if row.get("measurement_resolution_route") not in STAGE1_ROUTE_BUCKETS
                or not row.get("canonical_endpoint_name")
            ]
            if invalid_routes:
                raise ValueError(
                    "Stage 01 has invalid measurement routing; "
                    f"first={invalid_routes[0]!r}"
                )
        cleaned_attached = policy.attach_source_columns(cleaned)
        if policy.record_contract:
            for index, row in enumerate(cleaned_attached):
                cleaned_attached[index] = compact_persisted_record(
                    policy.record_contract.clean_projection(row)
                )
            cleaned_persisted = cleaned_attached
        else:
            cleaned_persisted = compact_persisted_records(cleaned_attached)
        assert_compact_schema(cleaned_persisted)
        with _temporary_stage_directory(out_dir, "clean") as stage_dir:
            write_parquet(stage_dir / CLEANED_FILENAME, cleaned_persisted)
            sidecars: dict[str, Path] = {}
            if policy.source_value_cleaner is not None:
                write_parquet(
                    stage_dir / SOURCE_VALUE_CLEANING_AUDIT_FILENAME,
                    source_value_cleaning_audit,
                )
                sidecars["source_value_cleaning_audit"] = (
                    stage_dir / SOURCE_VALUE_CLEANING_AUDIT_FILENAME
                )
                _write_json(
                    stage_dir / SOURCE_VALUE_CLEANING_MANIFEST_FILENAME,
                    source_value_cleaning_manifest,
                )
                sidecars["source_value_cleaning_manifest"] = (
                    stage_dir / SOURCE_VALUE_CLEANING_MANIFEST_FILENAME
                )
            if policy.record_contract:
                write_parquet(
                    stage_dir / STRUCTURE_REJECTIONS_FILENAME,
                    compact_persisted_records(structure_rejections),
                )
                sidecars["structure_rejections"] = (
                    stage_dir / STRUCTURE_REJECTIONS_FILENAME
                )
            if (
                policy.measurement_resolution_enabled
                or policy.stage1_measurement_routing_enabled
            ):
                sidecars["endpoint_unit_profile"] = _write_endpoint_unit_profile(
                    stage_dir / CLEANED_FILENAME,
                    stage_dir / ENDPOINT_UNIT_PROFILE_FILENAME,
                    task=policy.task_id,
                )
            route_counts = count_by(cleaned, "measurement_resolution_route")
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "clean",
                policy=policy,
                inputs={
                    "source_inventory": out_dir / SOURCE_INVENTORY_FILENAME,
                    **cleaning_inputs,
                },
                output_filename=CLEANED_FILENAME,
                row_counts={
                    "cleaned_records": len(cleaned),
                    "structure_rejections": len(structure_rejections),
                },
                validations={
                    "active_records_have_rdkit_valid_smiles": all(
                        row.get("canonical_smiles")
                        and row.get("structure_status") == "resolved"
                        for row in cleaned
                    ),
                    "structure_rejections_preserved": True,
                    "source_value_cleaning_audited": (
                        policy.source_value_cleaner is not None
                    ),
                    "source_value_cleaning_retained_order_and_identity_preserved": True,
                    "reviewed_source_row_drops": source_value_cleaning_manifest.get(
                        "n_dropped_records", 0
                    ),
                    "measurement_routing_is_partition": (
                        not (
                            policy.measurement_resolution_enabled
                            or policy.stage1_measurement_routing_enabled
                        )
                        or sum(route_counts.values()) == len(cleaned)
                    ),
                    "measurement_route_counts": route_counts,
                },
                sidecars=sidecars,
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(
                    out_dir,
                    "clean",
                    stage_dir,
                    policy,
                    invalidate_downstream=not args.preserve_downstream_artifacts,
                )
            )
        _log(f"clean: records={len(cleaned):,}")
    else:
        cleaned_persisted = (
            _load_verified_stage(out_dir, "clean", policy)
            if first_stage == STAGES.index("normalize")
            else []
        )
        cleaned = cleaned_persisted
        if cleaned and policy.record_contract:
            cleaned = [policy.record_contract.inflate_cleaned(row) for row in cleaned]

    if final_stage == STAGES.index("clean"):
        return _finish_partial(
            policy,
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= STAGES.index("normalize"):
        if not cleaned:
            cleaned_persisted = _load_verified_stage(out_dir, "clean", policy)
            cleaned = cleaned_persisted
            if policy.record_contract:
                cleaned = [
                    policy.record_contract.inflate_cleaned(row) for row in cleaned
                ]
        if policy.record_contract and not structure_rejection_ids:
            structure_rejection_ids = {
                str(row.get("cleaned_record_id") or "")
                for row in read_parquet_records(
                    out_dir / STRUCTURE_REJECTIONS_FILENAME,
                    columns=["cleaned_record_id"],
                )
            }
        if policy.record_contract:
            _validate_smiles_mapping_unchanged(
                policy, args, source_inventory=source_inventory
            )
        cleaned_parent_count = len(cleaned)
        resolution_mapping = _measurement_resolution_mapping(args)
        measurement_resolution_audit = (
            apply_measurement_resolution(
                cleaned,
                mapping_path=resolution_mapping,
                task=policy.task_id,
                unit_mapping_path=_exact_unit_mapping(policy),
                allow_partial=args.allow_partial_measurement_resolution,
                ignored_record_ids=structure_rejection_ids,
                expected_routing_version=MEASUREMENT_ROUTING_VERSION,
            )
            if policy.measurement_resolution_enabled
            else None
        )
        hooks = policy.build_hooks(args)
        measurement_input_ids = [
            str(record.get("cleaned_record_id") or "") for record in cleaned
        ]
        if policy.record_contract:
            normalized, normalized_persisted = normalize_and_project_records_ordered(
                cleaned,
                hooks=hooks,
                policy=policy,
                workers=args.workers,
                release_input=args.workers == 1,
                retain_working=False,
            )
        else:
            normalized = normalize_records_ordered(
                cleaned,
                hooks=hooks,
                task=policy.task_id,
                workers=args.workers,
            )
        identity_errors = validate_cleaned_normalized_identity_ids(
            measurement_input_ids, normalized
        )
        if identity_errors:
            raise ValueError(
                f"{len(identity_errors)} cleaned/normalized identity failure(s); "
                f"first={identity_errors[0]}"
            )
        if args.validation_level == "full":
            pair_errors = validate_measurement_pairs(
                normalized,
                hooks.endpoint_standardizer,
                hooks.source_measurement_resolver,
                hooks.contextual_standardizer,
                task=policy.task_id,
            )
            if pair_errors:
                raise ValueError(
                    f"{len(pair_errors)} measurement/unit pair invariant failure(s); "
                    f"first={pair_errors[0]}"
                )
            if policy.assay_transfer_measurement_policy is not None:
                from data.processing.evidence_library.shared.v1.assay_transfer_measurements import (
                    validate_final_assay_transfer_measurements,
                )

                final_errors = validate_final_assay_transfer_measurements(normalized)
                if final_errors:
                    raise ValueError(
                        f"{len(final_errors)} final assay-transfer measurement "
                        f"failure(s); first={final_errors[0]}"
                    )
        if not policy.record_contract:
            _require_valid_schema(normalized, "normalize")
        if not policy.record_contract:
            normalized = policy.attach_source_columns(normalized)
            normalized_persisted = compact_persisted_records(normalized)
        assert_compact_schema(normalized_persisted)
        documents = policy.stage_documents(
            args=args,
            hooks=hooks,
            normalized=normalized,
            persisted=normalized_persisted,
            unit_policy_manifest=unit_policy_manifest,
        )
        with _temporary_stage_directory(out_dir, "normalize") as stage_dir:
            write_parquet(
                stage_dir / normalized_records_filename, normalized_persisted
            )
            _write_json(
                stage_dir / stage_paths["validity_policy"], documents.validity_policy
            )
            _write_json(
                stage_dir / stage_paths["auxiliary_manifest"],
                documents.auxiliary_mapping_manifest,
            )
            _write_json(
                stage_dir / stage_paths["source_contract"],
                documents.source_column_contract,
            )
            _write_json(
                stage_dir / stage_paths["endpoint_registry"],
                documents.endpoint_registry
                or {
                    source_id: inventory["endpoints"]
                    for source_id, inventory in endpoint_inventories.items()
                },
            )
            if policy.reference_semantics_enabled:
                if documents.reference_semantics_manifest is None:
                    raise ValueError(
                        "reference semantics are enabled but Stage 02 supplied no manifest"
                    )
                _write_json(
                    stage_dir / stage_paths["reference_semantics_manifest"],
                    documents.reference_semantics_manifest,
                )
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "normalize",
                policy=policy,
                inputs={
                    "cleaned_records": out_dir / CLEANED_FILENAME,
                    "contextual_unit_policy": unit_policy_manifest["path"],
                    "qualifier_vocabulary": qualifier_manifest["path"],
                    **(
                        {"auxiliary_mapping": Path(args.auxiliary_mapping)}
                        if getattr(args, "auxiliary_mapping", None)
                        else {}
                    ),
                    **(
                        {
                            "reference_semantics_mapping": Path(
                                args.reference_semantics_mapping
                            )
                        }
                        if getattr(args, "reference_semantics_mapping", None)
                        else {}
                    ),
                    **(
                        {"measurement_resolution_mapping": resolution_mapping}
                        if measurement_resolution_audit and resolution_mapping
                        else {}
                    ),
                    **(
                        {
                            "exact_measurement_unit_map": _exact_unit_mapping(
                                policy
                            )
                        }
                        if measurement_resolution_audit and resolution_mapping
                        else {}
                    ),
                    **(
                        {
                            "assay_transfer_measurement_policy": (
                                policy.assay_transfer_measurement_policy
                            )
                        }
                        if policy.assay_transfer_measurement_policy is not None
                        else {}
                    ),
                },
                output_filename=normalized_records_filename,
                row_counts={
                    "cleaned_records": cleaned_parent_count,
                    "measurement_input_records": len(cleaned),
                    "normalized_records": len(normalized),
                    "finite_scalars": sum(
                        row.get("finite_scalar_value") is not None
                        for row in normalized
                    ),
                    "absolute_and_continuous": sum(
                        is_absolute_continuous(row) for row in normalized
                    ),
                    "normalization_valid": sum(
                        (
                            row.get("normalization_validity_status")
                            or row.get("canonicalization_status")
                        )
                        == "valid"
                        for row in normalized
                    ),
                },
                validations={
                    **documents.validations,
                    "measurement_unit_pair_validation": (
                        "passed" if args.validation_level == "full" else "not_run"
                    ),
                    "final_assay_transfer_measurement_validation": (
                        "passed"
                        if args.validation_level == "full"
                        and policy.assay_transfer_measurement_policy is not None
                        else "not_applicable"
                        if policy.assay_transfer_measurement_policy is None
                        else "not_run"
                    ),
                    **(
                        {"measurement_resolution": measurement_resolution_audit}
                        if measurement_resolution_audit is not None
                        else {}
                    ),
                },
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "normalize", stage_dir, policy)
            )
        _log(
            "normalize: "
            f"records={len(normalized):,} "
            f"scalars={sum(row.get('finite_scalar_value') is not None for row in normalized):,}"
        )
        if final_stage > STAGES.index("normalize") and policy.record_contract:
            for index, row in enumerate(normalized):
                normalized[index] = policy.record_contract.inflate_canonical(row)
            normalized_persisted = []
    else:
        normalized_persisted = (
            _load_verified_stage(out_dir, "normalize", policy)
            if first_stage == STAGES.index("organize")
            else []
        )
        normalized = normalized_persisted
        if normalized and policy.record_contract:
            for index, row in enumerate(normalized):
                normalized[index] = policy.record_contract.inflate_canonical(row)
            normalized_persisted = []

    if final_stage == STAGES.index("normalize"):
        return _finish_partial(
            policy,
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= STAGES.index("organize"):
        if not normalized:
            normalized_persisted = _load_verified_stage(out_dir, "normalize", policy)
            normalized = normalized_persisted
            if policy.record_contract:
                for index, row in enumerate(normalized):
                    normalized[index] = policy.record_contract.inflate_canonical(row)
                normalized_persisted = []
        if args.frozen_retrieval_normalized_records:
            from data.processing.evidence_library.shared.v1.retrieval_boundary import (
                freeze_normalized_retrieval_identity,
            )

            normalized = freeze_normalized_retrieval_identity(
                normalized, args.frozen_retrieval_normalized_records
            )
        records, duplicates, exclusions, organization_stats = organize_normalized_records(
            normalized,
            endpoint_identity_required_sources=(
                policy.endpoint_identity_required_sources
            ),
            reuse_mutable_records=True,
        )
        _require_valid_schema(records, "organize")
        distribution_rows = scalar_distribution_audit(records)
        census_extras = policy.census_extras(records) if policy.census_extras else {}
        normalization_validity_status_counts = count_by(
            records, "normalization_validity_status"
        )
        if normalized is not records and isinstance(normalized, list):
            normalized.clear()
        if policy.record_contract:
            required_contract_fields = {
                field
                for bucket in policy.record_contract.pair_buckets.values()
                for field in (
                    *bucket.canonical_dimensions,
                    *bucket.core_context_dimensions,
                    *bucket.variance_candidates,
                )
            }
            for index, record in enumerate(records):
                projected = policy.record_contract.canonical_projection(record)
                if index == 0:
                    for field in required_contract_fields:
                        projected.setdefault(field, None)
                records[index] = compact_persisted_record(projected)
            records_persisted = records
        else:
            records_persisted = compact_persisted_records(records)
        if args.frozen_retrieval_records:
            from data.processing.evidence_library.shared.v1.retrieval_boundary import (
                freeze_retrieval_boundary,
            )

            records_persisted = freeze_retrieval_boundary(
                records_persisted, args.frozen_retrieval_records
            )
        records = records_persisted
        assert_compact_schema(records_persisted)
        with _temporary_stage_directory(out_dir, "organize") as stage_dir:
            write_parquet(stage_dir / RECORDS_FILENAME, records_persisted)
            write_parquet(stage_dir / DUPLICATES_FILENAME, duplicates)
            write_parquet(stage_dir / REJECTIONS_FILENAME, exclusions)
            write_parquet(stage_dir / DISTRIBUTION_AUDIT_FILENAME, distribution_rows)
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "organize",
                policy=policy,
                inputs={
                    "normalized_records": out_dir / normalized_records_filename,
                    **(
                        {"frozen_retrieval_records": args.frozen_retrieval_records}
                        if args.frozen_retrieval_records
                        else {}
                    ),
                    **(
                        {
                            "frozen_retrieval_normalized_records": (
                                args.frozen_retrieval_normalized_records
                            )
                        }
                        if args.frozen_retrieval_normalized_records
                        else {}
                    ),
                },
                output_filename=RECORDS_FILENAME,
                row_counts=organization_stats,
                validations={
                    "cross_source_deduplication": False,
                    "row_deduplication": False,
                    "row_cardinality_preserved": (
                        organization_stats["n_records"]
                        == organization_stats["n_normalized_records"]
                    ),
                    "unusable_structures_retained_in_records": True,
                },
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "organize", stage_dir, policy)
            )
        _log(
            f"organize: records={len(records):,} "
            f"retrieval_eligible={organization_stats['n_retrieval_eligible']:,}"
        )
    else:
        records = _load_verified_stage(out_dir, "organize", policy)
        if policy.record_contract:
            records = [
                policy.record_contract.inflate_canonical(row) for row in records
            ]
        duplicates = read_parquet_records(out_dir / DUPLICATES_FILENAME)
        organization_stats = {
            "n_normalized_records": len(records),
            "n_records": len(records),
            "n_duplicates_removed": len(duplicates),
            "row_deduplication_deferred": True,
            "n_retrieval_eligible": sum(
                bool(row.get("retrieval_eligible")) for row in records
            ),
            "n_organization_exclusions": sum(
                not bool(row.get("retrieval_eligible")) for row in records
            ),
            "n_missing_endpoint_identity": sum(
                row.get("organization_status") == "missing_endpoint_identity"
                for row in records
            ),
            "n_absolute_and_continuous": sum(
                is_absolute_continuous(row) for row in records
            ),
        }
        census_extras = policy.census_extras(records) if policy.census_extras else {}
        normalization_validity_status_counts = count_by(
            records, "normalization_validity_status"
        )

    frozen_census = {
        "n_cleaned_records": sum(
            int(value)
            for value in (source_inventory.get("source_row_counts") or {}).values()
        ),
        **organization_stats,
        **census_extras,
        "normalization_validity_status_counts": normalization_validity_status_counts,
    }

    if final_stage == STAGES.index("organize"):
        return _finish_partial(
            policy,
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    evidence_families, evidence_bridge = build_relational_evidence_catalog(
        records,
        max_record_examples=args.max_record_examples,
        family_resolver=policy.family_resolver,
    )
    compact_profile = policy.compact_profile_for_contract(
        policy.record_contract.version if policy.record_contract else ""
    )
    with _temporary_stage_directory(out_dir, "index") as stage_dir:
        write_parquet(stage_dir / EVIDENCE_FAMILIES_FILENAME, evidence_families)
        write_parquet(stage_dir / EVIDENCE_BRIDGE_FILENAME, evidence_bridge)
        index_manifest = write_compact_neighbor_index(
            profile=compact_profile,
            families=evidence_families,
            output_dir=stage_dir / "05_neighbor_index",
            workers=args.workers,
            progress_every=args.progress_every,
        )
        _write_json(
            stage_dir / EVIDENCE_MANIFEST_FILENAME,
            {
                "artifact_version": compact_profile.artifact_version,
                "families": len(evidence_families),
                "record_references": len(evidence_bridge),
                "files": {
                    "molecule_families.parquet": {
                        "sha256": file_sha256(stage_dir / EVIDENCE_FAMILIES_FILENAME)
                    },
                    "molecule_family_records.parquet": {
                        "sha256": file_sha256(stage_dir / EVIDENCE_BRIDGE_FILENAME)
                    },
                },
                "validations": {
                    "evidence_records_are_references_only": True,
                    "embedded_normalized_records": False,
                },
            },
        )

        artifact_filenames = {
            "cleaned_records": CLEANED_FILENAME,
            "normalized_records": normalized_records_filename,
            "records": RECORDS_FILENAME,
            "duplicates": DUPLICATES_FILENAME,
            "organization_exclusions": REJECTIONS_FILENAME,
            "molecule_families": EVIDENCE_FAMILIES_FILENAME,
            "molecule_family_records": EVIDENCE_BRIDGE_FILENAME,
            "scalar_distribution_audit": DISTRIBUTION_AUDIT_FILENAME,
            "record_validity_policy": stage_paths["validity_policy"],
            "auxiliary_mapping_manifest": stage_paths["auxiliary_manifest"],
            "source_column_contracts": stage_paths["source_contract"],
            "index_molecules": INDEX_MOLECULES_FILENAME,
            "index_fingerprints": INDEX_FINGERPRINTS_FILENAME,
            "index_membership": INDEX_MEMBERSHIP_FILENAME,
        }
        if policy.reference_semantics_enabled:
            artifact_filenames["reference_semantics_manifest"] = stage_paths[
                "reference_semantics_manifest"
            ]
        if policy.source_value_cleaner is not None:
            artifact_filenames["source_value_cleaning_audit"] = (
                SOURCE_VALUE_CLEANING_AUDIT_FILENAME
            )
        manifest = {
            "artifact_version": (
                CANONICAL_ARTIFACT_VERSION
                if policy.record_contract
                else NORMALIZED_ARTIFACT_VERSION
            ),
            "compact_artifact_version": compact_profile.artifact_version,
            "record_version": (
                CANONICAL_RECORD_VERSION
                if policy.record_contract
                else NORMALIZED_RECORD_VERSION
            ),
            "record_contract": (
                policy.record_contract.manifest()
                if policy.record_contract
                else None
            ),
            "scalar_parser_version": SCALAR_PARSER_VERSION,
            "unit_normalizer_version": UNIT_NORMALIZER_VERSION,
            "measurement_resolution": _measurement_resolution_manifest(policy, args),
            "contextual_unit_policy": unit_policy_manifest,
            "qualifier_vocabulary": qualifier_manifest,
            "index_version": compact_profile.index_version,
            **policy.manifest_versions(),
            "completed_stages": list(STAGES),
            "rebuild_request": {
                "from_stage": args.from_stage,
                "through_stage": args.through_stage,
            },
            "invalidated_artifacts": sorted(set(invalidated_artifacts)),
            "source_inventory": source_inventory,
            "endpoint_inventory": endpoint_inventories,
            "stats": {
                **frozen_census,
                "n_input_rows": sum(
                    int(value)
                    for value in (
                        source_inventory.get("source_row_counts") or {}
                    ).values()
                ),
                "n_molecule_family_evidence_rows": len(evidence_families),
                "n_index_molecules": index_manifest["molecules"],
            },
            "groups": index_manifest["groups"],
            "fingerprint": fingerprint_metadata(),
            "artifact_hashes": {
                name: file_sha256(
                    stage_dir / filename
                    if (stage_dir / filename).exists()
                    else out_dir / filename
                )
                for name, filename in artifact_filenames.items()
            },
            "record_build_cache": _record_build_cache(
                policy,
                out_dir,
                args,
                completed_stage="index",
                digests=digests,
                staged_stage_dir=stage_dir,
            ),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        _write_json(stage_dir / MANIFEST_FILENAME, manifest)
        invalidated_artifacts.extend(
            _commit_stage_outputs(out_dir, "index", stage_dir, policy)
        )
    _log(
        f"complete: records={len(records):,} "
        f"molecule-family rows={len(evidence_families):,} "
        f"index molecules={index_manifest['molecules']:,} out={out_dir}"
    )
    return 0


def _prepare_source_inputs(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
) -> tuple[
    list[tuple[Any, Any, str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    data_dir = Path(args.starling_data_dir)
    mapping_spec = policy.smiles_mapping(args) if policy.smiles_mapping else None
    inputs: dict[str, Path] = {}
    mapping_sha = ""
    if mapping_spec is not None:
        if not mapping_spec.path.exists():
            raise FileNotFoundError(
                f"authoritative SMILES mapping not found: {mapping_spec.path}"
            )
        mapping_sha = file_sha256(mapping_spec.path)
        if (
            mapping_spec.expected_sha256
            and not mapping_spec.allow_unpinned
            and mapping_sha != mapping_spec.expected_sha256
        ):
            raise ValueError(
                "SMILES mapping hash mismatch: "
                f"expected {mapping_spec.expected_sha256}, found {mapping_sha}"
            )
        inputs["smiles_mapping"] = mapping_spec.path

    profiles = list(policy.source_profiles(data_dir))
    frames: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    endpoint_inventories: dict[str, Any] = {}
    needed_identifiers: set[str] = set()
    for profile in profiles:
        path = Path(profile.source_path)
        inputs[profile.source_id] = path
        if policy.verify_source_digest is not None:
            policy.verify_source_digest(profile.source_id, path)
        source_hashes[profile.source_id] = file_sha256(path)
        frame = pd.read_parquet(path)
        expected_rows = policy.expected_source_rows[profile.source_id]
        if not args.max_rows_per_source and len(frame) != expected_rows:
            raise ValueError(
                f"source row-count drift for {profile.source_id}: "
                f"expected {expected_rows:,}, found {len(frame):,}"
            )
        if args.max_rows_per_source:
            frame = frame.head(args.max_rows_per_source)
        frames[profile.source_id] = frame
        if profile.structure_mode == "mapped":
            identity_field = "global_identifier"
            if identity_field in frame:
                needed_identifiers.update(
                    str(value)
                    for value in frame[identity_field].dropna().unique().tolist()
                    if str(value).strip()
                )
        endpoints = [
            normalize_endpoint_name(value) or ""
            for value in frame[profile.endpoint_field].drop_duplicates().tolist()
        ]
        endpoint_inventories[profile.source_id] = policy.endpoint_inventory(
            profile.source_id, endpoints, strict=args.strict_endpoint_inventory
        )

    smiles_mapping = (
        load_smiles_mapping(mapping_spec.path, needed_identifiers)
        if mapping_spec is not None
        else {}
    )
    clean_batches: list[tuple[Any, Any, str, Any]] = [
        (
            profile,
            frames[profile.source_id],
            source_hashes[profile.source_id],
            smiles_mapping or None,
        )
        for profile in profiles
    ]

    extra_entries: dict[str, dict[str, Any] | None] = {}
    if policy.load_extra_source is not None:
        loaded = policy.load_extra_source(args)
        extras = (
            []
            if loaded is None
            else [loaded]
            if isinstance(loaded, ExtraSourceBatch)
            else list(loaded)
        )
        source_ids = [extra.source_id for extra in extras]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("extra source batches contain duplicate source IDs")
        for extra in extras:
            inputs[extra.inventory_key] = extra.source_path
            extra_entries[extra.inventory_key] = extra.inventory_entry
            endpoint_inventories[extra.source_id] = policy.endpoint_inventory(
                extra.source_id,
                extra.endpoint_names,
                strict=args.strict_endpoint_inventory,
            )
            clean_batches.append(
                (extra.profile, extra.rows, extra.source_sha256, None)
            )
            source_hashes[extra.source_id] = extra.source_sha256

    source_inventory: dict[str, Any] = {
        "dataset": policy.dataset_name,
        "source_hashes": source_hashes,
        "source_row_counts": {
            batch[0].source_id: len(batch[1]) for batch in clean_batches
        },
        **extra_entries,
    }
    if mapping_spec is not None:
        source_inventory["smiles_mapping"] = {
            "path": str(mapping_spec.path),
            "sha256": mapping_sha,
            "expected_sha256": mapping_spec.expected_sha256,
            "n_loaded_identifiers": len(smiles_mapping),
        }
    return (
        clean_batches,
        source_inventory,
        endpoint_inventories,
        inputs,
    )


def _clean_prepared_sources(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    prepared: tuple[
        list[tuple[Any, Any, str, Any]],
        dict[str, Any],
        dict[str, Any],
        dict[str, Path],
    ],
) -> tuple[list[dict[str, Any]], dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    clean_batches, _, _, _ = prepared
    cleaned = clean_sources_ordered(clean_batches, workers=args.workers)
    if policy.source_value_cleaner is None:
        return cleaned, {}, [], {}
    before_ids = [str(row.get("cleaned_record_id") or "") for row in cleaned]
    result = policy.source_value_cleaner(cleaned, args)
    after_ids = [str(row.get("cleaned_record_id") or "") for row in result.records]
    dropped_ids = {
        str(row.get("cleaned_record_id") or "")
        for row in result.audit_rows
        if row.get("field") == "record" and row.get("after") == "dropped"
    }
    expected_after_ids = [row_id for row_id in before_ids if row_id not in dropped_ids]
    if expected_after_ids != after_ids:
        raise ValueError(
            "source-value cleaning changed retained Stage-01 row order or identity"
        )
    if len(before_ids) - len(after_ids) != len(dropped_ids):
        raise ValueError("source-value cleaning dropped a row without an audit entry")
    inputs = {
        f"source_value_cleaning_asset_{index}": path
        for index, path in enumerate(result.input_paths, start=1)
    }
    return result.records, inputs, result.audit_rows, result.manifest


def _retain_valid_structures(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate missing or RDKit-invalid structures without losing lineage."""
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        if record.get("canonical_smiles") and record.get("structure_status") == "resolved":
            retained.append(record)
            continue
        rejected.append(
            {
                **record,
                "stage1_rejection_reason": str(
                    record.get("structure_status") or "missing_or_invalid_structure"
                ),
            }
        )
    return retained, rejected


def load_smiles_mapping(path: Path, needed_identifiers: set[str]) -> dict[str, str]:
    frame = pd.read_parquet(path, columns=["global_identifier", "smiles"])
    if needed_identifiers:
        frame = frame[frame["global_identifier"].isin(needed_identifiers)]
    mapping: dict[str, str] = {}
    conflicts: set[str] = set()
    for identifier, smiles in frame[["global_identifier", "smiles"]].itertuples(index=False):
        key, value = str(identifier), str(smiles)
        existing = mapping.get(key)
        if existing is not None and existing != value:
            conflicts.add(key)
        else:
            mapping[key] = value
    if conflicts:
        raise ValueError(
            f"authoritative SMILES mapping has {len(conflicts)} conflicting identifier(s)"
        )
    return mapping


def _validate_smiles_mapping_unchanged(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    *,
    source_inventory: dict[str, Any],
) -> None:
    """Fail a Stage-02 resume if the mapping used by Stage 01 has drifted."""
    mapping_spec = policy.smiles_mapping(args) if policy.smiles_mapping else None
    if mapping_spec is None:
        return
    if not mapping_spec.path.is_file():
        raise FileNotFoundError(
            f"authoritative SMILES mapping not found: {mapping_spec.path}"
        )
    mapping_sha = file_sha256(mapping_spec.path)
    if (
        mapping_spec.expected_sha256
        and not mapping_spec.allow_unpinned
        and mapping_sha != mapping_spec.expected_sha256
    ):
        raise ValueError(
            "SMILES mapping hash mismatch during resume: "
            f"expected {mapping_spec.expected_sha256}, found {mapping_sha}"
        )
    inventory_sha = str(
        (source_inventory.get("smiles_mapping") or {}).get("sha256") or ""
    )
    if not inventory_sha:
        raise ValueError("clean-stage inventory lacks the SMILES mapping digest")
    if mapping_sha != inventory_sha:
        raise ValueError(
            "SMILES mapping changed since Stage 01: "
            f"expected {inventory_sha}, found {mapping_sha}"
        )


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_verified_stage(
    out_dir: Path, stage: str, policy: StarlingTaskPolicy | None = None
) -> list[dict[str, Any]]:
    return read_parquet_records(_verify_stage_artifact(out_dir, stage, policy))


def _verify_stage_artifact(
    out_dir: Path, stage: str, policy: StarlingTaskPolicy | None = None
) -> Path:
    filename, manifest_filename, version = _stage_artifacts(policy)[stage]
    artifact = out_dir / filename
    manifest_path = out_dir / manifest_filename
    if not artifact.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"cannot restart from {stage}: missing {artifact} or {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != version:
        raise ValueError(
            f"{stage} stage version mismatch: expected {version}, "
            f"found {manifest.get('version')}"
        )
    expected_hash = manifest.get("output", {}).get("sha256")
    actual_hash = file_sha256(artifact)
    if expected_hash != actual_hash:
        raise ValueError(
            f"{stage} stage artifact hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    upstream = STAGE_UPSTREAM_INPUTS.get(stage)
    if upstream is not None:
        input_name, upstream_filename = upstream
        if stage == "organize":
            upstream_filename = _stage_paths(policy)["normalized_records"]
        upstream_artifact = out_dir / upstream_filename
        expected_upstream_hash = (
            (manifest.get("inputs") or {}).get(input_name, {}).get("sha256")
        )
        if not upstream_artifact.exists() or not expected_upstream_hash:
            raise ValueError(
                f"{stage} stage manifest is missing a verifiable upstream "
                f"input: {input_name}"
            )
        actual_upstream_hash = file_sha256(upstream_artifact)
        if expected_upstream_hash != actual_upstream_hash:
            raise ValueError(
                f"{stage} stage upstream input hash mismatch for {input_name}: "
                f"expected {expected_upstream_hash}, found {actual_upstream_hash}"
            )
    if (
        stage == "normalize"
        and policy is not None
        and policy.assay_transfer_measurement_policy is not None
    ):
        expected_policy_hash = (
            (manifest.get("inputs") or {})
            .get("assay_transfer_measurement_policy", {})
            .get("sha256")
        )
        actual_policy_hash = file_sha256(policy.assay_transfer_measurement_policy)
        if expected_policy_hash != actual_policy_hash:
            raise ValueError(
                "normalize stage assay-transfer policy hash mismatch: "
                f"expected {expected_policy_hash}, found {actual_policy_hash}"
            )
    return artifact


def _write_endpoint_unit_profile(
    records_path: Path, output_path: Path, *, task: str
) -> Path:
    from data.processing.evidence_library.versions.v8.build_endpoint_unit_profile import (
        build_profile,
    )

    config = import_task_module(task, "starling_measurement_resolution")
    profile = build_profile(
        records_path,
        config.source_routing_rules(),
        task=task,
    )
    _write_json(output_path, profile)
    return output_path


def _write_staged_stage_manifest(
    stage_dir: Path,
    out_dir: Path,
    stage: str,
    *,
    policy: StarlingTaskPolicy,
    inputs: dict[str, Path],
    output_filename: str,
    row_counts: dict[str, int],
    validations: dict[str, Any],
    sidecars: Mapping[str, Path] | None = None,
) -> None:
    _, manifest_filename, version = _stage_artifacts(policy)[stage]
    payload = stage_manifest(
        stage=stage,
        version=version,
        inputs=inputs,
        output=stage_dir / output_filename,
        row_counts=row_counts,
        validations=validations,
    )
    payload["output"]["path"] = str(out_dir / output_filename)
    if sidecars:
        payload["sidecars"] = {
            name: {
                "path": str(out_dir / path.relative_to(stage_dir)),
                "sha256": file_sha256(path),
            }
            for name, path in sorted(sidecars.items())
        }
    _write_json(stage_dir / manifest_filename, payload)


@contextmanager
def _temporary_stage_directory(out_dir: Path, stage: str):
    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=f".{stage}-stage-"
    ) as directory:
        yield Path(directory)


def _commit_stage_outputs(
    out_dir: Path,
    stage: str,
    stage_dir: Path,
    policy: StarlingTaskPolicy,
    *,
    invalidate_downstream: bool = True,
) -> list[str]:
    expected_outputs = _stage_output_filenames(policy)[stage]
    missing = [
        filename
        for filename in expected_outputs
        if not (stage_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"cannot publish {stage} stage; missing staged outputs: {missing}"
        )

    invalidated = (
        _invalidate_downstream_artifacts(out_dir, stage, policy)
        if invalidate_downstream
        else []
    )
    if stage != "index":
        (out_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    for filename in expected_outputs:
        destination = out_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir / filename, destination)
    return invalidated


def _invalidate_downstream_artifacts(
    out_dir: Path, stage: str, policy: StarlingTaskPolicy | None = None
) -> list[str]:
    invalidated: list[str] = []
    if stage in {"clean", "normalize"}:
        legacy_root = out_dir / "historical/pre_three_stage_core"
        for name in ("03_records", "04_pair_buckets", "05_deduplicated_records"):
            active, archived = out_dir / name, legacy_root / name
            if active.exists() and not archived.exists():
                legacy_root.mkdir(parents=True, exist_ok=True)
                os.replace(active, archived)
                invalidated.append(
                    f"{name} -> historical/pre_three_stage_core/{name}"
                )
    stage_index = STAGES.index(stage)
    targets: list[Path] = []
    stage_outputs = _stage_output_filenames(policy)
    for downstream_stage in STAGES[stage_index + 1 :]:
        if downstream_stage not in stage_outputs:
            continue
        targets.extend(
            out_dir / filename
            for filename in stage_outputs[downstream_stage]
            if filename != MANIFEST_FILENAME
        )
    if stage_index <= STAGES.index("organize"):
        targets.extend(out_dir / filename for filename in RECORD_DEPENDENT_FILES)
        targets.extend(out_dir / directory for directory in RECORD_DEPENDENT_DIRECTORIES)
    if stage == "clean":
        targets.extend(out_dir / filename for filename in LEGACY_FLAT_ARTIFACTS)
    historical = out_dir / "historical/pre_unified_row_dedup"
    historical_names = {
        "04_pair_buckets",
        "05_collapsed_records",
        "06_distance_calibration",
    }
    legacy_collapsed_layout = any(
        (out_dir / name).exists()
        for name in ("05_collapsed_records", "06_distance_calibration")
    )
    for target in targets:
        if not target.exists():
            continue
        relative = str(target.relative_to(out_dir))
        if (
            target.name in historical_names
            and (target.name != "04_pair_buckets" or legacy_collapsed_layout)
            and not (historical / target.name).exists()
        ):
            historical.mkdir(parents=True, exist_ok=True)
            os.replace(target, historical / target.name)
            invalidated.append(
                f"{relative} -> historical/pre_unified_row_dedup/{target.name}"
            )
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        invalidated.append(relative)
    if invalidated:
        _log(f"invalidate after {stage}: " + ", ".join(sorted(invalidated)))
    return sorted(invalidated)


def _require_valid_schema(rows: list[dict[str, Any]], stage: str) -> None:
    errors = validate_stage_schema(rows, stage)
    if errors:
        raise ValueError(errors[0])


def _finish_partial(
    policy: StarlingTaskPolicy,
    out_dir: Path,
    args: argparse.Namespace,
    source_inventory: dict[str, Any],
    endpoint_inventories: dict[str, Any],
    started: float,
    invalidated_artifacts: list[str],
) -> int:
    payload = {
        "artifact_version": (
            CANONICAL_ARTIFACT_VERSION
            if policy.record_contract
            else NORMALIZED_ARTIFACT_VERSION
        ),
        "record_version": (
            CANONICAL_RECORD_VERSION
            if policy.record_contract
            else NORMALIZED_RECORD_VERSION
        ),
        "record_contract": (
            policy.record_contract.manifest() if policy.record_contract else None
        ),
        "scalar_parser_version": SCALAR_PARSER_VERSION,
        "measurement_resolution": _measurement_resolution_manifest(policy, args),
        "contextual_unit_policy": contextual_unit_policy_manifest(),
        "qualifier_vocabulary": qualifier_vocabulary_manifest(),
        **policy.manifest_versions(complete=False),
        "completed_stages": list(STAGES[: STAGES.index(args.through_stage) + 1]),
        "rebuild_request": {
            "from_stage": args.from_stage,
            "through_stage": args.through_stage,
        },
        "invalidated_artifacts": sorted(set(invalidated_artifacts)),
        "downstream_artifacts_physically_preserved": bool(
            args.preserve_downstream_artifacts
        ),
        "source_inventory": source_inventory,
        "endpoint_inventory": endpoint_inventories,
        "record_build_cache": _record_build_cache(
            policy,
            out_dir,
            args,
            completed_stage=args.through_stage,
            digests=FileDigestCache(),
        ),
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    if not getattr(args, "core_only", False):
        compact_profile = policy.compact_profile_for_contract(
            policy.record_contract.version if policy.record_contract else ""
        )
        payload.update(
            {
                "compact_artifact_version": compact_profile.artifact_version,
                "index_version": compact_profile.index_version,
            }
        )
    with _temporary_stage_directory(out_dir, "manifest") as stage_dir:
        _write_json(stage_dir / MANIFEST_FILENAME, payload)
        os.replace(stage_dir / MANIFEST_FILENAME, out_dir / MANIFEST_FILENAME)
    _log(f"partial build complete through {args.through_stage}: out={out_dir}")
    return 0


def _record_build_cache(
    policy: StarlingTaskPolicy,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    completed_stage: str,
    digests: FileDigestCache,
    staged_stage_dir: Path | None = None,
) -> dict[str, Any]:
    stage_index = STAGES.index(completed_stage)
    physical_output_paths: list[Path] = []
    staged_to_published: dict[str, str] = {}
    for stage in STAGES[: stage_index + 1]:
        for filename in _stage_output_filenames(policy).get(stage, ()):
            if filename == MANIFEST_FILENAME:
                continue
            published = out_dir / filename
            physical = (
                staged_stage_dir / filename
                if staged_stage_dir is not None and stage == completed_stage
                else published
            )
            physical_output_paths.append(physical)
            if physical != published:
                staged_to_published[str(physical)] = str(published)
    input_paths: list[Path] = []
    for stage in STAGES[: min(stage_index, STAGES.index("organize")) + 1]:
        artifact = _stage_artifacts(policy).get(stage)
        if artifact is None:
            continue
        manifest_path = out_dir / artifact[1]
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_paths.extend(_manifest_file_paths(manifest.get("inputs")))
    metadata = build_cache_metadata(
        task_id=policy.task_id,
        completed_stage=completed_stage,
        args=args,
        input_paths=input_paths,
        output_paths=physical_output_paths,
        digests=digests,
        scientific_assets=_scientific_assets(policy, args),
    )
    if staged_to_published:
        metadata["outputs"] = {
            staged_to_published.get(path, path): sha256
            for path, sha256 in metadata["outputs"].items()
        }
    return metadata


def _manifest_file_paths(value: Any) -> list[Path]:
    if isinstance(value, Mapping):
        paths: list[Path] = []
        path = value.get("path")
        if isinstance(path, str) and Path(path).is_file():
            paths.append(Path(path))
        for item in value.values():
            paths.extend(_manifest_file_paths(item))
        return paths
    if isinstance(value, (list, tuple)):
        return [path for item in value for path in _manifest_file_paths(item)]
    return []


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _log(message: str) -> None:
    print(f"[build_normalized_starling_evidence_library] {message}", flush=True)


def parse_args(
    policy: StarlingTaskPolicy,
    argv: list[str] | None,
    default_through_stage: str = "normalize",
    core_only: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stage_choices = CORE_STAGES if core_only else STAGES
    parser.set_defaults(core_only=core_only)
    parser.add_argument("--starling-data-dir", default=policy.default_data_dir)
    parser.add_argument("--out-dir", default=policy.default_out_dir)
    parser.add_argument("--from-stage", choices=stage_choices, default="source")
    parser.add_argument(
        "--through-stage", choices=stage_choices, default=default_through_stage
    )
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument(
        "--strict-endpoint-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--preserve-downstream-artifacts",
        action="store_true",
        help=(
            "Keep physical downstream files during an isolated Stage-01 rebuild; "
            "the resulting top-level manifest remains partial through clean."
        ),
    )
    if core_only:
        parser.set_defaults(
            frozen_retrieval_records=None,
            frozen_retrieval_normalized_records=None,
            legacy_task_local_downstream=False,
        )
    else:
        parser.add_argument("--frozen-retrieval-records", type=Path)
        parser.add_argument("--frozen-retrieval-normalized-records", type=Path)
        parser.add_argument("--legacy-task-local-downstream", action="store_true")
    parser.add_argument(
        "--validation-level",
        choices=("strict", "full"),
        default="strict",
        help=(
            "strict validates producer invariants once; full additionally "
            "recomputes every measurement/unit pair"
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("auto", "off"),
        default="auto",
        help="reuse a complete content-matched build, or force recomputation",
    )
    if not core_only:
        parser.add_argument("--semantic-aggregation-max-new-groups", type=int)
        parser.add_argument(
            "--semantic-aggregation-budget-max-tokens", type=int, default=10_000_000
        )
    if policy.measurement_resolution_enabled:
        default_mapping = import_task_module(
            policy.task_id, "starling_measurement_resolution"
        ).DEFAULT_MAPPING_PATH
        parser.add_argument(
            "--measurement-resolution-mapping",
            default=str(default_mapping) if Path(default_mapping).is_file() else "",
            help=(
                "Frozen offline measurement extraction applied as Stage-02 input. "
                "Empty retains numeric source rows as evidence without a scalar."
            ),
        )
        parser.add_argument(
            "--allow-partial-measurement-resolution",
            action="store_true",
            help=(
                "Permit extract-routed rows the frozen extraction does not cover; "
                "they remain evidence without a scalar."
            ),
        )
    if policy.add_cli_arguments is not None:
        policy.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    if stage_choices.index(args.from_stage) > stage_choices.index(args.through_stage):
        parser.error("--from-stage cannot be later than --through-stage")
    if args.preserve_downstream_artifacts and not (
        args.from_stage == "clean" and args.through_stage == "clean"
    ):
        parser.error(
            "--preserve-downstream-artifacts is limited to --from-stage clean "
            "--through-stage clean"
        )
    if args.max_rows_per_source and args.strict_endpoint_inventory:
        parser.error("bounded source runs require --no-strict-endpoint-inventory")
    if not core_only and (
        args.semantic_aggregation_max_new_groups is not None
        and args.semantic_aggregation_max_new_groups < 1
    ):
        parser.error("--semantic-aggregation-max-new-groups must be positive")
    if not core_only and args.semantic_aggregation_budget_max_tokens < 1:
        parser.error("--semantic-aggregation-budget-max-tokens must be positive")
    if policy.validate_arguments is not None:
        policy.validate_arguments(parser, args)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
