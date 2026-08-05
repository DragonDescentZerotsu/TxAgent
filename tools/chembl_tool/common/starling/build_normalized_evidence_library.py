"""Staged builder for a task's layered normalized-Starling evidence library.

Stages are ``clean -> normalize -> organize -> index``.  Each stage writes into
a temporary directory and is published only after its outputs exist and its
validations pass, so a failed computation leaves the previous coherent build in
place.  Resuming a stage re-verifies both the stage artifact hash and the
recorded immediate-upstream input hash, so a self-consistent but stale
downstream stage cannot be resumed.

Everything task-specific arrives through a
:class:`~tools.chembl_tool.common.starling.normalization.task_policy.StarlingTaskPolicy`.
Run it for a task with ``--task``, or through the task's own thin wrapper.
"""

from __future__ import annotations

import argparse
import importlib
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

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.starling.compact_artifacts import (
    assert_compact_schema,
    build_relational_evidence_catalog,
    compact_persisted_records,
    write_compact_neighbor_index,
)
from tools.chembl_tool.common.starling.canonicalization_v7 import (
    CANONICAL_ARTIFACT_VERSION,
    CANONICAL_RECORD_VERSION,
)
from tools.chembl_tool.common.starling.build_runtime import (
    FileDigestCache,
    build_cache_metadata,
    cache_metadata_matches,
    clean_sources_ordered,
    normalize_and_project_records_ordered,
    normalize_records_ordered,
)
from tools.chembl_tool.common.starling.normalization.audit import (
    read_parquet_records,
    scalar_distribution_audit,
    stage_manifest,
    validate_cleaned_normalized_identity,
    validate_measurement_pairs,
    validate_stage_schema,
    write_parquet,
)
from tools.chembl_tool.common.starling.normalization.cleaning import (
    file_sha256,
    normalize_endpoint_name,
    resolve_structure_value,
)
from tools.chembl_tool.common.starling.normalization.contracts import (
    CLEANING_STAGE_VERSION,
    NORMALIZATION_STAGE_VERSION,
    NORMALIZED_ARTIFACT_VERSION,
    NORMALIZED_RECORD_VERSION,
    ORGANIZATION_STAGE_VERSION,
    SCALAR_PARSER_VERSION,
)
from tools.chembl_tool.common.starling.normalization.organization import (
    organize_normalized_records,
)
from tools.chembl_tool.common.starling.normalization.task_policy import (
    ExtraSourceBatch,
    StarlingTaskPolicy,
)
from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id
from tools.chembl_tool.common.task_workflows.evidence_library import fingerprint_metadata
from tools.chembl_tool.common.units import (
    UNIT_NORMALIZER_VERSION,
    contextual_unit_policy_manifest,
    qualifier_vocabulary_manifest,
)


CLEANED_FILENAME = "01_cleaned/records.parquet"
NORMALIZED_RECORDS_FILENAME = "02_normalized/records.parquet"
CANONICALIZED_RECORDS_FILENAME = "02_canonicalized/records.parquet"
RECORDS_FILENAME = "03_records/records.parquet"
REJECTIONS_FILENAME = "03_records/exclusions.parquet"
DUPLICATES_FILENAME = "03_records/duplicates.parquet"
DISTRIBUTION_AUDIT_FILENAME = "03_records/scalar_distribution.parquet"
ENDPOINT_REGISTRY_FILENAME = "02_normalized/endpoint_registry.json"
ENDPOINT_INVENTORY_FILENAME = "01_cleaned/endpoint_inventory.json"
SOURCE_INVENTORY_FILENAME = "01_cleaned/source_inventory.json"
SOURCE_VALUE_CLEANING_AUDIT_FILENAME = (
    "01_cleaned/source_value_cleaning_audit.parquet"
)
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
        (SOURCE_VALUE_CLEANING_AUDIT_FILENAME,)
        if policy is not None and policy.source_value_cleaner is not None
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
        ),
    }

STAGES = ("clean", "normalize", "organize", "index")
STAGE_ARTIFACTS = {
    "clean": (CLEANED_FILENAME, "01_cleaned/manifest.json", CLEANING_STAGE_VERSION),
    "normalize": (
        NORMALIZED_RECORDS_FILENAME,
        "02_normalized/manifest.json",
        NORMALIZATION_STAGE_VERSION,
    ),
    "organize": (RECORDS_FILENAME, "03_records/manifest.json", ORGANIZATION_STAGE_VERSION),
}
STAGE_OUTPUT_FILENAMES = {
    "clean": (
        CLEANED_FILENAME,
        "01_cleaned/manifest.json",
        SOURCE_INVENTORY_FILENAME,
        ENDPOINT_INVENTORY_FILENAME,
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
    "normalize": ("cleaned_records", CLEANED_FILENAME),
    "organize": ("normalized_records", NORMALIZED_RECORDS_FILENAME),
}
RECORD_DEPENDENT_DIRECTORIES = (
    "04_pair_buckets",
    "05_distance_calibration",
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
    "endpoint_inventory.json",
    "source_inventory.json",
)


def load_task_policy(task_id: str) -> StarlingTaskPolicy:
    """Resolve ``tools.chembl_tool.tasks.<task>.starling_policy:POLICY``."""
    module = importlib.import_module(
        f"tools.chembl_tool.tasks.{task_id}.starling_policy"
    )
    policy = getattr(module, "POLICY", None)
    if not isinstance(policy, StarlingTaskPolicy):
        raise TypeError(
            f"tasks.{task_id}.starling_policy must publish POLICY as a StarlingTaskPolicy"
        )
    return policy


def build(policy: StarlingTaskPolicy, argv: list[str] | None = None) -> int:
    return _run(policy, parse_args(policy, argv))


def run_with_args(policy: StarlingTaskPolicy, args: argparse.Namespace) -> int:
    """Run an already-validated argument namespace through the shared stages."""
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

    digests = FileDigestCache()
    active_manifest = _read_optional_json(out_dir / MANIFEST_FILENAME)
    cached = active_manifest.get("record_build_cache")
    cached_stage = str((cached or {}).get("completed_stage") or "")
    if (
        args.cache_mode == "auto"
        and args.validation_level == "strict"
        and first_stage == 0
        and cached_stage in STAGES
        and cached_stage == args.through_stage
        and cache_metadata_matches(
            cached,
            task_id=policy.task_id,
            completed_stage=cached_stage,
            args=args,
            digests=digests,
            scientific_assets=policy.scientific_assets,
        )
    ):
        _log(
            f"cache hit through {cached_stage}: content_key={cached['content_key']}"
        )
        return 0

    source_inventory = _read_optional_json(out_dir / SOURCE_INVENTORY_FILENAME)
    endpoint_inventories = _read_optional_json(out_dir / ENDPOINT_INVENTORY_FILENAME)

    if first_stage == 0:
        (
            cleaned,
            source_inventory,
            endpoint_inventories,
            cleaning_inputs,
            source_value_cleaning_audit,
        ) = _build_cleaned_records(policy, args)
        _require_valid_schema(cleaned, "clean")
        cleaned_attached = policy.attach_source_columns(cleaned)
        cleaned_persisted = compact_persisted_records(
            [
                policy.record_contract.clean_projection(row)
                for row in cleaned_attached
            ]
            if policy.record_contract
            else cleaned_attached
        )
        assert_compact_schema(cleaned_persisted)
        with _temporary_stage_directory(out_dir, "clean") as stage_dir:
            write_parquet(stage_dir / CLEANED_FILENAME, cleaned_persisted)
            if policy.source_value_cleaner is not None:
                write_parquet(
                    stage_dir / SOURCE_VALUE_CLEANING_AUDIT_FILENAME,
                    source_value_cleaning_audit,
                )
            _write_json(stage_dir / SOURCE_INVENTORY_FILENAME, source_inventory)
            _write_json(stage_dir / ENDPOINT_INVENTORY_FILENAME, endpoint_inventories)
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "clean",
                policy=policy,
                inputs=cleaning_inputs,
                output_filename=CLEANED_FILENAME,
                row_counts={"cleaned_records": len(cleaned)},
                validations={
                    "one_cleaned_record_per_source_row": True,
                    "source_value_cleaning_audited": (
                        policy.source_value_cleaner is not None
                    ),
                    "source_value_cleaning_row_identity_preserved": True,
                },
                sidecars=(
                    {
                        "source_value_cleaning_audit": (
                            stage_dir / SOURCE_VALUE_CLEANING_AUDIT_FILENAME
                        )
                    }
                    if policy.source_value_cleaner is not None
                    else None
                ),
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "clean", stage_dir, policy)
            )
        _log(f"clean: records={len(cleaned):,}")
    else:
        cleaned_persisted = (
            _load_verified_stage(out_dir, "clean", policy) if first_stage == 1 else []
        )
        cleaned = cleaned_persisted
        if cleaned and policy.record_contract:
            cleaned = [policy.record_contract.inflate_cleaned(row) for row in cleaned]
            _restore_v7_cleaned_structures(
                policy, args, cleaned, source_inventory=source_inventory
            )

    if final_stage == 0:
        return _finish_partial(
            policy,
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= 1:
        if not cleaned:
            cleaned_persisted = _load_verified_stage(out_dir, "clean", policy)
            cleaned = cleaned_persisted
            if policy.record_contract:
                cleaned = [
                    policy.record_contract.inflate_cleaned(row) for row in cleaned
                ]
                _restore_v7_cleaned_structures(
                    policy, args, cleaned, source_inventory=source_inventory
                )
        hooks = policy.build_hooks(args)
        if policy.record_contract:
            normalized, normalized_persisted = normalize_and_project_records_ordered(
                cleaned,
                hooks=hooks,
                policy=policy,
                workers=args.workers,
            )
        else:
            normalized = normalize_records_ordered(
                cleaned,
                hooks=hooks,
                task=policy.task_id,
                workers=args.workers,
            )
        identity_errors = validate_cleaned_normalized_identity(cleaned, normalized)
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
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "normalize",
                policy=policy,
                inputs={
                    "cleaned_records": out_dir / CLEANED_FILENAME,
                    "contextual_unit_policy": unit_policy_manifest["path"],
                    "qualifier_vocabulary": qualifier_manifest["path"],
                },
                output_filename=normalized_records_filename,
                row_counts={
                    "cleaned_records": len(cleaned),
                    "normalized_records": len(normalized),
                    "finite_scalars": sum(
                        row.get("finite_scalar_value") is not None
                        for row in normalized
                    ),
                    "absolute_and_continuous": sum(
                        bool(row.get("is_absolute_and_continuous"))
                        for row in normalized
                    ),
                    "normalization_valid": sum(
                        row.get("normalization_validity_status") == "valid"
                        for row in normalized
                    ),
                },
                validations=documents.validations,
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "normalize", stage_dir, policy)
            )
        _log(
            "normalize: "
            f"records={len(normalized):,} "
            f"scalars={sum(row.get('finite_scalar_value') is not None for row in normalized):,}"
        )
    else:
        normalized_persisted = (
            _load_verified_stage(out_dir, "normalize", policy)
            if first_stage == 2
            else []
        )
        normalized = normalized_persisted
        if normalized and policy.record_contract:
            normalized = [
                policy.record_contract.inflate_canonical(row) for row in normalized
            ]

    if final_stage == 1:
        return _finish_partial(
            policy,
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= 2:
        if not normalized:
            normalized_persisted = _load_verified_stage(out_dir, "normalize", policy)
            normalized = normalized_persisted
            if policy.record_contract:
                normalized = [
                    policy.record_contract.inflate_canonical(row)
                    for row in normalized
                ]
        records, duplicates, exclusions, organization_stats = organize_normalized_records(
            normalized
        )
        _require_valid_schema(records, "organize")
        distribution_rows = scalar_distribution_audit(records)
        if policy.record_contract:
            canonical_by_id = {
                str(row.get("canonical_record_id") or ""): row
                for row in normalized_persisted
            }
            records_persisted = []
            for record in records:
                record_id = str(record.get("normalized_record_id") or "")
                base = canonical_by_id.get(record_id)
                if base is None:
                    raise ValueError(
                        f"organized record lacks its canonical Stage-02 row: {record_id}"
                    )
                persisted = dict(base)
                for field in (
                    "duplicate_group_id",
                    "duplicate_group_size",
                    "retrieval_eligible",
                    "organization_status",
                ):
                    persisted[field] = record.get(field)
                records_persisted.append(persisted)
            # Preserve the established global Parquet column order without
            # re-projecting every record.  Pandas fixes the initial columns
            # from the first mapping and appends source-specific fields as
            # they first appear in later mappings.
            if records_persisted:
                records_persisted[0] = compact_persisted_records(
                    [policy.record_contract.canonical_projection(records[0])]
                )[0]
        else:
            records_persisted = compact_persisted_records(records)
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
                inputs={"normalized_records": out_dir / normalized_records_filename},
                output_filename=RECORDS_FILENAME,
                row_counts=organization_stats,
                validations={
                    "cross_source_deduplication": False,
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
            "n_normalized_records_before_deduplication": len(records) + len(duplicates),
            "n_records": len(records),
            "n_duplicates_removed": len(duplicates),
            "n_retrieval_eligible": sum(
                bool(row.get("retrieval_eligible")) for row in records
            ),
            "n_organization_exclusions": sum(
                not bool(row.get("retrieval_eligible")) for row in records
            ),
            "n_absolute_and_continuous": sum(
                bool(row.get("is_absolute_and_continuous")) for row in records
            ),
        }

    frozen_census = {
        "n_cleaned_records": sum(
            int(value)
            for value in (source_inventory.get("source_row_counts") or {}).values()
        ),
        **organization_stats,
        **(policy.census_extras(records) if policy.census_extras else {}),
        "normalization_validity_status_counts": count_by(
            records, "normalization_validity_status"
        ),
    }

    if final_stage == 2:
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


def _build_cleaned_records(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    list[dict[str, Any]],
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

    cleaned = clean_sources_ordered(clean_batches, workers=args.workers)
    source_value_cleaning_audit: list[dict[str, Any]] = []
    source_value_cleaning_manifest: dict[str, Any] | None = None
    if policy.source_value_cleaner is not None:
        before_ids = [str(row.get("cleaned_record_id") or "") for row in cleaned]
        cleaning_result = policy.source_value_cleaner(cleaned, args)
        cleaned = cleaning_result.records
        after_ids = [str(row.get("cleaned_record_id") or "") for row in cleaned]
        if before_ids != after_ids:
            raise ValueError(
                "source-value cleaning changed Stage-01 row count, order, or identity"
            )
        source_value_cleaning_audit = cleaning_result.audit_rows
        source_value_cleaning_manifest = cleaning_result.manifest
        for index, path in enumerate(cleaning_result.input_paths, start=1):
            inputs[f"source_value_cleaning_asset_{index}"] = path

    source_inventory: dict[str, Any] = {
        "dataset": policy.dataset_name,
        "source_hashes": source_hashes,
        "source_row_counts": count_by(cleaned, "source_id"),
        **extra_entries,
    }
    if source_value_cleaning_manifest is not None:
        source_inventory["source_value_cleaning"] = source_value_cleaning_manifest
    if mapping_spec is not None:
        source_inventory["smiles_mapping"] = {
            "path": str(mapping_spec.path),
            "sha256": mapping_sha,
            "expected_sha256": mapping_spec.expected_sha256,
            "n_loaded_identifiers": len(smiles_mapping),
        }
    return (
        cleaned,
        source_inventory,
        endpoint_inventories,
        inputs,
        source_value_cleaning_audit,
    )


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


def _restore_v7_cleaned_structures(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    source_inventory: dict[str, Any],
) -> None:
    """Recompute private structure helpers omitted from persisted Stage 01."""
    contract = policy.record_contract
    if contract is None:
        return
    mapping_spec = policy.smiles_mapping(args) if policy.smiles_mapping else None
    if mapping_spec is not None:
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
        clean_manifest_path = Path(args.out_dir) / "01_cleaned/manifest.json"
        clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
        manifest_sha = str(
            ((clean_manifest.get("inputs") or {}).get("smiles_mapping") or {}).get(
                "sha256"
            )
            or ""
        )
        if not manifest_sha or mapping_sha != manifest_sha:
            raise ValueError(
                "SMILES mapping differs from the Stage 01 input contract: "
                f"expected {manifest_sha or '<missing>'}, found {mapping_sha}"
            )
    identifiers = {
        str(row.get("global_identifier") or "")
        for row in records
        if contract.source(str(row.get("source_id") or "")).structure_mode == "mapped"
        and row.get("global_identifier")
    }
    mapping = (
        load_smiles_mapping(mapping_spec.path, identifiers)
        if mapping_spec is not None and identifiers
        else {}
    )
    for row in records:
        profile = contract.source(str(row.get("source_id") or ""))
        raw_smiles = (
            mapping.get(str(row.get(profile.structure_identity_field) or ""))
            if profile.structure_mode == "mapped"
            else row.get("smiles")
        )
        canonical, structure_status = resolve_structure_value(
            raw_smiles,
            structure_mode=profile.structure_mode,
        )
        row["source_smiles"] = raw_smiles
        row["canonical_smiles"] = canonical
        row["structure_status"] = structure_status
        row["molecule_id"] = starling_molecule_id(canonical) if canonical else None


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_verified_stage(
    out_dir: Path, stage: str, policy: StarlingTaskPolicy | None = None
) -> list[dict[str, Any]]:
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
    return read_parquet_records(artifact)


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

    invalidated = _invalidate_downstream_artifacts(out_dir, stage, policy)
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
    stage_index = STAGES.index(stage)
    targets: list[Path] = []
    for downstream_stage in STAGES[stage_index + 1 :]:
        targets.extend(
            out_dir / filename
            for filename in _stage_output_filenames(policy)[downstream_stage]
            if filename != MANIFEST_FILENAME
        )
    if stage_index <= STAGES.index("organize"):
        targets.extend(out_dir / filename for filename in RECORD_DEPENDENT_FILES)
        targets.extend(out_dir / directory for directory in RECORD_DEPENDENT_DIRECTORIES)
    if stage == "clean":
        targets.extend(out_dir / filename for filename in LEGACY_FLAT_ARTIFACTS)

    invalidated: list[str] = []
    for target in targets:
        if not target.exists():
            continue
        relative = str(target.relative_to(out_dir))
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
        "contextual_unit_policy": contextual_unit_policy_manifest(),
        "qualifier_vocabulary": qualifier_vocabulary_manifest(),
        **policy.manifest_versions(complete=False),
        "completed_stages": list(STAGES[: STAGES.index(args.through_stage) + 1]),
        "rebuild_request": {
            "from_stage": args.from_stage,
            "through_stage": args.through_stage,
        },
        "invalidated_artifacts": sorted(set(invalidated_artifacts)),
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
        scientific_assets=policy.scientific_assets,
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
    policy: StarlingTaskPolicy, argv: list[str] | None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-data-dir", default=policy.default_data_dir)
    parser.add_argument("--out-dir", default=policy.default_out_dir)
    parser.add_argument("--from-stage", choices=STAGES, default="clean")
    parser.add_argument("--through-stage", choices=STAGES, default="index")
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
    if policy.add_cli_arguments is not None:
        policy.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    if STAGES.index(args.from_stage) > STAGES.index(args.through_stage):
        parser.error("--from-stage cannot be later than --through-stage")
    if args.max_rows_per_source and args.strict_endpoint_inventory:
        parser.error("bounded source runs require --no-strict-endpoint-inventory")
    if policy.validate_arguments is not None:
        policy.validate_arguments(parser, args)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
