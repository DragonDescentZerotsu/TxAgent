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
from typing import Any

import pandas as pd

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.starling.compact_artifacts import (
    assert_compact_schema,
    build_relational_evidence_catalog,
    compact_persisted_records,
    write_compact_neighbor_index,
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
    clean_source_rows,
    file_sha256,
    normalize_endpoint_name,
)
from tools.chembl_tool.common.starling.normalization.contracts import (
    CLEANING_STAGE_VERSION,
    NORMALIZATION_STAGE_VERSION,
    NORMALIZED_ARTIFACT_VERSION,
    NORMALIZED_RECORD_VERSION,
    ORGANIZATION_STAGE_VERSION,
    SCALAR_PARSER_VERSION,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_cleaned_records,
)
from tools.chembl_tool.common.starling.normalization.organization import (
    organize_normalized_records,
)
from tools.chembl_tool.common.starling.normalization.task_policy import StarlingTaskPolicy
from tools.chembl_tool.common.task_workflows.evidence_library import fingerprint_metadata
from tools.chembl_tool.common.units import (
    UNIT_NORMALIZER_VERSION,
    contextual_unit_policy_manifest,
)


CLEANED_FILENAME = "01_cleaned/records.parquet"
NORMALIZED_RECORDS_FILENAME = "02_normalized/records.parquet"
RECORDS_FILENAME = "03_records/records.parquet"
REJECTIONS_FILENAME = "03_records/exclusions.parquet"
DUPLICATES_FILENAME = "03_records/duplicates.parquet"
DISTRIBUTION_AUDIT_FILENAME = "03_records/scalar_distribution.parquet"
ENDPOINT_REGISTRY_FILENAME = "02_normalized/endpoint_registry.json"
ENDPOINT_INVENTORY_FILENAME = "01_cleaned/endpoint_inventory.json"
SOURCE_INVENTORY_FILENAME = "01_cleaned/source_inventory.json"
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
    out_dir = Path(ensure_dir(args.out_dir))
    first_stage = STAGES.index(args.from_stage)
    final_stage = STAGES.index(args.through_stage)
    invalidated_artifacts: list[str] = []

    source_inventory = _read_optional_json(out_dir / SOURCE_INVENTORY_FILENAME)
    endpoint_inventories = _read_optional_json(out_dir / ENDPOINT_INVENTORY_FILENAME)

    if first_stage == 0:
        (
            cleaned,
            source_inventory,
            endpoint_inventories,
            cleaning_inputs,
        ) = _build_cleaned_records(policy, args)
        _require_valid_schema(cleaned, "clean")
        cleaned_persisted = compact_persisted_records(
            policy.attach_source_columns(cleaned)
        )
        assert_compact_schema(cleaned_persisted)
        with _temporary_stage_directory(out_dir, "clean") as stage_dir:
            write_parquet(stage_dir / CLEANED_FILENAME, cleaned_persisted)
            _write_json(stage_dir / SOURCE_INVENTORY_FILENAME, source_inventory)
            _write_json(stage_dir / ENDPOINT_INVENTORY_FILENAME, endpoint_inventories)
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "clean",
                inputs=cleaning_inputs,
                output_filename=CLEANED_FILENAME,
                row_counts={"cleaned_records": len(cleaned)},
                validations={"one_cleaned_record_per_source_row": True},
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "clean", stage_dir)
            )
        _log(f"clean: records={len(cleaned):,}")
    else:
        cleaned = _load_verified_stage(out_dir, "clean") if first_stage == 1 else []

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
            cleaned = _load_verified_stage(out_dir, "clean")
        hooks = policy.build_hooks(args)
        normalized = normalize_cleaned_records(
            cleaned,
            endpoint_normalizer=hooks.endpoint_normalizer,
            endpoint_standardizer=hooks.endpoint_standardizer,
            source_measurement_resolver=hooks.source_measurement_resolver,
            family_resolver=hooks.family_resolver,
            record_enricher=hooks.record_enricher,
        )
        identity_errors = validate_cleaned_normalized_identity(cleaned, normalized)
        if identity_errors:
            raise ValueError(
                f"{len(identity_errors)} cleaned/normalized identity failure(s); "
                f"first={identity_errors[0]}"
            )
        pair_errors = validate_measurement_pairs(
            normalized,
            hooks.endpoint_standardizer,
            hooks.source_measurement_resolver,
            hooks.contextual_standardizer,
        )
        if pair_errors:
            raise ValueError(
                f"{len(pair_errors)} measurement/unit pair invariant failure(s); "
                f"first={pair_errors[0]}"
            )
        _require_valid_schema(normalized, "normalize")
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
                stage_dir / NORMALIZED_RECORDS_FILENAME, normalized_persisted
            )
            _write_json(stage_dir / VALIDITY_POLICY_FILENAME, documents.validity_policy)
            _write_json(
                stage_dir / AUXILIARY_MAPPING_MANIFEST_FILENAME,
                documents.auxiliary_mapping_manifest,
            )
            _write_json(
                stage_dir / SOURCE_COLUMN_CONTRACT_FILENAME,
                documents.source_column_contract,
            )
            _write_json(
                stage_dir / ENDPOINT_REGISTRY_FILENAME,
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
                inputs={
                    "cleaned_records": out_dir / CLEANED_FILENAME,
                    "contextual_unit_policy": unit_policy_manifest["path"],
                },
                output_filename=NORMALIZED_RECORDS_FILENAME,
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
                _commit_stage_outputs(out_dir, "normalize", stage_dir)
            )
        _log(
            "normalize: "
            f"records={len(normalized):,} "
            f"scalars={sum(row.get('finite_scalar_value') is not None for row in normalized):,}"
        )
    else:
        normalized = (
            _load_verified_stage(out_dir, "normalize") if first_stage == 2 else []
        )

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
            normalized = _load_verified_stage(out_dir, "normalize")
        records, duplicates, exclusions, organization_stats = organize_normalized_records(
            normalized
        )
        _require_valid_schema(records, "organize")
        distribution_rows = scalar_distribution_audit(records)
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
                inputs={"normalized_records": out_dir / NORMALIZED_RECORDS_FILENAME},
                output_filename=RECORDS_FILENAME,
                row_counts=organization_stats,
                validations={
                    "cross_source_deduplication": False,
                    "unusable_structures_retained_in_records": True,
                },
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "organize", stage_dir)
            )
        _log(
            f"organize: records={len(records):,} "
            f"retrieval_eligible={organization_stats['n_retrieval_eligible']:,}"
        )
    else:
        records = _load_verified_stage(out_dir, "organize")
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
    with _temporary_stage_directory(out_dir, "index") as stage_dir:
        write_parquet(stage_dir / EVIDENCE_FAMILIES_FILENAME, evidence_families)
        write_parquet(stage_dir / EVIDENCE_BRIDGE_FILENAME, evidence_bridge)
        index_manifest = write_compact_neighbor_index(
            profile=policy.compact,
            families=evidence_families,
            output_dir=stage_dir / "05_neighbor_index",
            workers=args.workers,
            progress_every=args.progress_every,
        )
        _write_json(
            stage_dir / EVIDENCE_MANIFEST_FILENAME,
            {
                "artifact_version": policy.compact.artifact_version,
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
            "normalized_records": NORMALIZED_RECORDS_FILENAME,
            "records": RECORDS_FILENAME,
            "duplicates": DUPLICATES_FILENAME,
            "organization_exclusions": REJECTIONS_FILENAME,
            "molecule_families": EVIDENCE_FAMILIES_FILENAME,
            "molecule_family_records": EVIDENCE_BRIDGE_FILENAME,
            "scalar_distribution_audit": DISTRIBUTION_AUDIT_FILENAME,
            "record_validity_policy": VALIDITY_POLICY_FILENAME,
            "auxiliary_mapping_manifest": AUXILIARY_MAPPING_MANIFEST_FILENAME,
            "source_column_contracts": SOURCE_COLUMN_CONTRACT_FILENAME,
            "index_molecules": INDEX_MOLECULES_FILENAME,
            "index_fingerprints": INDEX_FINGERPRINTS_FILENAME,
            "index_membership": INDEX_MEMBERSHIP_FILENAME,
        }
        manifest = {
            "artifact_version": NORMALIZED_ARTIFACT_VERSION,
            "compact_artifact_version": policy.compact.artifact_version,
            "record_version": NORMALIZED_RECORD_VERSION,
            "scalar_parser_version": SCALAR_PARSER_VERSION,
            "unit_normalizer_version": UNIT_NORMALIZER_VERSION,
            "contextual_unit_policy": unit_policy_manifest,
            "index_version": policy.compact.index_version,
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
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        _write_json(stage_dir / MANIFEST_FILENAME, manifest)
        invalidated_artifacts.extend(_commit_stage_outputs(out_dir, "index", stage_dir))
    _log(
        f"complete: records={len(records):,} "
        f"molecule-family rows={len(evidence_families):,} "
        f"index molecules={index_manifest['molecules']:,} out={out_dir}"
    )
    return 0


def _build_cleaned_records(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Path]]:
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
        needed_identifiers.update(
            str(value)
            for value in frame.get("global_identifier", [])
            if value is not None and str(value).strip()
        )
        endpoints = [
            normalize_endpoint_name(value) or ""
            for value in frame[profile.endpoint_field].tolist()
        ]
        endpoint_inventories[profile.source_id] = policy.endpoint_inventory(
            profile.source_id, endpoints, strict=args.strict_endpoint_inventory
        )

    smiles_mapping = (
        load_smiles_mapping(mapping_spec.path, needed_identifiers)
        if mapping_spec is not None
        else {}
    )
    cleaned: list[dict[str, Any]] = []
    for profile in profiles:
        cleaned.extend(
            clean_source_rows(
                frames[profile.source_id].to_dict(orient="records"),
                profile,
                smiles_mapping=smiles_mapping or None,
                source_sha256=source_hashes[profile.source_id],
            )
        )

    extra_entry: dict[str, Any] | None = None
    extra_key = "direct_hf"
    if policy.load_extra_source is not None:
        extra = policy.load_extra_source(args)
        if extra is not None:
            extra_key = extra.source_id
            inputs[extra.inventory_key] = extra.source_path
            extra_entry = extra.inventory_entry
            endpoint_inventories[extra.source_id] = policy.endpoint_inventory(
                extra.source_id,
                extra.endpoint_names,
                strict=args.strict_endpoint_inventory,
            )
            cleaned.extend(
                clean_source_rows(
                    extra.rows,
                    extra.profile,
                    smiles_mapping=None,
                    source_sha256=extra.source_sha256,
                )
            )
            source_hashes[extra.source_id] = extra.source_sha256

    source_inventory: dict[str, Any] = {
        "dataset": policy.dataset_name,
        "source_hashes": source_hashes,
        "source_row_counts": count_by(cleaned, "source_id"),
        extra_key: extra_entry,
    }
    if mapping_spec is not None:
        source_inventory["smiles_mapping"] = {
            "path": str(mapping_spec.path),
            "sha256": mapping_sha,
            "expected_sha256": mapping_spec.expected_sha256,
            "n_loaded_identifiers": len(smiles_mapping),
        }
    return cleaned, source_inventory, endpoint_inventories, inputs


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


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_verified_stage(out_dir: Path, stage: str) -> list[dict[str, Any]]:
    filename, manifest_filename, version = STAGE_ARTIFACTS[stage]
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
    inputs: dict[str, Path],
    output_filename: str,
    row_counts: dict[str, int],
    validations: dict[str, Any],
) -> None:
    _, manifest_filename, version = STAGE_ARTIFACTS[stage]
    payload = stage_manifest(
        stage=stage,
        version=version,
        inputs=inputs,
        output=stage_dir / output_filename,
        row_counts=row_counts,
        validations=validations,
    )
    payload["output"]["path"] = str(out_dir / output_filename)
    _write_json(stage_dir / manifest_filename, payload)


@contextmanager
def _temporary_stage_directory(out_dir: Path, stage: str):
    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=f".{stage}-stage-"
    ) as directory:
        yield Path(directory)


def _commit_stage_outputs(out_dir: Path, stage: str, stage_dir: Path) -> list[str]:
    expected_outputs = STAGE_OUTPUT_FILENAMES[stage]
    missing = [
        filename
        for filename in expected_outputs
        if not (stage_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"cannot publish {stage} stage; missing staged outputs: {missing}"
        )

    invalidated = _invalidate_downstream_artifacts(out_dir, stage)
    if stage != "index":
        (out_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    for filename in expected_outputs:
        destination = out_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir / filename, destination)
    return invalidated


def _invalidate_downstream_artifacts(out_dir: Path, stage: str) -> list[str]:
    stage_index = STAGES.index(stage)
    targets: list[Path] = []
    for downstream_stage in STAGES[stage_index + 1 :]:
        targets.extend(
            out_dir / filename
            for filename in STAGE_OUTPUT_FILENAMES[downstream_stage]
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
        "artifact_version": NORMALIZED_ARTIFACT_VERSION,
        "record_version": NORMALIZED_RECORD_VERSION,
        "scalar_parser_version": SCALAR_PARSER_VERSION,
        "contextual_unit_policy": contextual_unit_policy_manifest(),
        **policy.manifest_versions(complete=False),
        "completed_stages": list(STAGES[: STAGES.index(args.through_stage) + 1]),
        "rebuild_request": {
            "from_stage": args.from_stage,
            "through_stage": args.through_stage,
        },
        "invalidated_artifacts": sorted(set(invalidated_artifacts)),
        "source_inventory": source_inventory,
        "endpoint_inventory": endpoint_inventories,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    with _temporary_stage_directory(out_dir, "manifest") as stage_dir:
        _write_json(stage_dir / MANIFEST_FILENAME, payload)
        os.replace(stage_dir / MANIFEST_FILENAME, out_dir / MANIFEST_FILENAME)
    _log(f"partial build complete through {args.through_stage}: out={out_dir}")
    return 0


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
