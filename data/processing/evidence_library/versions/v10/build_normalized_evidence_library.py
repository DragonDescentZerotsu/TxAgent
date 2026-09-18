"""Staged builder for the first two normalized-Starling core stages.

Stages are ``source -> clean -> normalize``. Pair buckets and transfer
calibration are built separately as the split-independent Stage 3. Each stage
writes into a temporary directory and is published only after its outputs exist
and its validations pass, so a failed computation leaves the previous coherent
build in place.  Resuming a stage re-verifies both the stage artifact hash and
the recorded immediate-upstream input hash, so a self-consistent but stale
downstream stage cannot be resumed.

Everything task-specific arrives through a
:class:`~data.processing.evidence_library.shared.v2.normalization.task_policy.StarlingTaskPolicy`.
Run it for a task with ``--task``, or through the task's own thin wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from data.processing.evidence_library.compact_artifacts import (
    assert_compact_schema,
    build_relational_evidence_catalog,
    compact_persisted_record,
    compact_persisted_records,
    write_compact_neighbor_index,
)
from data.processing.evidence_library.shared.v2.build_runtime import (
    FileDigestCache,
    assert_unpublished_build_root,
    build_cache_metadata,
    cache_metadata_matches,
    clean_sources_ordered,
    normalize_and_project_records_ordered,
    normalize_records_ordered,
    starling_build_session,
)
from data.processing.evidence_library.shared.v2.canonicalization_v7 import (
    CANONICAL_ARTIFACT_VERSION,
    CANONICAL_RECORD_VERSION,
)
from data.processing.evidence_library.shared.v2.normalization.audit import (
    read_parquet_records,
    scalar_distribution_audit,
    stage_manifest,
    validate_cleaned_normalized_identity_ids,
    validate_measurement_pairs,
    validate_stage_schema,
    write_parquet,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_literal_text,
    file_sha256,
    normalize_endpoint_name,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import (
    CLEANING_STAGE_VERSION,
    NormalizedSourceProfile,
    NORMALIZATION_STAGE_VERSION,
    NORMALIZED_ARTIFACT_VERSION,
    NORMALIZED_RECORD_VERSION,
    ORGANIZATION_STAGE_VERSION,
    SCALAR_PARSER_VERSION,
    SOURCE_STAGE_VERSION,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    RESOLUTION_APPLY_VERSION,
    apply_measurement_resolution,
)
from data.processing.evidence_library.shared.v2.normalization.organization import (
    is_absolute_continuous,
    organize_normalized_records,
)
from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    ExtraSourceBatch,
    StarlingTaskPolicy,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)
from data.processing.evidence_library.versions.v10.stage1_exact_deduplication import load_voter_protection
from data.processing.evidence_library.versions.v10.task_registry import (
    import_task_module,
)
from data.processing.paths import assert_canonical_library_output
from data.processing.starling_source_row_identity import (
    UID_FIELD,
    validate_task_sources,
)
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.task_workflows.evidence_library import (
    fingerprint_metadata,
)
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
SOURCE_VALUE_CLEANING_AUDIT_FILENAME = "01_cleaned/source_value_cleaning_audit.parquet"
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


def _exact_unit_mapping(
    policy: StarlingTaskPolicy, args: argparse.Namespace | None = None
) -> Path | None:
    path = getattr(args, "exact_unit_mapping", None) or policy.exact_unit_mapping_path
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
    unit_mapping = _exact_unit_mapping(policy, args) if enabled else None
    return {
        "enabled": enabled,
        "active": enabled,
        "application_stage": "clean",
        "stage1_routing_enabled": routing_enabled,
        "frozen_extraction_loaded": mapping is not None,
        "apply_version": RESOLUTION_APPLY_VERSION,
        "exact_unit_mapping_version": (
            json.loads(unit_mapping.read_text(encoding="utf-8"))["version"]
            if unit_mapping
            else EXACT_UNIT_MAPPING_VERSION
        ),
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
    universe = _source_universe_paths(policy)
    if universe is not None:
        manifest_path, part_paths, _ = universe
        assets.extend((manifest_path, *part_paths))
    if policy.measurement_resolution_enabled:
        unit_mapping = _exact_unit_mapping(policy, args)
        default_unit_mapping = (
            Path(policy.exact_unit_mapping_path)
            if policy.exact_unit_mapping_path is not None
            else None
        )
        if unit_mapping != default_unit_mapping and default_unit_mapping in assets:
            assets.remove(default_unit_mapping)
        if unit_mapping not in assets:
            assets.append(unit_mapping)
        mapping = _measurement_resolution_mapping(args)
        if mapping is not None:
            assets.append(mapping)
    endpoint_concept_root = str(getattr(args, "endpoint_concept_root", "") or "")
    if endpoint_concept_root:
        root = Path(endpoint_concept_root)
        assets.extend(sorted(root.glob("*.json")))
    return tuple(assets)


def _source_universe_paths(
    policy: StarlingTaskPolicy,
) -> tuple[Path, tuple[Path, ...], dict[str, Any]] | None:
    mapping_path = getattr(policy, "source_universe_mapping", None)
    if mapping_path is None:
        return None
    mapping_path = Path(mapping_path)
    manifest_path = mapping_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") == "source_uid_levels.v3":
        spec = manifest.get("tasks", {}).get(policy.task_id)
    elif (
        manifest.get("version") == "gold_original_level_mapping.v1"
        and manifest.get("task") == policy.task_id
    ):
        output = manifest.get("outputs", {}).get("level_mapping", {})
        spec = {
            "path": output.get("path"),
            "parts": output.get("parts", []),
            "mapped_rows": output.get("rows"),
            "unique_uids": output.get("rows"),
            "upstream_commit": manifest.get("upstream", {}).get("upstream_commit"),
            "rows_by_level": output.get("rows_by_level", {}),
        }
    else:
        raise ValueError("source universe manifest version changed")
    if spec is None or spec.get("path") != mapping_path.name:
        raise ValueError(f"source universe manifest mismatch for {policy.task_id}")
    part_paths = tuple(mapping_path / part["path"] for part in spec.get("parts", ()))
    if not part_paths:
        raise ValueError(f"source universe has no parts for {policy.task_id}")
    return manifest_path, part_paths, spec


def _load_source_universe(
    policy: StarlingTaskPolicy, inputs: dict[str, Path]
) -> tuple[set[str] | None, dict[str, Any] | None]:
    resolved = _source_universe_paths(policy)
    if resolved is None:
        return None, None
    manifest_path, part_paths, spec = resolved
    expected_schema = (
        "source_row_uid",
        "canonical_record_id",
        "source_group_id",
        "family_key",
        "level",
    )
    for index, (path, part) in enumerate(zip(part_paths, spec["parts"], strict=True)):
        if (
            path.stat().st_size != part["size_bytes"]
            or file_sha256(path) != part["sha256"]
        ):
            raise ValueError(f"source universe part changed: {path}")
        if tuple(pq.read_schema(path).names) != expected_schema:
            raise ValueError(f"source universe schema changed: {path}")
        inputs[f"source_universe_part_{index:05d}"] = path
    inputs["source_universe_manifest"] = manifest_path
    values = pq.read_table(
        Path(policy.source_universe_mapping), columns=[UID_FIELD]
    )[UID_FIELD].to_pandas()
    if values.isna().any() or values.duplicated().any():
        raise ValueError("source universe contains missing or duplicate UIDs")
    if not values.str.fullmatch(r"sr_[0-9a-f]{32}").all():
        raise ValueError("source universe contains an invalid UID")
    if len(values) != spec["mapped_rows"] or values.nunique() != spec["unique_uids"]:
        raise ValueError("source universe row counts disagree with its manifest")
    return set(values), {
        "version": "source_uid_level_membership.v1",
        "mapping_path": str(policy.source_universe_mapping),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "upstream_commit": spec["upstream_commit"],
        "mapped_rows": int(spec["mapped_rows"]),
        "rows_by_level": spec["rows_by_level"],
        "selection_key": UID_FIELD,
        "levels_attached_to_records": False,
    }


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
    clean_outputs = (
        STAGE_OUTPUT_FILENAMES["clean"]
        + (
            (
                SOURCE_VALUE_CLEANING_AUDIT_FILENAME,
                SOURCE_VALUE_CLEANING_MANIFEST_FILENAME,
            )
            if policy is not None
            and (
                policy.source_value_cleaner is not None
                or policy.stage1_canonical_deduplicator is not None
            )
            else ()
        )
        + (
            (ENDPOINT_UNIT_PROFILE_FILENAME,)
            if policy is not None
            and (
                policy.measurement_resolution_enabled
                or policy.stage1_measurement_routing_enabled
            )
            else ()
        )
        + (
            (STRUCTURE_REJECTIONS_FILENAME,)
            if policy is not None and policy.record_contract is not None
            else ()
        )
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
    "organize": (
        RECORDS_FILENAME,
        "03_records/manifest.json",
        ORGANIZATION_STAGE_VERSION,
    ),
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
    state = _new_run_state(policy, args)
    if _cache_hit(state):
        return 0
    _run_source_stage(state)
    if _stage_finished(state, "source"):
        return _finish_run_stage(state)
    _run_clean_stage(state)
    if _stage_finished(state, "clean"):
        return _finish_run_stage(state)
    _run_normalize_stage(state)
    if _stage_finished(state, "normalize"):
        return _finish_run_stage(state)
    _run_organize_stage(state)
    if _stage_finished(state, "organize"):
        return _finish_run_stage(state)
    return _run_index_stage(state)


def _new_run_state(
    policy: StarlingTaskPolicy, args: argparse.Namespace
) -> SimpleNamespace:
    out_dir = Path(ensure_dir(args.out_dir))
    paths = _stage_paths(policy)
    return SimpleNamespace(
        policy=policy,
        args=args,
        started=time.monotonic(),
        unit_policy_manifest=contextual_unit_policy_manifest(),
        qualifier_manifest=qualifier_vocabulary_manifest(),
        out_dir=out_dir,
        first_stage=STAGES.index(args.from_stage),
        final_stage=STAGES.index(args.through_stage),
        invalidated_artifacts=[],
        stage_paths=paths,
        normalized_records_filename=paths["normalized_records"],
        cleaned=[],
        cleaned_persisted=[],
        normalized=[],
        normalized_persisted=[],
        structure_rejection_ids=set(),
        digests=FileDigestCache(),
        source_inventory=_read_optional_json(out_dir / SOURCE_INVENTORY_FILENAME),
        endpoint_inventories=_read_optional_json(out_dir / ENDPOINT_INVENTORY_FILENAME),
        prepared_sources=None,
        resolution_mapping=None,
        measurement_resolution_audit=None,
    )


def _cache_hit(state: SimpleNamespace) -> bool:
    cached = _read_optional_json(state.out_dir / MANIFEST_FILENAME).get(
        "record_build_cache"
    )
    cached_stage = str((cached or {}).get("completed_stage") or "")
    matches = (
        state.args.cache_mode == "auto"
        and state.args.validation_level == "strict"
        and state.first_stage == STAGES.index("source")
        and cached_stage in STAGES
        and cached_stage == state.args.through_stage
        and cache_metadata_matches(
            cached,
            task_id=state.policy.task_id,
            completed_stage=cached_stage,
            args=state.args,
            digests=state.digests,
            scientific_assets=_scientific_assets(state.policy, state.args),
        )
    )
    if matches:
        _log(f"cache hit through {cached_stage}: content_key={cached['content_key']}")
    return matches


def _stage_finished(state: SimpleNamespace, stage: str) -> bool:
    return state.final_stage == STAGES.index(stage)


def _finish_run_stage(state: SimpleNamespace) -> int:
    return _finish_partial(
        state.policy,
        state.out_dir,
        state.args,
        state.source_inventory,
        state.endpoint_inventories,
        state.started,
        state.invalidated_artifacts,
    )


def _run_source_stage(state: SimpleNamespace) -> None:
    if state.first_stage > STAGES.index("source"):
        _verify_stage_artifact(state.out_dir, "source", state.policy)
        return
    state.prepared_sources = _prepare_source_inputs(state.policy, state.args)
    _, state.source_inventory, state.endpoint_inventories, inputs = (
        state.prepared_sources
    )
    _publish_source_stage(state, inputs)
    count = sum(state.source_inventory["source_row_counts"].values())
    _log(f"source: records={count:,}")


def _publish_source_stage(state: SimpleNamespace, inputs: dict[str, Path]) -> None:
    with _temporary_stage_directory(state.out_dir, "source") as stage_dir:
        _write_json(stage_dir / SOURCE_INVENTORY_FILENAME, state.source_inventory)
        _write_json(stage_dir / ENDPOINT_INVENTORY_FILENAME, state.endpoint_inventories)
        _write_staged_stage_manifest(
            stage_dir,
            state.out_dir,
            "source",
            policy=state.policy,
            inputs=inputs,
            output_filename=SOURCE_INVENTORY_FILENAME,
            row_counts={
                "source_records": sum(
                    state.source_inventory["source_row_counts"].values()
                )
            },
            validations={
                "source_hashes_recorded": True,
                "source_row_counts_verified": True,
                "source_rows_immutable": True,
            },
            sidecars={"endpoint_inventory": stage_dir / ENDPOINT_INVENTORY_FILENAME},
        )
        state.invalidated_artifacts.extend(
            _commit_stage_outputs(state.out_dir, "source", stage_dir, state.policy)
        )


def _run_clean_stage(state: SimpleNamespace) -> None:
    if state.first_stage > STAGES.index("clean"):
        state.cleaned_persisted = (
            _load_verified_stage(state.out_dir, "clean", state.policy)
            if state.first_stage == STAGES.index("normalize")
            else []
        )
        state.cleaned = state.cleaned_persisted
        if state.cleaned and state.policy.record_contract:
            state.cleaned = [
                state.policy.record_contract.inflate_cleaned(row)
                for row in state.cleaned
            ]
        return
    _require_current_source_snapshot(state)
    _prepare_clean_stage(state)
    _publish_clean_stage(state)
    _log(f"clean: records={len(state.cleaned):,}")


def _require_current_source_snapshot(state: SimpleNamespace) -> None:
    if state.prepared_sources is not None:
        return
    state.prepared_sources = _prepare_source_inputs(state.policy, state.args)
    _, current_inventory, current_endpoints, _ = state.prepared_sources
    if (
        current_inventory != state.source_inventory
        or current_endpoints != state.endpoint_inventories
    ):
        raise ValueError("current sources differ from the frozen Stage-00 snapshot")


def _prepare_clean_stage(state: SimpleNamespace) -> None:
    (
        state.cleaned,
        state.cleaning_inputs,
        state.source_value_cleaning_audit,
        state.source_value_cleaning_manifest,
    ) = _clean_prepared_sources(state.policy, state.args, state.prepared_sources)
    _apply_gold_v1_voter_protection(state)
    state.structure_rejections = []
    if state.policy.record_contract:
        state.cleaned, state.structure_rejections = _retain_valid_structures(
            state.cleaned
        )
        state.structure_rejection_ids = {
            str(row.get("cleaned_record_id") or "")
            for row in state.structure_rejections
        }
    state.cleaned = state.policy.attach_source_columns(state.cleaned)
    _validate_stage1_routes(state.policy, state.cleaned)
    if state.policy.measurement_resolution_enabled:
        state.resolution_mapping = _measurement_resolution_mapping(state.args)
        if (
            state.resolution_mapping is None
            and not state.args.allow_partial_measurement_resolution
        ):
            raise ValueError(
                "Finalized Stage 01 requires --measurement-resolution-mapping; "
                "use --allow-partial-measurement-resolution only for a "
                "pre-generation artifact"
            )
        state.measurement_resolution_audit = _apply_stage_measurement_resolution(state)
        _attach_stage1_canonical_measurements(state.cleaned)
    _apply_stage1_canonical_deduplication(state)
    _require_valid_schema(state.cleaned, "clean")
    state.cleaned_persisted = _persist_cleaned_records(state.policy, state.cleaned)
    assert_compact_schema(state.cleaned_persisted)


def _apply_gold_v1_voter_protection(state: SimpleNamespace) -> None:
    protected, path = load_voter_protection(state.args, state.policy.task_id)
    if not protected:
        return
    records = {str(row.get(UID_FIELD) or ""): row for row in state.cleaned}
    missing = set(protected) - set(records)
    if missing:
        raise ValueError(
            "Protected gold-v1 voters are absent after source cleaning: "
            f"{sorted(missing)[:20]}"
        )
    invalid = sorted(
        uid
        for uid in protected
        if not records[uid].get("canonical_smiles")
        or records[uid].get("structure_status") != "resolved"
    )
    if invalid:
        raise ValueError(
            "Protected gold-v1 voters lack a valid main-universe structure: "
            f"{invalid[:20]}"
        )
    state.source_value_cleaning_manifest["gold_v1_voter_protection"] = {
        "version": "gold_v1_voter_structure_protection.v2",
        "path": str(path),
        "sha256": file_sha256(path),
        "protected_rows": len(protected),
        "present_rows": len(set(protected) & set(records)),
        "all_protected_rows_present": True,
        "structure_authority": "pinned_main_source_universe",
        "membership_columns_used": ["task", "source_row_uid"],
        "source_values_mutated": False,
    }
    state.cleaning_inputs["gold_v1_voter_protection"] = path


def _attach_stage1_canonical_measurements(records: list[dict[str, Any]]) -> None:
    for record in records:
        mapped = record.get("measurement_unit_mapping_status") == "mapped"
        record["canonical_measurement_text"] = (
            record.get("resolved_measurement_text") if mapped else None
        )
        record["canonical_unit_text"] = (
            record.get("resolved_unit_text") if mapped else None
        )


def _apply_stage1_canonical_deduplication(state: SimpleNamespace) -> None:
    deduplicator = state.policy.stage1_canonical_deduplicator
    if deduplicator is None:
        return
    before_ids = [str(row.get("cleaned_record_id") or "") for row in state.cleaned]
    result = deduplicator(state.cleaned, state.args)
    after_ids = [str(row.get("cleaned_record_id") or "") for row in result.records]
    dropped_ids = {
        str(row.get("cleaned_record_id") or "")
        for row in result.audit_rows
        if row.get("field") == "record" and row.get("after") == "dropped"
    }
    if [row_id for row_id in before_ids if row_id not in dropped_ids] != after_ids:
        raise ValueError("Stage-01 canonical deduplication changed retained row order")
    if len(before_ids) - len(after_ids) != len(dropped_ids):
        raise ValueError("Stage-01 canonical deduplication lacks row-level audit")
    state.cleaned = result.records
    state.source_value_cleaning_audit.extend(result.audit_rows)
    state.source_value_cleaning_manifest["canonical_deduplication"] = result.manifest
    state.source_value_cleaning_manifest["n_stage1_final_records"] = len(state.cleaned)
    state.source_value_cleaning_manifest["n_stage1_duplicate_records"] = len(
        dropped_ids
    )
    state.cleaning_inputs.update(
        {
            f"stage1_canonical_deduplication_asset_{index}": path
            for index, path in enumerate(result.input_paths, start=1)
        }
    )


def _validate_stage1_routes(
    policy: StarlingTaskPolicy, cleaned: list[dict[str, Any]]
) -> None:
    if not (
        policy.measurement_resolution_enabled
        or policy.stage1_measurement_routing_enabled
    ):
        return
    from data.processing.evidence_library.versions.v10.measurement_routing import (
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
            f"Stage 01 has invalid measurement routing; first={invalid_routes[0]!r}"
        )


def _persist_cleaned_records(
    policy: StarlingTaskPolicy, cleaned: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    attached = policy.attach_source_columns(cleaned)
    if not policy.record_contract:
        return compact_persisted_records(attached)
    for index, row in enumerate(attached):
        attached[index] = compact_persisted_record(
            policy.record_contract.clean_projection(row)
        )
    return attached


def _publish_clean_stage(state: SimpleNamespace) -> None:
    with _temporary_stage_directory(state.out_dir, "clean") as stage_dir:
        write_parquet(stage_dir / CLEANED_FILENAME, state.cleaned_persisted)
        sidecars = _write_clean_sidecars(state, stage_dir)
        route_counts = count_by(state.cleaned, "measurement_resolution_route")
        _write_staged_stage_manifest(
            stage_dir,
            state.out_dir,
            "clean",
            policy=state.policy,
            inputs={
                "source_inventory": state.out_dir / SOURCE_INVENTORY_FILENAME,
                **state.cleaning_inputs,
                **(
                    {
                        "measurement_resolution_mapping": state.resolution_mapping,
                        "exact_measurement_unit_map": _exact_unit_mapping(
                            state.policy, state.args
                        ),
                    }
                    if state.measurement_resolution_audit
                    and state.resolution_mapping
                    else {}
                ),
            },
            output_filename=CLEANED_FILENAME,
            row_counts=_clean_row_counts(state),
            validations=_clean_validations(state, route_counts),
            sidecars=sidecars,
        )
        state.invalidated_artifacts.extend(
            _commit_stage_outputs(
                state.out_dir,
                "clean",
                stage_dir,
                state.policy,
                invalidate_downstream=not state.args.preserve_downstream_artifacts,
            )
        )


def _write_clean_sidecars(state: SimpleNamespace, stage_dir: Path) -> dict[str, Path]:
    sidecars: dict[str, Path] = {}
    if (
        state.policy.source_value_cleaner is not None
        or state.policy.stage1_canonical_deduplicator is not None
    ):
        audit_path = stage_dir / SOURCE_VALUE_CLEANING_AUDIT_FILENAME
        manifest_path = stage_dir / SOURCE_VALUE_CLEANING_MANIFEST_FILENAME
        write_parquet(audit_path, state.source_value_cleaning_audit)
        _write_json(manifest_path, state.source_value_cleaning_manifest)
        sidecars["source_value_cleaning_audit"] = audit_path
        sidecars["source_value_cleaning_manifest"] = manifest_path
    if state.policy.record_contract:
        path = stage_dir / STRUCTURE_REJECTIONS_FILENAME
        write_parquet(path, compact_persisted_records(state.structure_rejections))
        sidecars["structure_rejections"] = path
    if (
        state.policy.measurement_resolution_enabled
        or state.policy.stage1_measurement_routing_enabled
    ):
        sidecars["endpoint_unit_profile"] = _write_endpoint_unit_profile(
            stage_dir / CLEANED_FILENAME,
            stage_dir / ENDPOINT_UNIT_PROFILE_FILENAME,
            task=state.policy.task_id,
            published_records_path=state.out_dir / CLEANED_FILENAME,
        )
    return sidecars


def _clean_row_counts(state: SimpleNamespace) -> dict[str, int]:
    return {
        "cleaned_records": len(state.cleaned),
        "structure_rejections": len(state.structure_rejections),
    }


def _clean_validations(
    state: SimpleNamespace, route_counts: dict[str, int]
) -> dict[str, Any]:
    routed = (
        state.policy.measurement_resolution_enabled
        or state.policy.stage1_measurement_routing_enabled
    )
    duplicate_rows = int(
        (state.source_value_cleaning_manifest.get("canonical_deduplication") or {}).get(
            "duplicates_removed", 0
        )
    )
    reviewed_drops = int(
        state.source_value_cleaning_manifest.get("n_dropped_records", 0)
    )
    source_rows = int(
        state.source_value_cleaning_manifest.get(
            "n_input_records",
            len(state.cleaned) + len(state.structure_rejections) + duplicate_rows,
        )
    )
    return {
        "active_records_have_rdkit_valid_smiles": all(
            row.get("canonical_smiles") and row.get("structure_status") == "resolved"
            for row in state.cleaned
        ),
        "structure_rejections_preserved": True,
        "source_value_cleaning_audited": (
            state.policy.source_value_cleaner is not None
            or state.policy.stage1_canonical_deduplicator is not None
        ),
        "source_value_cleaning_retained_order_and_identity_preserved": True,
        "reviewed_source_row_drops": reviewed_drops,
        "canonical_duplicate_rows": duplicate_rows,
        "source_row_conservation": source_rows
        == len(state.cleaned)
        + reviewed_drops
        + len(state.structure_rejections)
        + duplicate_rows,
        "measurement_routing_is_partition": (
            not routed or sum(route_counts.values()) == len(state.cleaned)
        ),
        "measurement_route_counts": route_counts,
        "measurement_resolution": state.measurement_resolution_audit,
        "canonical_measurement_rows": sum(
            row.get("canonical_measurement_text") is not None
            and row.get("canonical_unit_text") is not None
            for row in state.cleaned
        ),
    }


def _run_normalize_stage(state: SimpleNamespace) -> None:
    if state.first_stage > STAGES.index("normalize"):
        state.normalized_persisted = (
            _load_verified_stage(state.out_dir, "normalize", state.policy)
            if state.first_stage == STAGES.index("organize")
            else []
        )
        state.normalized = state.normalized_persisted
        if state.normalized and state.policy.record_contract:
            _inflate_canonical_records(state.policy, state.normalized)
            state.normalized_persisted = []
        return
    _prepare_normalization_inputs(state)
    documents = _normalize_records_for_stage(state)
    _publish_normalize_stage(state, documents)
    scalar_count = sum(
        row.get("finite_scalar_value") is not None for row in state.normalized
    )
    _log(f"normalize: records={len(state.normalized):,} scalars={scalar_count:,}")
    if state.final_stage > STAGES.index("normalize") and state.policy.record_contract:
        _inflate_canonical_records(state.policy, state.normalized)
        state.normalized_persisted = []


def _prepare_normalization_inputs(state: SimpleNamespace) -> None:
    if not state.cleaned:
        state.cleaned_persisted = _load_verified_stage(
            state.out_dir, "clean", state.policy
        )
        state.cleaned = state.cleaned_persisted
        if state.policy.record_contract:
            state.cleaned = [
                state.policy.record_contract.inflate_cleaned(row)
                for row in state.cleaned
            ]
    if state.policy.record_contract:
        if not state.structure_rejection_ids:
            state.structure_rejection_ids = {
                str(row.get("cleaned_record_id") or "")
                for row in read_parquet_records(
                    state.out_dir / STRUCTURE_REJECTIONS_FILENAME,
                    columns=["cleaned_record_id"],
                )
            }
        _validate_smiles_mapping_unchanged(
            state.policy, state.args, source_inventory=state.source_inventory
        )
    state.cleaned_parent_count = len(state.cleaned)


def _apply_stage_measurement_resolution(state: SimpleNamespace) -> dict | None:
    if not state.policy.measurement_resolution_enabled:
        return None
    return apply_measurement_resolution(
        state.cleaned,
        mapping_path=state.resolution_mapping,
        task=state.policy.task_id,
        unit_mapping_path=_exact_unit_mapping(state.policy, state.args),
        allow_partial=state.args.allow_partial_measurement_resolution,
        allow_out_of_scope_mapping_rows=(
            state.policy.source_universe_mapping is not None
            or bool(getattr(state.args, "authoritative_source_root", None))
        ),
        ignored_record_ids=state.structure_rejection_ids,
        expected_routing_version=MEASUREMENT_ROUTING_VERSION,
        allow_unmapped_source_exact_units=(
            state.policy.allow_unmapped_source_exact_units
        ),
        allow_unmapped_extracted_units=state.args.allow_partial_measurement_resolution,
    )


def _normalize_records_for_stage(state: SimpleNamespace) -> Any:
    hooks = state.policy.build_hooks(state.args)
    input_ids = [str(record.get("cleaned_record_id") or "") for record in state.cleaned]
    if state.policy.record_contract:
        state.normalized, state.normalized_persisted = (
            normalize_and_project_records_ordered(
                state.cleaned,
                hooks=hooks,
                policy=state.policy,
                workers=state.args.workers,
                release_input=state.args.workers == 1,
                retain_working=False,
            )
        )
    else:
        state.normalized = normalize_records_ordered(
            state.cleaned,
            hooks=hooks,
            task=state.policy.task_id,
            workers=state.args.workers,
        )
    _validate_normalized_records(state, hooks, input_ids)
    if not state.policy.record_contract:
        state.normalized = state.policy.attach_source_columns(state.normalized)
        state.normalized_persisted = compact_persisted_records(state.normalized)
    assert_compact_schema(state.normalized_persisted)
    return state.policy.stage_documents(
        args=state.args,
        hooks=hooks,
        normalized=state.normalized,
        persisted=state.normalized_persisted,
        unit_policy_manifest=state.unit_policy_manifest,
    )


def _validate_normalized_records(
    state: SimpleNamespace, hooks: Any, input_ids: list[str]
) -> None:
    identity_errors = validate_cleaned_normalized_identity_ids(
        input_ids, state.normalized
    )
    if identity_errors:
        raise ValueError(
            f"{len(identity_errors)} cleaned/normalized identity failure(s); "
            f"first={identity_errors[0]}"
        )
    if state.args.validation_level == "full":
        pair_errors = validate_measurement_pairs(
            state.normalized,
            hooks.endpoint_standardizer,
            hooks.source_measurement_resolver,
            hooks.contextual_standardizer,
            task=state.policy.task_id,
        )
        if pair_errors:
            raise ValueError(
                f"{len(pair_errors)} measurement/unit pair invariant failure(s); "
                f"first={pair_errors[0]}"
            )
        _validate_final_assay_transfer_measurements(state)
    if not state.policy.record_contract:
        _require_valid_schema(state.normalized, "normalize")


def _validate_final_assay_transfer_measurements(state: SimpleNamespace) -> None:
    if state.policy.assay_transfer_measurement_policy is None:
        return
    from data.processing.evidence_library.shared.v2.assay_transfer_measurements import (
        validate_final_assay_transfer_measurements,
    )

    errors = validate_final_assay_transfer_measurements(state.normalized)
    if errors:
        raise ValueError(
            f"{len(errors)} final assay-transfer measurement failure(s); "
            f"first={errors[0]}"
        )


def _publish_normalize_stage(state: SimpleNamespace, documents: Any) -> None:
    with _temporary_stage_directory(state.out_dir, "normalize") as stage_dir:
        write_parquet(
            stage_dir / state.normalized_records_filename,
            state.normalized_persisted,
        )
        _write_stage2_documents(state, stage_dir, documents)
        _write_staged_stage_manifest(
            stage_dir,
            state.out_dir,
            "normalize",
            policy=state.policy,
            inputs=_normalization_manifest_inputs(state),
            output_filename=state.normalized_records_filename,
            row_counts=_normalization_row_counts(state),
            validations=_normalization_validations(state, documents),
        )
        state.invalidated_artifacts.extend(
            _commit_stage_outputs(state.out_dir, "normalize", stage_dir, state.policy)
        )


def _write_stage2_documents(
    state: SimpleNamespace, stage_dir: Path, documents: Any
) -> None:
    paths = state.stage_paths
    _write_json(stage_dir / paths["validity_policy"], documents.validity_policy)
    _write_json(
        stage_dir / paths["auxiliary_manifest"],
        documents.auxiliary_mapping_manifest,
    )
    _write_json(stage_dir / paths["source_contract"], documents.source_column_contract)
    endpoint_registry = documents.endpoint_registry or {
        source_id: inventory["endpoints"]
        for source_id, inventory in state.endpoint_inventories.items()
    }
    _write_json(stage_dir / paths["endpoint_registry"], endpoint_registry)
    if not state.policy.reference_semantics_enabled:
        return
    if documents.reference_semantics_manifest is None:
        raise ValueError(
            "reference semantics are enabled but Stage 02 supplied no manifest"
        )
    _write_json(
        stage_dir / paths["reference_semantics_manifest"],
        documents.reference_semantics_manifest,
    )


def _normalization_manifest_inputs(state: SimpleNamespace) -> dict[str, Path]:
    inputs = {
        "cleaned_records": state.out_dir / CLEANED_FILENAME,
        "contextual_unit_policy": state.unit_policy_manifest["path"],
        "qualifier_vocabulary": state.qualifier_manifest["path"],
    }
    if getattr(state.args, "auxiliary_mapping", None):
        inputs["auxiliary_mapping"] = Path(state.args.auxiliary_mapping)
    if getattr(state.args, "reference_semantics_mapping", None):
        inputs["reference_semantics_mapping"] = Path(
            state.args.reference_semantics_mapping
        )
    if state.policy.assay_transfer_measurement_policy is not None:
        inputs["assay_transfer_measurement_policy"] = (
            state.policy.assay_transfer_measurement_policy
        )
    return inputs


def _normalization_row_counts(state: SimpleNamespace) -> dict[str, int]:
    return {
        "cleaned_records": state.cleaned_parent_count,
        "measurement_input_records": len(state.cleaned),
        "normalized_records": len(state.normalized),
        "finite_scalars": sum(
            row.get("finite_scalar_value") is not None for row in state.normalized
        ),
        "absolute_and_continuous": sum(
            is_absolute_continuous(row) for row in state.normalized
        ),
        "normalization_valid": sum(
            (
                row.get("normalization_validity_status")
                or row.get("canonicalization_status")
            )
            == "valid"
            for row in state.normalized
        ),
    }


def _normalization_validations(
    state: SimpleNamespace, documents: Any
) -> dict[str, Any]:
    policy = state.policy.assay_transfer_measurement_policy
    validations = {
        **documents.validations,
        "measurement_unit_pair_validation": (
            "passed" if state.args.validation_level == "full" else "not_run"
        ),
        "final_assay_transfer_measurement_validation": (
            "passed"
            if state.args.validation_level == "full" and policy is not None
            else "not_applicable"
            if policy is None
            else "not_run"
        ),
    }
    return validations


def _inflate_canonical_records(
    policy: StarlingTaskPolicy, records: list[dict[str, Any]]
) -> None:
    for index, row in enumerate(records):
        records[index] = policy.record_contract.inflate_canonical(row)


def _run_organize_stage(state: SimpleNamespace) -> None:
    if state.first_stage <= STAGES.index("organize"):
        _prepare_records_for_organization(state)
        _organize_records(state)
        _publish_organize_stage(state)
        _log(
            f"organize: records={len(state.records):,} "
            f"retrieval_eligible={state.organization_stats['n_retrieval_eligible']:,}"
        )
    else:
        _load_organized_stage(state)
    state.frozen_census = {
        "n_cleaned_records": sum(
            int(value)
            for value in (
                state.source_inventory.get("source_row_counts") or {}
            ).values()
        ),
        **state.organization_stats,
        **state.census_extras,
        "normalization_validity_status_counts": (
            state.normalization_validity_status_counts
        ),
    }


def _prepare_records_for_organization(state: SimpleNamespace) -> None:
    if not state.normalized:
        state.normalized_persisted = _load_verified_stage(
            state.out_dir, "normalize", state.policy
        )
        state.normalized = state.normalized_persisted
        if state.policy.record_contract:
            _inflate_canonical_records(state.policy, state.normalized)
            state.normalized_persisted = []
    if not state.args.frozen_retrieval_normalized_records:
        return
    from data.processing.evidence_library.shared.v2.retrieval_boundary import (
        freeze_normalized_retrieval_identity,
    )

    state.normalized = freeze_normalized_retrieval_identity(
        state.normalized, state.args.frozen_retrieval_normalized_records
    )


def _organize_records(state: SimpleNamespace) -> None:
    (
        state.records,
        state.duplicates,
        state.exclusions,
        state.organization_stats,
    ) = organize_normalized_records(
        state.normalized,
        endpoint_identity_required_sources=(
            state.policy.endpoint_identity_required_sources
        ),
        reuse_mutable_records=True,
    )
    _require_valid_schema(state.records, "organize")
    state.distribution_rows = scalar_distribution_audit(state.records)
    state.census_extras = (
        state.policy.census_extras(state.records) if state.policy.census_extras else {}
    )
    state.normalization_validity_status_counts = count_by(
        state.records, "normalization_validity_status"
    )
    if state.normalized is not state.records and isinstance(state.normalized, list):
        state.normalized.clear()
    state.records = _persist_organized_records(state)
    assert_compact_schema(state.records)


def _persist_organized_records(state: SimpleNamespace) -> list[dict[str, Any]]:
    if state.policy.record_contract:
        required = {
            field
            for bucket in state.policy.record_contract.pair_buckets.values()
            for field in (
                *bucket.canonical_dimensions,
                *bucket.core_context_dimensions,
                *bucket.variance_candidates,
            )
        }
        for index, record in enumerate(state.records):
            projected = state.policy.record_contract.canonical_projection(record)
            if index == 0:
                for field in required:
                    projected.setdefault(field, None)
            state.records[index] = compact_persisted_record(projected)
        persisted = state.records
    else:
        persisted = compact_persisted_records(state.records)
    return _freeze_retrieval_records(state.args, persisted)


def _freeze_retrieval_records(
    args: argparse.Namespace, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not args.frozen_retrieval_records:
        return records
    from data.processing.evidence_library.shared.v2.retrieval_boundary import (
        freeze_retrieval_boundary,
    )

    return freeze_retrieval_boundary(records, args.frozen_retrieval_records)


def _publish_organize_stage(state: SimpleNamespace) -> None:
    with _temporary_stage_directory(state.out_dir, "organize") as stage_dir:
        write_parquet(stage_dir / RECORDS_FILENAME, state.records)
        write_parquet(stage_dir / DUPLICATES_FILENAME, state.duplicates)
        write_parquet(stage_dir / REJECTIONS_FILENAME, state.exclusions)
        write_parquet(stage_dir / DISTRIBUTION_AUDIT_FILENAME, state.distribution_rows)
        _write_staged_stage_manifest(
            stage_dir,
            state.out_dir,
            "organize",
            policy=state.policy,
            inputs=_organization_manifest_inputs(state),
            output_filename=RECORDS_FILENAME,
            row_counts=state.organization_stats,
            validations={
                "cross_source_deduplication": False,
                "row_deduplication": False,
                "row_cardinality_preserved": (
                    state.organization_stats["n_records"]
                    == state.organization_stats["n_normalized_records"]
                ),
                "unusable_structures_retained_in_records": True,
            },
        )
        state.invalidated_artifacts.extend(
            _commit_stage_outputs(state.out_dir, "organize", stage_dir, state.policy)
        )


def _organization_manifest_inputs(state: SimpleNamespace) -> dict[str, Path]:
    inputs = {"normalized_records": state.out_dir / state.normalized_records_filename}
    if state.args.frozen_retrieval_records:
        inputs["frozen_retrieval_records"] = state.args.frozen_retrieval_records
    if state.args.frozen_retrieval_normalized_records:
        inputs["frozen_retrieval_normalized_records"] = (
            state.args.frozen_retrieval_normalized_records
        )
    return inputs


def _load_organized_stage(state: SimpleNamespace) -> None:
    state.records = _load_verified_stage(state.out_dir, "organize", state.policy)
    if state.policy.record_contract:
        state.records = [
            state.policy.record_contract.inflate_canonical(row) for row in state.records
        ]
    duplicates = read_parquet_records(state.out_dir / DUPLICATES_FILENAME)
    state.organization_stats = _loaded_organization_stats(state.records, duplicates)
    state.census_extras = (
        state.policy.census_extras(state.records) if state.policy.census_extras else {}
    )
    state.normalization_validity_status_counts = count_by(
        state.records, "normalization_validity_status"
    )


def _loaded_organization_stats(
    records: list[dict[str, Any]], duplicates: list[dict[str, Any]]
) -> dict[str, int | bool]:
    return {
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


def _run_index_stage(state: SimpleNamespace) -> int:
    families, bridge = build_relational_evidence_catalog(
        state.records,
        max_record_examples=state.args.max_record_examples,
        family_resolver=state.policy.family_resolver,
    )
    profile = state.policy.compact_profile_for_contract(
        state.policy.record_contract.version if state.policy.record_contract else ""
    )
    with _temporary_stage_directory(state.out_dir, "index") as stage_dir:
        index_manifest = _write_index_artifacts(
            state, stage_dir, profile, families, bridge
        )
        manifest = _final_manifest(state, stage_dir, profile, families, index_manifest)
        _write_json(stage_dir / MANIFEST_FILENAME, manifest)
        state.invalidated_artifacts.extend(
            _commit_stage_outputs(state.out_dir, "index", stage_dir, state.policy)
        )
    _log(
        f"complete: records={len(state.records):,} "
        f"molecule-family rows={len(families):,} "
        f"index molecules={index_manifest['molecules']:,} out={state.out_dir}"
    )
    return 0


def _write_index_artifacts(
    state: SimpleNamespace,
    stage_dir: Path,
    profile: Any,
    families: list[dict[str, Any]],
    bridge: list[dict[str, Any]],
) -> dict[str, Any]:
    write_parquet(stage_dir / EVIDENCE_FAMILIES_FILENAME, families)
    write_parquet(stage_dir / EVIDENCE_BRIDGE_FILENAME, bridge)
    index_manifest = write_compact_neighbor_index(
        profile=profile,
        families=families,
        output_dir=stage_dir / "05_neighbor_index",
        workers=state.args.workers,
        progress_every=state.args.progress_every,
    )
    _write_json(
        stage_dir / EVIDENCE_MANIFEST_FILENAME,
        {
            "artifact_version": profile.artifact_version,
            "families": len(families),
            "record_references": len(bridge),
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
    return index_manifest


def _index_artifact_filenames(state: SimpleNamespace) -> dict[str, str]:
    paths = state.stage_paths
    filenames = {
        "cleaned_records": CLEANED_FILENAME,
        "normalized_records": state.normalized_records_filename,
        "records": RECORDS_FILENAME,
        "duplicates": DUPLICATES_FILENAME,
        "organization_exclusions": REJECTIONS_FILENAME,
        "molecule_families": EVIDENCE_FAMILIES_FILENAME,
        "molecule_family_records": EVIDENCE_BRIDGE_FILENAME,
        "scalar_distribution_audit": DISTRIBUTION_AUDIT_FILENAME,
        "record_validity_policy": paths["validity_policy"],
        "auxiliary_mapping_manifest": paths["auxiliary_manifest"],
        "source_column_contracts": paths["source_contract"],
        "index_molecules": INDEX_MOLECULES_FILENAME,
        "index_fingerprints": INDEX_FINGERPRINTS_FILENAME,
        "index_membership": INDEX_MEMBERSHIP_FILENAME,
    }
    if state.policy.reference_semantics_enabled:
        filenames["reference_semantics_manifest"] = paths[
            "reference_semantics_manifest"
        ]
    if state.policy.source_value_cleaner is not None:
        filenames["source_value_cleaning_audit"] = SOURCE_VALUE_CLEANING_AUDIT_FILENAME
    return filenames


def _final_manifest(
    state: SimpleNamespace,
    stage_dir: Path,
    profile: Any,
    families: list[dict[str, Any]],
    index_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = _release_manifest_metadata(state, profile)
    manifest.update(
        {
            "invalidated_artifacts": sorted(set(state.invalidated_artifacts)),
            "source_inventory": state.source_inventory,
            "endpoint_inventory": state.endpoint_inventories,
            "stats": _final_stats(state, families, index_manifest),
            "groups": index_manifest["groups"],
            "fingerprint": fingerprint_metadata(),
            "artifact_hashes": _artifact_hashes(state, stage_dir),
            "record_build_cache": _record_build_cache(
                state.policy,
                state.out_dir,
                state.args,
                completed_stage="index",
                digests=state.digests,
                staged_stage_dir=stage_dir,
            ),
            "elapsed_s": round(time.monotonic() - state.started, 3),
        }
    )
    return manifest


def _release_manifest_metadata(state: SimpleNamespace, profile: Any) -> dict[str, Any]:
    return {
        "artifact_version": (
            CANONICAL_ARTIFACT_VERSION
            if state.policy.record_contract
            else NORMALIZED_ARTIFACT_VERSION
        ),
        "compact_artifact_version": profile.artifact_version,
        "record_version": (
            CANONICAL_RECORD_VERSION
            if state.policy.record_contract
            else NORMALIZED_RECORD_VERSION
        ),
        "record_contract": (
            state.policy.record_contract.manifest()
            if state.policy.record_contract
            else None
        ),
        "scalar_parser_version": SCALAR_PARSER_VERSION,
        "unit_normalizer_version": UNIT_NORMALIZER_VERSION,
        "measurement_resolution": _measurement_resolution_manifest(
            state.policy, state.args
        ),
        "contextual_unit_policy": state.unit_policy_manifest,
        "qualifier_vocabulary": state.qualifier_manifest,
        "index_version": profile.index_version,
        **state.policy.manifest_versions(),
        "completed_stages": list(STAGES),
        "rebuild_request": {
            "from_stage": state.args.from_stage,
            "through_stage": state.args.through_stage,
        },
    }


def _final_stats(
    state: SimpleNamespace,
    families: list[dict[str, Any]],
    index_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        **state.frozen_census,
        "n_input_rows": sum(
            int(value)
            for value in (
                state.source_inventory.get("source_row_counts") or {}
            ).values()
        ),
        "n_molecule_family_evidence_rows": len(families),
        "n_index_molecules": index_manifest["molecules"],
    }


def _artifact_hashes(state: SimpleNamespace, stage_dir: Path) -> dict[str, str]:
    return {
        name: file_sha256(
            stage_dir / filename
            if (stage_dir / filename).exists()
            else state.out_dir / filename
        )
        for name, filename in _index_artifact_filenames(state).items()
    }


def _prepare_source_inputs(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
) -> tuple[
    list[tuple[Any, Any, str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    if getattr(args, "authoritative_source_root", None):
        return _prepare_authoritative_source_inputs(policy, args)
    legacy_ids = validate_task_sources(policy.task_id)
    inputs: dict[str, Path] = {}
    universe_uids, universe_receipt = _load_source_universe(policy, inputs)
    mapping_spec, mapping_sha = _validated_smiles_mapping_spec(policy, args, inputs)
    profiles = list(policy.source_profiles(Path(args.starling_data_dir)))
    frames, source_hashes, endpoints, identifiers = _load_profile_sources(
        policy, args, profiles, legacy_ids, inputs, universe_uids
    )
    smiles_mapping = (
        load_smiles_mapping(mapping_spec.path, identifiers) if mapping_spec else {}
    )
    batches = [
        (
            profile,
            frames[profile.source_id],
            source_hashes[profile.source_id],
            smiles_mapping or None,
        )
        for profile in profiles
    ]
    extras = _append_extra_sources(
        policy, args, legacy_ids, inputs, endpoints, source_hashes, batches
    )
    inventory = _source_inventory(
        policy,
        batches,
        source_hashes,
        extras,
        mapping_spec,
        mapping_sha,
        smiles_mapping,
        universe_receipt,
    )
    return batches, inventory, endpoints, inputs


def _prepare_authoritative_source_inputs(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
) -> tuple[
    list[tuple[Any, Any, str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    """Adapt one imported main-tree record universe back to source rows."""
    root = Path(args.authoritative_source_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "main_source_universe.v1":
        raise ValueError(f"unsupported authoritative source manifest: {manifest_path}")
    spec = (manifest.get("tasks") or {}).get(policy.task_id)
    if spec is None:
        raise ValueError(f"authoritative source manifest lacks {policy.task_id}")
    records_path = root / str(spec["path"])
    if file_sha256(records_path) != spec.get("sha256"):
        raise ValueError(f"authoritative source hash mismatch: {records_path}")
    parquet = pq.ParquetFile(records_path)
    if parquet.metadata.num_rows != int(spec.get("rows") or -1):
        raise ValueError(f"authoritative source row-count mismatch: {records_path}")
    if policy.record_contract is None:
        raise ValueError("authoritative source adaptation requires a record contract")

    inputs = {
        "authoritative_source_manifest": manifest_path,
        "authoritative_source_records": records_path,
    }
    protected = set(load_voter_protection(args, policy.task_id)[0])
    mapping_spec, mapping_sha = _validated_smiles_mapping_spec(policy, args, inputs)
    profiles = {profile.source_id: profile for profile in policy.source_profiles(Path("."))}
    for source_id, contract_profile in policy.record_contract.sources.items():
        if source_id in profiles:
            continue
        name_fields = tuple(
            field
            for field in ("molecule_name", "agent_name", "entity_name", "global_identifier")
            if field in contract_profile.source_columns
        )
        record_id_field = next(
            (
                field
                for field in ("source_index", "extraction_id")
                if field in contract_profile.source_columns
            ),
            "source_record_id",
        )
        excluded = {
            contract_profile.endpoint_field,
            contract_profile.measurement_field,
            contract_profile.unit_field,
            contract_profile.smiles_field,
            record_id_field,
            "confidence",
            "support_text",
            *name_fields,
        }
        profiles[source_id] = NormalizedSourceProfile(
            source_id=source_id,
            source_name=f"main/{policy.task_id}/{source_id}",
            endpoint_field=contract_profile.endpoint_field,
            endpoint_constant=contract_profile.endpoint_constant,
            measurement_field=contract_profile.measurement_field,
            unit_field=contract_profile.unit_field,
            unit_constant=contract_profile.unit_constant,
            embedded_unit=not (
                contract_profile.unit_field or contract_profile.unit_constant
            ),
            smiles_field=contract_profile.smiles_field,
            structure_mode="direct",
            record_id_field=record_id_field,
            name_fields=name_fields,
            context_fields=tuple(
                field
                for field in contract_profile.source_columns
                if field not in excluded
            ),
        )
    source_manifest_path = (
        Path(__file__).parent / "tasks" / policy.task_id / "source_manifest.json"
    )
    source_aliases = {}
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_aliases = {
            str(source_spec["directory"]): source_id
            for source_id, source_spec in source_manifest.get("sources", {}).items()
        }
    batches: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in profiles}
    source_columns = set(parquet.schema_arrow.names)
    smiles_mapping = (
        load_smiles_mapping(mapping_spec.path, set())
        if mapping_spec
        else {}
    )
    structure_sources: dict[str, int] = {
        "protected_main_universe": 0,
        "frozen_source_smiles_mapping": 0,
        "main_universe_unmapped_fallback": 0,
        "main_universe_base": 0,
    }

    def source_value(
        record: dict[str, Any],
        source_name: str,
        canonical_name: str,
    ) -> Any:
        """Preserve main's completed source projection, including explicit nulls."""
        return (
            record.get(source_name)
            if source_name in source_columns
            else record.get(canonical_name)
        )

    for record in (
        record
        for arrow_batch in parquet.iter_batches(batch_size=25_000)
        for record in arrow_batch.to_pylist()
    ):
        upstream_source_id = str(record.get("source_id") or "")
        source_id = source_aliases.get(upstream_source_id, upstream_source_id)
        if source_id not in profiles:
            raise ValueError(f"authoritative record has unknown source_id={source_id!r}")
        profile = profiles[source_id]
        contract_profile = policy.record_contract.source(source_id)
        role_values = {
            field: value
            for field, value in (
                (
                    contract_profile.endpoint_field,
                    source_value(
                        record,
                        "endpoint_name",
                        "canonical_endpoint_name",
                    ),
                ),
                (
                    contract_profile.measurement_field,
                    source_value(
                        record,
                        "measurement_text",
                        "canonical_measurement_text",
                    ),
                ),
                (
                    contract_profile.unit_field,
                    source_value(
                        record,
                        "unit_text",
                        "canonical_unit_text",
                    ),
                ),
                (contract_profile.smiles_field, record.get("canonical_smiles")),
            )
            if field
        }
        payload = {
            field: role_values.get(field, record.get(field))
            for field in contract_profile.source_columns
        }
        for field in (
            contract_profile.endpoint_field,
            contract_profile.measurement_field,
            contract_profile.unit_field,
        ):
            if field and record.get(field) != role_values.get(field):
                payload[f"_authoritative_original_{field}"] = record.get(field)
        uid = str(record.get(UID_FIELD) or "")
        if not uid:
            raise ValueError("authoritative record lacks source_row_uid")
        payload[UID_FIELD] = uid
        payload["_source_row_number"] = int(record.get("source_row_number") or 0)
        payload["_legacy_cleaned_record_id"] = str(
            record.get("cleaned_record_id") or ""
        )
        if uid in protected:
            structure = record.get("canonical_smiles")
            structure_sources["protected_main_universe"] += 1
        elif profile.structure_mode == "mapped":
            identifier = str(record.get("global_identifier") or "")
            structure = smiles_mapping.get(identifier)
            if not structure:
                structure = record.get("canonical_smiles")
                structure_sources["main_universe_unmapped_fallback"] += 1
            else:
                structure_sources["frozen_source_smiles_mapping"] += 1
        else:
            structure = record.get("canonical_smiles")
            structure_sources["main_universe_base"] += 1
        payload[profile.smiles_field] = structure
        batches[source_id].append(payload)

    ordered_batches = [
        (
            replace(profile, structure_mode="direct"),
            batches[profile.source_id],
            spec["sha256"],
            None,
        )
        for profile in profiles.values()
        if batches[profile.source_id]
    ]
    endpoints = {}
    for profile in profiles.values():
        if not batches[profile.source_id]:
            continue
        names = _profile_endpoint_names(
            profile, pd.DataFrame(batches[profile.source_id])
        )
        if "endpoint_name" not in source_columns:
            unique = sorted(set(names))
            endpoints[profile.source_id] = {
                "source_id": profile.source_id,
                "count": len(unique),
                "coverage": 1.0,
                "endpoints": unique,
                "status": "authoritative_main_canonical_endpoints",
            }
        else:
            endpoints[profile.source_id] = policy.endpoint_inventory(
                profile.source_id,
                names,
                # The imported universe is already a pinned retrieval subset,
                # not the complete acquisition file used by task-local census checks.
                strict=False,
            )
    counts = {
        source_id: len(rows) for source_id, rows in batches.items() if rows
    }
    inventory = {
        "dataset": policy.dataset_name,
        "source_hashes": {source_id: spec["sha256"] for source_id in profiles},
        "source_row_counts": counts,
        "authoritative_source_seed": {
            "version": manifest["version"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "records_path": str(records_path),
            "records_sha256": spec["sha256"],
            "upstream_commit": manifest["upstream_commit"],
            "rows": sum(counts.values()),
            "structure_policy": "protected_main_then_frozen_nonvoter_repairs.v1",
            "structure_source_counts": structure_sources,
            **({"source_aliases": source_aliases} if source_aliases else {}),
        },
    }
    if mapping_spec is not None:
        inventory["smiles_mapping"] = {
            "path": str(mapping_spec.path),
            "sha256": mapping_sha,
            "expected_sha256": mapping_spec.expected_sha256,
            "n_loaded_identifiers": len(smiles_mapping),
        }
    return ordered_batches, inventory, endpoints, inputs


def _validated_smiles_mapping_spec(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    inputs: dict[str, Path],
) -> tuple[Any, str]:
    spec = policy.smiles_mapping(args) if policy.smiles_mapping else None
    if spec is None:
        return None, ""
    if not spec.path.exists():
        raise FileNotFoundError(f"authoritative SMILES mapping not found: {spec.path}")
    mapping_sha = file_sha256(spec.path)
    if (
        spec.expected_sha256
        and not spec.allow_unpinned
        and mapping_sha != spec.expected_sha256
    ):
        raise ValueError(
            "SMILES mapping hash mismatch: "
            f"expected {spec.expected_sha256}, found {mapping_sha}"
        )
    inputs["smiles_mapping"] = spec.path
    return spec, mapping_sha


def _load_profile_sources(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    profiles: list[Any],
    legacy_ids: dict[str, str],
    inputs: dict[str, Path],
    universe_uids: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any], set[str]]:
    frames: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    endpoint_inventories: dict[str, Any] = {}
    needed_identifiers: set[str] = set()
    seen_universe_uids: set[str] = set()
    for profile in profiles:
        path = Path(profile.source_path)
        inputs[profile.source_id] = path
        if policy.verify_source_digest is not None:
            policy.verify_source_digest(profile.source_id, path)
        source_hashes[profile.source_id] = file_sha256(path)
        frame = pd.read_parquet(path)
        _validate_source_frame(policy, args, profile, path, frame)
        if universe_uids is not None:
            expected_rows = int(policy.expected_source_rows[profile.source_id])
            if len(frame) != expected_rows:
                raise ValueError(
                    f"source row-count drift for {profile.source_id}: "
                    f"expected {expected_rows:,}, found {len(frame):,}"
                )
            if frame[UID_FIELD].duplicated().any():
                raise ValueError(
                    f"source contains duplicate permanent row identity: {path}"
                )
            endpoint_inventories[profile.source_id] = policy.endpoint_inventory(
                profile.source_id,
                _profile_endpoint_names(profile, frame),
                strict=args.strict_endpoint_inventory,
            )
            frame["_source_row_number"] = range(1, len(frame) + 1)
            selected = frame[UID_FIELD].isin(universe_uids)
            selected_uids = set(frame.loc[selected, UID_FIELD])
            duplicate_uids = seen_universe_uids & selected_uids
            if duplicate_uids:
                raise ValueError(
                    f"source universe UID occurs in multiple sources: "
                    f"{min(duplicate_uids)}"
                )
            seen_universe_uids.update(selected_uids)
            frame = frame.loc[selected].copy()
        if args.max_rows_per_source:
            frame = frame.head(args.max_rows_per_source)
        frame["_legacy_cleaned_record_id"] = frame[UID_FIELD].map(legacy_ids)
        frames[profile.source_id] = frame
        needed_identifiers.update(_mapped_profile_identifiers(profile, frame))
        if universe_uids is None:
            endpoint_inventories[profile.source_id] = policy.endpoint_inventory(
                profile.source_id,
                _profile_endpoint_names(profile, frame),
                strict=args.strict_endpoint_inventory,
            )
    if universe_uids is not None and seen_universe_uids != universe_uids:
        missing = universe_uids - seen_universe_uids
        raise ValueError(
            f"source universe has {len(missing)} UIDs absent from raw sources; "
            f"first={min(missing)}"
        )
    return frames, source_hashes, endpoint_inventories, needed_identifiers


def _validate_source_frame(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    profile: Any,
    path: Path,
    frame: pd.DataFrame,
) -> None:
    if UID_FIELD not in frame or frame[UID_FIELD].isna().any():
        raise ValueError(f"source lacks permanent row identity: {path}")
    expected = policy.expected_source_rows[profile.source_id]
    if not args.max_rows_per_source and len(frame) != expected:
        raise ValueError(
            f"source row-count drift for {profile.source_id}: "
            f"expected {expected:,}, found {len(frame):,}"
        )


def _mapped_profile_identifiers(profile: Any, frame: pd.DataFrame) -> set[str]:
    field = "global_identifier"
    if profile.structure_mode != "mapped" or field not in frame:
        return set()
    return {
        str(value)
        for value in frame[field].dropna().unique().tolist()
        if str(value).strip()
    }


def _profile_endpoint_names(profile: Any, frame: pd.DataFrame) -> list[str]:
    if profile.endpoint_constant:
        return [normalize_endpoint_name(profile.endpoint_constant) or ""]
    cleaner = (
        clean_literal_text
        if profile.endpoint_field in profile.literal_text_fields
        else normalize_endpoint_name
    )
    return [
        cleaner(value) or ""
        for value in frame[profile.endpoint_field].drop_duplicates().tolist()
    ]


def _append_extra_sources(
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    legacy_ids: dict[str, str],
    inputs: dict[str, Path],
    endpoint_inventories: dict[str, Any],
    source_hashes: dict[str, str],
    clean_batches: list[tuple[Any, Any, str, Any]],
) -> dict[str, dict[str, Any] | None]:
    if policy.load_extra_source is None:
        return {}
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
    entries = {}
    for extra in extras:
        _prepare_extra_source_rows(extra, legacy_ids)
        inputs[extra.inventory_key] = extra.source_path
        entries[extra.inventory_key] = extra.inventory_entry
        endpoint_inventories[extra.source_id] = policy.endpoint_inventory(
            extra.source_id,
            extra.endpoint_names,
            strict=args.strict_endpoint_inventory,
        )
        clean_batches.append((extra.profile, extra.rows, extra.source_sha256, None))
        source_hashes[extra.source_id] = extra.source_sha256
    return entries


def _prepare_extra_source_rows(
    extra: ExtraSourceBatch, legacy_ids: dict[str, str]
) -> None:
    for row in extra.rows:
        uid = str(row.get(UID_FIELD) or "")
        if not uid:
            raise ValueError(
                f"extra source lacks permanent row identity: {extra.source_path}"
            )
        row["_legacy_cleaned_record_id"] = legacy_ids.get(uid)


def _source_inventory(
    policy: StarlingTaskPolicy,
    batches: list[tuple[Any, Any, str, Any]],
    source_hashes: dict[str, str],
    extra_entries: dict[str, dict[str, Any] | None],
    mapping_spec: Any,
    mapping_sha: str,
    smiles_mapping: dict[str, str],
    universe_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_counts = {batch[0].source_id: len(batch[1]) for batch in batches}
    inventory = {
        "dataset": policy.dataset_name,
        "source_hashes": source_hashes,
        "source_row_counts": selected_counts,
        **extra_entries,
    }
    if universe_receipt is not None:
        raw_rows = sum(int(value) for value in policy.expected_source_rows.values())
        inventory["raw_source_row_counts"] = dict(policy.expected_source_rows)
        inventory["source_universe"] = {
            **universe_receipt,
            "processed_rows": sum(selected_counts.values()),
            "outside_source_universe_rows": raw_rows
            - int(universe_receipt["mapped_rows"]),
        }
    if mapping_spec is not None:
        inventory["smiles_mapping"] = {
            "path": str(mapping_spec.path),
            "sha256": mapping_sha,
            "expected_sha256": mapping_spec.expected_sha256,
            "n_loaded_identifiers": len(smiles_mapping),
        }
    return inventory


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
        if (
            record.get("canonical_smiles")
            and record.get("structure_status") == "resolved"
        ):
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
    for identifier, smiles in frame[["global_identifier", "smiles"]].itertuples(
        index=False
    ):
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
    authoritative = source_inventory.get("authoritative_source_seed") or {}
    if authoritative:
        if authoritative.get("version") != "main_source_universe.v1":
            raise ValueError("unsupported authoritative source structure provenance")
        structure_policy = authoritative.get("structure_policy")
        if structure_policy is None:
            if authoritative.get("structure_source") != "upstream_canonical_smiles":
                raise ValueError("unsupported authoritative source structure provenance")
            return
        if structure_policy != "protected_main_then_frozen_nonvoter_repairs.v1":
            raise ValueError("unsupported authoritative source structure policy")
        if policy.smiles_mapping is None:
            return
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
    records_path: Path,
    output_path: Path,
    *,
    task: str,
    published_records_path: Path | None = None,
) -> Path:
    from data.processing.evidence_library.versions.v10.build_endpoint_unit_profile import (
        build_profile,
    )

    config = import_task_module(task, "starling_measurement_resolution")
    profile = build_profile(
        records_path,
        config.source_routing_rules(),
        task=task,
    )
    profile["records_path"] = str(published_records_path or records_path)
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
        filename for filename in expected_outputs if not (stage_dir / filename).exists()
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
    invalidated = _archive_pre_three_stage_outputs(out_dir, stage)
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
        targets.extend(
            out_dir / directory for directory in RECORD_DEPENDENT_DIRECTORIES
        )
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


def _archive_pre_three_stage_outputs(out_dir: Path, stage: str) -> list[str]:
    if stage not in {"clean", "normalize"}:
        return []
    legacy_root = out_dir / "historical/pre_three_stage_core"
    invalidated = []
    for name in ("03_records", "04_pair_buckets", "05_deduplicated_records"):
        active, archived = out_dir / name, legacy_root / name
        if active.exists() and not archived.exists():
            legacy_root.mkdir(parents=True, exist_ok=True)
            os.replace(active, archived)
            invalidated.append(f"{name} -> historical/pre_three_stage_core/{name}")
    return invalidated


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
    payload = _partial_manifest(
        policy,
        out_dir,
        args,
        source_inventory,
        endpoint_inventories,
        started,
        invalidated_artifacts,
    )
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


def _partial_manifest(
    policy: StarlingTaskPolicy,
    out_dir: Path,
    args: argparse.Namespace,
    source_inventory: dict[str, Any],
    endpoint_inventories: dict[str, Any],
    started: float,
    invalidated_artifacts: list[str],
) -> dict[str, Any]:
    return {
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
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
    _add_build_arguments(parser, policy, stage_choices, default_through_stage)
    _add_downstream_arguments(parser, core_only)
    _add_validation_arguments(parser)
    _add_optional_generation_arguments(parser, core_only)
    _add_measurement_resolution_arguments(parser, policy)
    if policy.add_cli_arguments is not None:
        policy.add_cli_arguments(parser)
    args = parser.parse_args(argv)
    _validate_build_arguments(parser, policy, args, stage_choices, core_only)
    return args


def _add_build_arguments(
    parser: argparse.ArgumentParser,
    policy: StarlingTaskPolicy,
    stage_choices: tuple[str, ...],
    default_through_stage: str,
) -> None:
    parser.add_argument("--starling-data-dir", default=policy.default_data_dir)
    parser.add_argument(
        "--authoritative-source-root",
        type=Path,
        help="Imported main-tree source universe; bypasses task-local raw snapshots.",
    )
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


def _add_downstream_arguments(parser: argparse.ArgumentParser, core_only: bool) -> None:
    if core_only:
        parser.set_defaults(
            frozen_retrieval_records=None,
            frozen_retrieval_normalized_records=None,
            legacy_task_local_downstream=False,
        )
        return
    parser.add_argument("--frozen-retrieval-records", type=Path)
    parser.add_argument("--frozen-retrieval-normalized-records", type=Path)
    parser.add_argument("--legacy-task-local-downstream", action="store_true")


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
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


def _add_optional_generation_arguments(
    parser: argparse.ArgumentParser, core_only: bool
) -> None:
    if core_only:
        return
    parser.add_argument("--semantic-aggregation-max-new-groups", type=int)
    parser.add_argument(
        "--semantic-aggregation-budget-max-tokens", type=int, default=10_000_000
    )


def _add_measurement_resolution_arguments(
    parser: argparse.ArgumentParser, policy: StarlingTaskPolicy
) -> None:
    if not policy.measurement_resolution_enabled:
        return
    default_mapping = import_task_module(
        policy.task_id, "starling_measurement_resolution"
    ).DEFAULT_MAPPING_PATH
    parser.add_argument(
        "--measurement-resolution-mapping",
        default=str(default_mapping) if Path(default_mapping).is_file() else "",
        help=(
            "Frozen offline measurement extraction applied while finalizing "
            "Stage 01 canonical measurement and unit fields."
        ),
    )
    parser.add_argument(
        "--exact-unit-mapping",
        default=str(policy.exact_unit_mapping_path or ""),
        help="Exact unit reconciliation map paired with the measurement extraction.",
    )
    parser.add_argument(
        "--allow-partial-measurement-resolution",
        action="store_true",
        help=(
            "Permit a pre-generation Stage 01 whose extract-routed rows are not "
            "fully canonicalized or whose exact units still require reconciliation; "
            "this output is not publication-ready."
        ),
    )


def _validate_build_arguments(
    parser: argparse.ArgumentParser,
    policy: StarlingTaskPolicy,
    args: argparse.Namespace,
    stage_choices: tuple[str, ...],
    core_only: bool,
) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
