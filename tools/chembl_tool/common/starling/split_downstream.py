"""Shared transactional stages 04-09 for split-aware Starling artifacts.

V7 publishes distance calibration at Stage 05. Frozen v6 inputs continue to
publish their historical assay-transfer policy at the historical stage name.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import inspect
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    write_deterministic_gzip,
)
from tools.chembl_tool.common.starling.build_pair_bucket_distance_calibration import (
    CALIBRATION_FILENAME,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    CompactArtifactProfile,
    build_relational_evidence_catalog,
    load_compact_neighbor_index,
    validate_compact_neighbor_index,
    write_compact_neighbor_index,
)
from tools.chembl_tool.common.starling.build_runtime import (
    FileDigestCache,
    build_cache_metadata,
    cache_metadata_matches,
    parent_identity_map,
)
from tools.chembl_tool.common.starling.heldout_index import load_heldout_identity_keys
from tools.chembl_tool.common.starling.normalization.audit import (
    PARQUET_COMPRESSION_LEVEL,
    read_parquet_records,
    write_parquet,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.task_policy import StarlingTaskPolicy
from tools.chembl_tool.common.task_workflows.evidence_library import (
    standardize_index_molecules,
)


PAIR_BUCKET_STAGE = "04_pair_buckets"
TRANSFER_POLICY_STAGE = "05_assay_transfer_policy"
DISTANCE_CALIBRATION_STAGE = "05_distance_calibration"
HELDOUT_STAGE = "06_remove_heldout_overlap"
MOLECULE_EVIDENCE_STAGE = "07_molecule_evidence"
NEIGHBOR_INDEX_STAGE = "08_neighbor_index"
AUDIT_STAGE = "09_audits"
DOWNSTREAM_STAGES = (
    PAIR_BUCKET_STAGE,
    TRANSFER_POLICY_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    AUDIT_STAGE,
)
V7_DOWNSTREAM_STAGES = (
    PAIR_BUCKET_STAGE,
    DISTANCE_CALIBRATION_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    AUDIT_STAGE,
)


def _downstream_stages(*, v7: bool) -> tuple[str, ...]:
    return V7_DOWNSTREAM_STAGES if v7 else DOWNSTREAM_STAGES
DEFAULT_LEGACY_DOWNSTREAM_STAGES = (
    "04_evidence_catalog",
    "05_neighbor_index",
    "06_pair_buckets",
    "07_assay_transfer_policy",
    "08_audits",
)
# Historical name retained for the existing Bioavailability binding.
LEGACY_DOWNSTREAM_STAGES = DEFAULT_LEGACY_DOWNSTREAM_STAGES
PAIR_BUCKET_RECORDS_FILENAME = "pair_bucket_records.parquet"
PAIR_BUCKET_METADATA_FILENAME = "pair_bucket_metadata.json"
FILTERED_RECORDS_FILENAME = "records.parquet"
MANIFEST_FILENAME = "manifest.json"
EVIDENCE_FAMILIES_FILENAME = "molecule_families.parquet"
EVIDENCE_BRIDGE_FILENAME = "molecule_family_records.parquet"


@dataclass(frozen=True)
class SplitDownstreamSpec:
    task_id: str
    policy: StarlingTaskPolicy
    pipeline_layout_version: str
    heldout_overlap_version: str
    pair_bucket_version: str
    filter_source_id: str
    exclusions_filename: str
    build_sidecar: Callable[..., dict[str, Any]]
    build_transfer_policy: Callable[..., dict[str, Any]]
    benchmark_splits: tuple[str, ...]
    filter_scope_field: str = ""
    filter_scope_value: str = ""
    legacy_downstream_stages: tuple[str, ...] = DEFAULT_LEGACY_DOWNSTREAM_STAGES
    policy_statistics_scope: str = "complete_unfiltered_records"

    @property
    def compact_profile(self) -> CompactArtifactProfile:
        return self.policy.compact

    @property
    def pair_bucket_records_filename(self) -> str:
        return PAIR_BUCKET_RECORDS_FILENAME

    @property
    def pair_bucket_metadata_filename(self) -> str:
        return PAIR_BUCKET_METADATA_FILENAME


def _compact_profile(
    spec: SplitDownstreamSpec, *, v7: bool
) -> CompactArtifactProfile:
    return spec.policy.compact_profile_for_contract(
        spec.policy.record_contract.version
        if v7 and spec.policy.record_contract is not None
        else ""
    )


def build_downstream_artifacts(
    spec: SplitDownstreamSpec,
    *,
    normalized_root: str | Path,
    benchmark_split_root: str | Path,
    workers: int = 1,
    progress_every: int = 10_000,
    max_record_examples: int = 6,
    rebuild_request: Mapping[str, Any] | None = None,
    validation_level: str = "strict",
    cache_mode: str = "auto",
) -> dict[str, Any]:
    """Build all split-aware downstream stages and publish them atomically."""
    started = time.monotonic()
    root = Path(normalized_root)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "03_records/records.parquet"
    v7 = "canonical_record_id" in pq.read_schema(records_path).names
    canonical_stage = (
        "02_canonicalized"
        if (root / "02_canonicalized/auxiliary_mapping_manifest.json").is_file()
        else "02_normalized"
    )
    auxiliary_manifest = root / canonical_stage / "auxiliary_mapping_manifest.json"
    root_manifest_path = root / MANIFEST_FILENAME
    previous_manifest = root_manifest_path.read_bytes() if root_manifest_path.exists() else None
    manifest = json.loads(previous_manifest.decode()) if previous_manifest else {}
    cache_args = SimpleNamespace(
        benchmark_split_root=str(benchmark_split_root),
        max_record_examples=max_record_examples,
    )
    digests = FileDigestCache()
    cached = manifest.get("downstream_build_cache")
    if (
        cache_mode == "auto"
        and validation_level == "strict"
        and cache_metadata_matches(
            cached,
            task_id=spec.task_id,
            completed_stage=AUDIT_STAGE,
            args=cache_args,
            digests=digests,
            scientific_assets=spec.policy.scientific_assets,
        )
    ):
        expected_hashes = manifest.get("downstream_artifact_hashes")
        if expected_hashes == _downstream_hashes(spec, root, v7=v7):
            _validate_published_indices(spec, root, v7=v7)
            print(
                "[build_starling_downstream] cache hit: "
                f"content_key={cached['content_key']}",
                flush=True,
            )
            return manifest
    inputs = _preflight_inputs(
        records_path=records_path,
        auxiliary_manifest=auxiliary_manifest,
        benchmark_split_root=Path(benchmark_split_root),
        splits=spec.benchmark_splits,
    )

    with tempfile.TemporaryDirectory(dir=root, prefix=".downstream-build-") as name:
        candidate = Path(name)
        paths = _stage_paths(candidate, v7=v7)
        _seed_audits(root, paths["audits"])
        pair_metadata = spec.build_sidecar(
            records_path=records_path,
            out_dir=paths["pair_buckets"],
        )
        pair_metadata = _project_paths(pair_metadata, candidate, root)
        _write_json(
            paths["pair_buckets"] / spec.pair_bucket_metadata_filename,
            pair_metadata,
        )
        transfer_kwargs = {
            "records_path": records_path,
            "pair_bucket_records_path": (
                paths["pair_buckets"] / spec.pair_bucket_records_filename
            ),
            "pair_bucket_metadata_path": (
                paths["pair_buckets"] / spec.pair_bucket_metadata_filename
            ),
            "auxiliary_manifest_path": auxiliary_manifest,
            "out_dir": paths["transfer_policy"],
        }
        if "workers" in inspect.signature(spec.build_transfer_policy).parameters:
            transfer_kwargs["workers"] = workers
        transfer = spec.build_transfer_policy(
            **transfer_kwargs,
        )
        transfer = _project_paths(transfer, candidate, root)
        stage_payload_filename = CALIBRATION_FILENAME if v7 else POLICY_FILENAME
        write_deterministic_gzip(
            paths["transfer_policy"] / stage_payload_filename, transfer
        )
        filtered = materialize_filtered_record_views(
            spec,
            records_path=records_path,
            benchmark_split_root=benchmark_split_root,
            out_dir=paths["heldout"],
            workers=workers,
        )
        filtered = _project_paths(filtered, candidate, root)
        _write_json(paths["heldout"] / MANIFEST_FILENAME, filtered)
        evidence = build_filtered_molecule_evidence(
            spec,
            filtered_records_root=paths["heldout"],
            out_dir=paths["evidence"],
            max_record_examples=max_record_examples,
            workers=workers,
        )
        for split, value in evidence.items():
            evidence[split] = _project_paths(value, candidate, root)
            _write_json(paths["evidence"] / split / MANIFEST_FILENAME, evidence[split])
        indices = build_filtered_neighbor_indices(
            spec,
            evidence_root=paths["evidence"],
            overlap_manifest=paths["heldout"] / MANIFEST_FILENAME,
            out_dir=paths["index"],
            workers=workers,
            progress_every=progress_every,
        )
        for split, value in indices.items():
            indices[split] = _project_paths(value, candidate, root)
            _write_json(paths["index"] / split / MANIFEST_FILENAME, indices[split])
        audit = _heldout_audit(
            spec, pair_metadata, filtered, evidence, indices, transfer
        )
        _write_json(paths["audits"] / "heldout_overlap.json", audit)
        manifest.update(
            _manifest_update(
                spec=spec,
                pair_metadata=pair_metadata,
                transfer=transfer,
                audit=audit,
                candidate=candidate,
                rebuild_request=rebuild_request,
                elapsed_s=round(time.monotonic() - started, 3),
                v7=v7,
            )
        )
        downstream_cache = build_cache_metadata(
            task_id=spec.task_id,
            completed_stage=AUDIT_STAGE,
            args=cache_args,
            input_paths=inputs,
            output_paths=_downstream_output_paths(spec, candidate, v7=v7),
            digests=digests,
            scientific_assets=spec.policy.scientific_assets,
        )
        downstream_cache["outputs"] = {
            str(root / Path(path).relative_to(candidate)): sha256
            for path, sha256 in downstream_cache["outputs"].items()
        }
        manifest["downstream_build_cache"] = downstream_cache
        _write_json(candidate / MANIFEST_FILENAME, manifest)
        _validate_candidate(
            spec,
            candidate,
            root,
            v7=v7,
            validation_level=validation_level,
        )
        _verify_inputs(inputs)
        actual_manifest = root_manifest_path.read_bytes() if root_manifest_path.exists() else None
        if actual_manifest != previous_manifest:
            raise RuntimeError("root manifest changed during downstream candidate build")
        _publish_downstream_candidate(spec, root, candidate, v7=v7)
    return manifest


# Descriptive alias retained for callers that adopted the first shared API name.
build_split_downstream_artifacts = build_downstream_artifacts


def materialize_filtered_record_views(
    spec: SplitDownstreamSpec,
    *,
    records_path: str | Path,
    benchmark_split_root: str | Path,
    out_dir: str | Path,
    workers: int = 1,
) -> dict[str, Any]:
    return materialize_split_record_views(
        records_path=records_path,
        benchmark_split_root=benchmark_split_root,
        out_dir=out_dir,
        filter_source_id=spec.filter_source_id,
        filter_scope_field=spec.filter_scope_field,
        filter_scope_value=spec.filter_scope_value,
        splits=spec.benchmark_splits,
        exclusions_filename=spec.exclusions_filename,
        stage_version=spec.heldout_overlap_version,
        policy_statistics_scope=spec.policy_statistics_scope,
        workers=workers,
    )


def materialize_split_record_views(
    *,
    records_path: str | Path,
    benchmark_split_root: str | Path,
    out_dir: str | Path,
    filter_source_id: str,
    filter_scope_field: str = "",
    filter_scope_value: str = "",
    splits: Sequence[str],
    exclusions_filename: str,
    stage_version: str,
    policy_statistics_scope: str,
    workers: int = 1,
) -> dict[str, Any]:
    records_path = Path(records_path)
    split_root = Path(benchmark_split_root)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(records_path)
    record_id_field = (
        "canonical_record_id"
        if "canonical_record_id" in table.column_names
        else "normalized_record_id"
    )
    required = {record_id_field, "source_id", "canonical_smiles"}
    if filter_scope_field:
        required.add(filter_scope_field)
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"Stage-03 records lack heldout-filter columns: {sorted(missing)}")
    record_ids = [str(value or "") for value in table[record_id_field].to_pylist()]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError(
            f"Stage-03 {record_id_field} values must be nonempty and unique"
        )
    source_ids = [str(value or "") for value in table["source_id"].to_pylist()]
    smiles = [str(value or "") for value in table["canonical_smiles"].to_pylist()]
    scope_values = (
        [str(value or "") for value in table[filter_scope_field].to_pylist()]
        if filter_scope_field
        else [""] * table.num_rows
    )
    scoped_indices = [
        index
        for index, source_id in enumerate(source_ids)
        if source_id == filter_source_id
        and (
            not filter_scope_field
            or scope_values[index] == filter_scope_value
        )
    ]
    identities = _bounded_parent_identity_map(
        (smiles[index] for index in scoped_indices), workers=workers
    )
    key_by_index = {
        index: identities.get(smiles[index], "") for index in scoped_indices
    }
    exclusion_rows: list[dict[str, str]] = []
    split_payloads: dict[str, Any] = {}
    prepared_writes: dict[str, tuple[pa.Table, Path]] = {}
    for split in splits:
        split_dir = split_root / split
        train_path = split_dir / "train_molecule_labels.jsonl"
        heldout_path = split_dir / "heldout_molecule_labels.jsonl"
        train_keys = load_heldout_identity_keys(train_path)
        heldout_keys = load_heldout_identity_keys(heldout_path)
        normalize_molecule_identity.cache_clear()
        if train_keys & heldout_keys:
            raise ValueError(f"{split} train and heldout parent identities overlap")
        excluded_indices = {
            index for index, key in key_by_index.items() if key and key in heldout_keys
        }
        excluded_ids = {record_ids[index] for index in excluded_indices}
        for index in sorted(excluded_indices, key=lambda item: record_ids[item]):
            exclusion_rows.append(
                {
                    "benchmark_split": split,
                    record_id_field: record_ids[index],
                    "molecule_identity_key": key_by_index[index],
                }
            )
        mask = np.ones(table.num_rows, dtype=bool)
        if excluded_indices:
            mask[list(excluded_indices)] = False
        filtered = table.filter(pa.array(mask))
        output = target / split / FILTERED_RECORDS_FILENAME
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared_writes[split] = (filtered, output)
        if filtered.schema != table.schema:
            raise ValueError(f"{split} filtered record schema differs from Stage 03")
        filtered_ids = {
            str(value or "") for value in filtered[record_id_field].to_pylist()
        }
        if filtered_ids & excluded_ids:
            raise AssertionError(f"{split} filtered records retain excluded IDs")
        if len(filtered_ids) != table.num_rows - len(excluded_indices):
            raise AssertionError(f"{split} filtering changed records outside the filter scope")
        remaining_overlap = {
            key_by_index[index]
            for index in scoped_indices
            if index not in excluded_indices
            and key_by_index.get(index)
            and key_by_index[index] in heldout_keys
        }
        if remaining_overlap:
            raise AssertionError(f"{split} retains heldout parent identities")
        split_payloads[split] = {
            "train_molecule_labels": {
                "path": str(train_path),
                "sha256": file_sha256(train_path),
                "parent_identities": len(train_keys),
            },
            "heldout_molecule_labels": {
                "path": str(heldout_path),
                "sha256": file_sha256(heldout_path),
                "parent_identities": len(heldout_keys),
            },
            "matched_heldout_parent_identities": len(
                {key_by_index[index] for index in excluded_indices}
            ),
            "heldout_parent_overlap_after_filter": len(remaining_overlap),
            "excluded_filter_source_records": len(excluded_indices),
            "excluded_filter_scope_records": len(excluded_indices),
            f"excluded_{filter_source_id}_records": len(excluded_indices),
            "filtered_records": filtered.num_rows,
            "output": {"path": str(output)},
            "validations": {
                "train_heldout_parent_overlap": 0,
                "only_filter_source_removed": True,
                "only_filter_scope_removed": True,
                f"only_{filter_source_id}_removed": True,
                "excluded_ids_absent": True,
                "schema_matches_stage_03": True,
                "heldout_parent_overlap_zero": not remaining_overlap,
            },
        }
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(2, len(prepared_writes) or 1)
    ) as executor:
        futures = {
            split: executor.submit(_write_filtered_records, filtered, output)
            for split, (filtered, output) in prepared_writes.items()
        }
        for split in splits:
            futures[split].result()
            split_payloads[split]["output"]["sha256"] = file_sha256(
                prepared_writes[split][1]
            )
    exclusions_path = target / exclusions_filename
    frame = pd.DataFrame(
        exclusion_rows,
        columns=("benchmark_split", record_id_field, "molecule_identity_key"),
    ).sort_values(["benchmark_split", record_id_field])
    frame.to_parquet(
        exclusions_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )
    manifest = {
        "stage_version": stage_version,
        "identity_normalizer_version": "rdkit_fragment_parent.v1",
        "filter_source_id": filter_source_id,
        "filter_scope": {
            "field": filter_scope_field or None,
            "value": filter_scope_value if filter_scope_field else None,
        },
        "filter_identity": "parent_inchi_key_then_parent_smiles",
        "input": {
            "path": str(records_path),
            "sha256": file_sha256(records_path),
            "records": table.num_rows,
            "filter_source_records": len(scoped_indices),
            "filter_source_records_total": sum(
                source_id == filter_source_id for source_id in source_ids
            ),
            "filter_scope_records": len(scoped_indices),
        },
        "splits": split_payloads,
        "exclusions": {
            "path": str(exclusions_path),
            "sha256": file_sha256(exclusions_path),
            "rows": len(exclusion_rows),
        },
        "policy_statistics_scope": policy_statistics_scope,
        "validations": {
            "source_complete_stage_03_unchanged": True,
            "only_declared_source_is_filterable": True,
            "only_declared_scope_is_filterable": True,
            "one_materialized_view_per_split": True,
        },
    }
    _write_json(target / MANIFEST_FILENAME, manifest)
    return manifest


def _write_filtered_records(table: pa.Table, output: Path) -> None:
    pq.write_table(
        table,
        output,
        compression="zstd",
        compression_level=PARQUET_COMPRESSION_LEVEL,
        use_dictionary=True,
        write_statistics=True,
    )


def build_filtered_molecule_evidence(
    spec: SplitDownstreamSpec,
    *,
    filtered_records_root: str | Path,
    out_dir: str | Path,
    max_record_examples: int = 6,
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    global _EVIDENCE_BUILD_CONTEXT
    _EVIDENCE_BUILD_CONTEXT = (
        spec,
        Path(filtered_records_root),
        target,
        max_record_examples,
    )
    try:
        split_workers = min(max(1, int(workers or 1)), len(spec.benchmark_splits))
        if split_workers > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(processes=split_workers) as pool:
                results = pool.map(
                    _build_filtered_molecule_evidence_split,
                    spec.benchmark_splits,
                    chunksize=1,
                )
        else:
            results = [
                _build_filtered_molecule_evidence_split(split)
                for split in spec.benchmark_splits
            ]
        return {split: manifest for split, manifest in results}
    finally:
        _EVIDENCE_BUILD_CONTEXT = None


_EVIDENCE_BUILD_CONTEXT: tuple[
    SplitDownstreamSpec, Path, Path, int
] | None = None


def evidence_catalog_projection_columns(
    schema: Iterable[str],
    *,
    policy: StarlingTaskPolicy,
    v7: bool,
) -> set[str]:
    """Select persisted fields needed to rederive relational family labels."""
    available = set(schema)
    record_id_field = "canonical_record_id" if v7 else "normalized_record_id"
    useful_columns = {
        record_id_field,
        "retrieval_eligible",
        "canonical_smiles",
        "group_id",
        "finite_scalar_value",
        "confidence",
        "endpoint_name",
        "canonical_endpoint_name" if v7 else "canonical_endpoint",
        "source_id",
        "source_name",
        "source_row_number",
        "assay_tier",
        "endpoint_group",
        "evidence_role",
        "target_pref_name",
        *policy.family_resolver_input_fields,
    }
    return useful_columns & available


def _build_filtered_molecule_evidence_split(
    split: str,
) -> tuple[str, dict[str, Any]]:
    if _EVIDENCE_BUILD_CONTEXT is None:
        raise RuntimeError("split evidence worker has no build context")
    spec, filtered_records_root, target, max_record_examples = _EVIDENCE_BUILD_CONTEXT
    records_path = filtered_records_root / split / FILTERED_RECORDS_FILENAME
    schema = set(pq.read_schema(records_path).names)
    v7 = "canonical_record_id" in schema
    record_id_field = "canonical_record_id" if v7 else "normalized_record_id"
    useful_columns = evidence_catalog_projection_columns(
        schema,
        policy=spec.policy,
        v7=v7,
    )
    records = read_parquet_records(
        records_path, columns=sorted(useful_columns)
    )
    families, bridge = build_relational_evidence_catalog(
        records,
        max_record_examples=max_record_examples,
        family_resolver=spec.policy.family_resolver,
    )
    split_dir = target / split
    split_dir.mkdir(parents=True, exist_ok=True)
    families_path = split_dir / EVIDENCE_FAMILIES_FILENAME
    bridge_path = split_dir / EVIDENCE_BRIDGE_FILENAME
    write_parquet(families_path, families)
    write_parquet(bridge_path, bridge)
    excluded = _excluded_ids(filtered_records_root, split, spec.exclusions_filename)
    retained = {str(row[record_id_field]) for row in bridge}
    if retained & excluded:
        raise AssertionError(f"{split} molecule evidence references excluded records")
    manifest = {
        "artifact_version": _compact_profile(spec, v7=v7).artifact_version,
        "benchmark_split": split,
        "input": {
            "path": str(records_path),
            "sha256": file_sha256(records_path),
            "records": len(records),
        },
        "families": len(families),
        "record_references": len(bridge),
        "files": {
            EVIDENCE_FAMILIES_FILENAME: file_sha256(families_path),
            EVIDENCE_BRIDGE_FILENAME: file_sha256(bridge_path),
        },
        "validations": {
            "excluded_filter_source_records_absent": True,
            "evidence_records_are_references_only": True,
        },
    }
    return split, manifest


build_split_molecule_evidence = build_filtered_molecule_evidence


def build_filtered_neighbor_indices(
    spec: SplitDownstreamSpec,
    *,
    evidence_root: str | Path,
    overlap_manifest: str | Path,
    out_dir: str | Path,
    workers: int = 1,
    progress_every: int = 0,
) -> dict[str, dict[str, Any]]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    overlap_hash = file_sha256(overlap_manifest)
    manifests: dict[str, dict[str, Any]] = {}
    families_by_split: dict[str, list[dict[str, Any]]] = {}
    profiles_by_split: dict[str, CompactArtifactProfile] = {}
    molecule_smiles: dict[str, str] = {}
    for split in spec.benchmark_splits:
        evidence_dir = Path(evidence_root) / split
        families = read_parquet_records(evidence_dir / EVIDENCE_FAMILIES_FILENAME)
        families_by_split[split] = families
        bridge_columns = pq.read_schema(
            evidence_dir / EVIDENCE_BRIDGE_FILENAME
        ).names
        profile = _compact_profile(
            spec, v7="canonical_record_id" in bridge_columns
        )
        profiles_by_split[split] = profile
        for family in families:
            molecule_id = str(family.get("molecule_id") or "")
            smiles = str(family.get("canonical_smiles") or "")
            previous = molecule_smiles.setdefault(molecule_id, smiles)
            if previous != smiles:
                raise ValueError(
                    f"molecule {molecule_id!r} has inconsistent split SMILES"
                )
    standardized = standardize_index_molecules(
        molecule_smiles,
        workers=workers,
        progress_every=progress_every,
    )
    for split in spec.benchmark_splits:
        evidence_dir = Path(evidence_root) / split
        families = families_by_split[split]
        profile = profiles_by_split[split]
        index_dir = target / split
        manifest = write_compact_neighbor_index(
            profile=profile,
            families=families,
            output_dir=index_dir,
            workers=workers,
            progress_every=progress_every,
            standardized_by_molecule=standardized,
        )
        manifest.update(
            {
                "benchmark_split": split,
                "heldout_overlap_manifest_sha256": overlap_hash,
                "evidence_dir": f"../../{MOLECULE_EVIDENCE_STAGE}/{split}",
                "records_path": f"../../{HELDOUT_STAGE}/{split}/{FILTERED_RECORDS_FILENAME}",
                "filtered_source_overlap": True,
                "filter_source_id": spec.filter_source_id,
                "filter_scope": {
                    "field": spec.filter_scope_field or None,
                    "value": (
                        spec.filter_scope_value
                        if spec.filter_scope_field
                        else None
                    ),
                },
                "unfiltered_index": False,
            }
        )
        _write_json(index_dir / MANIFEST_FILENAME, manifest)
        manifests[split] = manifest
    return manifests


build_split_neighbor_indices = build_filtered_neighbor_indices


def _preflight_inputs(
    *,
    records_path: Path,
    auxiliary_manifest: Path,
    benchmark_split_root: Path,
    splits: Sequence[str],
) -> dict[Path, str]:
    _require_file(records_path)
    _require_file(auxiliary_manifest)
    json.loads(auxiliary_manifest.read_text(encoding="utf-8"))
    paths = [records_path, auxiliary_manifest]
    for split in splits:
        train = benchmark_split_root / split / "train_molecule_labels.jsonl"
        heldout = benchmark_split_root / split / "heldout_molecule_labels.jsonl"
        _require_file(train)
        _require_file(heldout)
        train_keys = load_heldout_identity_keys(train)
        heldout_keys = load_heldout_identity_keys(heldout)
        normalize_molecule_identity.cache_clear()
        if train_keys & heldout_keys:
            raise ValueError(f"{split} train and heldout parent identities overlap")
        paths.extend((train, heldout))
    return {path: file_sha256(path) for path in paths}


def _heldout_audit(
    spec: SplitDownstreamSpec,
    pair_metadata: Mapping[str, Any],
    filtered: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    indices: Mapping[str, Mapping[str, Any]],
    transfer: Mapping[str, Any],
) -> dict[str, Any]:
    pair_bucket_version = str(pair_metadata.get("contract_version") or "")
    if not pair_bucket_version:
        raise ValueError("Stage-04 pair-bucket metadata lacks contract_version")
    calibration_pair_bucket_version = str(
        transfer.get("pair_bucket_version") or ""
    )
    if (
        calibration_pair_bucket_version
        and calibration_pair_bucket_version != pair_bucket_version
    ):
        raise ValueError(
            "Stage-04/Stage-05 pair-bucket contract mismatch: "
            f"{pair_bucket_version!r} != {calibration_pair_bucket_version!r}"
        )
    calibration_kind = (
        "distance_calibration"
        if transfer.get("calibration_version")
        else "assay_transfer_policy"
    )
    scope_matches = _calibration_scope_matches(
        spec.policy_statistics_scope,
        heldout_exclusion=transfer.get("heldout_exclusion"),
    )
    return {
        "audit_version": spec.heldout_overlap_version,
        "filter_source_id": spec.filter_source_id,
        "filter_scope": {
            "field": spec.filter_scope_field or None,
            "value": (
                spec.filter_scope_value if spec.filter_scope_field else None
            ),
        },
        "policy_statistics_scope": spec.policy_statistics_scope,
        "calibration_kind": calibration_kind,
        "excluded_filter_source_records_present_in_filtered_views": False,
        "other_sources_are_filtered": False,
        "out_of_scope_records_are_filtered": False,
        "pair_bucket_version": pair_bucket_version,
        "transfer_policy_heldout_exclusion": transfer.get("heldout_exclusion"),
        "splits": {
            split: {
                "excluded_filter_source_records": filtered["splits"][split][
                    "excluded_filter_source_records"
                ],
                "filtered_records": filtered["splits"][split]["filtered_records"],
                "molecule_families": evidence[split]["families"],
                "index_molecules": indices[split]["molecules"],
            }
            for split in spec.benchmark_splits
        },
        "validations": {
            "pair_buckets_precede_split_filter": True,
            "calibration_statistics_scope_matches_artifact": scope_matches,
            "filtered_evidence_only": True,
            "filtered_neighbor_indices_only": True,
        },
    }


def _calibration_scope_matches(
    policy_statistics_scope: str,
    *,
    heldout_exclusion: Any,
) -> bool:
    expected_heldout_filter = {
        "complete_unfiltered_records": False,
        "heldout_gold_filtered_transfer_calibration": True,
    }.get(policy_statistics_scope)
    if expected_heldout_filter is None:
        return False
    return bool(heldout_exclusion) is expected_heldout_filter


def _require_true_audit_validations(audit: Mapping[str, Any]) -> None:
    validations = audit.get("validations")
    if not isinstance(validations, Mapping) or not validations:
        raise ValueError("held-out audit has no required validations")
    failed = sorted(
        str(name) for name, value in validations.items() if value is not True
    )
    if failed:
        raise ValueError(
            "held-out audit validations failed: " + ", ".join(failed)
        )


def _manifest_update(
    *,
    spec: SplitDownstreamSpec,
    pair_metadata: Mapping[str, Any],
    transfer: Mapping[str, Any],
    audit: Mapping[str, Any],
    candidate: Path,
    rebuild_request: Mapping[str, Any] | None,
    elapsed_s: float,
    v7: bool,
) -> dict[str, Any]:
    return {
        **spec.policy.manifest_versions(),
        "pipeline_layout_version": spec.pipeline_layout_version,
        "compact_artifact_version": _compact_profile(
            spec,
            v7=str(pair_metadata.get("contract_version") or "").endswith(".v7"),
        ).artifact_version,
        "index_version": _compact_profile(
            spec,
            v7=str(pair_metadata.get("contract_version") or "").endswith(".v7"),
        ).index_version,
        "completed_artifact_stages": [
            "01_cleaned",
            (
                "02_canonicalized"
                if str(pair_metadata.get("contract_version") or "").endswith(".v7")
                else "02_normalized"
            ),
            "03_records",
            *_downstream_stages(v7=v7),
        ],
        "rebuild_request": dict(rebuild_request or {"from_stage": "index", "through_stage": "index"}),
        "downstream_scope": {
            "unfiltered_record_stage": "03_records",
            "unfiltered_pair_bucket_stage": PAIR_BUCKET_STAGE,
            (
                "distance_calibration_stage"
                if v7
                else "transfer_policy_stage"
            ): DISTANCE_CALIBRATION_STAGE if v7 else TRANSFER_POLICY_STAGE,
            "filtered_splits": list(spec.benchmark_splits),
            "filter_source_id": spec.filter_source_id,
            "filter_scope": {
                "field": spec.filter_scope_field or None,
                "value": (
                    spec.filter_scope_value
                    if spec.filter_scope_field
                    else None
                ),
            },
            "unfiltered_molecule_evidence_retained": False,
            "unfiltered_neighbor_index_retained": False,
        },
        "downstream_stats": {
            "pair_buckets": pair_metadata["stats"]["buckets"],
            (
                "calibration_valid_buckets"
                if v7
                else "assay_transfer_eligible_buckets"
            ): transfer["summary"][
                "calibration_valid_buckets"
                if v7
                else "assay_transfer_eligible_buckets"
            ],
            "splits": audit["splits"],
        },
        "downstream_artifact_hashes": _downstream_hashes(spec, candidate, v7=v7),
        "elapsed_downstream_s": elapsed_s,
    }


def _downstream_hashes(
    spec: SplitDownstreamSpec, root: Path, *, v7: bool
) -> dict[str, str]:
    paths = {
        "pair_bucket_records": root / PAIR_BUCKET_STAGE / spec.pair_bucket_records_filename,
        "pair_bucket_metadata": root / PAIR_BUCKET_STAGE / spec.pair_bucket_metadata_filename,
        (
            "distance_calibration" if v7 else "assay_transfer_policy"
        ): root
        / (DISTANCE_CALIBRATION_STAGE if v7 else TRANSFER_POLICY_STAGE)
        / (CALIBRATION_FILENAME if v7 else POLICY_FILENAME),
        "heldout_overlap_manifest": root / HELDOUT_STAGE / MANIFEST_FILENAME,
        "heldout_exclusions": root / HELDOUT_STAGE / spec.exclusions_filename,
    }
    for split in spec.benchmark_splits:
        paths[f"{split}_filtered_records"] = root / HELDOUT_STAGE / split / FILTERED_RECORDS_FILENAME
        paths[f"{split}_molecule_families"] = root / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_FAMILIES_FILENAME
        paths[f"{split}_molecule_family_records"] = root / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_BRIDGE_FILENAME
        paths[f"{split}_neighbor_index_manifest"] = root / NEIGHBOR_INDEX_STAGE / split / MANIFEST_FILENAME
    return {key: file_sha256(path) for key, path in sorted(paths.items())}


def _validate_candidate(
    spec: SplitDownstreamSpec,
    candidate: Path,
    published: Path,
    *,
    v7: bool,
    validation_level: str = "strict",
) -> None:
    required = [
        candidate / PAIR_BUCKET_STAGE / spec.pair_bucket_records_filename,
        candidate / PAIR_BUCKET_STAGE / spec.pair_bucket_metadata_filename,
        candidate
        / (DISTANCE_CALIBRATION_STAGE if v7 else TRANSFER_POLICY_STAGE)
        / (CALIBRATION_FILENAME if v7 else POLICY_FILENAME),
        candidate / HELDOUT_STAGE / MANIFEST_FILENAME,
        candidate / HELDOUT_STAGE / spec.exclusions_filename,
        candidate / AUDIT_STAGE / "heldout_overlap.json",
        candidate / MANIFEST_FILENAME,
    ]
    for split in spec.benchmark_splits:
        required.extend(
            (
                candidate / HELDOUT_STAGE / split / FILTERED_RECORDS_FILENAME,
                candidate / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_FAMILIES_FILENAME,
                candidate / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_BRIDGE_FILENAME,
                candidate / NEIGHBOR_INDEX_STAGE / split / "molecules.parquet",
                candidate / NEIGHBOR_INDEX_STAGE / split / "fingerprints.npz",
                candidate / NEIGHBOR_INDEX_STAGE / split / "group_membership.parquet",
                candidate / NEIGHBOR_INDEX_STAGE / split / MANIFEST_FILENAME,
            )
        )
    for path in required:
        _require_file(path)
    manifest = json.loads((candidate / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("downstream_artifact_hashes") != _downstream_hashes(
        spec, candidate, v7=v7
    ):
        raise ValueError("candidate downstream hashes differ from its manifest")
    token = str(candidate)
    for path in candidate.rglob("*.json"):
        if token in path.read_text(encoding="utf-8"):
            raise ValueError(f"candidate path leaked into published JSON: {path}")
    policy_text = gzip.decompress(
        (
            candidate
            / (DISTANCE_CALIBRATION_STAGE if v7 else TRANSFER_POLICY_STAGE)
            / (CALIBRATION_FILENAME if v7 else POLICY_FILENAME)
        ).read_bytes()
    ).decode("utf-8")
    if token in policy_text:
        raise ValueError("candidate path leaked into transfer policy")
    audit = json.loads(
        (candidate / AUDIT_STAGE / "heldout_overlap.json").read_text(encoding="utf-8")
    )
    _require_true_audit_validations(audit)
    for split in spec.benchmark_splits:
        bridge_columns = pq.read_schema(
            candidate
            / MOLECULE_EVIDENCE_STAGE
            / split
            / EVIDENCE_BRIDGE_FILENAME
        ).names
        profile = _compact_profile(
            spec, v7="canonical_record_id" in bridge_columns
        )
        if validation_level == "full":
            index = load_compact_neighbor_index(
                candidate / NEIGHBOR_INDEX_STAGE / split,
                profile=profile,
            )
            molecule_count = len(index["molecules"])
        else:
            validated = validate_compact_neighbor_index(
                candidate / NEIGHBOR_INDEX_STAGE / split,
                profile=profile,
                evidence_dir=candidate / MOLECULE_EVIDENCE_STAGE / split,
            )
            molecule_count = validated["molecules"]
        if molecule_count != int(audit["splits"][split]["index_molecules"]):
            raise ValueError(f"{split} candidate index molecule count mismatch")
    metadata = json.loads(
        (candidate / PAIR_BUCKET_STAGE / spec.pair_bucket_metadata_filename).read_text(
            encoding="utf-8"
        )
    )
    expected = str(published / PAIR_BUCKET_STAGE / spec.pair_bucket_records_filename)
    if metadata.get("output", {}).get("path") != expected:
        raise ValueError("candidate pair metadata lacks its published output path")


def _validate_published_indices(
    spec: SplitDownstreamSpec, root: Path, *, v7: bool
) -> None:
    audit = json.loads(
        (root / AUDIT_STAGE / "heldout_overlap.json").read_text(encoding="utf-8")
    )
    _require_true_audit_validations(audit)
    for split in spec.benchmark_splits:
        bridge_columns = pq.read_schema(
            root / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_BRIDGE_FILENAME
        ).names
        validated = validate_compact_neighbor_index(
            root / NEIGHBOR_INDEX_STAGE / split,
            profile=_compact_profile(
                spec, v7="canonical_record_id" in bridge_columns
            ),
            evidence_dir=root / MOLECULE_EVIDENCE_STAGE / split,
        )
        if validated["molecules"] != int(
            audit["splits"][split]["index_molecules"]
        ):
            raise ValueError(f"{split} published index molecule count mismatch")


def _downstream_output_paths(
    spec: SplitDownstreamSpec, root: Path, *, v7: bool
) -> list[Path]:
    paths = [
        root / PAIR_BUCKET_STAGE / spec.pair_bucket_records_filename,
        root / PAIR_BUCKET_STAGE / spec.pair_bucket_metadata_filename,
        root
        / (DISTANCE_CALIBRATION_STAGE if v7 else TRANSFER_POLICY_STAGE)
        / (CALIBRATION_FILENAME if v7 else POLICY_FILENAME),
        root / HELDOUT_STAGE / MANIFEST_FILENAME,
        root / HELDOUT_STAGE / spec.exclusions_filename,
        root / AUDIT_STAGE / "heldout_overlap.json",
    ]
    for split in spec.benchmark_splits:
        paths.extend(
            (
                root / HELDOUT_STAGE / split / FILTERED_RECORDS_FILENAME,
                root / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_FAMILIES_FILENAME,
                root / MOLECULE_EVIDENCE_STAGE / split / EVIDENCE_BRIDGE_FILENAME,
                root / MOLECULE_EVIDENCE_STAGE / split / MANIFEST_FILENAME,
                root / NEIGHBOR_INDEX_STAGE / split / "molecules.parquet",
                root / NEIGHBOR_INDEX_STAGE / split / "fingerprints.npz",
                root / NEIGHBOR_INDEX_STAGE / split / "group_membership.parquet",
                root / NEIGHBOR_INDEX_STAGE / split / MANIFEST_FILENAME,
            )
        )
    return paths


def _publish_downstream_candidate(
    spec: SplitDownstreamSpec, root: Path, candidate: Path, *, v7: bool | None = None
) -> None:
    if v7 is None:
        v7 = (candidate / DISTANCE_CALIBRATION_STAGE).is_dir()
    stages = _downstream_stages(v7=v7)
    backup = Path(tempfile.mkdtemp(dir=root, prefix=".downstream-backup-"))
    active_manifest = root / MANIFEST_FILENAME
    had_manifest = active_manifest.exists()
    if had_manifest:
        shutil.copyfile(active_manifest, backup / MANIFEST_FILENAME)
    moved: list[str] = []
    moved_legacy: list[str] = []
    published: list[str] = []
    try:
        for stage in stages:
            active = root / stage
            if active.exists():
                os.replace(active, backup / stage)
                moved.append(stage)
        legacy = backup / "legacy"
        for stage in spec.legacy_downstream_stages:
            active = root / stage
            if active.exists():
                legacy.mkdir(parents=True, exist_ok=True)
                os.replace(active, legacy / stage)
                moved_legacy.append(stage)
        for stage in stages:
            os.replace(candidate / stage, root / stage)
            published.append(stage)
        os.replace(candidate / MANIFEST_FILENAME, active_manifest)
    except BaseException:
        for stage in reversed(published):
            if (root / stage).exists():
                os.replace(root / stage, candidate / stage)
        for stage in reversed(moved):
            os.replace(backup / stage, root / stage)
        for stage in reversed(moved_legacy):
            os.replace(backup / "legacy" / stage, root / stage)
        if had_manifest:
            os.replace(backup / MANIFEST_FILENAME, active_manifest)
        elif active_manifest.exists():
            active_manifest.unlink()
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _publish_candidate(root: Path, candidate: Path) -> None:
    """Compatibility helper for the original default legacy-stage layout."""
    raise RuntimeError(
        "_publish_candidate requires a task spec; call _publish_downstream_candidate"
    )


def _stage_paths(root: Path, *, v7: bool = False) -> dict[str, Path]:
    return {
        "pair_buckets": root / PAIR_BUCKET_STAGE,
        "transfer_policy": root
        / (DISTANCE_CALIBRATION_STAGE if v7 else TRANSFER_POLICY_STAGE),
        "heldout": root / HELDOUT_STAGE,
        "evidence": root / MOLECULE_EVIDENCE_STAGE,
        "index": root / NEIGHBOR_INDEX_STAGE,
        "audits": root / AUDIT_STAGE,
    }


def _seed_audits(root: Path, target: Path) -> None:
    source = root / AUDIT_STAGE
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)


def _project_paths(value: Any, candidate: Path, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _project_paths(item, candidate, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_project_paths(item, candidate, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_project_paths(item, candidate, root) for item in value)
    return value.replace(str(candidate), str(root)) if isinstance(value, str) else value


def _bounded_parent_identity_map(
    smiles_values: Iterable[str], *, workers: int = 1
) -> dict[str, str]:
    output = parent_identity_map(smiles_values, workers=workers)
    normalize_molecule_identity.cache_clear()
    return output


def _excluded_ids(root: Path, split: str, filename: str) -> set[str]:
    path = root / filename
    columns = pq.read_schema(path).names
    record_id_field = (
        "canonical_record_id"
        if "canonical_record_id" in columns
        else "normalized_record_id"
    )
    frame = pd.read_parquet(path, columns=["benchmark_split", record_id_field])
    return set(
        frame[frame["benchmark_split"] == split][record_id_field].astype(str)
    )


def _verify_inputs(inputs: Mapping[Path, str]) -> None:
    for path, expected in inputs.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"downstream input changed during build: {path}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "AUDIT_STAGE",
    "DOWNSTREAM_STAGES",
    "EVIDENCE_BRIDGE_FILENAME",
    "EVIDENCE_FAMILIES_FILENAME",
    "FILTERED_RECORDS_FILENAME",
    "HELDOUT_STAGE",
    "MOLECULE_EVIDENCE_STAGE",
    "NEIGHBOR_INDEX_STAGE",
    "PAIR_BUCKET_STAGE",
    "PAIR_BUCKET_METADATA_FILENAME",
    "PAIR_BUCKET_RECORDS_FILENAME",
    "DEFAULT_LEGACY_DOWNSTREAM_STAGES",
    "DISTANCE_CALIBRATION_STAGE",
    "LEGACY_DOWNSTREAM_STAGES",
    "SplitDownstreamSpec",
    "TRANSFER_POLICY_STAGE",
    "V7_DOWNSTREAM_STAGES",
    "build_downstream_artifacts",
    "build_split_downstream_artifacts",
    "build_filtered_molecule_evidence",
    "build_filtered_neighbor_indices",
    "materialize_filtered_record_views",
    "build_split_molecule_evidence",
    "build_split_neighbor_indices",
    "materialize_split_record_views",
]
