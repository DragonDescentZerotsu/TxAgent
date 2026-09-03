"""Recover the exact BBB cluster-local auxiliary mapping with provenance.

The original clustered builder publishes a mapping only after its narrow
cleaned-key reconciliation step.  Its ignored caches still contain the exact
cluster membership and accepted response for every first-pass item.  This
module turns those caches into durable, deterministic artifacts before any
broader vocabulary review is performed.

This module deliberately contains no semantic merge or adjudication logic.
It only reconstructs what the model returned, verifies it against the frozen
generation audit and currently published mapping, and serializes that lineage.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


PROVISIONAL_MAPPING_VERSION = "bbb_martins_auxiliary.provisional_pre_reconciliation.v1"
PROVISIONAL_CLUSTER_ARTIFACT_VERSION = (
    "bbb_martins_auxiliary.provisional_clusters.v1"
)
PROVISIONAL_MANIFEST_VERSION = "bbb_martins_auxiliary.provisional_manifest.v1"

EXPECTED_SECTION_CLUSTERS = 1_726
EXPECTED_NON_NULL_ASSIGNMENTS = 172_308
EXPECTED_CURRENT_PUBLISHED_DELTAS = 2

PROVISIONAL_CLUSTER_FILENAME = "provisional_cluster_mapping.json"
CLUSTER_ASSIGNMENTS_FILENAME = "cluster_assignments.parquet"
PRE_RECONCILIATION_FILENAME = "pre_reconciliation_mapping.json"
PROVISIONAL_MANIFEST_FILENAME = "provisional_manifest.json"

_ASSIGNMENT_COLUMNS = (
    "source_id",
    "output_field",
    "input_column",
    "tuple_key",
    "raw_value",
    "cluster_order",
    "cluster_id",
    "item_order",
    "item_id",
    "cache_identity",
    "provisional_label",
    "served_model",
    "response_sha256",
    "generation_attempt_count",
)


def _json_bytes(value: Any) -> bytes:
    """Return the canonical on-disk JSON representation used by this module."""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tuple_key(raw_value: str | None) -> str:
    return json.dumps([raw_value], ensure_ascii=False, separators=(",", ":"))


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_response_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid cache record at {path}:{line_number}")
            identity = record.get("identity")
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"cache identity missing at {path}:{line_number}")
            if identity in records:
                raise ValueError(f"duplicate cache identity in {path}: {identity}")
            if not isinstance(record.get("mapping"), dict):
                raise ValueError(f"cache mapping missing at {path}:{line_number}")
            if not isinstance(record.get("generation"), dict):
                raise ValueError(f"cache generation missing at {path}:{line_number}")
            records[identity] = record
    if not records:
        raise ValueError(f"response cache is empty: {path}")
    return records


def _cache_identity(
    *,
    prompt_version: str,
    prompt: str,
    source_id: str,
    input_column: str,
    output_field: str,
    cluster_id: str,
    values: list[str],
    model: str,
    reasoning_effort: str,
) -> str:
    """Reproduce the clustered builder's immutable response-cache identity."""
    payload = {
        "prompt_version": prompt_version,
        "prompt": prompt,
        "source_id": source_id,
        "input_column": input_column,
        "output_field": output_field,
        "cluster_id": cluster_id,
        "values": values,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _accepted_response_metadata(generation: Mapping[str, Any]) -> tuple[str, str, int]:
    attempts = generation.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("cache generation has no attempts")
    accepted = [
        attempt
        for attempt in attempts
        if isinstance(attempt, Mapping) and attempt.get("status") == "valid"
    ]
    if len(accepted) != 1:
        raise ValueError(
            f"cache generation must contain exactly one valid response, found {len(accepted)}"
        )
    served_model = accepted[0].get("served_model")
    response_sha256 = accepted[0].get("response_sha256")
    if not isinstance(served_model, str) or not served_model:
        raise ValueError("accepted response lacks served_model")
    if (
        not isinstance(response_sha256, str)
        or len(response_sha256) != 64
        or any(character not in "0123456789abcdef" for character in response_sha256)
    ):
        raise ValueError("accepted response lacks a valid response_sha256")
    attempt_count = generation.get("attempt_count")
    if attempt_count != len(attempts):
        raise ValueError("generation attempt_count does not match attempts")
    return served_model, response_sha256, int(attempt_count)


def reconstruct_provisional(
    *,
    cache_dir: str | Path,
    cluster_cache_dir: str | Path | None = None,
    generation_audit_path: str | Path,
    published_mapping_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Reconstruct and validate the exact mapping before reconciliation.

    Returns ``(provisional_mapping, assignments, audit)``.  The provisional
    mapping is the exact base-cluster output and uses the same
    source/output/tuple structure as the published mapping.  ``assignments``
    retains cluster and accepted-response provenance for every non-null raw
    value.  The audit records the two currently published cleaned-key changes
    without applying them to the provisional mapping.
    """
    response_root = Path(cache_dir)
    cluster_root = Path(cluster_cache_dir) if cluster_cache_dir else response_root
    generation_path = Path(generation_audit_path)
    published_path = Path(published_mapping_path)
    generation_audit = _load_json_object(generation_path)
    published = _load_json_object(published_path)

    published_sha256 = _file_sha256(published_path)
    if generation_audit.get("mapping_sha256") != published_sha256:
        raise ValueError(
            "published mapping hash does not match the frozen generation audit"
        )
    prompt_version = generation_audit.get("prompt_version")
    model = generation_audit.get("model")
    reasoning_effort = generation_audit.get("reasoning_effort")
    if not all(isinstance(value, str) for value in (prompt_version, model, reasoning_effort)):
        raise ValueError("generation audit lacks prompt/model/reasoning provenance")

    # Import lazily so this provenance reader remains independent of the CLI's
    # import path while still hashing the exact frozen prompt text used by it.
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing import (
        build_embedding_bucket_mapping,
    )

    specs = sorted(
        build_embedding_bucket_mapping.extraction_specs(),
        key=lambda item: (item.source_id, item.output_field),
    )
    section_names = {f"{spec.source_id}/{spec.output_field}" for spec in specs}
    generation_sections = generation_audit.get("sections")
    if not isinstance(generation_sections, dict) or set(generation_sections) != section_names:
        raise ValueError("generation-audit section inventory mismatch")
    published_sources = published.get("sources")
    if not isinstance(published_sources, dict):
        raise ValueError("published mapping lacks sources")

    rows: list[dict[str, Any]] = []
    pre_sources: dict[str, dict[str, Any]] = defaultdict(dict)
    input_files: list[dict[str, Any]] = []
    section_cluster_count = 0

    recorded_cluster_files: set[Path] = set()
    for spec in specs:
        section_name = f"{spec.source_id}/{spec.output_field}"
        cluster_path = cluster_root / (
            f"clusters__{spec.source_id}__{spec.input_column}.json"
        )
        cache_path = response_root / f"{spec.source_id}__{spec.output_field}.jsonl"
        cluster_payload = _load_json_object(cluster_path)
        clusters = cluster_payload.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ValueError(f"cluster cache has no clusters: {cluster_path}")
        response_cache = _load_response_cache(cache_path)
        section_audit = generation_sections[section_name]
        audited_generation = section_audit.get("cluster_generation")
        if not isinstance(audited_generation, dict):
            raise ValueError(f"generation audit lacks clusters for {section_name}")
        if section_audit.get("clusters") != len(clusters):
            raise ValueError(f"cluster count differs from audit for {section_name}")

        raw_mapping: dict[str, str | None] = {}
        consumed_identities: set[str] = set()
        cluster_ids: list[str] = []
        for cluster_order, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                raise ValueError(f"invalid cluster in {cluster_path}")
            cluster_id = cluster.get("cluster_id")
            values = cluster.get("values")
            if (
                not isinstance(cluster_id, str)
                or not cluster_id
                or not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) for value in values)
            ):
                raise ValueError(f"invalid cluster membership in {cluster_path}")
            cluster_ids.append(cluster_id)
            identity = _cache_identity(
                prompt_version=prompt_version,
                prompt=spec.prompt,
                source_id=spec.source_id,
                input_column=spec.input_column,
                output_field=spec.output_field,
                cluster_id=cluster_id,
                values=values,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            if identity not in response_cache:
                raise ValueError(
                    f"no response cache matches {section_name}/{cluster_id}"
                )
            consumed_identities.add(identity)
            cached = response_cache[identity]
            cached_generation = cached["generation"]
            if audited_generation.get(cluster_id) != cached_generation:
                raise ValueError(
                    f"cache generation differs from audit for {section_name}/{cluster_id}"
                )
            served_model, response_sha256, attempt_count = (
                _accepted_response_metadata(cached_generation)
            )
            cluster_mapping = cached["mapping"]
            expected_item_ids = [f"v{index:04d}" for index in range(len(values))]
            if set(cluster_mapping) != set(expected_item_ids):
                raise ValueError(
                    f"item-ID coverage mismatch for {section_name}/{cluster_id}"
                )
            for item_order, (item_id, raw_value) in enumerate(
                zip(expected_item_ids, values, strict=True)
            ):
                label = cluster_mapping[item_id]
                if label is not None and not isinstance(label, str):
                    raise ValueError(
                        f"non-string provisional label for {section_name}/{cluster_id}/{item_id}"
                    )
                tuple_json = _tuple_key(raw_value)
                if tuple_json in raw_mapping:
                    raise ValueError(
                        f"raw value appears in multiple clusters for {section_name}: {raw_value!r}"
                    )
                raw_mapping[tuple_json] = label
                rows.append(
                    {
                        "source_id": spec.source_id,
                        "output_field": spec.output_field,
                        "input_column": spec.input_column,
                        "tuple_key": tuple_json,
                        "raw_value": raw_value,
                        "cluster_order": cluster_order,
                        "cluster_id": cluster_id,
                        "item_order": item_order,
                        "item_id": item_id,
                        "cache_identity": identity,
                        "provisional_label": label,
                        "served_model": served_model,
                        "response_sha256": response_sha256,
                        "generation_attempt_count": attempt_count,
                    }
                )
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError(f"duplicate cluster IDs for {section_name}")
        if consumed_identities != set(response_cache):
            raise ValueError(
                f"unmatched response cache records for {section_name}: "
                f"{len(set(response_cache) - consumed_identities)}"
            )
        if set(audited_generation) != set(cluster_ids):
            raise ValueError(f"generation cluster inventory mismatch for {section_name}")
        if section_audit.get("raw_values") != len(raw_mapping):
            raise ValueError(f"raw-value count differs from audit for {section_name}")
        section_cluster_count += len(clusters)

        # Null-like source records did not enter embedding clusters.  Retain the
        # explicit null tuple that the published attacher requires, but do not
        # misrepresent it as a model assignment in the flat provenance table.
        raw_mapping[_tuple_key(None)] = None
        try:
            published_section = published_sources[spec.source_id][spec.output_field]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"published section missing: {section_name}") from exc
        source_columns = published_section.get("source_columns")
        if source_columns != [spec.input_column]:
            raise ValueError(f"published source-column mismatch for {section_name}")
        pre_sources[spec.source_id][spec.output_field] = {
            "source_columns": [spec.input_column],
            "mapping": dict(sorted(raw_mapping.items())),
        }

        input_files.append(
            {
                "role": "response_cache",
                "section": section_name,
                "filename": cache_path.name,
                "sha256": _file_sha256(cache_path),
            }
        )
        if cluster_path not in recorded_cluster_files:
            input_files.append(
                {
                    "role": "cluster_cache",
                    "inventory": f"{spec.source_id}/{spec.input_column}",
                    "filename": cluster_path.name,
                    "sha256": _file_sha256(cluster_path),
                }
            )
            recorded_cluster_files.add(cluster_path)

    assignments = pd.DataFrame(rows, columns=_ASSIGNMENT_COLUMNS).sort_values(
        ["source_id", "output_field", "cluster_order", "item_order"],
        kind="stable",
        ignore_index=True,
    )
    pre_mapping = {
        "mapping_version": PROVISIONAL_MAPPING_VERSION,
        "sources": {
            source_id: dict(sorted(outputs.items()))
            for source_id, outputs in sorted(pre_sources.items())
        },
    }

    deltas: list[dict[str, Any]] = []
    for source_id, outputs in pre_mapping["sources"].items():
        if source_id not in published_sources:
            raise ValueError(f"provisional source missing from published mapping: {source_id}")
        for output_field, section in outputs.items():
            published_section = published_sources[source_id].get(output_field)
            if not isinstance(published_section, dict):
                raise ValueError(f"published output missing: {source_id}/{output_field}")
            provisional_values = section["mapping"]
            published_values = published_section.get("mapping")
            if not isinstance(published_values, dict):
                raise ValueError(f"published mapping missing: {source_id}/{output_field}")
            if set(provisional_values) != set(published_values):
                raise ValueError(
                    f"provisional/published tuple coverage differs for {source_id}/{output_field}"
                )
            assignment_lookup = {
                row.tuple_key: row
                for row in assignments.loc[
                    (assignments["source_id"] == source_id)
                    & (assignments["output_field"] == output_field)
                ].itertuples(index=False)
            }
            for tuple_json in sorted(provisional_values):
                provisional_label = provisional_values[tuple_json]
                published_label = published_values[tuple_json]
                if provisional_label == published_label:
                    continue
                assignment = assignment_lookup.get(tuple_json)
                if assignment is None:
                    raise ValueError(
                        "published reconciliation unexpectedly changed the null tuple"
                    )
                deltas.append(
                    {
                        "source_id": source_id,
                        "output_field": output_field,
                            "tuple_key": tuple_json,
                        "raw_value": assignment.raw_value,
                        "cluster_id": assignment.cluster_id,
                        "item_id": assignment.item_id,
                        "cache_identity": assignment.cache_identity,
                        "provisional_label": provisional_label,
                        "published_label": published_label,
                    }
                )

    if section_cluster_count != EXPECTED_SECTION_CLUSTERS:
        raise ValueError(
            f"expected {EXPECTED_SECTION_CLUSTERS:,} section clusters, "
            f"found {section_cluster_count:,}"
        )
    if len(assignments) != EXPECTED_NON_NULL_ASSIGNMENTS:
        raise ValueError(
            f"expected {EXPECTED_NON_NULL_ASSIGNMENTS:,} non-null assignments, "
            f"found {len(assignments):,}"
        )
    if len(deltas) != EXPECTED_CURRENT_PUBLISHED_DELTAS:
        raise ValueError(
            f"expected {EXPECTED_CURRENT_PUBLISHED_DELTAS} current published deltas, "
            f"found {len(deltas)}"
        )

    audit = {
        "artifact_version": PROVISIONAL_MANIFEST_VERSION,
        "generation": {
            "mapping_version": generation_audit.get("mapping_version"),
            "prompt_version": prompt_version,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "embedding_model": generation_audit.get("embedding_model"),
            "cluster_target_size": generation_audit.get("cluster_target_size"),
            "cluster_random_seed": generation_audit.get("cluster_random_seed"),
        },
        "inputs": sorted(
            input_files
            + [
                {
                    "role": "generation_audit",
                    "filename": generation_path.name,
                    "sha256": _file_sha256(generation_path),
                },
                {
                    "role": "published_mapping",
                    "filename": published_path.name,
                    "sha256": published_sha256,
                },
            ],
            key=lambda item: (
                str(item.get("role")),
                str(item.get("section") or item.get("inventory") or ""),
                str(item.get("filename")),
            ),
        ),
        "counts": {
            "sections": len(specs),
            "section_clusters": section_cluster_count,
            "non_null_assignments": len(assignments),
            "explicit_null_tuples": len(specs),
            "published_deltas": len(deltas),
        },
        "published_deltas": deltas,
        "validations": {
            "cache_identities_match_frozen_prompts": True,
            "cache_generation_matches_generation_audit": True,
            "all_cluster_items_reconstructed": True,
            "provisional_and_published_tuple_coverage_match": True,
            "expected_section_cluster_count": True,
            "expected_non_null_assignment_count": True,
            "expected_current_published_delta_count": True,
        },
    }
    return pre_mapping, assignments, audit


def _cluster_artifact(
    pre_mapping: Mapping[str, Any],
    assignments: pd.DataFrame,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the nested, human-inspectable view of the flat assignments."""
    sources: dict[str, dict[str, Any]] = defaultdict(dict)
    grouped = assignments.groupby(["source_id", "output_field"], sort=True)
    for (source_id, output_field), section_frame in grouped:
        clusters: list[dict[str, Any]] = []
        for _, cluster_frame in section_frame.groupby("cluster_order", sort=True):
            first = cluster_frame.iloc[0]
            clusters.append(
                {
                    "cluster_id": first["cluster_id"],
                    "cache_identity": first["cache_identity"],
                    "served_model": first["served_model"],
                    "response_sha256": first["response_sha256"],
                    "generation_attempt_count": int(
                        first["generation_attempt_count"]
                    ),
                    "items": [
                        {
                            "item_id": row.item_id,
                            "raw_value": row.raw_value,
                            "tuple_key": row.tuple_key,
                            "provisional_label": row.provisional_label,
                        }
                        for row in cluster_frame.sort_values(
                            "item_order", kind="stable"
                        ).itertuples(index=False)
                    ],
                }
            )
        source_columns = pre_mapping["sources"][source_id][output_field][
            "source_columns"
        ]
        sources[source_id][output_field] = {
            "source_columns": source_columns,
            "mapping": pre_mapping["sources"][source_id][output_field]["mapping"],
            "cluster_count": len(clusters),
            "assignment_count": len(section_frame),
            "clusters": clusters,
        }
    return {
        "artifact_version": PROVISIONAL_CLUSTER_ARTIFACT_VERSION,
        "generation": audit["generation"],
        "sources": {
            source_id: dict(sorted(outputs.items()))
            for source_id, outputs in sorted(sources.items())
        },
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_provisional_artifacts(
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    cluster_cache_dir: str | Path | None = None,
    generation_audit_path: str | Path,
    published_mapping_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically publish the reconstructed mapping, assignments, and manifest.

    The manifest is replaced last and acts as the commit marker for the other
    three files.  Its hashes make the complete reconstruction independently
    verifiable.  Existing artifacts require ``overwrite=True``.
    """
    destination = Path(output_dir)
    paths = {
        "provisional_cluster_mapping": destination / PROVISIONAL_CLUSTER_FILENAME,
        "cluster_assignments": destination / CLUSTER_ASSIGNMENTS_FILENAME,
        "pre_reconciliation_mapping": destination / PRE_RECONCILIATION_FILENAME,
        "provisional_manifest": destination / PROVISIONAL_MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "provisional artifact already exists; pass overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    provisional_mapping, assignments, reconstruction_audit = reconstruct_provisional(
        cache_dir=cache_dir,
        cluster_cache_dir=cluster_cache_dir,
        generation_audit_path=generation_audit_path,
        published_mapping_path=published_mapping_path,
    )
    cluster_payload = _cluster_artifact(
        provisional_mapping, assignments, reconstruction_audit
    )
    destination.mkdir(parents=True, exist_ok=True)

    cluster_bytes = _json_bytes(cluster_payload)
    # This file is the rollback/current-v6 baseline before future global
    # semantic review, so preserve the currently published mapping byte for
    # byte.  The genuinely pre-conflict projection lives in the cluster
    # artifact above and is returned by ``reconstruct_provisional``.
    pre_mapping_bytes = Path(published_mapping_path).read_bytes()
    _atomic_write_bytes(paths["provisional_cluster_mapping"], cluster_bytes)
    _atomic_write_bytes(paths["pre_reconciliation_mapping"], pre_mapping_bytes)

    parquet_temporary = paths["cluster_assignments"].with_name(
        f".{paths['cluster_assignments'].name}.tmp"
    )
    assignments.to_parquet(
        parquet_temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    os.replace(parquet_temporary, paths["cluster_assignments"])

    manifest = dict(reconstruction_audit)
    manifest["outputs"] = {
        name: {
            "filename": path.name,
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
        if name != "provisional_manifest"
    }
    manifest["validations"] = {
        **manifest["validations"],
        "output_hashes_recorded": True,
    }
    _atomic_write_bytes(paths["provisional_manifest"], _json_bytes(manifest))
    return manifest


__all__ = [
    "CLUSTER_ASSIGNMENTS_FILENAME",
    "PRE_RECONCILIATION_FILENAME",
    "PROVISIONAL_CLUSTER_FILENAME",
    "PROVISIONAL_MANIFEST_FILENAME",
    "reconstruct_provisional",
    "write_provisional_artifacts",
]
