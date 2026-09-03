"""Freeze the completed Skin_Reaction cluster-local auxiliary mappings.

The embedding builder deliberately retained its paid-response caches instead
of publishing the runtime mapping.  This module reconstructs those responses,
verifies each cache identity against the frozen prompt/model/source contract,
and writes immutable provenance for the later cross-cluster review.

No function in this module writes the runtime canonical mapping.
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

from data.processing.evidence_library.versions.v7.prompts import PROMPT_ROOT


PROVISIONAL_MAPPING_VERSION = (
    "skin_reaction_auxiliary.aggregated_local_pre_reconciliation.v1"
)
PROVISIONAL_CLUSTER_ARTIFACT_VERSION = (
    "skin_reaction_auxiliary.local_clusters.v1"
)
PROVISIONAL_MANIFEST_VERSION = "skin_reaction_auxiliary.provisional_manifest.v1"

EXPECTED_SECTION_CLUSTERS = 1_649
EXPECTED_LLM_ASSIGNMENTS = 228_639
EXPECTED_LOCAL_TO_AGGREGATED_DELTAS = 2_395
EXPECTED_AGGREGATED_MAPPING_SHA256 = (
    "29ca01c3ae54644d0bf6a8bbdc2079222a69200b911bf8db8ca709a13eb24e4a"
)

MODEL_BY_SOURCE = {
    "direct_skin_reaction": "gpt-5.4",
    "sensitization_aop": "gpt-5.4",
    "phototoxicity_irritation_local_damage": "gpt-5.4-mini",
    "skin_exposure": "gpt-5.4-mini",
}
REASONING_EFFORT = "low"

REVIEW_NAMESPACES = frozenset(
    {
        "direct_skin_reaction/global_context",
        "direct_skin_reaction/global_species_context",
        "sensitization_aop/global_context",
        "sensitization_aop/global_species_context",
        "sensitization_aop/global_endpoint_context",
        "phototoxicity_irritation_local_damage/global_context",
        "phototoxicity_irritation_local_damage/global_species_context",
        "skin_exposure/global_species_context",
    }
)
FROZEN_NAMESPACES = frozenset(
    {
        "direct_skin_reaction/global_severity_grade",
        "skin_exposure/global_context",
    }
)

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
    "cluster_local_label",
    "provisional_label",
    "review_eligible",
    "requested_model",
    "response_sha256",
    "generation_attempt_count",
    "generation_total_tokens",
)


def _json_bytes(value: Any, *, sort_keys: bool = True) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n"
    ).encode("utf-8")


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


def _load_response_cache(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid cache row at {path}:{line_number}")
            identity = row.get("identity")
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"cache identity missing at {path}:{line_number}")
            if identity in identities:
                raise ValueError(f"duplicate cache identity in {path}: {identity}")
            identities.add(identity)
            if not isinstance(row.get("members"), list) or not row["members"]:
                raise ValueError(f"cache members missing at {path}:{line_number}")
            if not isinstance(row.get("mapping"), dict):
                raise ValueError(f"cache mapping missing at {path}:{line_number}")
            if not isinstance(row.get("generation"), dict):
                raise ValueError(f"cache generation missing at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"response cache is empty: {path}")
    return rows


def _cluster_id(members: list[str]) -> str:
    digest = hashlib.sha256(
        json.dumps(tuple(members), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"cluster_{digest}"


def _accepted_response_metadata(generation: Mapping[str, Any]) -> tuple[str, int, int]:
    attempts = generation.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("cache generation has no attempts")
    accepted = [
        item
        for item in attempts
        if isinstance(item, Mapping) and item.get("status") == "valid"
    ]
    if len(accepted) != 1:
        raise ValueError(
            f"cache generation must have one valid response, found {len(accepted)}"
        )
    response_sha256 = accepted[0].get("response_sha256")
    if (
        not isinstance(response_sha256, str)
        or len(response_sha256) != 64
        or any(character not in "0123456789abcdef" for character in response_sha256)
    ):
        raise ValueError("accepted response lacks response_sha256")
    attempt_count = generation.get("attempt_count")
    if attempt_count != len(attempts):
        raise ValueError("generation attempt_count does not match attempts")
    usage = generation.get("usage")
    if not isinstance(usage, Mapping) or not isinstance(usage.get("total_tokens"), int):
        raise ValueError("cache generation lacks total-token usage")
    return response_sha256, int(attempt_count), int(usage["total_tokens"])


def _downstream_manifests(data_processing_dir: Path) -> list[dict[str, Any]]:
    """Hash the complete existing downstream tree, not only its manifests."""
    repo_root = data_processing_dir.parents[4]
    root = (
        repo_root
        / "outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v6"
    )
    if not root.exists():
        return []
    return [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def reconstruct_provisional(
    *, cache_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Return exact clusters, aggregated local mapping, assignments, and audit."""
    from data.processing.evidence_library.versions.v7.tasks.skin_reaction.data_processing import (
        build_embedding_bucket_mapping as builder,
    )
    from data.processing.evidence_library.versions.v7.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers import (
        reconciliation,
    )

    response_root = Path(cache_dir)
    data_processing_dir = Path(__file__).resolve().parent
    runtime_mapping_path = (
        data_processing_dir / "globally_reconciled_auxiliary_value_mapping.json"
    )
    if runtime_mapping_path.exists():
        raise ValueError(
            "runtime mapping already exists; reconciliation snapshot must remain unpublished"
        )
    first_stage: dict[str, Any] = {}
    input_files: list[dict[str, Any]] = []
    snapshot_models: dict[str, str] = {}

    for source_id in reconciliation.SOURCE_SPECS:
        model = MODEL_BY_SOURCE[source_id]
        snapshot_path = response_root / f"{source_id}.complete.json"
        snapshot = builder._load_snapshot(
            response_root,
            source_id,
            model=model,
            reasoning_effort=REASONING_EFFORT,
            embedding_model=builder.EMBEDDING_MODEL,
        )
        if snapshot is None:
            raise ValueError(f"completed snapshot identity mismatch: {source_id}")
        first_stage[source_id] = snapshot
        snapshot_models[source_id] = model
        input_files.append(
            {
                "role": "complete_source_snapshot",
                "source_id": source_id,
                "filename": snapshot_path.name,
                "sha256": _file_sha256(snapshot_path),
                "requested_model": model,
            }
        )
        input_files.append(
            {
                "role": "source_parquet",
                "source_id": source_id,
                "path": str(reconciliation.SOURCE_SPECS[source_id].parquet_path),
                "sha256": _file_sha256(
                    reconciliation.SOURCE_SPECS[source_id].parquet_path
                ),
            }
        )

    aggregated = reconciliation.build_mapping(first_stage)
    reconciliation.validate_mapping(aggregated)
    # This is the exact serialization used by the existing builder.
    aggregated_bytes = _json_bytes(aggregated, sort_keys=False)
    aggregated_sha256 = hashlib.sha256(aggregated_bytes).hexdigest()
    if aggregated_sha256 != EXPECTED_AGGREGATED_MAPPING_SHA256:
        raise ValueError(
            "aggregated local mapping changed: "
            f"expected={EXPECTED_AGGREGATED_MAPPING_SHA256} actual={aggregated_sha256}"
        )

    rows: list[dict[str, Any]] = []
    cluster_sections: dict[str, dict[str, Any]] = defaultdict(dict)
    cluster_count = 0
    local_to_aggregated_deltas = 0
    seen_namespaces: set[str] = set()

    for source_id, source_spec in reconciliation.SOURCE_SPECS.items():
        model = snapshot_models[source_id]
        for output in source_spec.outputs:
            namespace = f"{source_id}/{output.output_name}"
            seen_namespaces.add(namespace)
            section = aggregated["sources"][source_id][output.output_name]
            candidate_mapping = section["mapping"]
            if output.label_source == "reviewed_table":
                if output.reviewed_mapping_path is None:
                    raise ValueError(f"reviewed table path missing: {namespace}")
                cluster_sections[source_id][output.output_name] = {
                    "source_columns": list(output.source_columns),
                    "mapping": candidate_mapping,
                    "label_source": "reviewed_table",
                    "reviewed_mapping_sha256": _file_sha256(
                        output.reviewed_mapping_path
                    ),
                    "review_eligible": False,
                    "clusters": [],
                }
                input_files.append(
                    {
                        "role": "reviewed_table",
                        "section": namespace,
                        "filename": output.reviewed_mapping_path.name,
                        "sha256": _file_sha256(output.reviewed_mapping_path),
                    }
                )
                continue

            extraction = builder._extraction(source_id, output)
            cache_path = response_root / f"{source_id}__{output.output_name}.jsonl"
            cache_rows = _load_response_cache(cache_path)
            flattened: dict[str, str | None] = {}
            clusters: list[dict[str, Any]] = []
            for cluster_order, cache_row in enumerate(cache_rows):
                members = cache_row["members"]
                if not all(isinstance(value, str) for value in members):
                    raise ValueError(f"non-string cluster member in {namespace}")
                cluster_id = _cluster_id(members)
                cluster = builder.Cluster(cluster_id=cluster_id, values=tuple(members))
                identity = builder._cache_identity(
                    extraction,
                    cluster,
                    model=model,
                    reasoning_effort=REASONING_EFFORT,
                )
                if cache_row["identity"] != identity:
                    raise ValueError(f"cache identity mismatch: {namespace}/{cluster_id}")
                item_ids = [f"v{index:04d}" for index in range(len(members))]
                mapping = cache_row["mapping"]
                if set(mapping) != set(item_ids):
                    raise ValueError(f"item-ID coverage mismatch: {namespace}/{cluster_id}")
                response_sha256, attempt_count, total_tokens = (
                    _accepted_response_metadata(cache_row["generation"])
                )
                items: list[dict[str, Any]] = []
                for item_order, (item_id, raw_value) in enumerate(
                    zip(item_ids, members, strict=True)
                ):
                    if raw_value in flattened:
                        raise ValueError(
                            f"raw value occurs in multiple clusters: {namespace}/{raw_value!r}"
                        )
                    local_label = mapping[item_id]
                    if local_label is not None and not isinstance(local_label, str):
                        raise ValueError(f"invalid local label: {namespace}/{item_id}")
                    flattened[raw_value] = local_label
                    tuple_key = _tuple_key(raw_value)
                    if tuple_key not in candidate_mapping:
                        raise ValueError(f"candidate mapping lacks {namespace}/{raw_value!r}")
                    provisional_label = candidate_mapping[tuple_key]
                    local_to_aggregated_deltas += int(local_label != provisional_label)
                    review_eligible = namespace in REVIEW_NAMESPACES
                    rows.append(
                        {
                            "source_id": source_id,
                            "output_field": output.output_name,
                            "input_column": output.current_input_column,
                            "tuple_key": tuple_key,
                            "raw_value": raw_value,
                            "cluster_order": cluster_order,
                            "cluster_id": cluster_id,
                            "item_order": item_order,
                            "item_id": item_id,
                            "cache_identity": identity,
                            "cluster_local_label": local_label,
                            "provisional_label": provisional_label,
                            "review_eligible": review_eligible,
                            "requested_model": model,
                            "response_sha256": response_sha256,
                            "generation_attempt_count": attempt_count,
                            "generation_total_tokens": total_tokens,
                        }
                    )
                    items.append(
                        {
                            "item_id": item_id,
                            "raw_value": raw_value,
                            "tuple_key": tuple_key,
                            "cluster_local_label": local_label,
                            "provisional_label": provisional_label,
                        }
                    )
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "cache_identity": identity,
                        "requested_model": model,
                        "response_sha256": response_sha256,
                        "generation_attempt_count": attempt_count,
                        "generation_total_tokens": total_tokens,
                        "items": items,
                    }
                )

            snapshot_mapping = first_stage[source_id][output.current_input_column][
                output.current_output_column
            ]
            if flattened != snapshot_mapping:
                raise ValueError(f"cache/snapshot mapping mismatch: {namespace}")
            expected_tuple_keys = {_tuple_key(value) for value in flattened} | {
                _tuple_key(None)
            }
            if set(candidate_mapping) != expected_tuple_keys:
                raise ValueError(f"candidate tuple coverage mismatch: {namespace}")
            cluster_count += len(clusters)
            cluster_sections[source_id][output.output_name] = {
                "source_columns": list(output.source_columns),
                "mapping": candidate_mapping,
                "label_source": "llm_cluster",
                "review_eligible": namespace in REVIEW_NAMESPACES,
                "cluster_count": len(clusters),
                "assignment_count": len(flattened),
                "clusters": clusters,
            }
            input_files.append(
                {
                    "role": "response_cache",
                    "section": namespace,
                    "filename": cache_path.name,
                    "sha256": _file_sha256(cache_path),
                    "requested_model": model,
                }
            )

    if seen_namespaces != REVIEW_NAMESPACES | FROZEN_NAMESPACES:
        raise ValueError("output namespace inventory mismatch")
    assignments = pd.DataFrame(rows, columns=_ASSIGNMENT_COLUMNS).sort_values(
        ["source_id", "output_field", "cluster_order", "item_order"],
        kind="stable",
        ignore_index=True,
    )
    if cluster_count != EXPECTED_SECTION_CLUSTERS:
        raise ValueError(
            f"expected {EXPECTED_SECTION_CLUSTERS:,} clusters, found {cluster_count:,}"
        )
    if len(assignments) != EXPECTED_LLM_ASSIGNMENTS:
        raise ValueError(
            f"expected {EXPECTED_LLM_ASSIGNMENTS:,} assignments, found {len(assignments):,}"
        )
    if local_to_aggregated_deltas != EXPECTED_LOCAL_TO_AGGREGATED_DELTAS:
        raise ValueError(
            "local-to-aggregated delta count changed: "
            f"expected={EXPECTED_LOCAL_TO_AGGREGATED_DELTAS} "
            f"actual={local_to_aggregated_deltas}"
        )

    prompt_path = PROMPT_ROOT / "auxiliary_canonicalization/skin_reaction.json"
    input_files.append(
        {
            "role": "prompt_registry",
            "filename": prompt_path.name,
            "sha256": _file_sha256(prompt_path),
        }
    )
    for code_path in (
        data_processing_dir / "build_embedding_bucket_mapping.py",
        data_processing_dir / "auxiliary_mapping_helpers/reconciliation.py",
    ):
        input_files.append(
            {
                "role": "mapping_implementation",
                "filename": code_path.relative_to(data_processing_dir).as_posix(),
                "sha256": _file_sha256(code_path),
            }
        )
    for ledger_name in (
        "openai_guard.neighborhood_v3.token_ledger.json",
        "openai_guard.neighborhood_v3.mini.token_ledger.json",
    ):
        ledger_path = data_processing_dir / ledger_name
        input_files.append(
            {
                "role": "token_ledger",
                "filename": ledger_path.name,
                "sha256": _file_sha256(ledger_path),
            }
        )

    cluster_artifact = {
        "artifact_version": PROVISIONAL_CLUSTER_ARTIFACT_VERSION,
        "aggregated_mapping_version": aggregated["mapping_version"],
        "sources": {
            source: dict(sorted(outputs.items()))
            for source, outputs in sorted(cluster_sections.items())
        },
    }
    audit = {
        "artifact_version": PROVISIONAL_MANIFEST_VERSION,
        "generation": {
            "prompt_version": builder.PROMPT_VERSION,
            "embedding_model": builder.EMBEDDING_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "models_by_source": snapshot_models,
            "cluster_random_seed": builder.CLUSTER_RANDOM_SEED,
        },
        "inputs": sorted(
            input_files,
            key=lambda item: (
                str(item.get("role")),
                str(item.get("section") or item.get("source_id") or ""),
                str(item.get("filename")),
            ),
        ),
        "counts": {
            "output_namespaces": len(seen_namespaces),
            "review_namespaces": len(REVIEW_NAMESPACES),
            "frozen_namespaces": len(FROZEN_NAMESPACES),
            "section_clusters": cluster_count,
            "llm_assignments": len(assignments),
            "review_assignments": int(assignments["review_eligible"].sum()),
            "local_to_aggregated_deltas": local_to_aggregated_deltas,
        },
        "aggregated_mapping": {
            "mapping_version": aggregated["mapping_version"],
            "sha256": aggregated_sha256,
            "bytes": len(aggregated_bytes),
        },
        "runtime_mapping": {
            "path": str(runtime_mapping_path),
            "exists": False,
        },
        "downstream_artifacts_before": _downstream_manifests(data_processing_dir),
        "validations": {
            "all_cache_identities_match_frozen_inputs": True,
            "all_cache_items_match_complete_source_snapshots": True,
            "all_source_tuples_covered": True,
            "aggregated_local_mapping_valid": True,
            "expected_cluster_count": True,
            "expected_assignment_count": True,
            "expected_local_to_aggregated_delta_count": True,
            "runtime_mapping_absent": True,
        },
    }
    return cluster_artifact, aggregated, assignments, audit


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_provisional_artifacts(
    *, output_dir: str | Path, cache_dir: str | Path, overwrite: bool = False
) -> dict[str, Any]:
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
            "provisional artifacts exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    cluster_artifact, aggregated, assignments, audit = reconstruct_provisional(
        cache_dir=cache_dir
    )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write(paths["provisional_cluster_mapping"], _json_bytes(cluster_artifact))
    _atomic_write(
        paths["pre_reconciliation_mapping"],
        _json_bytes(aggregated, sort_keys=False),
    )
    parquet_tmp = paths["cluster_assignments"].with_name(
        f".{paths['cluster_assignments'].name}.tmp"
    )
    assignments.to_parquet(parquet_tmp, index=False, compression="zstd")
    os.replace(parquet_tmp, paths["cluster_assignments"])
    manifest = dict(audit)
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
    _atomic_write(paths["provisional_manifest"], _json_bytes(manifest))
    return manifest


__all__ = [
    "CLUSTER_ASSIGNMENTS_FILENAME",
    "FROZEN_NAMESPACES",
    "PRE_RECONCILIATION_FILENAME",
    "PROVISIONAL_CLUSTER_FILENAME",
    "PROVISIONAL_MANIFEST_FILENAME",
    "REVIEW_NAMESPACES",
    "reconstruct_provisional",
    "write_provisional_artifacts",
]
