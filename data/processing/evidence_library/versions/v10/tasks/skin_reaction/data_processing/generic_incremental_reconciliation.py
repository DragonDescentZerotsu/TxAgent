#!/usr/bin/env python3
"""Review and immutably append non-species Skin auxiliary mapping deltas.

This workflow never calls a model or publishes the runtime mapping.  It freezes
the incremental builder output, prepares cluster-atomic packets for two complete
reviews, adjudicates every disagreement or proposed change, and writes an
unpublished v5 base-plus-delta proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_measurement_text,
    clean_scalar,
)

from . import build_embedding_bucket_mapping as builder
from .auxiliary_reconciliation import (
    _accepted_response_metadata,
    _cluster_id,
    _load_response_cache,
)
from .auxiliary_mapping_helpers.reconciliation import NULL_LIKE
from .species_context_reconciliation import (
    _atomic_json,
    _atomic_jsonl,
    _json_bytes,
    _load_jsonl,
    file_sha256,
    reviewer_roles,
)


BASE_VERSION = "starling_auxiliary.skin_reaction.globally_reconciled.v4"
PROPOSAL_VERSION = "starling_auxiliary.skin_reaction.globally_reconciled.v5"
SNAPSHOT_VERSION = "skin_auxiliary.generic_incremental_snapshot.v1"
PACKET_VERSION = "skin_auxiliary.generic_incremental_review_packet.v1"
REVIEW_VERSION = "skin_auxiliary.generic_incremental_assignment_review.v1"
PROPOSAL_MANIFEST_VERSION = "skin_auxiliary.generic_incremental_proposal.v1"
MERGE_MANIFEST_VERSION = "skin_auxiliary.incremental_v5_merge.v1"
SPECIALIZED_NAMESPACE = "sensitization_aop/global_species_context"
FROZEN_NAMESPACES = frozenset(
    {
        "direct_skin_reaction/global_severity_grade",
        "skin_exposure/global_context",
    }
)
OPEN_NAMESPACES = frozenset(
    {
        "direct_skin_reaction/global_context",
        "direct_skin_reaction/global_species_context",
        "sensitization_aop/global_context",
        "sensitization_aop/global_endpoint_context",
        "phototoxicity_irritation_local_damage/global_context",
        "phototoxicity_irritation_local_damage/global_species_context",
        "skin_exposure/global_species_context",
    }
)
EXPECTED_NONEMPTY_OPEN_NAMESPACES = OPEN_NAMESPACES - {
    "phototoxicity_irritation_local_damage/global_species_context"
}
ALLOWED_SEVERITY = frozenset({None, "0", "1", "2", "3", "4"})
LABEL_FORBIDDEN = builder.FORBIDDEN_OUTPUT
SOURCE_COLUMN_ALIASES = {
    "direct_skin_reaction": {"effect_metric": "measurement_text"},
    "sensitization_aop": {"endpoint_or_target": "endpoint_name"},
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _namespace(source_id: str, output_field: str) -> str:
    return f"{source_id}/{output_field}"


def _cleaned_tuple_key(
    source_id: str, source_columns: Sequence[str], serialized_key: str
) -> tuple[Any, ...]:
    values = json.loads(serialized_key)
    if not isinstance(values, list) or len(values) != len(source_columns):
        raise ValueError(f"invalid tuple key for {source_id}: {serialized_key!r}")
    aliases = SOURCE_COLUMN_ALIASES.get(source_id, {})
    cleaned: list[Any] = []
    for column, value in zip(source_columns, values, strict=True):
        target = aliases.get(column, column)
        item = (
            clean_measurement_text(value)
            if target in {"measurement_text", "unit_text"}
            else clean_scalar(value)
        )
        if isinstance(item, str) and item.casefold() in NULL_LIKE:
            item = None
        cleaned.append(item)
    return tuple(cleaned)


def _inherit_cleaned_base_labels(
    base: Mapping[str, Any],
    additions: Mapping[tuple[str, str], dict[str, Any]],
) -> int:
    """Keep a reviewed label when a new raw spelling cleans to its v4 key."""
    inherited = 0
    for (source_id, output_field), delta in additions.items():
        section = base["sources"][source_id][output_field]
        columns = section["source_columns"]
        reviewed: dict[tuple[Any, ...], Any] = {}
        for raw_key, label in section["mapping"].items():
            key = _cleaned_tuple_key(source_id, columns, raw_key)
            if key in reviewed and reviewed[key] != label:
                raise ValueError(
                    f"reviewed v4 has a cleaned-key conflict for "
                    f"{source_id}/{output_field}: {key!r}"
                )
            reviewed[key] = label
        for raw_key, label in delta.items():
            key = _cleaned_tuple_key(source_id, columns, raw_key)
            if key in reviewed and label != reviewed[key]:
                delta[raw_key] = reviewed[key]
                inherited += 1
    return inherited


def _validate_cleaned_mapping(payload: Mapping[str, Any]) -> None:
    for source_id, outputs in payload["sources"].items():
        for output_field, section in outputs.items():
            columns = section["source_columns"]
            seen: dict[tuple[Any, ...], Any] = {}
            for raw_key, label in section["mapping"].items():
                key = _cleaned_tuple_key(source_id, columns, raw_key)
                if key in seen and seen[key] != label:
                    raise ValueError(
                        f"merged mapping has a cleaned-key conflict for "
                        f"{source_id}/{output_field}: {key!r}"
                    )
                seen[key] = label


def _assignment_id(identity: str, item_id: str, tuple_key: str) -> str:
    return hashlib.sha256(f"{identity}\0{item_id}\0{tuple_key}".encode()).hexdigest()


def _tuple_key(extraction: builder.ExtractionSpec, member: str) -> str:
    return member if extraction.is_composite else json.dumps(
        [member], ensure_ascii=False, separators=(",", ":")
    )


def _clean_label(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("label must be a string or null")
    label = re.sub(r"\s+", " ", value.strip()).casefold()
    if (
        not label
        or len(label) > 120
        or LABEL_FORBIDDEN.search(label)
        or any(token in label for token in ("```", "{", "}", "->", "→"))
    ):
        raise ValueError(f"invalid canonical label: {value!r}")
    return label


def _candidate_sections(candidate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if candidate.get("artifact_version") != builder.INCREMENTAL_ARTIFACT_VERSION:
        raise ValueError("candidate is not an incremental builder artifact")
    sections: dict[str, Mapping[str, Any]] = {}
    for source_id, outputs in candidate.get("sources", {}).items():
        for output_field, section in outputs.items():
            namespace = _namespace(str(source_id), str(output_field))
            if namespace != SPECIALIZED_NAMESPACE:
                sections[namespace] = section
    expected = OPEN_NAMESPACES | FROZEN_NAMESPACES
    if set(sections) != expected:
        raise ValueError(
            f"non-species candidate namespace mismatch: "
            f"missing={sorted(expected-set(sections))} extra={sorted(set(sections)-expected)}"
        )
    return sections


def _existing_label_anchors(
    base: Mapping[str, Any], namespaces: Sequence[str], examples_per_label: int = 3
) -> dict[str, list[dict[str, Any]]]:
    anchors: dict[str, list[dict[str, Any]]] = {}
    for namespace in sorted(namespaces):
        source_id, output_field = namespace.split("/", 1)
        mapping = base["sources"][source_id][output_field]["mapping"]
        grouped: dict[str, list[str]] = defaultdict(list)
        for tuple_key, label in mapping.items():
            if label is not None:
                grouped[str(label)].append(str(tuple_key))
        anchors[namespace] = [
            {
                "label": label,
                "reviewed_assignment_count": len(values),
                "tuple_examples": sorted(values)[:examples_per_label],
            }
            for label, values in sorted(grouped.items())
        ]
    return anchors


def _select_exact_cache_cover(
    target_keys: set[str], records: list[dict[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    """Select non-overlapping cached clusters that exactly reproduce a candidate."""
    by_key: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for key in record["keys"]:
            by_key[key].append(index)
    if missing := target_keys - set(by_key):
        raise ValueError(
            f"cache lacks candidate-compatible records for {namespace}: {len(missing)}"
        )
    failed: set[frozenset[str]] = set()

    def search(remaining: frozenset[str]) -> list[int] | None:
        if not remaining:
            return []
        if remaining in failed:
            return None
        key = min(
            remaining,
            key=lambda item: sum(
                records[index]["keys"] <= remaining for index in by_key[item]
            ),
        )
        options = sorted(
            (
                index
                for index in by_key[key]
                if records[index]["keys"] <= remaining
            ),
            key=lambda index: (
                -len(records[index]["keys"]),
                -records[index]["cluster_order"],
            ),
        )
        for index in options:
            suffix = search(remaining - records[index]["keys"])
            if suffix is not None:
                return [index, *suffix]
        failed.add(remaining)
        return None

    selected = search(frozenset(target_keys))
    if selected is None:
        raise ValueError(f"cache has no exact candidate-compatible cover for {namespace}")
    return [records[index] for index in selected]


def write_snapshot(
    *,
    work_root: str | Path,
    candidate_path: str | Path,
    cache_dir: str | Path,
    base_mapping_path: str | Path,
) -> dict[str, Any]:
    root = Path(work_root)
    provenance = root / "provenance"
    if provenance.exists():
        raise FileExistsError(f"incremental snapshot already exists: {provenance}")
    provenance.mkdir(parents=True, exist_ok=True)
    candidate = _read_json(candidate_path)
    base = _read_json(base_mapping_path)
    if base.get("mapping_version") != BASE_VERSION:
        raise ValueError("reviewed base mapping is not v4")
    sections = _candidate_sections(candidate)
    candidate_manifest_path = Path(candidate_path).with_name(
        "incremental_delta_manifest.json"
    )
    candidate_manifest = _read_json(candidate_manifest_path)
    if candidate_manifest.get("candidate", {}).get("sha256") != file_sha256(
        candidate_path
    ):
        raise ValueError("candidate manifest hash mismatch")
    if candidate_manifest.get("reviewed_base_mapping", {}).get(
        "sha256"
    ) != file_sha256(base_mapping_path):
        raise ValueError("candidate targets another reviewed base")

    rows: list[dict[str, Any]] = []
    cache_hashes: dict[str, str] = {}
    delta_counts: dict[str, int] = {}
    for namespace, section in sorted(sections.items()):
        source_id, output_field = namespace.split("/", 1)
        base_section = base["sources"][source_id][output_field]
        if section.get("source_columns") != base_section.get("source_columns"):
            raise ValueError(f"source columns differ for {namespace}")
        delta = section.get("mapping")
        if not isinstance(delta, dict):
            raise ValueError(f"candidate mapping is invalid for {namespace}")
        if overlap := set(delta) & set(base_section["mapping"]):
            raise ValueError(f"delta overlaps reviewed base for {namespace}: {len(overlap)}")
        delta_counts[namespace] = len(delta)
        if namespace == "skin_exposure/global_context":
            if delta:
                raise ValueError("reviewed study-design section unexpectedly has a delta")
            continue
        if not delta:
            continue
        cache_path = Path(cache_dir) / f"{source_id}__{output_field}.jsonl"
        cache_hashes[cache_path.name] = file_sha256(cache_path)
        output_spec = next(
            item
            for item in builder.SOURCE_SPECS[source_id].outputs
            if item.output_name == output_field
        )
        extraction = builder._extraction(source_id, output_spec)
        compatible_records: list[dict[str, Any]] = []
        for cluster_order, record in enumerate(_load_response_cache(cache_path)):
            members = record["members"]
            cluster_id = _cluster_id(members)
            cluster = builder.Cluster(cluster_id, tuple(members))
            identity = builder._cache_identity(
                extraction,
                cluster,
                model=str(candidate_manifest["model"]),
                reasoning_effort=str(candidate_manifest["reasoning_effort"]),
                max_tokens=int(candidate_manifest["max_tokens"]),
            )
            if record["identity"] != identity:
                raise ValueError(f"cache identity mismatch for {namespace}/{cluster_id}")
            response_sha, attempt_count, total_tokens = _accepted_response_metadata(
                record["generation"]
            )
            record_rows: list[dict[str, Any]] = []
            compatible = True
            for item_order, member in enumerate(members):
                item_id = f"v{item_order:04d}"
                tuple_key = _tuple_key(extraction, str(member))
                label = record["mapping"][item_id]
                if tuple_key not in delta or label != delta[tuple_key]:
                    compatible = False
                record_rows.append(
                    {
                        "source_id": source_id,
                        "output_field": output_field,
                        "namespace": namespace,
                        "cluster_id": cluster_id,
                        "cluster_order": cluster_order,
                        "item_id": item_id,
                        "item_order": item_order,
                        "assignment_id": _assignment_id(identity, item_id, tuple_key),
                        "tuple_key": tuple_key,
                        "raw_value": str(member),
                        "candidate_label": label,
                        "review_eligible": namespace in OPEN_NAMESPACES and label is not None,
                        "cache_identity": identity,
                        "response_sha256": response_sha,
                        "generation_attempt_count": attempt_count,
                        "generation_total_tokens": total_tokens,
                    }
                )
            if compatible:
                compatible_records.append(
                    {
                        "cluster_order": cluster_order,
                        "keys": frozenset(row["tuple_key"] for row in record_rows),
                        "rows": record_rows,
                    }
                )
        selected_records = _select_exact_cache_cover(
            set(delta), compatible_records, namespace=namespace
        )
        rows.extend(
            row for record in selected_records for row in record["rows"]
        )

    assignments = pd.DataFrame(rows)
    nonempty_open = {
        namespace for namespace in OPEN_NAMESPACES if delta_counts.get(namespace, 0)
    }
    if nonempty_open != EXPECTED_NONEMPTY_OPEN_NAMESPACES:
        raise ValueError(f"unexpected nonempty review namespaces: {sorted(nonempty_open)}")
    severity = assignments.loc[
        assignments.namespace == "direct_skin_reaction/global_severity_grade",
        "candidate_label",
    ]
    if any(
        not pd.isna(value) and value not in ALLOWED_SEVERITY
        for value in severity
    ):
        raise ValueError("frozen severity delta contains a label outside 0-4/null")
    if assignments.assignment_id.duplicated().any():
        raise ValueError("assignment IDs are not unique")
    assignments_path = provenance / "cluster_assignments.parquet"
    base_path = provenance / "reviewed_v4_base.json"
    candidate_copy = provenance / "incremental_delta_candidate.json"
    anchors_path = provenance / "existing_label_anchors.json"
    assignments.to_parquet(assignments_path, index=False, compression="zstd")
    _atomic_json(base_path, base)
    _atomic_json(candidate_copy, candidate)
    _atomic_json(
        anchors_path,
        _existing_label_anchors(base, sorted(EXPECTED_NONEMPTY_OPEN_NAMESPACES)),
    )
    outputs = {
        path.name: file_sha256(path)
        for path in (assignments_path, base_path, candidate_copy, anchors_path)
    }
    manifest = {
        "manifest_version": SNAPSHOT_VERSION,
        "publication_status": "candidate_only",
        "inputs": {
            "base_mapping": {"path": str(base_mapping_path), "sha256": file_sha256(base_mapping_path)},
            "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
            "candidate_manifest": {"path": str(candidate_manifest_path), "sha256": file_sha256(candidate_manifest_path)},
            "cluster_caches": cache_hashes,
        },
        "outputs": outputs,
        "delta_counts": delta_counts,
        "counts": {
            "assignments": len(assignments),
            "review_eligible": int(assignments.review_eligible.sum()),
            "frozen": int((~assignments.review_eligible).sum()),
        },
        "validations": {
            "reviewed_base_assignments_unchanged": True,
            "delta_keys_disjoint_from_reviewed_base": True,
            "cache_covers_delta_exactly": True,
            "sensitization_species_excluded": SPECIALIZED_NAMESPACE not in sections,
            "frozen_severity_is_0_to_4": True,
        },
    }
    _atomic_json(provenance / "manifest.json", manifest)
    return manifest


def _assignment_payload(row: Any) -> dict[str, Any]:
    return {
        "assignment_id": str(row.assignment_id),
        "item_id": str(row.item_id),
        "tuple_key": str(row.tuple_key),
        "raw_value": str(row.raw_value),
        "candidate_label": str(row.candidate_label),
    }


def write_review_packets(
    *, work_root: str | Path, max_assignments: int = 200
) -> dict[str, Any]:
    root = Path(work_root)
    provenance = root / "provenance"
    assignments = pd.read_parquet(provenance / "cluster_assignments.parquet")
    assignments = assignments.loc[assignments.review_eligible].copy()
    anchors_path = provenance / "existing_label_anchors.json"
    target = root / "review_packets"
    if target.exists():
        raise FileExistsError(f"review packets already exist: {target}")
    target.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []
    for namespace, namespace_rows in assignments.groupby("namespace", sort=True):
        clusters = [
            {
                "cluster_id": str(cluster_id),
                "cache_identity": str(group.cache_identity.iloc[0]),
                "assignments": [
                    _assignment_payload(row)
                    for row in group.sort_values("item_order").itertuples()
                ],
            }
            for cluster_id, group in namespace_rows.groupby("cluster_id", sort=False)
        ]
        current: list[dict[str, Any]] = []
        count = 0
        for cluster in clusters:
            if current and count + len(cluster["assignments"]) > max_assignments:
                packets.append((str(namespace), current))
                current, count = [], 0
            current.append(cluster)
            count += len(cluster["assignments"])
        if current:
            packets.append((str(namespace), current))
    records = []
    for index, (namespace, clusters) in enumerate(packets):
        packet_id = f"generic_delta_packet_{index + 1:04d}"
        roles = reviewer_roles(index)
        payload = {
            "packet_version": PACKET_VERSION,
            "packet_id": packet_id,
            "namespace": namespace,
            "reviewer_roles": roles,
            "review_policy": {
                "allowed_actions": ["retain", "correct", "abstain"],
                "unlisted_assignment_means": "retain",
                "scope": "new tuples only; reviewed v4 assignments are immutable",
                "existing_label_anchors": {
                    "path": str(anchors_path),
                    "sha256": file_sha256(anchors_path),
                    "namespace": namespace,
                },
            },
            "assignment_count": sum(len(cluster["assignments"]) for cluster in clusters),
            "clusters": clusters,
        }
        path = target / f"{packet_id}.json"
        _atomic_json(path, payload)
        records.append(
            {
                "packet_id": packet_id,
                "namespace": namespace,
                "path": path.name,
                "sha256": file_sha256(path),
                "clusters": len(clusters),
                "assignments": payload["assignment_count"],
                "reviewer_roles": roles,
            }
        )
    manifest = {
        "manifest_version": "skin_auxiliary.generic_incremental_review_packets.v1",
        "packets": records,
        "coverage": {
            "namespaces": int(assignments.namespace.nunique()),
            "assignments": len(assignments),
            "every_non_null_delta_assignment_owned_exactly_once": True,
            "reviewed_base_assignments_in_scope": 0,
        },
    }
    _atomic_json(target / "manifest.json", manifest)
    return manifest


def _reviews(
    paths: Sequence[str | Path], *, role: str, packet_meta: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for review in (row for path in paths for row in _load_jsonl(path)):
        packet_id = str(review.get("packet_id") or "")
        packet = packet_meta.get(packet_id)
        if packet is None or packet_id in output:
            raise ValueError(f"invalid/duplicate {role} packet review: {packet_id}")
        if (
            review.get("review_version") != REVIEW_VERSION
            or review.get("role") != role
            or review.get("reviewer_id") != packet["reviewer_roles"][role]
            or review.get("packet_sha256") != packet["sha256"]
            or review.get("complete") is not True
            or review.get("reviewed_assignment_count") != packet["assignments"]
            or not isinstance(review.get("exceptions"), list)
        ):
            raise ValueError(f"invalid {role} review contract for {packet_id}")
        output[packet_id] = review
    if set(output) != set(packet_meta):
        raise ValueError(f"{role} review packet coverage is incomplete")
    return output


def _review_decisions(
    assignments: pd.DataFrame,
    packets: Sequence[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = assignments.set_index("assignment_id")
    output: dict[str, dict[str, Any]] = {}
    for packet in packets:
        packet_id = str(packet["packet_id"])
        review = reviews[packet_id]
        exceptions = {str(item.get("assignment_id") or ""): item for item in review["exceptions"]}
        if len(exceptions) != len(review["exceptions"]):
            raise ValueError(f"duplicate exception in {packet_id}")
        packet_ids = [
            str(item["assignment_id"])
            for cluster in packet["clusters"]
            for item in cluster["assignments"]
        ]
        if set(exceptions) - set(packet_ids):
            raise ValueError(f"exception references another packet in {packet_id}")
        for assignment_id in packet_ids:
            candidate = str(rows.loc[assignment_id, "candidate_label"])
            item = exceptions.get(assignment_id)
            if item is None:
                action, label, rationale = "retain", candidate, "retain"
            else:
                action = str(item.get("action") or "")
                rationale = str(item.get("rationale") or "").strip()
                if action not in {"correct", "abstain"} or not rationale:
                    raise ValueError(f"invalid review exception for {assignment_id}")
                label = None if action == "abstain" else _clean_label(item.get("proposed_label"))
                if action == "abstain" and item.get("proposed_label") is not None:
                    raise ValueError("abstain must propose null")
                if action == "correct" and label == candidate:
                    raise ValueError("correction must differ from candidate")
            output[assignment_id] = {
                "action": action,
                "final_label": label,
                "rationale": rationale,
                "reviewer_id": review["reviewer_id"],
            }
    return output


def reconcile_reviews(
    assignments: pd.DataFrame,
    packets: Sequence[Mapping[str, Any]],
    primary_paths: Sequence[str | Path],
    checker_paths: Sequence[str | Path],
    adjudication_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    packet_meta = {
        str(packet["packet_id"]): {
            "sha256": hashlib.sha256(_json_bytes(packet)).hexdigest(),
            "assignments": int(packet["assignment_count"]),
            "reviewer_roles": packet["reviewer_roles"],
        }
        for packet in packets
    }
    primary_reviews = _reviews(primary_paths, role="primary", packet_meta=packet_meta)
    checker_reviews = _reviews(checker_paths, role="checker", packet_meta=packet_meta)
    primary = _review_decisions(assignments, packets, primary_reviews)
    checker = _review_decisions(assignments, packets, checker_reviews)
    routed = {
        assignment_id
        for assignment_id in primary
        if primary[assignment_id]["action"] != "retain"
        or checker[assignment_id]["action"] != "retain"
        or primary[assignment_id]["final_label"] != checker[assignment_id]["final_label"]
    }
    adjudication_rows = [
        row for path in adjudication_paths for row in _load_jsonl(path)
    ]
    adjudications = {
        str(row.get("assignment_id") or ""): row for row in adjudication_rows
    }
    if len(adjudications) != len(adjudication_rows):
        raise ValueError("adjudications repeat an assignment")
    if set(adjudications) != routed:
        raise ValueError("adjudication coverage differs from routed assignments")
    packet_by_assignment = {
        str(item["assignment_id"]): packet
        for packet in packets
        for cluster in packet["clusters"]
        for item in cluster["assignments"]
    }
    final = assignments.copy()
    final["final_label"] = final["candidate_label"]
    final["adjudicated"] = False
    final["review_rationale"] = None
    index = {str(value): idx for idx, value in enumerate(final.assignment_id)}
    for assignment_id in primary:
        idx = index[assignment_id]
        if assignment_id not in routed:
            final.at[idx, "review_rationale"] = "two independent retains"
            continue
        item = adjudications[assignment_id]
        packet = packet_by_assignment[assignment_id]
        if item.get("reviewer_id") != packet["reviewer_roles"]["adjudicator"]:
            raise ValueError(f"wrong adjudicator for {assignment_id}")
        rationale = str(item.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(f"adjudication lacks rationale for {assignment_id}")
        final.at[idx, "final_label"] = _clean_label(item.get("final_label"))
        final.at[idx, "adjudicated"] = True
        final.at[idx, "review_rationale"] = rationale
    return final, {
        "assignments_reviewed_twice": len(primary),
        "routed_to_adjudication": len(routed),
        "primary_complete": True,
        "checker_complete": True,
        "independent_reviewers": all(
            primary[assignment_id]["reviewer_id"]
            != checker[assignment_id]["reviewer_id"]
            for assignment_id in primary
        ),
    }


def write_adjudication_packets(
    *,
    work_root: str | Path,
    primary_paths: Sequence[str | Path],
    checker_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Package every proposed change or disagreement for the third reviewer."""
    root = Path(work_root)
    assignments = pd.read_parquet(root / "provenance/cluster_assignments.parquet")
    assignments = assignments.loc[assignments.review_eligible].copy()
    packet_dir = root / "review_packets"
    packet_manifest = _read_json(packet_dir / "manifest.json")
    packets = [_read_json(packet_dir / item["path"]) for item in packet_manifest["packets"]]
    packet_meta = {
        str(packet["packet_id"]): {
            "sha256": hashlib.sha256(_json_bytes(packet)).hexdigest(),
            "assignments": int(packet["assignment_count"]),
            "reviewer_roles": packet["reviewer_roles"],
        }
        for packet in packets
    }
    primary = _review_decisions(
        assignments,
        packets,
        _reviews(primary_paths, role="primary", packet_meta=packet_meta),
    )
    checker = _review_decisions(
        assignments,
        packets,
        _reviews(checker_paths, role="checker", packet_meta=packet_meta),
    )
    packet_by_assignment = {
        str(item["assignment_id"]): packet
        for packet in packets
        for cluster in packet["clusters"]
        for item in cluster["assignments"]
    }
    routed = sorted(
        assignment_id
        for assignment_id in primary
        if primary[assignment_id]["action"] != "retain"
        or checker[assignment_id]["action"] != "retain"
        or primary[assignment_id]["final_label"]
        != checker[assignment_id]["final_label"]
    )
    by_assignment = assignments.set_index("assignment_id", drop=False)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment_id in routed:
        packet = packet_by_assignment[assignment_id]
        reviewer_id = str(packet["reviewer_roles"]["adjudicator"])
        row = by_assignment.loc[assignment_id]
        grouped[reviewer_id].append(
            {
                "assignment_id": assignment_id,
                "packet_id": packet["packet_id"],
                "namespace": str(row.namespace),
                "tuple_key": str(row.tuple_key),
                "raw_value": str(row.raw_value),
                "candidate_label": str(row.candidate_label),
                "primary_decision": primary[assignment_id],
                "checker_decision": checker[assignment_id],
                "instruction": (
                    "Choose the shortest scientifically accurate canonical label, "
                    "prefer an equivalent reviewed-v4 anchor, or set final_label to "
                    "null only when the raw value does not support a label."
                ),
            }
        )
    target = root / "adjudication_packets"
    if target.exists():
        raise FileExistsError(f"adjudication packets already exist: {target}")
    target.mkdir(parents=True)
    records = []
    for reviewer_id, items in sorted(grouped.items()):
        payload = {
            "packet_version": "skin_auxiliary.generic_incremental_adjudication_packet.v1",
            "reviewer_id": reviewer_id,
            "assignment_count": len(items),
            "assignments": items,
        }
        path = target / f"{reviewer_id}.json"
        _atomic_json(path, payload)
        records.append(
            {
                "reviewer_id": reviewer_id,
                "path": path.name,
                "sha256": file_sha256(path),
                "assignments": len(items),
            }
        )
    manifest = {
        "manifest_version": "skin_auxiliary.generic_incremental_adjudication_packets.v1",
        "routed_assignments": len(routed),
        "reviewer_disagreements": sum(
            primary[value]["final_label"] != checker[value]["final_label"]
            for value in routed
        ),
        "packets": records,
    }
    _atomic_json(target / "manifest.json", manifest)
    return manifest


def build_proposal(
    base: Mapping[str, Any], candidate: Mapping[str, Any], reviewed: pd.DataFrame
) -> dict[str, Any]:
    proposal = json.loads(json.dumps(base, ensure_ascii=False))
    proposal["mapping_version"] = PROPOSAL_VERSION
    reviewed_by_key = {
        (str(row.namespace), str(row.tuple_key)): (
            None if pd.isna(row.final_label) else str(row.final_label)
        )
        for row in reviewed.itertuples()
    }
    for namespace, section in _candidate_sections(candidate).items():
        source_id, output_field = namespace.split("/", 1)
        base_values = base["sources"][source_id][output_field]["mapping"]
        delta = section["mapping"]
        if set(base_values) & set(delta):
            raise ValueError(f"delta overlaps reviewed base for {namespace}")
        if namespace == "direct_skin_reaction/global_severity_grade":
            reviewed_delta = delta
        elif namespace == "skin_exposure/global_context":
            reviewed_delta = {}
        else:
            reviewed_delta = {
                key: reviewed_by_key[(namespace, key)] for key in delta
            }
        proposal["sources"][source_id][output_field]["mapping"] = dict(
            sorted({**base_values, **reviewed_delta}.items())
        )
    if proposal["sources"]["sensitization_aop"]["global_species_context"] != base[
        "sources"
    ]["sensitization_aop"]["global_species_context"]:
        raise AssertionError("specialized sensitization species section changed")
    for source_id, outputs in base["sources"].items():
        for output_field, section in outputs.items():
            proposed = proposal["sources"][source_id][output_field]["mapping"]
            if any(proposed.get(key) != value for key, value in section["mapping"].items()):
                raise AssertionError("reviewed v4 base assignment changed")
    return proposal


def write_proposal(
    *,
    work_root: str | Path,
    primary_paths: Sequence[str | Path],
    checker_paths: Sequence[str | Path],
    adjudication_paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(work_root)
    provenance = root / "provenance"
    snapshot = _read_json(provenance / "manifest.json")
    for filename, expected in snapshot["outputs"].items():
        if file_sha256(provenance / filename) != expected:
            raise ValueError(f"snapshot output hash mismatch: {filename}")
    assignments = pd.read_parquet(provenance / "cluster_assignments.parquet")
    base = _read_json(provenance / "reviewed_v4_base.json")
    candidate = _read_json(provenance / "incremental_delta_candidate.json")
    packet_dir = root / "review_packets"
    packet_manifest = _read_json(packet_dir / "manifest.json")
    packets = [_read_json(packet_dir / row["path"]) for row in packet_manifest["packets"]]
    reviewed, review_audit = reconcile_reviews(
        assignments.loc[assignments.review_eligible].copy(),
        packets,
        primary_paths,
        checker_paths,
        adjudication_paths,
    )
    frozen = assignments.loc[~assignments.review_eligible].copy()
    frozen["final_label"] = frozen["candidate_label"]
    reviewed = pd.concat([reviewed, frozen], ignore_index=True)
    proposal = build_proposal(base, candidate, reviewed)
    target = root / "proposal"
    if target.exists():
        raise FileExistsError(f"incremental proposal already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)
    proposal_path = target / "proposed_globally_reconciled_mapping.v5.json"
    assignments_path = target / "reviewed_delta_assignments.parquet"
    _atomic_json(proposal_path, proposal)
    reviewed.to_parquet(assignments_path, index=False, compression="zstd")
    for filename, paths in (
        ("primary_review.jsonl", primary_paths),
        ("checker_review.jsonl", checker_paths),
        ("adjudication_review.jsonl", adjudication_paths),
    ):
        _atomic_jsonl(target / filename, [row for path in paths for row in _load_jsonl(path)])
    files = {
        path.name: file_sha256(path)
        for path in (
            proposal_path,
            assignments_path,
            target / "primary_review.jsonl",
            target / "checker_review.jsonl",
            target / "adjudication_review.jsonl",
        )
    }
    manifest = {
        "manifest_version": PROPOSAL_MANIFEST_VERSION,
        "publication_status": "awaiting_human_approval",
        "base_version": BASE_VERSION,
        "proposal_version": PROPOSAL_VERSION,
        "review": review_audit,
        "files": files,
        "validations": {
            "two_complete_independent_reviews": review_audit["independent_reviewers"],
            "all_nonretain_or_disputed_assignments_adjudicated": True,
            "reviewed_base_assignments_unchanged": True,
            "delta_keys_disjoint_from_reviewed_base": True,
            "sensitization_species_unchanged": True,
            "frozen_severity_applied_deterministically": True,
        },
        "publication_blockers": ["human_approval_not_recorded"],
    }
    _atomic_json(target / "INTEGRITY_MANIFEST.json", manifest)
    return manifest


def _validated_proposal(
    proposal_path: str | Path,
    manifest_path: str | Path,
    *,
    manifest_version: str,
) -> dict[str, Any]:
    proposal_path = Path(proposal_path)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("manifest_version") != manifest_version
        or manifest.get("base_version") != BASE_VERSION
        or manifest.get("proposal_version") != PROPOSAL_VERSION
        or manifest.get("publication_status") != "awaiting_human_approval"
        or not all(manifest.get("validations", {}).values())
        or manifest.get("files", {}).get(proposal_path.name)
        != file_sha256(proposal_path)
    ):
        raise ValueError(f"invalid proposal manifest: {manifest_path}")
    proposal = _read_json(proposal_path)
    if proposal.get("mapping_version") != PROPOSAL_VERSION:
        raise ValueError(f"proposal is not v5: {proposal_path}")
    return proposal


def _proposal_additions(
    base: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    allowed_namespaces: frozenset[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    if set(proposal.get("sources", {})) != set(base.get("sources", {})):
        raise ValueError("proposal source inventory differs from reviewed v4")
    additions: dict[tuple[str, str], dict[str, Any]] = {}
    for source_id, outputs in base["sources"].items():
        proposed_outputs = proposal["sources"].get(source_id, {})
        if set(proposed_outputs) != set(outputs):
            raise ValueError(f"proposal output inventory differs for {source_id}")
        for output_field, base_section in outputs.items():
            namespace = _namespace(source_id, output_field)
            proposed_section = proposed_outputs[output_field]
            if proposed_section.get("source_columns") != base_section.get("source_columns"):
                raise ValueError(f"proposal source columns differ for {namespace}")
            base_mapping = base_section.get("mapping")
            proposed_mapping = proposed_section.get("mapping")
            if not isinstance(base_mapping, dict) or not isinstance(proposed_mapping, dict):
                raise ValueError(f"invalid mapping section for {namespace}")
            if any(proposed_mapping.get(key) != value for key, value in base_mapping.items()):
                raise ValueError(f"proposal changed a reviewed v4 assignment: {namespace}")
            delta = {
                key: value
                for key, value in proposed_mapping.items()
                if key not in base_mapping
            }
            if delta and namespace not in allowed_namespaces:
                raise ValueError(f"proposal changed an unauthorized namespace: {namespace}")
            if delta:
                additions[(source_id, output_field)] = delta
    return additions


def merge_species_proposal(
    *,
    base_mapping_path: str | Path,
    generic_proposal_path: str | Path,
    generic_manifest_path: str | Path,
    species_proposal_path: str | Path,
    species_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compose independently reviewed generic and species deltas without publishing."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"merged proposal output already exists: {target}")
    base = _read_json(base_mapping_path)
    if base.get("mapping_version") != BASE_VERSION:
        raise ValueError("reviewed base mapping is not v4")
    generic = _validated_proposal(
        generic_proposal_path,
        generic_manifest_path,
        manifest_version=PROPOSAL_MANIFEST_VERSION,
    )
    species = _validated_proposal(
        species_proposal_path,
        species_manifest_path,
        manifest_version="skin_species_context_incremental_proposal.v1",
    )
    generic_delta = _proposal_additions(
        base,
        generic,
        allowed_namespaces=OPEN_NAMESPACES | FROZEN_NAMESPACES,
    )
    species_delta = _proposal_additions(
        base,
        species,
        allowed_namespaces=frozenset({SPECIALIZED_NAMESPACE}),
    )
    generic_keys = {
        (source_id, output_field, key)
        for (source_id, output_field), mapping in generic_delta.items()
        for key in mapping
    }
    species_keys = {
        (source_id, output_field, key)
        for (source_id, output_field), mapping in species_delta.items()
        for key in mapping
    }
    if generic_keys & species_keys:
        raise ValueError("generic and species proposal deltas overlap")
    if not species_keys:
        raise ValueError("specialized species proposal contains no delta")

    inherited_cleaned_aliases = _inherit_cleaned_base_labels(base, generic_delta)
    inherited_cleaned_aliases += _inherit_cleaned_base_labels(base, species_delta)

    merged = json.loads(json.dumps(base, ensure_ascii=False))
    merged["mapping_version"] = PROPOSAL_VERSION
    for delta in (generic_delta, species_delta):
        for (source_id, output_field), mapping in delta.items():
            section = merged["sources"][source_id][output_field]["mapping"]
            section.update(mapping)
            merged["sources"][source_id][output_field]["mapping"] = dict(
                sorted(section.items())
            )
    _validate_cleaned_mapping(merged)
    target.mkdir(parents=True)
    output_path = target / "globally_reconciled_auxiliary_value_mapping.v5.json"
    _atomic_json(output_path, merged)
    manifest = {
        "manifest_version": MERGE_MANIFEST_VERSION,
        "publication_status": "awaiting_human_approval",
        "base_version": BASE_VERSION,
        "output_version": PROPOSAL_VERSION,
        "inputs": {
            "reviewed_v4_base": {
                "path": str(base_mapping_path),
                "sha256": file_sha256(base_mapping_path),
            },
            "generic_proposal": {
                "path": str(generic_proposal_path),
                "sha256": file_sha256(generic_proposal_path),
                "manifest_path": str(generic_manifest_path),
                "manifest_sha256": file_sha256(generic_manifest_path),
            },
            "species_proposal": {
                "path": str(species_proposal_path),
                "sha256": file_sha256(species_proposal_path),
                "manifest_path": str(species_manifest_path),
                "manifest_sha256": file_sha256(species_manifest_path),
            },
        },
        "output": {"path": str(output_path), "sha256": file_sha256(output_path)},
        "counts": {
            "generic_delta_tuples": len(generic_keys),
            "species_delta_tuples": len(species_keys),
            "merged_delta_tuples": len(generic_keys | species_keys),
            "cleaned_aliases_inheriting_reviewed_v4_label": inherited_cleaned_aliases,
        },
        "validations": {
            "proposal_manifests_and_hashes_valid": True,
            "generic_changes_only_non_species_delta_keys": True,
            "species_changes_only_specialized_delta_keys": True,
            "generic_and_species_deltas_disjoint": True,
            "every_reviewed_v4_assignment_unchanged": True,
            "cleaned_tuple_keys_conflict_free": True,
        },
        "publication_blockers": ["human_approval_not_recorded"],
    }
    _atomic_json(target / "INTEGRITY_MANIFEST.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--work-root", required=True)
    snapshot.add_argument("--candidate", required=True)
    snapshot.add_argument("--cache-dir", required=True)
    snapshot.add_argument("--base-mapping", required=True)
    prepare = commands.add_parser("prepare-review")
    prepare.add_argument("--work-root", required=True)
    prepare.add_argument("--max-assignments", type=int, default=200)
    adjudicate = commands.add_parser("prepare-adjudication")
    adjudicate.add_argument("--work-root", required=True)
    adjudicate.add_argument("--primary", nargs="+", required=True)
    adjudicate.add_argument("--checker", nargs="+", required=True)
    propose = commands.add_parser("propose")
    propose.add_argument("--work-root", required=True)
    propose.add_argument("--primary", nargs="+", required=True)
    propose.add_argument("--checker", nargs="+", required=True)
    propose.add_argument("--adjudications", nargs="+", required=True)
    merge = commands.add_parser("merge-species")
    merge.add_argument("--base-mapping", required=True)
    merge.add_argument("--generic-proposal", required=True)
    merge.add_argument("--generic-manifest", required=True)
    merge.add_argument("--species-proposal", required=True)
    merge.add_argument("--species-manifest", required=True)
    merge.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "snapshot":
        result = write_snapshot(
            work_root=args.work_root,
            candidate_path=args.candidate,
            cache_dir=args.cache_dir,
            base_mapping_path=args.base_mapping,
        )
    elif args.command == "prepare-review":
        result = write_review_packets(
            work_root=args.work_root, max_assignments=args.max_assignments
        )
    elif args.command == "prepare-adjudication":
        result = write_adjudication_packets(
            work_root=args.work_root,
            primary_paths=args.primary,
            checker_paths=args.checker,
        )
    elif args.command == "propose":
        result = write_proposal(
            work_root=args.work_root,
            primary_paths=args.primary,
            checker_paths=args.checker,
            adjudication_paths=args.adjudications,
        )
    else:
        result = merge_species_proposal(
            base_mapping_path=args.base_mapping,
            generic_proposal_path=args.generic_proposal,
            generic_manifest_path=args.generic_manifest,
            species_proposal_path=args.species_proposal,
            species_manifest_path=args.species_manifest,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
