"""Human-gated global reconciliation for structured Skin species context.

This module never calls a model.  It reconstructs the exact completed local
GPT pass, packages complete structured evidence for independent Codex review,
validates two full reviews plus third-party adjudication, and writes an
unpublished v4 proposal.  Publication and downstream rebuilding are separate
human-gated operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .build_embedding_bucket_mapping import (
    _SPECIES_LABEL_ALIASES,
    _SPECIES_LITERAL_PATTERNS,
)


SOURCE_ID = "sensitization_aop"
OUTPUT_FIELD = "global_species_context"
NAMESPACE = f"{SOURCE_ID}/{OUTPUT_FIELD}"
CANDIDATE_VERSION = "starling_auxiliary.skin_reaction.globally_reconciled.v3"
PROPOSAL_VERSION = "starling_auxiliary.skin_reaction.globally_reconciled.v4"
PACKET_VERSION = "skin_species_context_review_packet.v1"
REVIEW_VERSION = "skin_species_context_assignment_review.v1"
AUDIT_VERSION = "skin_species_context_reconciliation_integrity.v1"

DATA_PROCESSING_DIR = Path(__file__).resolve().parent
DEFAULT_SPECIES_ROOT = DATA_PROCESSING_DIR / "species_context_v3"
DEFAULT_RECONCILIATION_ROOT = DEFAULT_SPECIES_ROOT / "reconciliation"
DEFAULT_CACHE = (
    DEFAULT_SPECIES_ROOT
    / "cluster_cache"
    / "sensitization_aop__global_species_context.jsonl"
)
DEFAULT_CANDIDATE_MAPPING = (
    DEFAULT_RECONCILIATION_ROOT / "provenance" / "candidate_v3_mapping.json"
)
DEFAULT_MAPPING_MANIFEST = DEFAULT_SPECIES_ROOT / "mapping_manifest.json"
DEFAULT_TOKEN_LEDGER = DEFAULT_SPECIES_ROOT / "token_ledger.json"
DEFAULT_PROMPT_REGISTRY = DATA_PROCESSING_DIR / "auxiliary_value_prompts.json"

REVIEWER_IDS = (
    "skin_species_audit_a",
    "skin_species_audit_b",
    "skin_species_audit_c",
)
CONTROLLED_SPECIES = frozenset(
    {
        "human",
        "rat",
        "mouse",
        "dog",
        "guinea pig",
        "pig",
        "rabbit",
        "cynomolgus monkey",
        "rhesus monkey",
        "monkey",
        "sheep",
        "cattle",
        "frog",
        "snake",
        "hamster",
        "chicken",
        "horse",
        "cat",
        "goat",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(_json_bytes(value))
        handle.flush()
    temporary.replace(target)


def _atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
    temporary.replace(target)


def encode_species_tuple(fields: Sequence[str | None]) -> str:
    if len(fields) != 3 or any(
        value is not None and not isinstance(value, str) for value in fields
    ):
        raise ValueError("species tuple must contain exactly three strings/nulls")
    return json.dumps(list(fields), ensure_ascii=False, separators=(",", ":"))


def decode_species_tuple(key: str) -> tuple[str | None, str | None, str | None]:
    try:
        fields = json.loads(key)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid species tuple JSON: {key!r}") from exc
    if not isinstance(fields, list) or len(fields) != 3 or any(
        value is not None and not isinstance(value, str) for value in fields
    ):
        raise ValueError("species tuple must contain exactly three strings/nulls")
    return fields[0], fields[1], fields[2]


def validate_literal_support(label: str | None, fields: Sequence[str | None]) -> bool:
    if label is None:
        return True
    if label not in CONTROLLED_SPECIES:
        return False
    canonical = _SPECIES_LABEL_ALIASES.get(label, label)
    pattern = _SPECIES_LITERAL_PATTERNS.get(canonical)
    text = " | ".join(value for value in fields if value)
    return pattern is not None and pattern.search(text) is not None


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _candidate_section(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    if mapping.get("mapping_version") != CANDIDATE_VERSION:
        raise ValueError("candidate mapping is not the frozen v3 mapping")
    try:
        section = mapping["sources"][SOURCE_ID][OUTPUT_FIELD]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"candidate mapping lacks {NAMESPACE}") from exc
    if section.get("source_columns") != [
        "assay_type",
        "experimental_conditions",
        "support_text",
    ]:
        raise ValueError("candidate species section has the wrong source columns")
    if not isinstance(section.get("mapping"), dict):
        raise ValueError("candidate species section lacks a mapping")
    return section


def build_candidate_assignments(
    cache_records: Sequence[Mapping[str, Any]], candidate_section: Mapping[str, Any]
) -> pd.DataFrame:
    """Join raw GPT outputs to the effective v3 candidate for every tuple."""
    candidate = candidate_section.get("mapping")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate_section.mapping must be an object")
    rows: list[dict[str, Any]] = []
    seen_tuples: set[str] = set()
    for cluster_index, record in enumerate(cache_records):
        members = record.get("members")
        raw_mapping = record.get("mapping")
        if not isinstance(members, list) or not isinstance(raw_mapping, Mapping):
            raise ValueError(f"invalid cache record {cluster_index}")
        expected_ids = {f"v{index:04d}" for index in range(len(members))}
        if set(raw_mapping) != expected_ids:
            raise ValueError(f"cache item IDs differ at cluster {cluster_index}")
        generation = record.get("generation") or {}
        attempts = generation.get("attempts") or []
        response_sha = attempts[-1].get("response_sha256") if attempts else None
        cache_identity = str(record.get("identity") or "")
        cluster_id = f"cluster_{cluster_index:04d}"
        for item_index, member in enumerate(members):
            if not isinstance(member, str):
                raise ValueError("cache member must be a tuple-key string")
            fields = decode_species_tuple(member)
            tuple_key = encode_species_tuple(fields)
            if tuple_key in seen_tuples:
                raise ValueError(f"tuple occurs in multiple cache clusters: {tuple_key}")
            seen_tuples.add(tuple_key)
            if tuple_key not in candidate:
                raise ValueError("cache tuple is absent from candidate mapping")
            item_id = f"v{item_index:04d}"
            raw_label = raw_mapping[item_id]
            final_candidate = candidate[tuple_key]
            if raw_label is not None and not isinstance(raw_label, str):
                raise ValueError("raw GPT label must be string/null")
            if final_candidate is not None and final_candidate not in CONTROLLED_SPECIES:
                raise ValueError(f"candidate label is uncontrolled: {final_candidate!r}")
            tuple_sha = hashlib.sha256(tuple_key.encode("utf-8")).hexdigest()
            assignment_id = hashlib.sha256(
                f"{cache_identity}\0{item_id}\0{tuple_sha}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "source_id": SOURCE_ID,
                    "output_field": OUTPUT_FIELD,
                    "namespace": NAMESPACE,
                    "cluster_id": cluster_id,
                    "cluster_order": cluster_index,
                    "cache_identity": cache_identity,
                    "response_sha256": response_sha,
                    "item_id": item_id,
                    "assignment_id": assignment_id,
                    "tuple_key": tuple_key,
                    "tuple_sha256": tuple_sha,
                    "assay_type": fields[0],
                    "experimental_conditions": fields[1],
                    "support_text": fields[2],
                    "raw_gpt_label": raw_label,
                    "candidate_label": final_candidate,
                    "literal_support_valid": validate_literal_support(
                        final_candidate, fields
                    ),
                    "review_eligible": final_candidate is not None,
                }
            )
    if set(candidate) != seen_tuples:
        raise ValueError(
            "candidate/cache tuple coverage differs: "
            f"candidate={len(candidate)} cache={len(seen_tuples)}"
        )
    frame = pd.DataFrame(rows).sort_values(
        ["cluster_order", "item_id"], ignore_index=True
    )
    if frame["assignment_id"].duplicated().any():
        raise ValueError("assignment IDs are not unique")
    if not frame.loc[frame["review_eligible"], "literal_support_valid"].all():
        raise ValueError("v3 candidate contains a non-null unsupported species")
    return frame


def _assignment_payload(row: Any) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        return None if pd.isna(value) else value

    return {
        "assignment_id": str(row.assignment_id),
        "item_id": str(row.item_id),
        "tuple_sha256": str(row.tuple_sha256),
        "candidate_label": str(row.candidate_label),
        "raw_gpt_label": clean(row.raw_gpt_label),
        "evidence": {
            "assay_type": clean(row.assay_type),
            "experimental_conditions": clean(row.experimental_conditions),
            "support_text": clean(row.support_text),
        },
    }


def packetize_assignments(
    assignments: pd.DataFrame,
    max_packet_bytes: int = 128 * 1024,
    max_assignments: int = 200,
) -> list[dict[str, Any]]:
    """Return cluster-atomic packets covering every eligible assignment once."""
    eligible = assignments.loc[assignments["review_eligible"]].copy()
    clusters: list[dict[str, Any]] = []
    for cluster_id, group in eligible.groupby("cluster_id", sort=False):
        items = [
            _assignment_payload(row)
            for row in group.sort_values("item_id").itertuples(index=False)
        ]
        clusters.append(
            {
                "cluster_id": str(cluster_id),
                "cache_identity": str(group["cache_identity"].iloc[0]),
                "assignments": items,
            }
        )
    packets: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_count = 0
    for cluster in clusters:
        candidate_clusters = current + [cluster]
        packet_index = len(packets)
        candidate_payload = _packet_payload(packet_index, candidate_clusters)
        candidate_count = current_count + len(cluster["assignments"])
        exceeds = (
            len(_json_bytes(candidate_payload)) > max_packet_bytes
            or candidate_count > max_assignments
        )
        if current and exceeds:
            packets.append(_packet_payload(packet_index, current))
            current = []
            current_count = 0
        current.append(cluster)
        current_count += len(cluster["assignments"])
    if current:
        packets.append(_packet_payload(len(packets), current))
    owned = [
        item["assignment_id"]
        for packet in packets
        for cluster in packet["clusters"]
        for item in cluster["assignments"]
    ]
    if len(owned) != len(set(owned)) or set(owned) != set(eligible["assignment_id"]):
        raise AssertionError("packet assignment ownership is not exact")
    return packets


def reviewer_roles(packet_index: int) -> dict[str, str]:
    offset = packet_index % 3
    return {
        "primary": REVIEWER_IDS[offset],
        "checker": REVIEWER_IDS[(offset + 1) % 3],
        "adjudicator": REVIEWER_IDS[(offset + 2) % 3],
    }


def _packet_payload(packet_index: int, clusters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    packet_id = f"species_context_packet_{packet_index + 1:04d}"
    roles = reviewer_roles(packet_index)
    count = sum(len(cluster["assignments"]) for cluster in clusters)
    return {
        "packet_version": PACKET_VERSION,
        "packet_id": packet_id,
        "namespace": NAMESPACE,
        "review_policy": {
            "eligible_subject": "unique measurement-producing subject, donor, or cell-system species",
            "invalid_roles": [
                "reagent",
                "background",
                "comparator",
                "citation",
                "acronym-only inference",
                "conflicting or unresolved multiple species",
            ],
            "allowed_actions": ["retain", "correct", "abstain"],
            "controlled_species": sorted(CONTROLLED_SPECIES),
            "unlisted_assignment_means": "retain",
        },
        "reviewer_roles": roles,
        "assignment_count": count,
        "clusters": list(clusters),
    }


def write_review_packets(
    assignments: pd.DataFrame,
    *,
    output_dir: str | Path,
    max_packet_bytes: int = 128 * 1024,
    max_assignments: int = 200,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    packets = packetize_assignments(
        assignments,
        max_packet_bytes=max_packet_bytes,
        max_assignments=max_assignments,
    )
    records: list[dict[str, Any]] = []
    for packet in packets:
        path = target / f"{packet['packet_id']}.json"
        _atomic_json(path, packet)
        records.append(
            {
                "packet_id": packet["packet_id"],
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "clusters": len(packet["clusters"]),
                "assignments": packet["assignment_count"],
                "reviewer_roles": packet["reviewer_roles"],
                "oversize_single_cluster": (
                    path.stat().st_size > max_packet_bytes
                    and len(packet["clusters"]) == 1
                ),
            }
        )
    manifest = {
        "manifest_version": "skin_species_context_review_packets.v1",
        "namespace": NAMESPACE,
        "candidate_version": CANDIDATE_VERSION,
        "max_packet_bytes": max_packet_bytes,
        "max_assignments": max_assignments,
        "packets": records,
        "coverage": {
            "packets": len(records),
            "clusters": int(assignments.loc[assignments.review_eligible, "cluster_id"].nunique()),
            "assignments": int(assignments["review_eligible"].sum()),
            "every_assignment_owned_exactly_once": True,
            "clusters_are_atomic": True,
        },
    }
    _atomic_json(target / "manifest.json", manifest)
    return manifest


def _review_index(
    reviews: Sequence[Mapping[str, Any]],
    *,
    role: str,
    packets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if role not in {"primary", "checker"}:
        raise ValueError("full review role must be primary/checker")
    output: dict[str, dict[str, Any]] = {}
    for review in reviews:
        packet_id = str(review.get("packet_id") or "")
        if packet_id not in packets or packet_id in output:
            raise ValueError(f"invalid/duplicate {role} packet review: {packet_id}")
        packet = packets[packet_id]
        if review.get("review_version") != REVIEW_VERSION:
            raise ValueError(f"wrong review version for {packet_id}")
        if review.get("role") != role:
            raise ValueError(f"wrong review role for {packet_id}")
        if review.get("reviewer_id") != packet["reviewer_roles"][role]:
            raise ValueError(f"wrong reviewer identity for {packet_id}/{role}")
        if review.get("packet_sha256") != packet["sha256"]:
            raise ValueError(f"packet hash mismatch for {packet_id}/{role}")
        if review.get("complete") is not True:
            raise ValueError(f"incomplete {role} review for {packet_id}")
        if review.get("reviewed_assignment_count") != packet["assignments"]:
            raise ValueError(f"reviewed assignment count mismatch for {packet_id}")
        exceptions = review.get("exceptions")
        if not isinstance(exceptions, list):
            raise ValueError(f"exceptions must be a list for {packet_id}")
        output[packet_id] = dict(review)
    if set(output) != set(packets):
        raise ValueError(
            f"{role} review coverage mismatch: missing={len(set(packets)-set(output))}"
        )
    return output


def validate_two_review_coverage(
    primary_reviews: Sequence[Mapping[str, Any]],
    checker_reviews: Sequence[Mapping[str, Any]],
    packet_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    packet_rows = packet_manifest.get("packets")
    if not isinstance(packet_rows, list) or not packet_rows:
        raise ValueError("packet manifest has no packets")
    packets = {str(row["packet_id"]): row for row in packet_rows}
    if len(packets) != len(packet_rows):
        raise ValueError("packet manifest repeats packet IDs")
    primary = _review_index(primary_reviews, role="primary", packets=packets)
    checker = _review_index(checker_reviews, role="checker", packets=packets)
    for packet_id, packet in packets.items():
        if primary[packet_id]["reviewer_id"] == checker[packet_id]["reviewer_id"]:
            raise ValueError(f"primary/checker must differ for {packet_id}")
    return {
        "packets": len(packets),
        "assignments_reviewed_twice": sum(int(row["assignments"]) for row in packet_rows),
        "primary_complete": True,
        "checker_complete": True,
        "independent_reviewers": True,
    }


def _validate_exception(
    item: Mapping[str, Any], *, candidate_label: str
) -> tuple[str | None, str]:
    action = str(item.get("action") or "")
    if action not in {"correct", "abstain"}:
        raise ValueError(f"invalid exception action: {action}")
    rationale = str(item.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("review exception lacks rationale")
    proposed = item.get("proposed_label")
    if action == "abstain":
        if proposed is not None:
            raise ValueError("abstain must propose null")
        return None, rationale
    if not isinstance(proposed, str) or proposed not in CONTROLLED_SPECIES:
        raise ValueError(f"correction has uncontrolled species: {proposed!r}")
    if proposed == candidate_label:
        raise ValueError("correction must differ from candidate")
    return proposed, rationale


def _full_decisions(
    assignments: pd.DataFrame,
    reviews: Sequence[Mapping[str, Any]],
    packets: Mapping[str, Mapping[str, Any]],
    assignment_packet: Mapping[str, str],
    *,
    role: str,
) -> dict[str, dict[str, Any]]:
    review_by_packet = _review_index(reviews, role=role, packets=packets)
    by_assignment = assignments.set_index("assignment_id", drop=False)
    decisions: dict[str, dict[str, Any]] = {}
    for packet_id, review in review_by_packet.items():
        exceptions: dict[str, Mapping[str, Any]] = {}
        for item in review["exceptions"]:
            assignment_id = str(item.get("assignment_id") or "")
            if assignment_id in exceptions:
                raise ValueError(f"duplicate exception for {assignment_id}")
            if assignment_id not in by_assignment.index:
                raise ValueError(f"exception references unknown assignment {assignment_id}")
            if assignment_packet.get(assignment_id) != packet_id:
                raise ValueError(f"exception belongs to another packet: {assignment_id}")
            exceptions[assignment_id] = item
        packet_ids = [
            assignment_id
            for assignment_id, owned_packet in assignment_packet.items()
            if owned_packet == packet_id
        ]
        for assignment_id in packet_ids:
            row = by_assignment.loc[assignment_id]
            candidate = str(row["candidate_label"])
            if assignment_id in exceptions:
                final_label, rationale = _validate_exception(
                    exceptions[assignment_id], candidate_label=candidate
                )
                fields = (
                    row["assay_type"],
                    row["experimental_conditions"],
                    row["support_text"],
                )
                if final_label is not None and not validate_literal_support(final_label, fields):
                    raise ValueError(
                        f"correction lacks literal support: {assignment_id}/{final_label}"
                    )
                action = str(exceptions[assignment_id]["action"])
            else:
                final_label, rationale, action = candidate, "retain", "retain"
            decisions[assignment_id] = {
                "assignment_id": assignment_id,
                "action": action,
                "final_label": final_label,
                "rationale": rationale,
                "reviewer_id": review["reviewer_id"],
                "packet_id": packet_id,
            }
    return decisions


def _packet_assignment_ownership(packet_payloads: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ownership: dict[str, str] = {}
    for packet in packet_payloads:
        packet_id = str(packet["packet_id"])
        for cluster in packet["clusters"]:
            for item in cluster["assignments"]:
                assignment_id = str(item["assignment_id"])
                if assignment_id in ownership:
                    raise ValueError("assignment appears in multiple packets")
                ownership[assignment_id] = packet_id
    return ownership


def reconcile_assignment_reviews(
    assignments: pd.DataFrame,
    packets: Sequence[Mapping[str, Any]],
    primary_reviews: Sequence[Mapping[str, Any]],
    checker_reviews: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Consolidate two complete reviews; every non-retain case needs a third."""
    packet_meta: dict[str, dict[str, Any]] = {}
    for packet in packets:
        packet_id = str(packet["packet_id"])
        payload_sha = hashlib.sha256(_json_bytes(packet)).hexdigest()
        packet_meta[packet_id] = {
            "packet_id": packet_id,
            "sha256": payload_sha,
            "assignments": int(packet["assignment_count"]),
            "reviewer_roles": dict(packet["reviewer_roles"]),
        }
    ownership = _packet_assignment_ownership(packets)
    eligible = assignments.loc[assignments.review_eligible].copy()
    if set(ownership) != set(eligible.assignment_id):
        raise ValueError("packet and eligible assignment coverage differ")
    primary = _full_decisions(
        eligible, primary_reviews, packet_meta, ownership, role="primary"
    )
    checker = _full_decisions(
        eligible, checker_reviews, packet_meta, ownership, role="checker"
    )
    routed = {
        assignment_id
        for assignment_id in ownership
        if primary[assignment_id]["final_label"] != primary[assignment_id]["action"]
        or checker[assignment_id]["final_label"] != checker[assignment_id]["action"]
        or primary[assignment_id]["final_label"] != checker[assignment_id]["final_label"]
        or primary[assignment_id]["action"] != "retain"
        or checker[assignment_id]["action"] != "retain"
    }
    # The first two comparisons above deliberately do not determine routing;
    # explicit action checks ensure every proposed change is adjudicated.
    routed = {
        assignment_id
        for assignment_id in ownership
        if primary[assignment_id]["action"] != "retain"
        or checker[assignment_id]["action"] != "retain"
        or primary[assignment_id]["final_label"] != checker[assignment_id]["final_label"]
    }
    adjudication_by_id: dict[str, Mapping[str, Any]] = {}
    for item in adjudications:
        assignment_id = str(item.get("assignment_id") or "")
        if assignment_id not in routed or assignment_id in adjudication_by_id:
            raise ValueError(f"invalid/duplicate adjudication: {assignment_id}")
        packet_id = ownership[assignment_id]
        if item.get("reviewer_id") != packet_meta[packet_id]["reviewer_roles"]["adjudicator"]:
            raise ValueError(f"wrong adjudicator for {assignment_id}")
        rationale = str(item.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(f"adjudication lacks rationale: {assignment_id}")
        final_label = item.get("final_label")
        if final_label is not None and final_label not in CONTROLLED_SPECIES:
            raise ValueError(f"adjudication has uncontrolled label: {final_label}")
        row = eligible.set_index("assignment_id").loc[assignment_id]
        fields = (row.assay_type, row.experimental_conditions, row.support_text)
        if final_label is not None and not validate_literal_support(final_label, fields):
            raise ValueError(f"adjudicated label lacks literal support: {assignment_id}")
        adjudication_by_id[assignment_id] = item
    if set(adjudication_by_id) != routed:
        raise ValueError(
            "adjudication coverage mismatch: "
            f"missing={len(routed-set(adjudication_by_id))} extra={len(set(adjudication_by_id)-routed)}"
        )
    final = assignments.copy()
    final["final_label"] = final["candidate_label"]
    final["primary_action"] = "not_reviewed_frozen_null"
    final["checker_action"] = "not_reviewed_frozen_null"
    final["adjudicated"] = False
    final["review_rationale"] = None
    index = {str(value): idx for idx, value in enumerate(final.assignment_id)}
    for assignment_id in ownership:
        idx = index[assignment_id]
        final.at[idx, "primary_action"] = primary[assignment_id]["action"]
        final.at[idx, "checker_action"] = checker[assignment_id]["action"]
        if assignment_id in adjudication_by_id:
            item = adjudication_by_id[assignment_id]
            final.at[idx, "final_label"] = item.get("final_label")
            final.at[idx, "adjudicated"] = True
            final.at[idx, "review_rationale"] = str(item["rationale"])
        else:
            final.at[idx, "final_label"] = final.at[idx, "candidate_label"]
            final.at[idx, "review_rationale"] = "two independent retains"
    frozen_nulls = ~final.review_eligible
    if final.loc[frozen_nulls, "final_label"].notna().any():
        raise ValueError("a frozen v3 null was promoted")
    audit = {
        "eligible_assignments": int(final.review_eligible.sum()),
        "frozen_null_assignments": int((~final.review_eligible).sum()),
        "routed_to_adjudication": len(routed),
        "retained": int(
            (final.candidate_label.fillna("<NULL>") == final.final_label.fillna("<NULL>"))
            .loc[final.review_eligible]
            .sum()
        ),
        "corrected": int(
            (
                final.candidate_label.notna()
                & final.final_label.notna()
                & (final.candidate_label != final.final_label)
            ).sum()
        ),
        "demoted_to_null": int(
            (final.candidate_label.notna() & final.final_label.isna()).sum()
        ),
        "null_promotions": int(
            (final.candidate_label.isna() & final.final_label.notna()).sum()
        ),
    }
    return final, audit


def build_proposed_mapping(
    base_mapping: Mapping[str, Any],
    final_labels_by_assignment: Mapping[str, str | None] | None,
    assignments: pd.DataFrame,
) -> dict[str, Any]:
    proposal = json.loads(json.dumps(base_mapping, ensure_ascii=False))
    _candidate_section(proposal)
    proposal["mapping_version"] = PROPOSAL_VERSION
    section = proposal["sources"][SOURCE_ID][OUTPUT_FIELD]
    by_assignment = final_labels_by_assignment or {
        str(row.assignment_id): (
            None if pd.isna(row.final_label) else str(row.final_label)
        )
        for row in assignments.itertuples()
    }
    mapping: dict[str, str | None] = {}
    for row in assignments.itertuples():
        assignment_id = str(row.assignment_id)
        if assignment_id not in by_assignment:
            raise ValueError(f"final label missing for {assignment_id}")
        label = by_assignment[assignment_id]
        if label is not None and label not in CONTROLLED_SPECIES:
            raise ValueError(f"final label is uncontrolled: {label}")
        if pd.isna(row.candidate_label) and label is not None:
            raise ValueError("cannot promote a v3 null")
        mapping[str(row.tuple_key)] = label
    if set(mapping) != set(section["mapping"]):
        raise ValueError("proposal tuple coverage differs from candidate")
    section["mapping"] = dict(sorted(mapping.items()))
    return proposal


def replay_proposal(
    base_mapping: Mapping[str, Any],
    assignments: pd.DataFrame,
    reviews: tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    adjudications: Sequence[Mapping[str, Any]],
    proposal: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    if packets is None:
        packets = packetize_assignments(assignments)
    final, _ = reconcile_assignment_reviews(
        assignments, packets, reviews[0], reviews[1], adjudications
    )
    replayed = build_proposed_mapping(base_mapping, None, final)
    return replayed == proposal


def write_snapshot(
    *,
    output_dir: str | Path = DEFAULT_RECONCILIATION_ROOT / "provenance",
    cache_path: str | Path = DEFAULT_CACHE,
    candidate_mapping_path: str | Path = DEFAULT_CANDIDATE_MAPPING,
    mapping_manifest_path: str | Path = DEFAULT_MAPPING_MANIFEST,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cache_records = _load_jsonl(cache_path)
    candidate_mapping = json.loads(Path(candidate_mapping_path).read_text(encoding="utf-8"))
    section = _candidate_section(candidate_mapping)
    assignments = build_candidate_assignments(cache_records, section)
    assignment_path = target / "cluster_assignments.parquet"
    candidate_path = target / "candidate_v3_mapping.json"
    assignments.to_parquet(assignment_path, index=False, compression="zstd")
    _atomic_json(candidate_path, candidate_mapping)
    inputs = {
        "cluster_cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        "candidate_mapping": {
            "path": str(candidate_mapping_path),
            "sha256": file_sha256(candidate_mapping_path),
        },
        "mapping_manifest": {
            "path": str(mapping_manifest_path),
            "sha256": file_sha256(mapping_manifest_path),
        },
        "token_ledger": {
            "path": str(DEFAULT_TOKEN_LEDGER),
            "sha256": file_sha256(DEFAULT_TOKEN_LEDGER),
        },
        "prompt_registry": {
            "path": str(DEFAULT_PROMPT_REGISTRY),
            "sha256": file_sha256(DEFAULT_PROMPT_REGISTRY),
        },
    }
    counts = {
        "clusters": len(cache_records),
        "assignments": len(assignments),
        "raw_gpt_non_null": int(assignments.raw_gpt_label.notna().sum()),
        "raw_gpt_null": int(assignments.raw_gpt_label.isna().sum()),
        "candidate_non_null": int(assignments.candidate_label.notna().sum()),
        "candidate_null": int(assignments.candidate_label.isna().sum()),
        "review_eligible": int(assignments.review_eligible.sum()),
    }
    manifest = {
        "manifest_version": "skin_species_context_reconciliation_snapshot.v1",
        "publication_status": "candidate_only",
        "candidate_version": CANDIDATE_VERSION,
        "proposed_version": PROPOSAL_VERSION,
        "inputs": inputs,
        "outputs": {
            assignment_path.name: file_sha256(assignment_path),
            candidate_path.name: file_sha256(candidate_path),
        },
        "counts": counts,
        "validations": {
            "cache_and_candidate_tuple_coverage_exact": True,
            "all_candidate_non_null_labels_have_literal_support": True,
            "candidate_nulls_frozen_against_promotion": True,
        },
    }
    _atomic_json(target / "manifest.json", manifest)
    return manifest


def _read_reviews(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_jsonl(path))
    return rows


def _load_packet_payloads(packet_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(packet_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payloads = []
    for row in manifest["packets"]:
        path = root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"packet hash mismatch: {path}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def write_proposal(
    *,
    work_root: str | Path,
    primary_paths: Sequence[str | Path],
    checker_paths: Sequence[str | Path],
    adjudication_paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(work_root)
    provenance = root / "provenance"
    packet_dir = root / "review_packets"
    proposal_dir = root / "proposal"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_parquet(provenance / "cluster_assignments.parquet")
    base_mapping = json.loads((provenance / "candidate_v3_mapping.json").read_text(encoding="utf-8"))
    packets = _load_packet_payloads(packet_dir)
    primary = _read_reviews(primary_paths)
    checker = _read_reviews(checker_paths)
    adjudications = _read_reviews(adjudication_paths)
    final, review_audit = reconcile_assignment_reviews(
        assignments, packets, primary, checker, adjudications
    )
    proposal = build_proposed_mapping(base_mapping, None, final)
    proposal_path = proposal_dir / "proposed_globally_reconciled_mapping.json"
    assignments_path = proposal_dir / "reviewed_assignments.parquet"
    changes_path = proposal_dir / "reconciliation_changes.parquet"
    _atomic_json(proposal_path, proposal)
    final.to_parquet(assignments_path, index=False, compression="zstd")
    changes = final.loc[
        final.candidate_label.fillna("<NULL>") != final.final_label.fillna("<NULL>")
    ].copy()
    changes.to_parquet(changes_path, index=False, compression="zstd")
    _atomic_jsonl(proposal_dir / "primary_review.jsonl", primary)
    _atomic_jsonl(proposal_dir / "checker_review.jsonl", checker)
    _atomic_jsonl(proposal_dir / "adjudication_review.jsonl", adjudications)
    label_counts_before = Counter(final.candidate_label.dropna().astype(str))
    label_counts_after = Counter(final.final_label.dropna().astype(str))
    manifest = {
        "manifest_version": "skin_species_context_reconciliation_proposal.v1",
        "publication_status": "awaiting_human_approval",
        "candidate_version": CANDIDATE_VERSION,
        "proposal_version": PROPOSAL_VERSION,
        "review": review_audit,
        "label_counts_before": dict(sorted(label_counts_before.items())),
        "label_counts_after": dict(sorted(label_counts_after.items())),
        "files": {
            path.name: file_sha256(path)
            for path in (
                proposal_path,
                assignments_path,
                changes_path,
                proposal_dir / "primary_review.jsonl",
                proposal_dir / "checker_review.jsonl",
                proposal_dir / "adjudication_review.jsonl",
            )
        },
        "publication_blockers": ["human_approval_not_recorded"],
    }
    _atomic_json(proposal_dir / "manifest.json", manifest)
    return manifest


def write_adjudication_packets(
    *,
    work_root: str | Path,
    primary_paths: Sequence[str | Path],
    checker_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Package the union of all proposed changes for the distinct third reviewer."""
    root = Path(work_root)
    assignments = pd.read_parquet(root / "provenance/cluster_assignments.parquet")
    eligible = assignments.loc[assignments.review_eligible].copy()
    packets = _load_packet_payloads(root / "review_packets")
    primary_reviews = _read_reviews(primary_paths)
    checker_reviews = _read_reviews(checker_paths)
    ownership = _packet_assignment_ownership(packets)
    packet_meta = {
        packet["packet_id"]: {
            "packet_id": packet["packet_id"],
            "sha256": hashlib.sha256(_json_bytes(packet)).hexdigest(),
            "assignments": packet["assignment_count"],
            "reviewer_roles": packet["reviewer_roles"],
        }
        for packet in packets
    }
    primary = _full_decisions(
        eligible, primary_reviews, packet_meta, ownership, role="primary"
    )
    checker = _full_decisions(
        eligible, checker_reviews, packet_meta, ownership, role="checker"
    )
    routed = sorted(
        assignment_id
        for assignment_id in ownership
        if primary[assignment_id]["action"] != "retain"
        or checker[assignment_id]["action"] != "retain"
        or primary[assignment_id]["final_label"]
        != checker[assignment_id]["final_label"]
    )
    by_assignment = eligible.set_index("assignment_id", drop=False)
    grouped: dict[str, list[dict[str, Any]]] = {
        reviewer: [] for reviewer in REVIEWER_IDS
    }
    for assignment_id in routed:
        packet_id = ownership[assignment_id]
        reviewer = packet_meta[packet_id]["reviewer_roles"]["adjudicator"]
        row = by_assignment.loc[assignment_id]
        grouped[reviewer].append(
            {
                "assignment_id": assignment_id,
                "packet_id": packet_id,
                "candidate_label": str(row.candidate_label),
                "evidence": {
                    "assay_type": None if pd.isna(row.assay_type) else row.assay_type,
                    "experimental_conditions": (
                        None
                        if pd.isna(row.experimental_conditions)
                        else row.experimental_conditions
                    ),
                    "support_text": (
                        None if pd.isna(row.support_text) else row.support_text
                    ),
                },
                "primary_decision": primary[assignment_id],
                "checker_decision": checker[assignment_id],
                "instruction": (
                    "Set final_label to the unique explicitly supported measurement-producing "
                    "species, or null for reagent/background/comparator/ambiguous evidence."
                ),
            }
        )
    target = root / "adjudication_packets"
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for reviewer, items in grouped.items():
        payload = {
            "packet_version": "skin_species_context_adjudication_packet.v1",
            "reviewer_id": reviewer,
            "assignment_count": len(items),
            "assignments": items,
        }
        path = target / f"{reviewer}.json"
        _atomic_json(path, payload)
        records.append(
            {
                "reviewer_id": reviewer,
                "path": path.name,
                "sha256": file_sha256(path),
                "assignments": len(items),
            }
        )
    manifest = {
        "manifest_version": "skin_species_context_adjudication_packets.v1",
        "routed_assignments": len(routed),
        "both_reviewers_proposed_null": sum(
            primary[value]["final_label"] is None
            and checker[value]["final_label"] is None
            for value in routed
        ),
        "reviewer_disagreements": sum(
            primary[value]["final_label"] != checker[value]["final_label"]
            for value in routed
        ),
        "packets": records,
    }
    _atomic_json(target / "manifest.json", manifest)
    return manifest


def audit_proposal(work_root: str | Path) -> dict[str, Any]:
    root = Path(work_root)
    provenance = root / "provenance"
    packet_dir = root / "review_packets"
    proposal_dir = root / "proposal"
    issues: list[str] = []
    snapshot = json.loads((provenance / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in snapshot["outputs"].items():
        path = provenance / filename
        if not path.exists() or file_sha256(path) != expected:
            issues.append(f"snapshot hash mismatch: {filename}")
    packet_manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    for row in packet_manifest["packets"]:
        path = packet_dir / row["path"]
        if not path.exists() or file_sha256(path) != row["sha256"]:
            issues.append(f"packet hash mismatch: {row['path']}")
    proposal_manifest = json.loads((proposal_dir / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in proposal_manifest["files"].items():
        path = proposal_dir / filename
        if not path.exists() or file_sha256(path) != expected:
            issues.append(f"proposal hash mismatch: {filename}")
    assignments = pd.read_parquet(provenance / "cluster_assignments.parquet")
    reviewed = pd.read_parquet(proposal_dir / "reviewed_assignments.parquet")
    base = json.loads((provenance / "candidate_v3_mapping.json").read_text(encoding="utf-8"))
    proposal = json.loads(
        (proposal_dir / "proposed_globally_reconciled_mapping.json").read_text(encoding="utf-8")
    )
    packets = _load_packet_payloads(packet_dir)
    primary = _load_jsonl(proposal_dir / "primary_review.jsonl")
    checker = _load_jsonl(proposal_dir / "checker_review.jsonl")
    adjudications = _load_jsonl(proposal_dir / "adjudication_review.jsonl")
    replayed_assignments, replay_review = reconcile_assignment_reviews(
        assignments, packets, primary, checker, adjudications
    )
    reviewed_exact = (
        list(replayed_assignments.columns) == list(reviewed.columns)
        and replayed_assignments.astype("string").fillna("<NULL>").equals(
            reviewed.astype("string").fillna("<NULL>")
        )
    )
    if not reviewed_exact:
        issues.append("reviewed assignments do not replay from reviewer decisions")
    replayed = build_proposed_mapping(base, None, replayed_assignments)
    if replayed != proposal:
        issues.append("proposal does not replay from reviewed assignments")
    before_sources = json.loads(json.dumps(base["sources"], ensure_ascii=False))
    after_sources = json.loads(json.dumps(proposal["sources"], ensure_ascii=False))
    before_species = before_sources[SOURCE_ID].pop(OUTPUT_FIELD)
    after_species = after_sources[SOURCE_ID].pop(OUTPUT_FIELD)
    if before_sources != after_sources:
        issues.append("a non-target mapping section changed")
    if set(before_species["mapping"]) != set(after_species["mapping"]):
        issues.append("species tuple coverage changed")
    if reviewed.loc[~reviewed.review_eligible, "final_label"].notna().any():
        issues.append("a frozen candidate null was promoted")
    if len(assignments) != len(reviewed):
        issues.append("reviewed assignment count changed")
    expected_candidate_sha = snapshot["outputs"]["candidate_v3_mapping.json"]
    candidate_source_unchanged = (
        (provenance / "candidate_v3_mapping.json").is_file()
        and file_sha256(provenance / "candidate_v3_mapping.json")
        == expected_candidate_sha
    )
    if not candidate_source_unchanged:
        issues.append("frozen v3 candidate changed after the snapshot")
    if replay_review != proposal_manifest["review"]:
        issues.append("recorded review audit differs from exact replay")
    result = {
        "audit_version": AUDIT_VERSION,
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "publication_status": "awaiting_human_approval",
        "validations": {
            "exact_proposal_replay": replayed == proposal,
            "exact_review_decision_replay": reviewed_exact,
            "non_target_sections_unchanged": before_sources == after_sources,
            "tuple_coverage_unchanged": set(before_species["mapping"])
            == set(after_species["mapping"]),
            "candidate_nulls_not_promoted": not reviewed.loc[
                ~reviewed.review_eligible, "final_label"
            ].notna().any(),
            "candidate_source_not_modified_by_this_workflow": candidate_source_unchanged,
            # Backward-compatible field name retained for existing audit readers.
            "runtime_mapping_not_modified_by_this_workflow": candidate_source_unchanged,
        },
        "review": proposal_manifest["review"],
        "proposal_sha256": file_sha256(
            proposal_dir / "proposed_globally_reconciled_mapping.json"
        ),
    }
    _atomic_json(proposal_dir / "FINAL_INTEGRITY_AUDIT.json", result)
    return result


def write_impact_report(
    *,
    work_root: str | Path,
    source_path: str | Path,
) -> dict[str, Any]:
    """Measure source-row impact and report pair effects without rebuilding."""
    from .auxiliary_mapping_helpers.reconciliation import (
        _clean_source_value,
        _tuple_key,
    )

    root = Path(work_root)
    reviewed = pd.read_parquet(root / "proposal/reviewed_assignments.parquet")
    changed = reviewed.loc[
        reviewed.candidate_label.notna() & reviewed.final_label.isna()
    ].copy()
    source = pd.read_parquet(source_path)
    fields = ("assay_type", "experimental_conditions", "support_text")
    source["tuple_key"] = [
        _tuple_key(tuple(_clean_source_value(value) for value in row))
        for row in source.loc[:, fields].itertuples(index=False, name=None)
    ]
    source["source_row_number"] = range(1, len(source) + 1)
    affected = source.loc[source.tuple_key.isin(set(changed.tuple_key))].copy()
    affected_rows = root / "proposal/affected_source_rows.parquet"
    affected.to_parquet(affected_rows, index=False, compression="zstd")

    current_bucket_impact: dict[str, Any] | None = None
    # The active artifacts may be packed; leave the exact comparison nullable
    # unless an unpacked sidecar is available.  This avoids quietly rebuilding.
    unpacked_root = Path(
        "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
        "starling_normalized_v7"
    )
    records_path = unpacked_root / "03_records/records.parquet"
    sidecar_path = unpacked_root / "04_pair_buckets/pair_bucket_records.parquet"
    if records_path.is_file() and sidecar_path.is_file():
        records = pd.read_parquet(records_path)
        sidecar = pd.read_parquet(sidecar_path)
        scoped = records.loc[
            (records.source_id == SOURCE_ID)
            & records.source_row_number.isin(set(affected.source_row_number))
        ]
        joined = scoped[["canonical_record_id"]].merge(
            sidecar, on="canonical_record_id", how="left"
        )
        eligible = joined.bucket_eligible.fillna(False)
        current_bucket_impact = {
            "affected_organized_records": len(scoped),
            "currently_bucket_eligible_records": int(eligible.sum()),
            "currently_represented_pair_buckets": int(
                joined.loc[eligible, "pair_bucket_key"].nunique()
            ),
        }
    result = {
        "report_version": "skin_species_context_reconciliation_impact.v1",
        "status": "dry_run_no_rebuild",
        "changed_tuples": len(changed),
        "affected_source_rows": len(affected),
        "affected_unique_smiles": int(affected["SMILES"].nunique(dropna=True)),
        "affected_pmids": int(affected["pmid"].nunique(dropna=True)),
        "affected_endpoint_values": int(
            affected["endpoint_or_target"].nunique(dropna=True)
        ),
        "source_rows_by_candidate_label": dict(
            sorted(
                affected.tuple_key.map(changed.set_index("tuple_key").candidate_label)
                .value_counts()
                .astype(int)
                .to_dict()
                .items()
            )
        ),
        "current_bucket_impact": current_bucket_impact,
        "assay_transfer_effect": (
            "All affected rows become unknown-species evidence. Any currently eligible "
            "row is excluded by required_known_fields_by_source; no new bucket is created."
        ),
        "output": {
            "affected_source_rows": str(affected_rows),
            "sha256": file_sha256(affected_rows),
        },
    }
    _atomic_json(root / "proposal/IMPACT_REPORT.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--work-root", default=str(DEFAULT_RECONCILIATION_ROOT))
    snapshot.add_argument(
        "--candidate-mapping",
        default=str(DEFAULT_CANDIDATE_MAPPING),
        help="Frozen v3 candidate mapping to snapshot; v4 runtime mappings are rejected.",
    )
    prepare = commands.add_parser("prepare-review")
    prepare.add_argument("--work-root", default=str(DEFAULT_RECONCILIATION_ROOT))
    prepare.add_argument("--max-packet-bytes", type=int, default=128 * 1024)
    prepare.add_argument("--max-assignments", type=int, default=200)
    propose = commands.add_parser("propose")
    propose.add_argument("--work-root", default=str(DEFAULT_RECONCILIATION_ROOT))
    propose.add_argument("--primary", nargs="+", required=True)
    propose.add_argument("--checker", nargs="+", required=True)
    propose.add_argument("--adjudications", nargs="+", required=True)
    adjudicate = commands.add_parser("prepare-adjudication")
    adjudicate.add_argument("--work-root", default=str(DEFAULT_RECONCILIATION_ROOT))
    adjudicate.add_argument("--primary", nargs="+", required=True)
    adjudicate.add_argument("--checker", nargs="+", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--work-root", default=str(DEFAULT_RECONCILIATION_ROOT))
    impact = commands.add_parser("impact-report")
    impact.add_argument("--work-root", default=str(DEFAULT_RECONCILIATION_ROOT))
    impact.add_argument(
        "--source-path",
        default=str(
            DATA_PROCESSING_DIR.parents[4]
            / "data/starling_data/skin_reaction/sensitization_aop/extractions.parquet"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.work_root)
    if args.command == "snapshot":
        result = write_snapshot(
            output_dir=root / "provenance",
            candidate_mapping_path=args.candidate_mapping,
        )
    elif args.command == "prepare-review":
        assignments = pd.read_parquet(root / "provenance/cluster_assignments.parquet")
        result = write_review_packets(
            assignments,
            output_dir=root / "review_packets",
            max_packet_bytes=args.max_packet_bytes,
            max_assignments=args.max_assignments,
        )
    elif args.command == "prepare-adjudication":
        result = write_adjudication_packets(
            work_root=root,
            primary_paths=args.primary,
            checker_paths=args.checker,
        )
    elif args.command == "propose":
        result = write_proposal(
            work_root=root,
            primary_paths=args.primary,
            checker_paths=args.checker,
            adjudication_paths=args.adjudications,
        )
    elif args.command == "impact-report":
        result = write_impact_report(
            work_root=root,
            source_path=args.source_path,
        )
    else:
        result = audit_proposal(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
