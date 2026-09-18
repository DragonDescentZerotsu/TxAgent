"""Reviewed second-pass reconciliation for Skin_Reaction auxiliary mappings.

This module never calls a model.  It packages the exact cluster-local output
for review, validates reviewer/checker/adjudicator records, and deterministically
applies accepted decisions to a provisional mapping.  Publication remains a
separate, human-gated operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    FORBIDDEN_OUTPUT,
)


POLICY_PATH = Path(__file__).with_name("auxiliary_reconciliation_policy.json")
PROPOSED_MAPPING_VERSION = "starling_auxiliary.skin_reaction.globally_reconciled.v2"
ALLOWED_ACTIONS = frozenset({"retain", "merge", "rename", "split"})
CHANGE_ACTIONS = frozenset({"merge", "rename", "split"})
CHECK_VERDICTS = frozenset({"agree", "disagree"})
ADJUDICATION_VERDICTS = frozenset({"accept", "reject"})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def namespace_for(source_id: str, output_field: str) -> str:
    return f"{source_id}/{output_field}"


def load_policy(path: str | Path = POLICY_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    namespace_policy = payload.get("namespace_policy", {})
    namespaces = namespace_policy.get("independent_namespaces")
    if not isinstance(namespaces, list) or not namespaces:
        raise ValueError("reconciliation policy must declare at least one namespace")
    if len(set(namespaces)) != len(namespaces):
        raise ValueError("reconciliation policy namespaces must be unique")
    frozen = namespace_policy.get("frozen_passthrough_namespaces")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("reconciliation policy must declare frozen namespaces")
    if len(set(frozen)) != len(frozen) or set(frozen) & set(namespaces):
        raise ValueError("review and frozen policy namespaces must be disjoint")
    return payload


def build_label_catalog(assignments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every provisional label with complete cluster provenance."""
    required = {
        "source_id",
        "output_field",
        "cluster_id",
        "raw_value",
        "provisional_label",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"cluster assignments lack columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    usable = assignments[assignments["provisional_label"].notna()].copy()
    usable["namespace"] = usable.apply(
        lambda row: namespace_for(str(row["source_id"]), str(row["output_field"])),
        axis=1,
    )
    for (namespace, label), group in usable.groupby(
        ["namespace", "provisional_label"], sort=True, dropna=False
    ):
        raw_values = sorted({str(value) for value in group["raw_value"]})
        cluster_ids = sorted({str(value) for value in group["cluster_id"]})
        rows.append(
            {
                "namespace": str(namespace),
                "provisional_label": str(label),
                "assignment_count": int(len(group)),
                "raw_value_count": len(raw_values),
                "cluster_count": len(cluster_ids),
                "cluster_ids_json": json.dumps(cluster_ids, separators=(",", ":")),
                "raw_examples_json": json.dumps(
                    _even_examples(raw_values, limit=12),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["namespace", "provisional_label"], ignore_index=True
    )


def build_review_packets(
    assignments: pd.DataFrame,
    *,
    output_dir: str | Path,
    max_packet_bytes: int = 750_000,
) -> dict[str, Any]:
    """Write disjoint review packets while keeping every local cluster atomic."""
    if max_packet_bytes < 10_000:
        raise ValueError("max_packet_bytes must be at least 10,000")
    policy = load_policy()
    allowed_namespaces = set(
        policy["namespace_policy"]["independent_namespaces"]
    )
    required = {
        "source_id",
        "output_field",
        "cluster_id",
        "item_id",
        "raw_value",
        "provisional_label",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"cluster assignments lack columns: {sorted(missing)}")
    frame = assignments.copy()
    frame["namespace"] = frame.apply(
        lambda row: namespace_for(str(row["source_id"]), str(row["output_field"])),
        axis=1,
    )
    if set(frame["namespace"]) != allowed_namespaces:
        raise ValueError("assignment namespaces differ from frozen policy")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    packet_records: list[dict[str, Any]] = []
    owned_clusters: set[tuple[str, str]] = set()
    owned_assignments = 0
    for namespace in sorted(allowed_namespaces):
        subset = frame[frame["namespace"] == namespace]
        clusters = [
            _cluster_packet(namespace, str(cluster_id), group)
            for cluster_id, group in subset.groupby("cluster_id", sort=True)
        ]
        packets: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for cluster in clusters:
            candidate = current + [cluster]
            candidate_id = (
                f"{namespace.replace('/', '__')}__packet_{len(packets) + 1:04d}"
            )
            candidate_payload = _review_packet_payload(
                packet_id=candidate_id,
                namespace=namespace,
                policy_version=str(policy["policy_version"]),
                clusters=candidate,
            )
            if current and len(_json_data(candidate_payload)) > max_packet_bytes:
                packets.append(current)
                current = []
            current.append(cluster)
        if current:
            packets.append(current)
        namespace_slug = namespace.replace("/", "__")
        for index, packet_clusters in enumerate(packets, start=1):
            packet_id = f"{namespace_slug}__packet_{index:04d}"
            payload = _review_packet_payload(
                packet_id=packet_id,
                namespace=namespace,
                policy_version=str(policy["policy_version"]),
                clusters=packet_clusters,
            )
            path = target / f"{packet_id}.json"
            _write_json(path, payload)
            if path.stat().st_size > max_packet_bytes and len(packet_clusters) != 1:
                raise AssertionError(f"review packet exceeds byte bound: {path}")
            cluster_keys = [
                (namespace, str(cluster["cluster_id"]))
                for cluster in packet_clusters
            ]
            if owned_clusters & set(cluster_keys):
                raise AssertionError("cluster assigned to more than one review packet")
            owned_clusters.update(cluster_keys)
            assignment_count = sum(
                int(cluster["assignment_count"]) for cluster in packet_clusters
            )
            owned_assignments += assignment_count
            packet_records.append(
                {
                    "packet_id": packet_id,
                    "namespace": namespace,
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "clusters": len(packet_clusters),
                    "assignments": assignment_count,
                    "bytes": path.stat().st_size,
                }
            )
    expected_clusters = {
        (str(row.namespace), str(row.cluster_id))
        for row in frame[["namespace", "cluster_id"]].drop_duplicates().itertuples()
    }
    if owned_clusters != expected_clusters:
        raise AssertionError("review packet cluster coverage mismatch")
    if owned_assignments != len(frame):
        raise AssertionError("review packet assignment coverage mismatch")
    manifest = {
        "manifest_version": "skin_reaction_auxiliary_cluster_review_packets.v1",
        "policy": {"path": str(POLICY_PATH), "sha256": file_sha256(POLICY_PATH)},
        "max_packet_bytes": max_packet_bytes,
        "packets": packet_records,
        "coverage": {
            "namespaces": len(allowed_namespaces),
            "section_clusters": len(expected_clusters),
            "assignments": len(frame),
            "packets": len(packet_records),
        },
        "validations": {
            "every_cluster_owned_exactly_once": True,
            "every_assignment_owned_exactly_once": True,
            "clusters_are_atomic": True,
            "namespaces_are_disjoint": True,
        },
    }
    _write_json(target / "manifest.json", manifest)
    return manifest


def validate_primary_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    assignments: pd.DataFrame,
    require_complete_label_coverage: bool = True,
) -> dict[str, Any]:
    """Validate primary review decisions against the provisional inventory."""
    policy = load_policy()
    allowed_namespaces = set(
        policy["namespace_policy"]["independent_namespaces"]
    )
    labels = _labels_by_namespace(assignments)
    evidence_by_label = _raw_evidence_by_label(assignments)
    seen_ids: set[str] = set()
    owned: dict[tuple[str, str], str] = {}
    action_counts: Counter[str] = Counter()
    for decision in decisions:
        decision_id = _required_string(decision, "decision_id")
        if decision_id in seen_ids:
            raise ValueError(f"duplicate decision_id: {decision_id}")
        seen_ids.add(decision_id)
        namespace = _required_string(decision, "namespace")
        if namespace not in allowed_namespaces:
            raise ValueError(f"decision has forbidden namespace: {namespace}")
        action = _required_string(decision, "action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"invalid action for {decision_id}: {action}")
        input_labels = _string_list(decision, "input_labels")
        final_labels = _string_list(decision, "final_labels")
        if not input_labels or not final_labels:
            raise ValueError(f"{decision_id} lacks input or final labels")
        if any(label not in labels[namespace] for label in input_labels):
            raise ValueError(f"{decision_id} references an unknown provisional label")
        for label in final_labels:
            normalized = re.sub(r"\s+", " ", label.strip()).casefold()
            if (
                label != normalized
                or len(label) > 120
                or FORBIDDEN_OUTPUT.search(label)
                or any(token in label for token in ("```", "{", "}", "->", "→"))
            ):
                raise ValueError(f"invalid final label for {decision_id}: {label!r}")
        if action == "retain" and (
            len(input_labels) != 1 or final_labels != input_labels
        ):
            raise ValueError(f"retain decision {decision_id} must be identity")
        if action == "merge" and (len(input_labels) < 2 or len(final_labels) != 1):
            raise ValueError(f"merge decision {decision_id} has invalid cardinality")
        if action == "rename" and (
            len(input_labels) != 1 or len(final_labels) != 1
        ):
            raise ValueError(f"rename decision {decision_id} has invalid cardinality")
        if action == "split":
            _validate_split(
                decision,
                namespace=namespace,
                evidence_by_label=evidence_by_label,
            )
        for label in input_labels:
            key = (namespace, label)
            if key in owned:
                raise ValueError(
                    f"provisional label {namespace}/{label} owned by both "
                    f"{owned[key]} and {decision_id}"
                )
            owned[key] = decision_id
        _require_review_evidence(
            decision,
            namespace=namespace,
            input_labels=input_labels,
            evidence_by_label=evidence_by_label,
        )
        action_counts[action] += 1
    expected = {(namespace, label) for namespace, values in labels.items() for label in values}
    missing = expected - set(owned)
    extra = set(owned) - expected
    if extra or (require_complete_label_coverage and missing):
        raise ValueError(
            f"primary label coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    return {
        "decisions": len(decisions),
        "action_counts": dict(sorted(action_counts.items())),
        "labels_expected": len(expected),
        "labels_owned": len(owned),
        "labels_missing": len(missing),
        "valid": not extra and (not require_complete_label_coverage or not missing),
    }


def validate_reviewer_packet_coverage(
    summaries: Sequence[Mapping[str, Any]],
    *,
    packet_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that primary reviewers inspected every atomic packet once."""
    packet_rows = packet_manifest.get("packets")
    if not isinstance(packet_rows, list) or not packet_rows:
        raise ValueError("packet manifest has no packets")
    expected = {str(row["packet_id"]): row for row in packet_rows}
    if len(expected) != len(packet_rows):
        raise ValueError("packet manifest repeats packet IDs")
    owners: dict[str, str] = {}
    reviewer_ids: set[str] = set()
    for summary in summaries:
        reviewer_id = _required_string(summary, "reviewer_id")
        if reviewer_id in reviewer_ids:
            raise ValueError(f"duplicate reviewer summary: {reviewer_id}")
        reviewer_ids.add(reviewer_id)
        packet_ids = _string_list(summary, "reviewed_packet_ids")
        unknown = set(packet_ids) - set(expected)
        if unknown:
            raise ValueError(f"reviewer {reviewer_id} claims unknown packets: {unknown}")
        duplicates = set(packet_ids) & set(owners)
        if duplicates:
            raise ValueError(f"review packets have multiple owners: {duplicates}")
        for packet_id in packet_ids:
            owners[packet_id] = reviewer_id
        expected_clusters = sum(int(expected[value]["clusters"]) for value in packet_ids)
        expected_assignments = sum(
            int(expected[value]["assignments"]) for value in packet_ids
        )
        if summary.get("reviewed_cluster_count") != expected_clusters:
            raise ValueError(f"reviewer {reviewer_id} cluster count mismatch")
        if summary.get("reviewed_assignment_count") != expected_assignments:
            raise ValueError(f"reviewer {reviewer_id} assignment count mismatch")
    missing = set(expected) - set(owners)
    if missing:
        raise ValueError(f"primary review packet coverage missing {len(missing)} packets")
    return {
        "reviewers": len(reviewer_ids),
        "packets": len(owners),
        "section_clusters": sum(int(row["clusters"]) for row in packet_rows),
        "assignments": sum(int(row["assignments"]) for row in packet_rows),
        "every_packet_owned_exactly_once": True,
    }


def consolidate_review(
    primary: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    *,
    assignments: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return accepted changes after independent checking and adjudication."""
    primary_audit = validate_primary_decisions(
        primary, assignments=assignments, require_complete_label_coverage=True
    )
    by_id = {str(item["decision_id"]): dict(item) for item in primary}
    checks_by_id: dict[str, dict[str, Any]] = {}
    for check in checks:
        decision_id = _required_string(check, "decision_id")
        if decision_id not in by_id:
            raise ValueError(f"checker references unknown decision: {decision_id}")
        if decision_id in checks_by_id:
            raise ValueError(f"duplicate checker decision: {decision_id}")
        verdict = _required_string(check, "verdict")
        if verdict not in CHECK_VERDICTS:
            raise ValueError(f"invalid checker verdict for {decision_id}")
        _required_string(check, "checker_id")
        _required_string(check, "rationale")
        checks_by_id[decision_id] = dict(check)
    adjudication_by_id: dict[str, dict[str, Any]] = {}
    for item in adjudications:
        decision_id = _required_string(item, "decision_id")
        if decision_id not in by_id:
            raise ValueError(f"adjudicator references unknown decision: {decision_id}")
        if decision_id in adjudication_by_id:
            raise ValueError(f"duplicate adjudication: {decision_id}")
        verdict = _required_string(item, "verdict")
        if verdict not in ADJUDICATION_VERDICTS:
            raise ValueError(f"invalid adjudication verdict for {decision_id}")
        _required_string(item, "adjudicator_id")
        _required_string(item, "rationale")
        adjudication_by_id[decision_id] = dict(item)

    change_ids = {
        decision_id
        for decision_id, decision in by_id.items()
        if decision["action"] in CHANGE_ACTIONS
    }
    missing_checks = change_ids - set(checks_by_id)
    missing_adjudications = change_ids - set(adjudication_by_id)
    if missing_checks or missing_adjudications:
        raise ValueError(
            "accepted-change provenance incomplete: "
            f"missing_checks={len(missing_checks)} "
            f"missing_adjudications={len(missing_adjudications)}"
        )
    accepted: list[dict[str, Any]] = []
    rejected = 0
    disagreements = 0
    for decision_id in sorted(change_ids):
        decision = dict(by_id[decision_id])
        check = checks_by_id[decision_id]
        adjudication = adjudication_by_id[decision_id]
        reviewer_id = _required_string(decision, "reviewer_id")
        checker_id = _required_string(check, "checker_id")
        adjudicator_id = _required_string(adjudication, "adjudicator_id")
        if len({reviewer_id, checker_id, adjudicator_id}) != 3:
            raise ValueError(
                f"reviewer, checker, and adjudicator must be distinct for {decision_id}"
            )
        disagreements += int(check["verdict"] == "disagree")
        if adjudication["verdict"] == "reject":
            rejected += 1
            continue
        decision["checker_id"] = check["checker_id"]
        decision["checker_verdict"] = check["verdict"]
        decision["checker_rationale"] = check["rationale"]
        decision["adjudicator_id"] = adjudication["adjudicator_id"]
        decision["adjudication_rationale"] = adjudication["rationale"]
        decision["decision_status"] = "adjudicated_accept"
        accepted.append(decision)
    return accepted, {
        "primary": primary_audit,
        "change_decisions": len(change_ids),
        "accepted_changes": len(accepted),
        "rejected_changes": rejected,
        "checker_disagreements": disagreements,
        "valid": True,
    }


def apply_accepted_decisions(
    provisional_mapping: Mapping[str, Any],
    assignments: pd.DataFrame,
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Apply adjudicated decisions and return a proposal plus exact delta."""
    frame = assignments.copy()
    frame["namespace"] = frame.apply(
        lambda row: namespace_for(str(row["source_id"]), str(row["output_field"])),
        axis=1,
    )
    frame["final_label"] = frame["provisional_label"]
    frame["decision_id"] = None
    frame["decision_action"] = "retain"
    seen_inputs: set[tuple[str, str]] = set()
    for decision in decisions:
        decision_id = _required_string(decision, "decision_id")
        namespace = _required_string(decision, "namespace")
        action = _required_string(decision, "action")
        if action not in CHANGE_ACTIONS:
            raise ValueError(f"proposal contains non-change decision: {decision_id}")
        input_labels = _string_list(decision, "input_labels")
        for label in input_labels:
            key = (namespace, label)
            if key in seen_inputs:
                raise ValueError(f"accepted decisions overlap at {namespace}/{label}")
            seen_inputs.add(key)
        scope = (frame["namespace"] == namespace) & frame["provisional_label"].isin(
            input_labels
        )
        if not scope.any():
            raise ValueError(f"accepted decision has no assignments: {decision_id}")
        if action in {"merge", "rename"}:
            final_label = _string_list(decision, "final_labels")[0]
            frame.loc[scope, "final_label"] = final_label
        else:
            raw_assignments = decision.get("raw_value_assignments")
            raw_to_final = {
                str(item["raw_value"]): str(item["final_label"])
                for item in raw_assignments
            }
            for raw_value, final_label in raw_to_final.items():
                raw_scope = scope & (frame["raw_value"].astype(str) == raw_value)
                if not raw_scope.any():
                    raise ValueError(
                        f"split {decision_id} raw value is absent: {raw_value}"
                    )
                frame.loc[raw_scope, "final_label"] = final_label
        frame.loc[scope, "decision_id"] = decision_id
        frame.loc[scope, "decision_action"] = action
    proposal = _mapping_from_assignments(provisional_mapping, frame)
    changed = frame[
        frame["provisional_label"].fillna("<NULL>")
        != frame["final_label"].fillna("<NULL>")
    ].copy()
    changed = changed[
        [
            "source_id",
            "output_field",
            "cluster_id",
            "item_id",
            "raw_value",
            "tuple_key",
            "provisional_label",
            "final_label",
            "decision_id",
            "decision_action",
        ]
    ].sort_values(["source_id", "output_field", "tuple_key"], ignore_index=True)
    audit = {
        "mapping_version": PROPOSED_MAPPING_VERSION,
        "assignments": len(frame),
        "changed_assignments": len(changed),
        "accepted_decisions": len(decisions),
        "before": _label_counts(frame, "provisional_label"),
        "after": _label_counts(frame, "final_label"),
        "validations": {
            "one_final_label_per_assignment": not frame["final_label"].isna().any()
            or frame["provisional_label"].isna().equals(frame["final_label"].isna()),
            "accepted_decisions_do_not_overlap": True,
            "proposal_replayed_from_provisional": True,
            "human_approved": False,
        },
    }
    return proposal, changed, audit


def write_proposal(
    *,
    output_dir: str | Path,
    provisional_mapping: Mapping[str, Any],
    assignments: pd.DataFrame,
    primary: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    review_summaries: Sequence[Mapping[str, Any]] | None = None,
    packet_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Consolidate reviews and write a non-published human-checkpoint proposal."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    accepted, review_audit = consolidate_review(
        primary, checks, adjudications, assignments=assignments
    )
    packet_coverage = None
    if review_summaries is not None or packet_manifest is not None:
        if review_summaries is None or packet_manifest is None:
            raise ValueError(
                "review_summaries and packet_manifest must be supplied together"
            )
        packet_coverage = validate_reviewer_packet_coverage(
            review_summaries, packet_manifest=packet_manifest
        )
    proposal, changes, proposal_audit = apply_accepted_decisions(
        provisional_mapping, assignments, accepted
    )
    proposal_path = target / "proposed_globally_reconciled_mapping.json"
    changes_path = target / "reconciliation_changes.parquet"
    decisions_path = target / "accepted_reconciliation_decisions.jsonl"
    primary_path = target / "primary_review.jsonl"
    checks_path = target / "checker_review.jsonl"
    adjudications_path = target / "adjudication_review.jsonl"
    summaries_path = target / "primary_review_summaries.json"
    report_path = target / "RECONCILIATION_PROPOSAL.md"
    _write_json(proposal_path, proposal)
    changes.to_parquet(changes_path, index=False, compression="zstd")
    _write_jsonl(decisions_path, accepted)
    _write_jsonl(primary_path, primary)
    _write_jsonl(checks_path, checks)
    _write_jsonl(adjudications_path, adjudications)
    _write_json(summaries_path, list(review_summaries or []))
    _atomic_write(
        report_path,
        _proposal_report(
            review_audit=review_audit,
            proposal_audit=proposal_audit,
            packet_coverage=packet_coverage,
        ).encode("utf-8"),
    )
    artifact_paths = (
        proposal_path,
        changes_path,
        decisions_path,
        primary_path,
        checks_path,
        adjudications_path,
        summaries_path,
        report_path,
    )
    manifest = {
        "manifest_version": "skin_reaction_auxiliary_reconciliation_proposal.v1",
        "publication_status": "awaiting_human_approval",
        "policy": {"path": str(POLICY_PATH), "sha256": file_sha256(POLICY_PATH)},
        "files": {path.name: file_sha256(path) for path in artifact_paths},
        "packet_manifest_sha256": (
            hashlib.sha256(_json_data(packet_manifest)).hexdigest()
            if packet_manifest is not None
            else None
        ),
        "review": review_audit,
        "primary_packet_coverage": packet_coverage,
        "proposal": proposal_audit,
        "publication_blockers": ["human_approval_not_recorded"],
    }
    _write_json(target / "reconciliation_manifest.json", manifest)
    return manifest


def _proposal_report(
    *,
    review_audit: Mapping[str, Any],
    proposal_audit: Mapping[str, Any],
    packet_coverage: Mapping[str, Any] | None,
) -> str:
    before = proposal_audit["before"]
    after = proposal_audit["after"]
    lines = [
        "# Skin_Reaction auxiliary reconciliation proposal",
        "",
        "Status: awaiting human approval. The runtime mapping remains absent and downstream stages have not been changed.",
        "",
        f"- Primary decisions: {review_audit['primary']['decisions']:,}",
        f"- Proposed change decisions: {review_audit['change_decisions']:,}",
        f"- Adjudicated accepted changes: {review_audit['accepted_changes']:,}",
        f"- Adjudicated rejected changes: {review_audit['rejected_changes']:,}",
        f"- Changed raw assignments: {proposal_audit['changed_assignments']:,}",
    ]
    if packet_coverage is not None:
        lines.extend(
            [
                f"- Reviewed packets: {packet_coverage['packets']:,}",
                f"- Reviewed section-clusters: {packet_coverage['section_clusters']:,}",
                f"- Reviewed assignments: {packet_coverage['assignments']:,}",
            ]
        )
    lines.extend(["", "## Vocabulary by namespace", "", "| Namespace | Before | Proposed |", "|---|---:|---:|"])
    for namespace in sorted(set(before) | set(after)):
        lines.append(f"| `{namespace}` | {before.get(namespace, 0):,} | {after.get(namespace, 0):,} |")
    lines.extend(
        [
            "",
            "## Publication blocker",
            "",
            "A human must review this proposal and explicitly approve publication. Until then, `globally_reconciled_auxiliary_value_mapping.json` remains absent and downstream stages remain frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def _cluster_packet(
    namespace: str, cluster_id: str, group: pd.DataFrame
) -> dict[str, Any]:
    items = [
        {
            "item_id": str(row.item_id),
            "raw_value": str(row.raw_value),
            "provisional_label": (
                None if pd.isna(row.provisional_label) else str(row.provisional_label)
            ),
        }
        for row in group.sort_values("item_id").itertuples()
    ]
    return {
        "namespace": namespace,
        "cluster_id": cluster_id,
        "cache_identity": _one(group.get("cache_identity")),
        "requested_model": _one(group.get("requested_model")),
        "response_sha256": _one(group.get("response_sha256")),
        "assignment_count": len(items),
        "items": items,
    }


def _review_packet_payload(
    *,
    packet_id: str,
    namespace: str,
    policy_version: str,
    clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "packet_version": "skin_reaction_auxiliary_cluster_review_packet.v1",
        "packet_id": packet_id,
        "namespace": namespace,
        "policy_version": policy_version,
        "clusters": list(clusters),
    }


def _mapping_from_assignments(
    provisional_mapping: Mapping[str, Any], frame: pd.DataFrame
) -> dict[str, Any]:
    proposal = json.loads(json.dumps(provisional_mapping, ensure_ascii=False))
    proposal["mapping_version"] = PROPOSED_MAPPING_VERSION
    for (source_id, output_field), group in frame.groupby(
        ["source_id", "output_field"], sort=True
    ):
        section = proposal["sources"][str(source_id)][str(output_field)]
        mapping = {
            str(row.tuple_key): (
                None if pd.isna(row.final_label) else str(row.final_label)
            )
            for row in group.itertuples()
        }
        # Explicit null tuples are outside the non-null cache assignments.
        for key, value in section["mapping"].items():
            if json.loads(key) == [None]:
                mapping[key] = value
        section["mapping"] = dict(sorted(mapping.items()))
    return proposal


def audit_proposal(
    *,
    work_root: str | Path,
    runtime_mapping_path: str | Path,
) -> dict[str, Any]:
    """Replay and verify the complete proposal without publishing it."""
    from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.auxiliary_reconciliation import (
        FROZEN_NAMESPACES,
        REVIEW_NAMESPACES,
        _downstream_manifests,
    )
    from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
        validate_mapping,
    )

    root = Path(work_root)
    provenance_dir = root / "provenance"
    packet_dir = root / "review_packets"
    proposal_dir = root / "proposal"
    provisional_manifest = json.loads(
        (provenance_dir / "provisional_manifest.json").read_text(encoding="utf-8")
    )
    packet_manifest = json.loads(
        (packet_dir / "manifest.json").read_text(encoding="utf-8")
    )
    proposal_manifest = json.loads(
        (proposal_dir / "reconciliation_manifest.json").read_text(encoding="utf-8")
    )

    issues: list[str] = []
    provisional_hashes_ok = True
    for info in provisional_manifest.get("outputs", {}).values():
        path = provenance_dir / str(info["filename"])
        if not path.exists() or file_sha256(path) != info["sha256"]:
            provisional_hashes_ok = False
            issues.append(f"provisional hash mismatch: {path.name}")
    packet_hashes_ok = True
    for info in packet_manifest.get("packets", []):
        path = packet_dir / str(info["path"])
        if not path.exists() or file_sha256(path) != info["sha256"]:
            packet_hashes_ok = False
            issues.append(f"review packet hash mismatch: {path.name}")
    proposal_hashes_ok = True
    for filename, expected in proposal_manifest.get("files", {}).items():
        path = proposal_dir / filename
        if not path.exists() or file_sha256(path) != expected:
            proposal_hashes_ok = False
            issues.append(f"proposal hash mismatch: {filename}")
    packet_manifest_sha256 = file_sha256(packet_dir / "manifest.json")
    packet_manifest_link_ok = (
        proposal_manifest.get("packet_manifest_sha256") == packet_manifest_sha256
    )
    if not packet_manifest_link_ok:
        issues.append("proposal does not pin the review packet manifest")
    policy_sha256 = file_sha256(POLICY_PATH)
    policy_links_ok = (
        packet_manifest.get("policy", {}).get("sha256") == policy_sha256
        and proposal_manifest.get("policy", {}).get("sha256") == policy_sha256
    )
    if not policy_links_ok:
        issues.append("review artifacts do not pin the current policy")

    assignments = pd.read_parquet(provenance_dir / "cluster_assignments.parquet")
    assignments = assignments.loc[assignments["review_eligible"]].copy()
    provisional = json.loads(
        (provenance_dir / "pre_reconciliation_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    proposal = json.loads(
        (proposal_dir / "proposed_globally_reconciled_mapping.json").read_text(
            encoding="utf-8"
        )
    )

    def read_jsonl(name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with (proposal_dir / name).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    primary = read_jsonl("primary_review.jsonl")
    checks = read_jsonl("checker_review.jsonl")
    adjudications = read_jsonl("adjudication_review.jsonl")
    recorded_accepted = read_jsonl("accepted_reconciliation_decisions.jsonl")
    accepted, review_audit = consolidate_review(
        primary, checks, adjudications, assignments=assignments
    )
    accepted_exact = accepted == recorded_accepted
    if not accepted_exact:
        issues.append("accepted decision file does not reconsolidate exactly")
    replayed, replayed_changes, replay_audit = apply_accepted_decisions(
        provisional, assignments, accepted
    )
    proposal_exact = replayed == proposal
    if not proposal_exact:
        issues.append("proposal does not replay exactly")
    recorded_changes = pd.read_parquet(proposal_dir / "reconciliation_changes.parquet")
    changes_exact = (
        list(recorded_changes.columns) == list(replayed_changes.columns)
        and len(recorded_changes) == len(replayed_changes)
        and recorded_changes.astype("string").fillna("<NULL>").equals(
            replayed_changes.astype("string").fillna("<NULL>")
        )
    )
    if not changes_exact:
        issues.append("reconciliation_changes.parquet differs from replay")

    validate_mapping(proposal)
    tuple_coverage_ok = True
    null_masks_ok = True
    frozen_sections_ok = True
    section_counts: dict[str, Any] = {}
    seen_namespaces: set[str] = set()
    for source_id, outputs in provisional["sources"].items():
        for output_field, before_section in outputs.items():
            namespace = namespace_for(source_id, output_field)
            seen_namespaces.add(namespace)
            after_section = proposal["sources"][source_id][output_field]
            before_mapping = before_section["mapping"]
            after_mapping = after_section["mapping"]
            same_keys = set(before_mapping) == set(after_mapping)
            same_nulls = {
                key for key, value in before_mapping.items() if value is None
            } == {key for key, value in after_mapping.items() if value is None}
            tuple_coverage_ok &= same_keys
            null_masks_ok &= same_nulls
            if namespace in FROZEN_NAMESPACES:
                frozen_sections_ok &= before_section == after_section
            section_counts[namespace] = {
                "tuples": len(before_mapping),
                "null_tuples": sum(value is None for value in before_mapping.values()),
            }
    if seen_namespaces != REVIEW_NAMESPACES | FROZEN_NAMESPACES:
        issues.append("proposal namespace inventory mismatch")
    if not tuple_coverage_ok:
        issues.append("proposal tuple coverage changed")
    if not null_masks_ok:
        issues.append("proposal null masks changed")
    if not frozen_sections_ok:
        issues.append("frozen passthrough section changed")

    runtime_path = Path(runtime_mapping_path)
    runtime_absent = not runtime_path.exists()
    if not runtime_absent:
        issues.append("runtime mapping was published")
    data_processing_dir = runtime_path.parent
    downstream_now = _downstream_manifests(data_processing_dir)
    downstream_before = provisional_manifest.get("downstream_artifacts_before", [])
    downstream_unchanged = downstream_now == downstream_before
    if not downstream_unchanged:
        issues.append("downstream stage manifests changed")

    result = {
        "audit_version": "skin_reaction_auxiliary_reconciliation.final_integrity.v1",
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "local_lineage": {
            "aggregated_local_mapping_sha256": file_sha256(
                provenance_dir / "pre_reconciliation_mapping.json"
            ),
            "runtime_mapping_absent": runtime_absent,
            "downstream_stage_artifacts_unchanged": downstream_unchanged,
        },
        "manifest_integrity": {
            "provisional_manifest_files_all_match": provisional_hashes_ok,
            "review_packet_files_all_match": packet_hashes_ok,
            "proposal_manifest_files_all_match": proposal_hashes_ok,
            "packet_manifest_link_matches": packet_manifest_link_ok,
            "policy_links_match": policy_links_ok,
        },
        "review_integrity": {
            "primary_labels_expected": review_audit["primary"]["labels_expected"],
            "primary_labels_owned": review_audit["primary"]["labels_owned"],
            "accepted_decisions": len(accepted),
            "accepted_decision_file_exact": accepted_exact,
            "roles_pairwise_distinct_for_every_change": True,
        },
        "mapping_integrity": {
            "sections": len(section_counts),
            "section_counts": section_counts,
            "tuple_coverage_preserved": tuple_coverage_ok,
            "null_masks_preserved": null_masks_ok,
            "frozen_passthrough_sections_unchanged": frozen_sections_ok,
            "proposal_mapping_valid": True,
        },
        "replay_integrity": {
            "proposal_mapping_exact": proposal_exact,
            "proposal_mapping_sha256": file_sha256(
                proposal_dir / "proposed_globally_reconciled_mapping.json"
            ),
            "changes_parquet_exact": changes_exact,
            "changed_assignments": len(replayed_changes),
            "accepted_decisions_do_not_overlap": replay_audit["validations"][
                "accepted_decisions_do_not_overlap"
            ],
        },
        "publication_gate": {
            "publication_status": proposal_manifest.get("publication_status"),
            "human_approved": False,
            "publication_blockers": proposal_manifest.get("publication_blockers"),
            "publication_remains_blocked": True,
        },
    }
    _write_json(proposal_dir / "FINAL_INTEGRITY_AUDIT.json", result)
    if issues:
        raise ValueError("final integrity audit failed: " + "; ".join(issues))
    return result


def _validate_split(
    decision: Mapping[str, Any],
    *,
    namespace: str,
    evidence_by_label: Mapping[tuple[str, str], Sequence[str]],
) -> None:
    decision_id = str(decision["decision_id"])
    input_labels = _string_list(decision, "input_labels")
    final_labels = _string_list(decision, "final_labels")
    if len(input_labels) != 1 or len(final_labels) < 2:
        raise ValueError(f"split decision {decision_id} has invalid cardinality")
    raw_assignments = decision.get("raw_value_assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError(f"split decision {decision_id} lacks raw assignments")
    supplied: dict[str, str] = {}
    for item in raw_assignments:
        raw_value = _required_string(item, "raw_value")
        final_label = _required_string(item, "final_label")
        if final_label not in final_labels:
            raise ValueError(f"split {decision_id} uses undeclared final label")
        if raw_value in supplied:
            raise ValueError(f"split {decision_id} repeats raw value: {raw_value}")
        supplied[raw_value] = final_label
    expected = set(evidence_by_label[(namespace, input_labels[0])])
    if set(supplied) != expected:
        raise ValueError(
            f"split {decision_id} raw coverage mismatch: "
            f"missing={len(expected - set(supplied))} extra={len(set(supplied) - expected)}"
        )
    if set(supplied.values()) != set(final_labels):
        raise ValueError(f"split {decision_id} does not use every final label")


def _require_review_evidence(
    decision: Mapping[str, Any],
    *,
    namespace: str,
    input_labels: Sequence[str],
    evidence_by_label: Mapping[tuple[str, str], Sequence[str]],
) -> None:
    _required_string(decision, "reviewer_id")
    _required_string(decision, "rationale")
    _required_string(decision, "evidence_summary")
    if decision.get("decision_status") != "proposed":
        raise ValueError(
            f"primary decision {decision.get('decision_id')} must have proposed status"
        )
    for field in ("preserved_dimensions", "removed_details"):
        values = decision.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(
                f"decision {decision.get('decision_id')} has invalid {field}"
            )
    raw_evidence = [
        raw_value
        for label in input_labels
        for raw_value in evidence_by_label[(namespace, label)]
    ]
    count = decision.get("affected_raw_value_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(raw_evidence):
        raise ValueError(
            f"decision {decision.get('decision_id')} affected_raw_value_count "
            f"must equal {len(raw_evidence)}"
        )
    examples = decision.get("supporting_raw_examples")
    if not isinstance(examples, (list, dict)) or not examples:
        raise ValueError(
            f"decision {decision.get('decision_id')} lacks supporting raw examples"
        )
    supplied_examples = set(_example_strings(examples))
    raw_values = set(raw_evidence)
    if not supplied_examples or not supplied_examples <= raw_values:
        raise ValueError(
            f"decision {decision.get('decision_id')} has examples outside its raw evidence"
        )


def _example_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_example_strings(item))
        return output
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(_example_strings(item))
        return output
    return []


def _labels_by_namespace(assignments: pd.DataFrame) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in assignments.itertuples():
        if pd.isna(row.provisional_label):
            continue
        output[namespace_for(str(row.source_id), str(row.output_field))].add(
            str(row.provisional_label)
        )
    return dict(output)


def _raw_evidence_by_label(
    assignments: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    output: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in assignments.itertuples():
        if pd.isna(row.provisional_label):
            continue
        key = (
            namespace_for(str(row.source_id), str(row.output_field)),
            str(row.provisional_label),
        )
        output[key].append(str(row.raw_value))
    return dict(output)


def _label_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for namespace, group in frame.groupby("namespace", sort=True):
        result[str(namespace)] = int(group[column].dropna().nunique())
    return result


def _even_examples(values: Sequence[str], *, limit: int) -> list[str]:
    if len(values) <= limit:
        return list(values)
    if limit <= 1:
        return [values[0]]
    return [
        values[round(index * (len(values) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _one(series: Any) -> Any:
    if series is None:
        return None
    values = {
        None if pd.isna(value) else str(value)
        for value in series
    }
    if len(values) > 1:
        raise ValueError(f"cluster provenance is not constant: {values}")
    return next(iter(values)) if values else None


def _required_string(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"missing nonempty string field: {field}")
    return item.strip()


def _string_list(value: Mapping[str, Any], field: str) -> list[str]:
    items = value.get(field)
    if not isinstance(items, list) or any(
        not isinstance(item, str) or not item.strip() for item in items
    ):
        raise ValueError(f"invalid string list field: {field}")
    normalized = [item.strip() for item in items]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate values in field: {field}")
    return normalized


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, _json_data(payload))


def _json_data(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "ALLOWED_ACTIONS",
    "POLICY_PATH",
    "PROPOSED_MAPPING_VERSION",
    "apply_accepted_decisions",
    "audit_proposal",
    "build_label_catalog",
    "build_review_packets",
    "consolidate_review",
    "file_sha256",
    "load_policy",
    "namespace_for",
    "validate_reviewer_packet_coverage",
    "validate_primary_decisions",
    "write_proposal",
]
