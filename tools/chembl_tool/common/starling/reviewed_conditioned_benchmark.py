"""Shared builder for record-reviewed context-conditioned benchmarks.

Task adapters are responsible for source semantics and for producing one
review ledger row per candidate source record.  This module never infers a
condition or a label from free text: it validates accepted review verdicts,
aggregates ``(parent, exact condition signature)`` votes, extends a frozen
molecule-only split, and publishes the common audit artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.starling.benchmark_dataset import sha256_file


NO_REPORTED_CONDITION = "no_reported_external_condition"
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class ConditionedBenchmarkConfig:
    task_name: str
    lineage: str
    protocol_version: str
    frozen_root: Path
    output_root: Path
    source_artifacts: tuple[Path, ...]
    row_id_prefix: str
    frozen_lineage: str
    external_agreement_threshold: float = 0.60
    minimum_group_parents: int = 3
    minimum_group_scaffolds: int = 3
    excluded_condition_groups: tuple[str, ...] = ()
    allowed_condition_groups: tuple[str, ...] | None = None
    publish_allowed_group_audit_only: bool = False
    review_policy: str = (
        "every candidate source record requires one terminal semantic verdict; "
        "reviewer provenance is retained per record"
    )
    seed: int = 20260818
    split_targets: tuple[float, float, float] = (0.8, 0.1, 0.1)


def build_reviewed_conditioned_benchmark(
    *,
    config: ConditionedBenchmarkConfig,
    review_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one task after exhaustive candidate-ledger review."""
    full_audits, full_votes = validate_review_ledger(review_rows)
    audits, votes = _published_review_scope(
        full_audits,
        full_votes,
        allowed_groups=config.allowed_condition_groups,
        restrict_to_allowed=config.publish_allowed_group_audit_only,
    )
    frozen_by_split = {
        split: _read_jsonl(config.frozen_root / f"{split}_molecule_labels.jsonl")
        for split in SPLITS
    }
    frozen_rows = _prepare_frozen_rows(frozen_by_split, config.frozen_lineage)
    parent_assignment, scaffold_assignment = _frozen_assignments(frozen_rows)
    frozen_by_parent = {row["molecule_identity_key"]: row for row in frozen_rows}

    accepted_units, rejected_units = aggregate_reviewed_votes(
        votes, agreement_threshold=config.external_agreement_threshold
    )
    _reuse_frozen_identity(accepted_units, frozen_by_parent)
    candidate_groups, preallocation = candidate_group_support(
        accepted_units,
        minimum_parents=config.minimum_group_parents,
        minimum_scaffolds=config.minimum_group_scaffolds,
        excluded_groups=set(config.excluded_condition_groups),
        allowed_groups=(
            set(config.allowed_condition_groups)
            if config.allowed_condition_groups is not None
            else None
        ),
    )
    allocation, selected_groups, allocation_audit = allocate_groups(
        accepted_units,
        candidate_groups=candidate_groups,
        parent_assignment=parent_assignment,
        scaffold_assignment=scaffold_assignment,
        config=config,
    )

    external_rows: list[dict[str, Any]] = []
    group_rejected: list[dict[str, Any]] = []
    for row in accepted_units:
        group = row["condition_group"]
        if group not in selected_groups:
            group_rejected.append(
                {
                    **row,
                    "drop_reason": allocation_audit["group_exclusion_reasons"].get(
                        group, "group_not_selected_by_split_allocation"
                    ),
                }
            )
            continue
        parent = row["molecule_identity_key"]
        scaffold = row["bemis_murcko_scaffold"]
        split = (
            parent_assignment.get(parent)
            or scaffold_assignment.get(scaffold)
            or allocation[scaffold]
        )
        external_rows.append(
            {
                **row,
                "split": split,
                "split_policy": config.lineage,
                "benchmark_row_id": _row_id(
                    config.row_id_prefix, parent, row["condition_group"]
                ),
            }
        )

    combined = {split: [] for split in SPLITS}
    for row in frozen_rows + external_rows:
        combined[row["split"]].append(row)
    for rows in combined.values():
        rows.sort(key=lambda row: (row["molecule_identity_key"], row["condition_group"]))
    validate_split_integrity(combined)

    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in combined.items():
        write_jsonl_atomic(root / f"{split}.jsonl", (_minimal_row(row) for row in rows))
        write_jsonl_atomic(root / f"{split}_molecule_condition_labels.jsonl", rows)
    write_jsonl_atomic(
        root / "heldout_molecule_condition_labels.jsonl",
        combined["valid"] + combined["test"],
    )
    write_jsonl_atomic(root / "source_condition_review.jsonl", audits)
    write_jsonl_atomic(root / "accepted_parent_conditions_before_group_gate.jsonl", accepted_units)
    write_jsonl_atomic(root / "rejected_parent_conditions.jsonl", rejected_units)
    write_jsonl_atomic(root / "group_gate_rejected_parent_conditions.jsonl", group_rejected)

    group_distribution = summarize_groups(combined)
    _write_csv(root / "group_distribution.csv", group_distribution)
    summary = _summary(
        config=config,
        n_full_review_ledger_records=len(full_audits),
        frozen_rows=frozen_rows,
        external_rows=external_rows,
        combined=combined,
        audits=audits,
        accepted_units=accepted_units,
        rejected_units=rejected_units,
        group_rejected=group_rejected,
        preallocation=preallocation,
        allocation_audit=allocation_audit,
        group_distribution=group_distribution,
    )
    write_json_atomic(root / "summary.json", summary)
    return summary


def _published_review_scope(
    audits: Sequence[Mapping[str, Any]],
    votes: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    allowed_groups: Sequence[str] | None,
    restrict_to_allowed: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Keep selected artifacts compact without weakening full-ledger validation.

    The complete review ledger is always validated first.  A selected build may
    then publish only rows concerning its exact allowlist; the canonical full
    ledger remains immutable under the task review root and is lineage-pinned
    through ``source_artifacts`` hashes.
    """
    copied_votes = {
        key: [dict(row) for row in rows]
        for key, rows in votes.items()
    }
    if not restrict_to_allowed:
        return [dict(row) for row in audits], copied_votes
    if allowed_groups is None:
        raise ValueError(
            "publish_allowed_group_audit_only requires allowed_condition_groups"
        )
    allowed = set(allowed_groups)
    scoped_audits = [
        dict(row)
        for row in audits
        if str(row.get("condition_group") or row.get("proposed_condition_group") or "")
        in allowed
    ]
    scoped_votes = {
        key: rows for key, rows in copied_votes.items() if key[1] in allowed
    }
    return scoped_audits, scoped_votes


def validate_review_ledger(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Require a unique terminal verdict for every candidate record."""
    required = {
        "source_record_id",
        "source_payload_sha256",
        "review_status",
        "reviewer",
        "review_reason",
    }
    audits: list[dict[str, Any]] = []
    votes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        missing = sorted(field for field in required if not str(row.get(field) or "").strip())
        if missing:
            raise ValueError(f"Review row lacks required fields {missing}: {row.get('source_record_id')}")
        record_id = str(row["source_record_id"])
        if record_id in seen:
            raise ValueError(f"Duplicate review verdict for {record_id}")
        seen.add(record_id)
        status = str(row["review_status"])
        if status not in {"accepted", "rejected"}:
            raise ValueError(f"Non-terminal review status for {record_id}: {status}")
        audits.append(row)
        if status == "rejected":
            continue
        accepted_required = {
            "drug",
            "molecule_identity_key",
            "condition_group",
            "condition_atoms",
            "Y",
            "label_method",
        }
        missing = sorted(field for field in accepted_required if row.get(field) in (None, "", []))
        if "bemis_murcko_scaffold" not in row or row["bemis_murcko_scaffold"] is None:
            missing.append("bemis_murcko_scaffold")
        if missing:
            raise ValueError(f"Accepted review row lacks {missing}: {record_id}")
        if int(row["Y"]) not in {0, 1}:
            raise ValueError(f"Accepted review row has non-binary label: {record_id}")
        if row["condition_group"] == NO_REPORTED_CONDITION:
            raise ValueError(f"External review row mapped to null group: {record_id}")
        atoms = [str(value) for value in row["condition_atoms"]]
        canonical_signature = "+".join(sorted(set(atoms)))
        if len(atoms) != len(set(atoms)) or str(row["condition_group"]) != canonical_signature:
            raise ValueError(f"Accepted review row has non-canonical condition signature: {record_id}")
        key = (str(row["molecule_identity_key"]), str(row["condition_group"]))
        votes[key].append(row)
    return audits, votes


def aggregate_reviewed_votes(
    votes: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    agreement_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for (parent, group), source_rows in sorted(votes.items()):
        rows = [dict(row) for row in source_rows]
        counts = Counter(int(row["Y"]) for row in rows)
        n0, n1 = counts.get(0, 0), counts.get(1, 0)
        majority = 1 if n1 > n0 else 0
        agreement = max(n0, n1) / len(rows)
        base = {
            "drug": rows[0]["drug"],
            "molecule_identity_key": parent,
            "molecule_identity": rows[0].get("molecule_identity", {}),
            "bemis_murcko_scaffold": rows[0]["bemis_murcko_scaffold"],
            "condition_scope": "external",
            "condition_group": group,
            "condition_atoms": list(rows[0]["condition_atoms"]),
            "label_counts": {"0": n0, "1": n1},
            "source_record_count": len(rows),
            "majority_label": majority,
            "majority_record_count": max(n0, n1),
            "minority_record_count": min(n0, n1),
            "agreement_fraction": agreement,
            "agreement_threshold": agreement_threshold,
            "vote_unit": "terminally_reviewed_source_record",
            "source_record_ids": _unique(row["source_record_id"] for row in rows),
            "source_pmids": _unique(row.get("pmid", "") for row in rows),
            "raw_value_examples": _unique(
                (row.get("raw_value", "") for row in rows), limit=20
            ),
            "condition_text_examples": _unique(
                (row.get("condition_text", "") for row in rows), limit=20
            ),
            "label_methods": dict(Counter(str(row["label_method"]) for row in rows)),
            "reviewers": sorted({str(row["reviewer"]) for row in rows}),
        }
        if n0 == n1:
            rejected.append({**base, "drop_reason": "parent_condition_label_tie"})
        elif agreement < agreement_threshold:
            rejected.append({**base, "drop_reason": "parent_condition_agreement_below_threshold"})
        else:
            accepted.append(
                {
                    **base,
                    "Y": majority,
                    "label_decision": "unanimous" if min(n0, n1) == 0 else "accepted_record_majority_60",
                }
            )
    return accepted, rejected


def candidate_group_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_parents: int,
    minimum_scaffolds: int,
    excluded_groups: set[str] | None = None,
    allowed_groups: set[str] | None = None,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    excluded_groups = excluded_groups or set()
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_group"])].append(row)
    candidates: set[str] = set()
    audit: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(grouped.items()):
        parents = {row["molecule_identity_key"] for row in group_rows}
        scaffolds = {row["bemis_murcko_scaffold"] for row in group_rows}
        reasons = []
        if allowed_groups is not None and group not in allowed_groups:
            reasons.append("not_in_task_condition_group_allowlist")
        if group in excluded_groups:
            reasons.append("task_semantic_group_exclusion")
        if len(parents) < minimum_parents:
            reasons.append("fewer_than_minimum_accepted_parents")
        if len(scaffolds) < minimum_scaffolds:
            reasons.append("fewer_than_minimum_distinct_scaffolds")
        if not reasons:
            candidates.add(group)
        audit[group] = {
            "accepted_parents": len(parents),
            "distinct_scaffolds": len(scaffolds),
            "label_counts": dict(Counter(str(row["Y"]) for row in group_rows)),
            "preallocation_eligible": not reasons,
            "preallocation_exclusion_reasons": reasons,
        }
    return candidates, audit


def allocate_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_groups: set[str],
    parent_assignment: Mapping[str, str],
    scaffold_assignment: Mapping[str, str],
    config: ConditionedBenchmarkConfig,
) -> tuple[dict[str, str], set[str], dict[str, Any]]:
    candidate_rows = [row for row in rows if row["condition_group"] in candidate_groups]
    novel_scaffolds = sorted(
        {
            row["bemis_murcko_scaffold"]
            for row in candidate_rows
            if row["molecule_identity_key"] not in parent_assignment
            and row["bemis_murcko_scaffold"] not in scaffold_assignment
        },
        key=lambda value: _stable_hash(f"{config.seed}\0{value}"),
    )
    groups = sorted(candidate_groups)
    scaffold_index = {value: index for index, value in enumerate(novel_scaffolds)}
    fixed = Counter()
    novel = Counter()
    for row in candidate_rows:
        group = row["condition_group"]
        parent = row["molecule_identity_key"]
        scaffold = row["bemis_murcko_scaffold"]
        split = parent_assignment.get(parent) or scaffold_assignment.get(scaffold)
        if split:
            fixed[(group, split)] += 1
        else:
            novel[(group, scaffold)] += 1

    n_scaffolds, n_groups = len(novel_scaffolds), len(groups)
    n_binary = n_scaffolds * len(SPLITS) + n_groups
    if n_binary:
        objective = np.zeros(n_binary)
        for index, group in enumerate(groups):
            support = sum(row["condition_group"] == group for row in candidate_rows)
            objective[n_scaffolds * len(SPLITS) + index] = -1_000_000 - support
        constraints: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        for scaffold, index in scaffold_index.items():
            vector = np.zeros(n_binary)
            for split_index in range(len(SPLITS)):
                vector[split_index * n_scaffolds + index] = 1
            constraints.append(vector); lower.append(1); upper.append(1)
        for group_index, group in enumerate(groups):
            y_index = n_scaffolds * len(SPLITS) + group_index
            for split_index, split in enumerate(SPLITS):
                vector = np.zeros(n_binary)
                for scaffold, scaffold_idx in scaffold_index.items():
                    vector[split_index * n_scaffolds + scaffold_idx] = novel[(group, scaffold)]
                vector[y_index] = -1
                constraints.append(vector); lower.append(-fixed[(group, split)]); upper.append(np.inf)
        result = milp(
            objective,
            integrality=np.ones(n_binary),
            bounds=Bounds(np.zeros(n_binary), np.ones(n_binary)),
            constraints=LinearConstraint(np.asarray(constraints), np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 300},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"Condition-group allocation failed: {result.message}")
        selected = {
            group for index, group in enumerate(groups)
            if result.x[n_scaffolds * len(SPLITS) + index] > 0.5
        }
    else:
        selected = {group for group in groups if all(fixed[(group, split)] for split in SPLITS)}

    allocation = _balanced_assignment(
        candidate_rows,
        selected_groups=selected,
        novel_scaffolds=novel_scaffolds,
        parent_assignment=parent_assignment,
        scaffold_assignment=scaffold_assignment,
        config=config,
    )
    distribution = _group_split_counts(
        candidate_rows,
        selected_groups=selected,
        allocation=allocation,
        parent_assignment=parent_assignment,
        scaffold_assignment=scaffold_assignment,
    )
    invalid = {group: counts for group, counts in distribution.items() if min(counts.values()) < 1}
    if invalid:
        raise RuntimeError(f"Selected groups lack three-way coverage: {invalid}")
    return allocation, selected, {
        "method": "frozen_parent_scaffold_plus_milp_group_coverage",
        "seed": config.seed,
        "n_candidate_groups": len(candidate_groups),
        "n_selected_groups": len(selected),
        "selected_groups": sorted(selected),
        "n_novel_scaffolds": len(novel_scaffolds),
        "group_split_counts": distribution,
        "group_exclusion_reasons": {
            group: "cannot_cover_all_three_splits_without_moving_frozen_scaffolds"
            for group in candidate_groups - selected
        },
    }


def _balanced_assignment(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_groups: set[str],
    novel_scaffolds: list[str],
    parent_assignment: Mapping[str, str],
    scaffold_assignment: Mapping[str, str],
    config: ConditionedBenchmarkConfig,
) -> dict[str, str]:
    if not novel_scaffolds:
        return {}
    selected_rows = [row for row in rows if row["condition_group"] in selected_groups]
    n_scaffolds = len(novel_scaffolds)
    scaffold_index = {value: index for index, value in enumerate(novel_scaffolds)}
    n_binary = n_scaffolds * len(SPLITS)
    n_variables = n_binary + len(SPLITS)
    objective = np.r_[np.zeros(n_binary), np.ones(len(SPLITS))]
    constraints: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for scaffold, index in scaffold_index.items():
        vector = np.zeros(n_variables)
        for split_index in range(len(SPLITS)):
            vector[split_index * n_scaffolds + index] = 1
        constraints.append(vector); lower.append(1); upper.append(1)
    for group in sorted(selected_groups):
        for split_index, split in enumerate(SPLITS):
            vector = np.zeros(n_variables)
            fixed = 0
            for row in selected_rows:
                if row["condition_group"] != group:
                    continue
                parent, scaffold = row["molecule_identity_key"], row["bemis_murcko_scaffold"]
                prior = parent_assignment.get(parent) or scaffold_assignment.get(scaffold)
                if prior == split:
                    fixed += 1
                elif not prior:
                    vector[split_index * n_scaffolds + scaffold_index[scaffold]] += 1
            constraints.append(vector); lower.append(1 - fixed); upper.append(np.inf)
    fixed_total = Counter()
    rows_per_scaffold = Counter()
    for row in selected_rows:
        parent, scaffold = row["molecule_identity_key"], row["bemis_murcko_scaffold"]
        prior = parent_assignment.get(parent) or scaffold_assignment.get(scaffold)
        if prior:
            fixed_total[prior] += 1
        else:
            rows_per_scaffold[scaffold] += 1
    for split_index, (split, target_fraction) in enumerate(zip(SPLITS, config.split_targets)):
        value = np.zeros(n_variables)
        for scaffold, index in scaffold_index.items():
            value[split_index * n_scaffolds + index] = rows_per_scaffold[scaffold]
        target = target_fraction * len(selected_rows) - fixed_total[split]
        for sign in (1.0, -1.0):
            vector = sign * value
            vector[n_binary + split_index] = -1
            constraints.append(vector); lower.append(-np.inf); upper.append(sign * target)
    result = milp(
        objective,
        integrality=np.r_[np.ones(n_binary), np.zeros(len(SPLITS))],
        bounds=Bounds(np.zeros(n_variables), np.r_[np.ones(n_binary), np.full(len(SPLITS), np.inf)]),
        constraints=LinearConstraint(np.asarray(constraints), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 300},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Balanced scaffold assignment failed: {result.message}")
    allocation = {}
    for scaffold, index in scaffold_index.items():
        chosen = [
            split for split_index, split in enumerate(SPLITS)
            if result.x[split_index * n_scaffolds + index] > 0.5
        ]
        if len(chosen) != 1:
            raise RuntimeError(f"Invalid split assignment for scaffold {scaffold}: {chosen}")
        allocation[scaffold] = chosen[0]
    return allocation


def _prepare_frozen_rows(
    by_split: Mapping[str, Sequence[Mapping[str, Any]]], frozen_lineage: str
) -> list[dict[str, Any]]:
    output = []
    for split, rows in by_split.items():
        for raw in rows:
            row = dict(raw)
            parent = str(row["molecule_identity_key"])
            output.append(
                {
                    **row,
                    "condition_scope": "none_reported",
                    "condition_group": NO_REPORTED_CONDITION,
                    "condition_atoms": [],
                    "split": split,
                    "split_policy": str(row.get("split_policy") or frozen_lineage),
                    "benchmark_row_id": _row_id("NULL", parent, NO_REPORTED_CONDITION),
                }
            )
    return output


def _frozen_assignments(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    parents: dict[str, str] = {}
    scaffolds: dict[str, str] = {}
    for row in rows:
        parent, scaffold, split = row["molecule_identity_key"], str(row.get("bemis_murcko_scaffold") or ""), row["split"]
        if parent in parents and parents[parent] != split:
            raise RuntimeError(f"Frozen parent crosses splits: {parent}")
        if scaffold in scaffolds and scaffolds[scaffold] != split:
            raise RuntimeError(f"Frozen scaffold crosses splits: {scaffold}")
        parents[parent] = split; scaffolds[scaffold] = split
    return parents, scaffolds


def _reuse_frozen_identity(rows: Sequence[dict[str, Any]], frozen: Mapping[str, Mapping[str, Any]]) -> None:
    for row in rows:
        prior = frozen.get(row["molecule_identity_key"])
        if not prior:
            continue
        row["drug"] = prior["drug"]
        row["bemis_murcko_scaffold"] = prior.get("bemis_murcko_scaffold", "")
        if prior.get("molecule_identity"):
            row["molecule_identity"] = prior["molecule_identity"]


def validate_split_integrity(combined: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    keys: set[tuple[str, str]] = set()
    parents: dict[str, str] = {}
    scaffolds: dict[str, str] = {}
    for split, rows in combined.items():
        for row in rows:
            key = (row["molecule_identity_key"], row["condition_group"])
            if key in keys:
                raise RuntimeError(f"Duplicate parent-condition row: {key}")
            keys.add(key)
            parent, scaffold = row["molecule_identity_key"], row["bemis_murcko_scaffold"]
            if parent in parents and parents[parent] != split:
                raise RuntimeError(f"Parent crosses splits: {parent}")
            if scaffold in scaffolds and scaffolds[scaffold] != split:
                raise RuntimeError(f"Scaffold crosses splits: {scaffold}")
            parents[parent] = split; scaffolds[scaffold] = split


def summarize_groups(combined: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    all_rows = [row for rows in combined.values() for row in rows]
    output = []
    for group in sorted({row["condition_group"] for row in all_rows}, key=lambda value: (value != NO_REPORTED_CONDITION, value)):
        rows = [row for row in all_rows if row["condition_group"] == group]
        output.append(
            {
                "condition_group": group,
                "condition_scope": rows[0]["condition_scope"],
                "n_molecule_condition_rows": len(rows),
                "n_unique_parents": len({row["molecule_identity_key"] for row in rows}),
                "n_distinct_scaffolds": len({row["bemis_murcko_scaffold"] for row in rows}),
                **{f"n_{split}": sum(row["split"] == split for row in rows) for split in SPLITS},
                "n_y0": sum(int(row["Y"]) == 0 for row in rows),
                "n_y1": sum(int(row["Y"]) == 1 for row in rows),
            }
        )
    return output


def _group_split_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_groups: set[str],
    allocation: Mapping[str, str],
    parent_assignment: Mapping[str, str],
    scaffold_assignment: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    output = {group: {split: 0 for split in SPLITS} for group in sorted(selected_groups)}
    for row in rows:
        group = row["condition_group"]
        if group not in selected_groups:
            continue
        parent, scaffold = row["molecule_identity_key"], row["bemis_murcko_scaffold"]
        split = parent_assignment.get(parent) or scaffold_assignment.get(scaffold) or allocation[scaffold]
        output[group][split] += 1
    return output


def _summary(
    *,
    config: ConditionedBenchmarkConfig,
    n_full_review_ledger_records: int,
    frozen_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
    combined: Mapping[str, Sequence[Mapping[str, Any]]],
    audits: Sequence[Mapping[str, Any]],
    accepted_units: Sequence[Mapping[str, Any]],
    rejected_units: Sequence[Mapping[str, Any]],
    group_rejected: Sequence[Mapping[str, Any]],
    preallocation: Mapping[str, Any],
    allocation_audit: Mapping[str, Any],
    group_distribution: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identities = {split: {row["molecule_identity_key"] for row in rows} for split, rows in combined.items()}
    scaffolds = {split: {row["bemis_murcko_scaffold"] for row in rows} for split, rows in combined.items()}
    return {
        "task": config.task_name,
        "lineage": config.lineage,
        "protocol_version": config.protocol_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "source_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in config.source_artifacts
        ],
        "frozen_molecule_only_source": {
            "path": str(config.frozen_root),
            "lineage": config.frozen_lineage,
            "n_rows": len(frozen_rows),
            "split_sha256": {
                split: sha256_file(config.frozen_root / f"{split}.jsonl") for split in SPLITS
            },
            "preservation_policy": "labels and split assignments copied without modification",
        },
        "label_policy": {
            "gold_unit": "parent_molecule_x_exact_condition_signature",
            "none_reported_group": NO_REPORTED_CONDITION,
            "external_condition_agreement_threshold": config.external_agreement_threshold,
            "tie_policy": "reject exact ties",
            "review_policy": config.review_policy,
        },
        "group_gate": {
            "minimum_accepted_parents": config.minimum_group_parents,
            "minimum_distinct_scaffolds": config.minimum_group_scaffolds,
            "minimum_per_split": 1,
            "task_semantic_excluded_groups": list(config.excluded_condition_groups),
            "condition_group_allowlist": (
                list(config.allowed_condition_groups)
                if config.allowed_condition_groups is not None
                else None
            ),
            "preallocation": preallocation,
            "allocation": allocation_audit,
        },
        "counts": {
            "n_full_review_ledger_records": n_full_review_ledger_records,
            "n_published_group_review_records": len(audits),
            "n_reviewed_source_records": len(audits),
            "n_review_accepted": sum(row["review_status"] == "accepted" for row in audits),
            "n_review_rejected": sum(row["review_status"] == "rejected" for row in audits),
            "n_frozen_no_reported_rows": len(frozen_rows),
            "n_added_external_condition_rows": len(external_rows),
            "n_total_benchmark_rows": sum(len(rows) for rows in combined.values()),
            "n_unique_parents": len({row["molecule_identity_key"] for rows in combined.values() for row in rows}),
            "n_external_groups": len(group_distribution) - 1,
            "n_parent_conditions_accepted_before_group_gate": len(accepted_units),
            "n_parent_conditions_rejected_by_vote": len(rejected_units),
            "n_parent_conditions_rejected_by_group_gate": len(group_rejected),
        },
        "review": {
            "reviewer_counts": dict(Counter(str(row["reviewer"]) for row in audits)),
            "status_counts": dict(Counter(str(row["review_status"]) for row in audits)),
            "reason_counts": dict(Counter(str(row["review_reason"]) for row in audits)),
        },
        "splits": {
            split: {
                "n_rows": len(rows),
                "n_unique_parents": len({row["molecule_identity_key"] for row in rows}),
                "n_no_reported": sum(row["condition_group"] == NO_REPORTED_CONDITION for row in rows),
                "n_external": sum(row["condition_group"] != NO_REPORTED_CONDITION for row in rows),
                "label_counts": dict(Counter(str(row["Y"]) for row in rows)),
            }
            for split, rows in combined.items()
        },
        "pairwise_identity_overlap": _pairwise_overlap(identities),
        "pairwise_scaffold_overlap": _pairwise_overlap(scaffolds),
        "group_distribution": list(group_distribution),
        "paths": {
            "root": str(config.output_root),
            "group_distribution": str(config.output_root / "group_distribution.csv"),
            "source_condition_review": str(config.output_root / "source_condition_review.jsonl"),
        },
    }


def _pairwise_overlap(values: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        "train_valid": len(values["train"] & values["valid"]),
        "train_test": len(values["train"] & values["test"]),
        "valid_test": len(values["valid"] & values["test"]),
    }


def _minimal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "drug": row["drug"],
        "Y": int(row["Y"]),
        "condition_group": row["condition_group"],
        "condition_scope": row["condition_scope"],
        "molecule_identity_key": row["molecule_identity_key"],
        "bemis_murcko_scaffold": row["bemis_murcko_scaffold"],
        "benchmark_row_id": row["benchmark_row_id"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader(); writer.writerows(rows)


def _row_id(prefix: str, parent: str, condition_group: str) -> str:
    digest = hashlib.sha256(f"{parent}\0{condition_group}".encode()).hexdigest()[:20]
    return f"{prefix}_{digest.upper()}"


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _unique(values: Iterable[Any], *, limit: int = 50) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))[:limit]
