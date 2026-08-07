"""Deterministic scaffold-aware train-only folds for task-local routers."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.molecule_identity import IDENTITY_NORMALIZER_VERSION

from .contract import FOLD_SCHEMA_VERSION, TaskSpec, fold_root, task_root
from .io import read_jsonl as _read_jsonl, sha256_file as _sha256_file


def prepare_task_folds(
    spec: TaskSpec,
    *,
    data_root: str | Path,
    output_root: str | Path,
    split: str,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    split_dir = Path(data_root) / spec.data_name / split
    train_path = split_dir / "train.jsonl"
    labels_path = split_dir / "train_molecule_labels.jsonl"
    benchmark_heldout_path = split_dir / "heldout_molecule_labels.jsonl"
    for path in (train_path, labels_path, benchmark_heldout_path):
        if not path.exists():
            raise FileNotFoundError(path)

    train_rows = _read_jsonl(train_path)
    detail_rows = _read_jsonl(labels_path)
    benchmark_heldout = _read_jsonl(benchmark_heldout_path)
    aligned_details = _align_train_provenance(train_rows, detail_rows, labels_path)

    enriched: list[dict[str, Any]] = []
    for train_index, (row, detail) in enumerate(zip(train_rows, aligned_details)):
        label = int(row["Y"])
        key = str(detail["molecule_identity_key"])
        if int(detail["Y"]) != label:
            raise ValueError(f"Label mismatch for train identity {key}")
        scaffold = str(detail.get("bemis_murcko_scaffold") or "")
        group_key = f"scaffold:{scaffold}" if scaffold else f"acyclic_parent:{key}"
        enriched.append(
            {
                "train_index": train_index,
                "drug": str(row["drug"]),
                "Y": label,
                "molecule_identity_key": key,
                "bemis_murcko_scaffold": scaffold,
                "fold_group_key": group_key,
                "detail": detail,
            }
        )

    assignments = assign_grouped_folds(enriched, n_folds=n_folds, seed=seed)
    benchmark_keys = {str(row["molecule_identity_key"]) for row in benchmark_heldout}
    train_keys = {row["molecule_identity_key"] for row in enriched}
    overlap = train_keys & benchmark_keys
    if overlap:
        raise AssertionError(f"Train overlaps frozen valid/test parents: {len(overlap)}")

    task_dir = task_root(output_root, spec.task)
    assignment_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(n_folds):
        query = [row for row in enriched if assignments[row["fold_group_key"]] == fold]
        query.sort(key=lambda row: row["train_index"])
        query_keys = {row["molecule_identity_key"] for row in query}
        query_groups = {row["fold_group_key"] for row in query}
        other_groups = {
            row["fold_group_key"]
            for row in enriched
            if assignments[row["fold_group_key"]] != fold
        }
        if query_groups & other_groups:
            raise AssertionError(f"Fold {fold} group leakage")

        current_root = fold_root(output_root, spec.task, fold)
        query_minimal = [
            {
                "drug": row["drug"],
                "Y": row["Y"],
                "oof_train_index": row["train_index"],
                "oof_fold": fold,
            }
            for row in query
        ]
        query_labels = [
            {
                **row["detail"],
                "split": "router_oof_query",
                "oof_train_index": row["train_index"],
                "oof_fold": fold,
                "fold_group_key": row["fold_group_key"],
            }
            for row in query
        ]
        heldout_union = [
            {**row, "router_oof_holdout_reason": "frozen_valid_or_test"}
            for row in benchmark_heldout
        ] + [
            {**row, "router_oof_holdout_reason": "oof_query_fold"}
            for row in query_labels
        ]
        union_keys = [str(row["molecule_identity_key"]) for row in heldout_union]
        if len(union_keys) != len(set(union_keys)):
            raise AssertionError(f"Fold {fold} held-out union contains duplicate parents")

        query_path = current_root / "agent_input" / "valid.jsonl"
        query_labels_path = current_root / "query_molecule_labels.jsonl"
        heldout_path = current_root / "heldout_molecule_labels.jsonl"
        write_jsonl_atomic(query_path, query_minimal)
        write_jsonl_atomic(query_labels_path, query_labels)
        write_jsonl_atomic(heldout_path, heldout_union)
        fold_summary = {
            "schema_version": FOLD_SCHEMA_VERSION,
            "task": spec.task,
            "split": split,
            "fold": fold,
            "n_folds": n_folds,
            "seed": seed,
            "n_query": len(query),
            "query_label_counts": _label_counts(query),
            "n_query_groups": len(query_groups),
            "n_query_parent_identities": len(query_keys),
            "n_benchmark_heldout_parent_identities": len(benchmark_keys),
            "n_heldout_union_parent_identities": len(union_keys),
            "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
            "paths": {
                "query": str(query_path),
                "query_molecule_labels": str(query_labels_path),
                "heldout_molecule_labels": str(heldout_path),
            },
            "sha256": {
                "query": _sha256_file(query_path),
                "query_molecule_labels": _sha256_file(query_labels_path),
                "heldout_molecule_labels": _sha256_file(heldout_path),
            },
            "zero_group_overlap": True,
            "zero_benchmark_parent_overlap": True,
        }
        write_json_atomic(current_root / "fold_manifest.json", fold_summary)
        fold_summaries.append(fold_summary)
        assignment_rows.extend(
            {
                "task": spec.task,
                "train_index": row["train_index"],
                "fold": fold,
                "drug": row["drug"],
                "Y": row["Y"],
                "molecule_identity_key": row["molecule_identity_key"],
                "bemis_murcko_scaffold": row["bemis_murcko_scaffold"],
                "fold_group_key": row["fold_group_key"],
            }
            for row in query
        )

    assignment_rows.sort(key=lambda row: int(row["train_index"]))
    if [row["train_index"] for row in assignment_rows] != list(range(len(train_rows))):
        raise AssertionError("OOF assignments do not cover every train row exactly once")
    assignments_path = task_dir / "folds.jsonl"
    write_jsonl_atomic(assignments_path, assignment_rows)
    task_manifest = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "task": spec.task,
        "data_name": spec.data_name,
        "split": split,
        "n_folds": n_folds,
        "seed": seed,
        "n_train": len(train_rows),
        "label_counts": _label_counts(enriched),
        "n_fold_groups": len(assignments),
        "n_acyclic_parent_groups": len(
            {row["fold_group_key"] for row in enriched if not row["bemis_murcko_scaffold"]}
        ),
        "folds": fold_summaries,
        "paths": {
            "source_train": str(train_path),
            "source_train_molecule_labels": str(labels_path),
            "source_benchmark_heldout": str(benchmark_heldout_path),
            "assignments": str(assignments_path),
        },
        "sha256": {
            "source_train": _sha256_file(train_path),
            "source_train_molecule_labels": _sha256_file(labels_path),
            "source_benchmark_heldout": _sha256_file(benchmark_heldout_path),
            "assignments": _sha256_file(assignments_path),
        },
        "all_train_rows_assigned_once": True,
        "zero_group_overlap": True,
        "zero_benchmark_parent_overlap": True,
    }
    write_json_atomic(task_dir / "fold_manifest.json", task_manifest)
    return task_manifest


def assign_grouped_folds(
    rows: Iterable[dict[str, Any]],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["fold_group_key"])].append(row)
    if len(groups) < n_folds:
        raise ValueError(f"Need at least {n_folds} groups, found {len(groups)}")

    total = sum(len(group) for group in groups.values())
    total_pos = sum(int(row["Y"]) for group in groups.values() for row in group)
    total_neg = total - total_pos
    targets = (total / n_folds, total_pos / n_folds, total_neg / n_folds)
    fold_stats = [{"n": 0, "pos": 0, "neg": 0} for _ in range(n_folds)]
    assignments: dict[str, int] = {}

    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -abs(sum(int(row["Y"]) for row in item[1]) - len(item[1]) / 2),
            _stable_hash(f"{seed}\0{item[0]}"),
        ),
    )
    for group_key, group_rows in ordered:
        group_n = len(group_rows)
        group_pos = sum(int(row["Y"]) for row in group_rows)
        group_neg = group_n - group_pos
        candidates: list[tuple[float, str, int]] = []
        for fold in range(n_folds):
            projected = [dict(stats) for stats in fold_stats]
            projected[fold]["n"] += group_n
            projected[fold]["pos"] += group_pos
            projected[fold]["neg"] += group_neg
            score = sum(
                ((stats["n"] - targets[0]) / max(targets[0], 1.0)) ** 2
                + ((stats["pos"] - targets[1]) / max(targets[1], 1.0)) ** 2
                + ((stats["neg"] - targets[2]) / max(targets[2], 1.0)) ** 2
                for stats in projected
            )
            tie = _stable_hash(f"{seed}\0{group_key}\0{fold}")
            candidates.append((score, tie, fold))
        _, _, selected = min(candidates)
        assignments[group_key] = selected
        fold_stats[selected]["n"] += group_n
        fold_stats[selected]["pos"] += group_pos
        fold_stats[selected]["neg"] += group_neg
    if any(stats["n"] == 0 for stats in fold_stats):
        raise AssertionError("Grouped fold assignment produced an empty fold")
    return assignments


def _label_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[str(int(row["Y"]))] += 1
    return counts


def _align_train_provenance(
    train_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    labels_path: Path,
) -> list[dict[str, Any]]:
    """Use frozen builder ordering and avoid re-normalizing 18k molecules with RDKit."""
    if len(train_rows) != len(detail_rows):
        raise ValueError(
            f"Train/provenance row-count mismatch: {len(train_rows)} != {len(detail_rows)}"
        )
    if all(
        str(train.get("drug")) == str(detail.get("drug"))
        and int(train.get("Y")) == int(detail.get("Y"))
        for train, detail in zip(train_rows, detail_rows)
    ):
        return detail_rows
    details_by_drug = {str(row.get("drug")): row for row in detail_rows}
    if len(details_by_drug) != len(detail_rows):
        raise ValueError(f"Duplicate train drugs prevent provenance alignment in {labels_path}")
    aligned = []
    for index, train in enumerate(train_rows):
        detail = details_by_drug.get(str(train.get("drug")))
        if detail is None or int(detail.get("Y")) != int(train.get("Y")):
            raise ValueError(f"No matching provenance for train row {index} in {labels_path}")
        aligned.append(detail)
    return aligned


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
