"""Publish deterministic scaffold-disjoint subsets of Gold validation rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import (
    BENCHMARK_ROOT,
    TASK_DIRECTORIES,
)
from data.processing.paths import REPO_ROOT
from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic


CONTRACT = "valid_small.scaffold_group_selection.v2"
DEFAULT_TARGET_SIZE = 100
DEFAULT_SEED = 20260723
PARENT_MAXIMIZED_TASKS = frozenset({"ames", "carcinogens", "dili"})


def select_scaffold_groups(
    rows: list[dict[str, Any]],
    *,
    target_size: int,
    seed: int,
    maximize_parents: bool = False,
) -> set[str]:
    """Choose whole scaffolds by size and the requested secondary objective."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if not rows:
        raise ValueError("validation rows are empty")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        if not scaffold:
            raise ValueError("valid_small requires a non-empty scaffold for every row")
        label = int(row["Y"])
        if label not in {0, 1}:
            raise ValueError(f"valid_small requires binary labels, found {label}")
        groups[scaffold].append(row)

    ordered = sorted(
        groups.items(),
        key=lambda item: (
            hashlib.sha256(
                f"{seed}\0valid_small\0{item[0]}".encode("utf-8")
            ).hexdigest(),
            item[0],
        ),
    )
    # A bit mask keeps one deterministic, first-seen solution per
    # (row count, positive count) without storing many scaffold tuples.
    reachable: dict[tuple[int, int], tuple[int, int]] = {(0, 0): (0, 0)}
    for index, (_, group_rows) in enumerate(ordered):
        group_size = len(group_rows)
        group_positives = sum(int(row["Y"]) for row in group_rows)
        group_parents = len(
            {str(row["molecule_identity_key"]) for row in group_rows}
        )
        additions: dict[tuple[int, int], tuple[int, int]] = {}
        for (size, positives), (parents, mask) in reachable.items():
            state = (size + group_size, positives + group_positives)
            candidate = (parents + group_parents, mask | (1 << index))
            previous = reachable.get(state) or additions.get(state)
            if previous is None or (maximize_parents and candidate[0] > previous[0]):
                additions[state] = candidate
        for state, candidate in additions.items():
            previous = reachable.get(state)
            if previous is None or (maximize_parents and candidate[0] > previous[0]):
                reachable[state] = candidate

    total_size = len(rows)
    total_positives = sum(int(row["Y"]) for row in rows)
    selected_state = min(
        (state for state in reachable if state[0] > 0),
        key=lambda state: (
            abs(state[0] - target_size),
            -reachable[state][0] if maximize_parents else 0,
            Fraction(
                abs(state[1] * total_size - total_positives * state[0]),
                state[0] * total_size,
            ),
        ),
    )
    mask = reachable[selected_state][1]
    return {
        scaffold
        for index, (scaffold, _) in enumerate(ordered)
        if mask & (1 << index)
    }


def build_all(
    *,
    source_root: Path = BENCHMARK_ROOT,
    output_root: Path = BENCHMARK_ROOT,
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for task, directory in TASK_DIRECTORIES.items():
        version = _active_version(source_root / directory)
        source = source_root / directory / version / "scaffold"
        destination = output_root / directory / version / "scaffold"
        manifests[task] = build_task(
            task=task,
            source=source,
            destination=destination,
            target_size=target_size,
            seed=seed,
        )
    return manifests


def build_task(
    *,
    task: str,
    source: Path,
    destination: Path,
    target_size: int,
    seed: int,
) -> dict[str, Any]:
    compact_path = source / "valid.jsonl"
    detail_path = source / "valid_molecule_condition_labels.jsonl"
    compact_lines, compact_rows = _read_lines(compact_path)
    detail_lines, detail_rows = _read_lines(detail_path)
    compact_ids = _row_ids(compact_rows, compact_path)
    detail_ids = _row_ids(detail_rows, detail_path)
    if compact_ids != detail_ids:
        raise ValueError(f"compact/detailed validation rows are not aligned for {task}")

    selected_scaffolds = select_scaffold_groups(
        compact_rows,
        target_size=target_size,
        seed=seed,
        maximize_parents=task in PARENT_MAXIMIZED_TASKS,
    )
    selected_indexes = [
        index
        for index, row in enumerate(compact_rows)
        if str(row["bemis_murcko_scaffold"]) in selected_scaffolds
    ]
    selected_rows = [compact_rows[index] for index in selected_indexes]
    remainder_rows = [
        row
        for row in compact_rows
        if str(row["bemis_murcko_scaffold"]) not in selected_scaffolds
    ]
    overlap = selected_scaffolds & {
        str(row["bemis_murcko_scaffold"]) for row in remainder_rows
    }
    if overlap:
        raise AssertionError(f"valid_small scaffold leakage for {task}: {len(overlap)}")

    destination.mkdir(parents=True, exist_ok=True)
    selected_path = destination / "valid_small.jsonl"
    selected_detail_path = destination / "valid_small_molecule_condition_labels.jsonl"
    _write_lines(selected_path, [compact_lines[index] for index in selected_indexes])
    _write_lines(selected_detail_path, [detail_lines[index] for index in selected_indexes])

    full_counts = Counter(int(row["Y"]) for row in compact_rows)
    selected_counts = Counter(int(row["Y"]) for row in selected_rows)
    manifest = {
        "contract": CONTRACT,
        "task": task,
        "selection": {
            "method": "whole_scaffold_dynamic_program",
            "objective_order": [
                "minimum_absolute_row_count_error",
                *(
                    [
                        "maximum_distinct_parent_count",
                        "minimum_validation_label_ratio_error",
                    ]
                    if task in PARENT_MAXIMIZED_TASKS
                    else ["minimum_validation_label_ratio_error"]
                ),
                "stable_scaffold_hash_tie_break",
            ],
            "target_rows": target_size,
            "actual_rows": len(selected_rows),
            "row_count_error": abs(len(selected_rows) - target_size),
            "seed": seed,
        },
        "source": {
            "valid": _portable_path(compact_path),
            "valid_sha256": sha256_file(compact_path),
            "valid_molecule_condition_labels": _portable_path(detail_path),
            "valid_molecule_condition_labels_sha256": sha256_file(detail_path),
            "rows": len(compact_rows),
            "label_counts": _label_counts(full_counts),
        },
        "valid_small": {
            "path": _portable_path(source / "valid_small.jsonl"),
            "sha256": sha256_file(selected_path),
            "molecule_condition_labels_path": _portable_path(
                source / "valid_small_molecule_condition_labels.jsonl"
            ),
            "molecule_condition_labels_sha256": sha256_file(selected_detail_path),
            "rows": len(selected_rows),
            "distinct_parents": len(
                {str(row["molecule_identity_key"]) for row in selected_rows}
            ),
            "distinct_scaffolds": len(selected_scaffolds),
            "label_counts": _label_counts(selected_counts),
            "label_ratio_absolute_error": abs(
                selected_counts[1] / len(selected_rows)
                - full_counts[1] / len(compact_rows)
            ),
            "scaffold_overlap_with_valid_remainder": 0,
        },
    }
    write_json_atomic(destination / "valid_small_manifest.json", manifest)
    return manifest


def _active_version(task_root: Path) -> str:
    current = task_root / "CURRENT"
    version = current.read_text(encoding="utf-8").strip()
    if not version or "/" in version or "\\" in version:
        raise ValueError(f"invalid gold-label release pointer: {current}")
    return version


def _read_lines(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return lines, [json.loads(line) for line in lines]


def _row_ids(rows: list[dict[str, Any]], path: Path) -> list[str]:
    row_ids = [str(row["benchmark_row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"duplicate benchmark_row_id in {path}")
    return row_ids


def _write_lines(path: Path, lines: list[str]) -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(label): counts[label] for label in (0, 1)}


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifests = build_all(
        source_root=args.source_root,
        output_root=args.output_root,
        target_size=args.target_size,
        seed=args.seed,
    )
    print(json.dumps(manifests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
