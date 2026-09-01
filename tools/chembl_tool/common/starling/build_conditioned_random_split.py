"""Build deterministic parent-grouped random splits for Conditioned Benchmark.

The input cohort is the exact union of the current scaffold train/valid/test
rows.  Labels and molecule-condition units are immutable.  All rows belonging
to one normalized parent stay together.  A deterministic lexicographic MILP
assigns an exact 80/10/10 row split.  It first minimizes singleton-vote labels
in valid+test, then balances unavoidable singletons across valid/test, and only
then balances labels and conditions.  Every condition must occur in all three
partitions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    BENCHMARK_ROOT,
    CONTRACT,
    TASK_DIRECTORIES,
    task_root,
)


RANDOM_SPLIT_CONTRACT = "conditioned_random_parent_grouped_quality_stratified.v2"
DEFAULT_SEED = 20260830
SPLITS = ("train", "valid", "test")
EVAL_FRACTION = 0.10
SOLVER_TIME_LIMIT_SECONDS = 300
VOTE_COUNT_FIELD = "source_record_count"
MINIMAL_FIELDS = (
    "drug",
    "Y",
    "condition_group",
    "condition_scope",
    "molecule_identity_key",
    "bemis_murcko_scaffold",
    "benchmark_row_id",
)


@dataclass(frozen=True)
class ParentGroup:
    parent: str
    rows: tuple[dict[str, Any], ...]
    condition_counts: Mapping[str, int]

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def positive_count(self) -> int:
        return sum(int(row["Y"]) for row in self.rows)

    @property
    def singleton_count(self) -> int:
        return sum(int(row[VOTE_COUNT_FIELD]) == 1 for row in self.rows)


def _rounded_fraction(total: int, fraction: float) -> int:
    return int(math.floor(total * fraction + 0.5))


def _stable_key(*values: object) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).hexdigest()


def _parent_groups(rows: list[dict[str, Any]]) -> list[ParentGroup]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parent = str(row.get("molecule_identity_key") or "").strip()
        if not parent:
            raise ValueError("conditioned row lacks molecule_identity_key")
        grouped[parent].append(row)
    return [
        ParentGroup(
            parent=parent,
            rows=tuple(parent_rows),
            condition_counts=Counter(
                str(row["condition_group"]) for row in parent_rows
            ),
        )
        for parent, parent_rows in sorted(grouped.items())
    ]


def _append_constraint(
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    row: np.ndarray,
    lower: float,
    upper: float,
) -> None:
    rows.append(row)
    low.append(lower)
    high.append(upper)


def _solve(
    objective: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
) -> np.ndarray:
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(np.asarray(rows), np.asarray(low), np.asarray(high)),
        options={"time_limit": SOLVER_TIME_LIMIT_SECONDS},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"random split allocation failed: {result.message}")
    return result.x


def _solve_absolute_difference(
    *,
    integrality: np.ndarray,
    bounds: Bounds,
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, int]:
    n = len(integrality)
    extended_rows = [np.r_[row, 0.0] for row in rows]
    extended_low = list(low)
    extended_high = list(high)
    for difference in (left - right, right - left):
        _append_constraint(
            extended_rows,
            extended_low,
            extended_high,
            np.r_[difference, -1.0],
            -np.inf,
            0.0,
        )
    solution = _solve(
        np.r_[np.zeros(n), 1.0],
        np.r_[integrality, 0],
        Bounds(np.r_[bounds.lb, 0.0], np.r_[bounds.ub, np.inf]),
        extended_rows,
        extended_low,
        extended_high,
    )
    return solution[:n], int(round(solution[n]))


def _append_absolute_difference_bound(
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    left: np.ndarray,
    right: np.ndarray,
    bound: int,
) -> None:
    for difference in (left - right, right - left):
        _append_constraint(rows, low, high, difference, -np.inf, bound)


def _solve_total_deviation(
    *,
    integrality: np.ndarray,
    bounds: Bounds,
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    values_and_targets: list[tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    n = len(integrality)
    n_slacks = len(values_and_targets)
    extended_rows = [np.r_[row, np.zeros(n_slacks)] for row in rows]
    extended_low = list(low)
    extended_high = list(high)
    for slack_index, (values, target) in enumerate(values_and_targets):
        for sign in (1.0, -1.0):
            row = np.r_[sign * values, np.zeros(n_slacks)]
            row[n + slack_index] = -1.0
            _append_constraint(
                extended_rows,
                extended_low,
                extended_high,
                row,
                -np.inf,
                sign * target,
            )
    objective = np.r_[np.zeros(n), np.ones(n_slacks)]
    solution = _solve(
        objective,
        np.r_[integrality, np.zeros(n_slacks)],
        Bounds(
            np.r_[bounds.lb, np.zeros(n_slacks)],
            np.r_[bounds.ub, np.full(n_slacks, np.inf)],
        ),
        extended_rows,
        extended_low,
        extended_high,
    )
    return solution[:n], int(round(objective @ solution))


def _append_total_deviation_bound(
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    values_and_targets: list[tuple[np.ndarray, int]],
    bound: int,
) -> tuple[list[np.ndarray], list[float], list[float], int]:
    n = len(values_and_targets[0][0])
    n_slacks = len(values_and_targets)
    extended_rows = [np.r_[row, np.zeros(n_slacks)] for row in rows]
    extended_low = list(low)
    extended_high = list(high)
    for slack_index, (values, target) in enumerate(values_and_targets):
        for sign in (1.0, -1.0):
            row = np.r_[sign * values, np.zeros(n_slacks)]
            row[n + slack_index] = -1.0
            _append_constraint(
                extended_rows,
                extended_low,
                extended_high,
                row,
                -np.inf,
                sign * target,
            )
    _append_constraint(
        extended_rows,
        extended_low,
        extended_high,
        np.r_[np.zeros(n), np.ones(n_slacks)],
        -np.inf,
        bound,
    )
    return extended_rows, extended_low, extended_high, n_slacks


def allocate_parent_groups(
    rows: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    optimize_record_support: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign complete parent groups to exact 80/10/10 row-count splits."""

    groups = _parent_groups(rows)
    if optimize_record_support:
        invalid_vote_counts = [
            row.get(VOTE_COUNT_FIELD)
            for row in rows
            if not str(row.get(VOTE_COUNT_FIELD) or "").isdigit()
            or int(row[VOTE_COUNT_FIELD]) < 1
        ]
        if invalid_vote_counts:
            raise ValueError(
                f"record-support allocation requires positive integer "
                f"{VOTE_COUNT_FIELD}; examples={invalid_vote_counts[:3]!r}"
            )
    n_rows = len(rows)
    target_valid = _rounded_fraction(n_rows, EVAL_FRACTION)
    target_test = _rounded_fraction(n_rows, EVAL_FRACTION)
    if target_valid + target_test >= n_rows:
        raise ValueError("evaluation targets leave no training rows")

    condition_totals = Counter(str(row["condition_group"]) for row in rows)
    condition_parent_totals = Counter(
        condition
        for group in groups
        for condition in group.condition_counts
    )
    for condition in condition_totals:
        if condition_parent_totals[condition] < 3:
            raise ValueError(
                f"condition {condition!r} has fewer than three parent groups"
            )
    n_groups = len(groups)
    n_variables = 2 * n_groups
    sizes = np.asarray([group.size for group in groups], dtype=float)
    positives = np.asarray([group.positive_count for group in groups], dtype=float)
    singletons = np.asarray([group.singleton_count for group in groups], dtype=float)
    integrality = np.ones(n_variables)
    bounds = Bounds(np.zeros(n_variables), np.ones(n_variables))
    constraints: list[np.ndarray] = []
    low: list[float] = []
    high: list[float] = []

    # Exact row counts for valid and test; train receives all unselected groups.
    for offset, target in ((0, target_valid), (n_groups, target_test)):
        row = np.zeros(n_variables)
        row[offset : offset + n_groups] = sizes
        _append_constraint(constraints, low, high, row, target, target)
    # A parent group can occur in at most one held-out split.
    for index in range(n_groups):
        row = np.zeros(n_variables)
        row[index] = 1.0
        row[n_groups + index] = 1.0
        _append_constraint(constraints, low, high, row, -np.inf, 1.0)
    # Every condition must retain at least one parent in train, valid, and test.
    ordered_conditions = sorted(condition_totals)
    for condition in ordered_conditions:
        present = np.asarray(
            [float(condition in group.condition_counts) for group in groups]
        )
        valid = np.r_[present, np.zeros(n_groups)]
        test = np.r_[np.zeros(n_groups), present]
        _append_constraint(constraints, low, high, valid, 1.0, np.inf)
        _append_constraint(constraints, low, high, test, 1.0, np.inf)
        _append_constraint(
            constraints,
            low,
            high,
            valid + test,
            -np.inf,
            condition_parent_totals[condition] - 1,
        )

    total_singletons = np.r_[singletons, singletons]
    valid_singletons = np.r_[singletons, np.zeros(n_groups)]
    test_singletons = np.r_[np.zeros(n_groups), singletons]
    minimum_singletons: int | None = None
    singleton_imbalance: int | None = None
    if optimize_record_support:
        # Stage 1: put as few one-vote labels as possible into valid+test.
        stage1 = _solve(
            total_singletons, integrality, bounds, constraints, low, high
        )
        minimum_singletons = int(round(total_singletons @ stage1))
        _append_constraint(
            constraints,
            low,
            high,
            total_singletons,
            minimum_singletons,
            minimum_singletons,
        )

        # Stage 2: distribute unavoidable singleton labels evenly across valid/test.
        _, singleton_imbalance = _solve_absolute_difference(
            integrality=integrality,
            bounds=bounds,
            rows=constraints,
            low=low,
            high=high,
            left=valid_singletons,
            right=test_singletons,
        )
        _append_absolute_difference_bound(
            constraints,
            low,
            high,
            valid_singletons,
            test_singletons,
            singleton_imbalance,
        )

    # Stage 3: balance binary labels only after record-support quality is fixed.
    positive_target = _rounded_fraction(
        int(sum(int(row["Y"]) for row in rows)), EVAL_FRACTION
    )
    label_values = [
        (np.r_[positives, np.zeros(n_groups)], positive_target),
        (np.r_[np.zeros(n_groups), positives], positive_target),
    ]
    _, label_deviation = _solve_total_deviation(
        integrality=integrality,
        bounds=bounds,
        rows=constraints,
        low=low,
        high=high,
        values_and_targets=label_values,
    )
    label_rows, label_low, label_high, label_slacks = _append_total_deviation_bound(
        constraints, low, high, label_values, label_deviation
    )

    # Stage 4: balance condition row counts, then use the seed only for stable ties.
    condition_values: list[tuple[np.ndarray, int]] = []
    for condition in ordered_conditions:
        counts = np.asarray(
            [float(group.condition_counts.get(condition, 0)) for group in groups]
        )
        target = _rounded_fraction(condition_totals[condition], EVAL_FRACTION)
        condition_values.extend(
            [
                (np.r_[counts, np.zeros(n_groups)], target),
                (np.r_[np.zeros(n_groups), counts], target),
            ]
        )
    condition_values_with_label_slacks = [
        (np.r_[values, np.zeros(label_slacks)], target)
        for values, target in condition_values
    ]
    extended_integrality = np.r_[integrality, np.zeros(label_slacks)]
    extended_bounds = Bounds(
        np.r_[bounds.lb, np.zeros(label_slacks)],
        np.r_[bounds.ub, np.full(label_slacks, np.inf)],
    )
    _, condition_deviation = _solve_total_deviation(
        integrality=extended_integrality,
        bounds=extended_bounds,
        rows=label_rows,
        low=label_low,
        high=label_high,
        values_and_targets=condition_values_with_label_slacks,
    )
    final_rows, final_low, final_high, condition_slacks = _append_total_deviation_bound(
        label_rows,
        label_low,
        label_high,
        condition_values_with_label_slacks,
        condition_deviation,
    )
    rank_order = sorted(
        range(n_groups), key=lambda index: _stable_key(seed, groups[index].parent)
    )
    rank_by_index = np.empty(n_groups, dtype=float)
    for rank, index in enumerate(rank_order, start=1):
        rank_by_index[index] = rank
    stable_objective = np.r_[
        rank_by_index,
        rank_by_index[::-1],
        np.zeros(label_slacks + condition_slacks),
    ]
    final_integrality = np.r_[
        integrality, np.zeros(label_slacks + condition_slacks)
    ]
    final_bounds = Bounds(
        np.r_[bounds.lb, np.zeros(label_slacks + condition_slacks)],
        np.r_[bounds.ub, np.full(label_slacks + condition_slacks, np.inf)],
    )
    selected = _solve(
        stable_objective,
        final_integrality,
        final_bounds,
        final_rows,
        final_low,
        final_high,
    )[:n_variables]

    assignment = {group.parent: "train" for group in groups}
    for index, group in enumerate(groups):
        if selected[index] > 0.5:
            assignment[group.parent] = "valid"
        elif selected[n_groups + index] > 0.5:
            assignment[group.parent] = "test"

    selected_valid_singletons = (
        int(round(valid_singletons @ selected)) if optimize_record_support else None
    )
    selected_test_singletons = (
        int(round(test_singletons @ selected)) if optimize_record_support else None
    )

    return assignment, {
        "method": "deterministic_lexicographic_parent_group_milp",
        "seed": seed,
        "target_fraction": {
            "train": 1.0 - 2 * EVAL_FRACTION,
            "valid": EVAL_FRACTION,
            "test": EVAL_FRACTION,
        },
        "target_rows": {
            "train": n_rows - target_valid - target_test,
            "valid": target_valid,
            "test": target_test,
        },
        "n_parent_groups": len(groups),
        "n_conditions": len(condition_totals),
        "record_support_objective_enabled": optimize_record_support,
        "record_support_field": VOTE_COUNT_FIELD if optimize_record_support else None,
        "multi_vote_definition": (
            f"{VOTE_COUNT_FIELD} >= 2" if optimize_record_support else None
        ),
        "objective_order": (
            [
                "minimum heldout singleton-vote rows",
                "minimum valid/test singleton imbalance",
            ]
            if optimize_record_support
            else []
        )
        + [
            "minimum valid/test positive-label deviation",
            "minimum valid/test condition-row deviation",
            "seeded stable tie-break",
        ],
        "minimum_singletons_in_heldout": minimum_singletons,
        "minimum_singleton_imbalance": singleton_imbalance,
        "selected_valid_singletons": selected_valid_singletons,
        "selected_test_singletons": selected_test_singletons,
        "minimum_positive_label_l1_deviation": label_deviation,
        "minimum_condition_row_l1_deviation": condition_deviation,
        "desired_positive_count_per_eval_split": positive_target,
        "condition_coverage_constraint": "every condition has at least one parent in every split",
    }


def preserve_parent_assignment(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Reuse an established parent split after a deletion-only source repair.

    This deliberately fails when the rebuilt cohort introduces a new parent or
    when the surviving assignment no longer has the required exact row counts.
    Those cases require a fresh optimized random split instead of silently
    extending an old assignment.
    """

    reference_by_parent: dict[str, str] = {}
    reference_by_id: dict[str, dict[str, Any]] = {}
    reference_ids: set[str] = set()
    for row in reference_rows:
        row_id = str(row["benchmark_row_id"])
        if row_id in reference_ids:
            raise ValueError(f"duplicate reference benchmark_row_id: {row_id}")
        reference_ids.add(row_id)
        reference_by_id[row_id] = row
        parent = str(row["molecule_identity_key"])
        split = str(row["split"])
        if split not in SPLITS:
            raise ValueError(f"invalid reference split {split!r} for {row_id}")
        previous = reference_by_parent.setdefault(parent, split)
        if previous != split:
            raise ValueError(f"reference parent {parent!r} spans multiple splits")

    assignment: dict[str, str] = {}
    current_ids: set[str] = set()
    for row in rows:
        row_id = str(row["benchmark_row_id"])
        if row_id in current_ids:
            raise ValueError(f"duplicate current benchmark_row_id: {row_id}")
        current_ids.add(row_id)
        if row_id not in reference_by_id:
            raise ValueError(
                "cannot preserve split after a non-deletion repair; new "
                f"benchmark_row_id {row_id!r} was introduced"
            )
        if _content_row(row) != _content_row(reference_by_id[row_id]):
            raise ValueError(
                "cannot preserve split because surviving benchmark row "
                f"{row_id!r} changed content"
            )
        parent = str(row["molecule_identity_key"])
        if parent not in reference_by_parent:
            raise ValueError(
                f"cannot preserve split because parent {parent!r} is new"
            )
        assignment[parent] = reference_by_parent[parent]

    counts = Counter(
        assignment[str(row["molecule_identity_key"])] for row in rows
    )
    target_eval = _rounded_fraction(len(rows), EVAL_FRACTION)
    expected = {
        "train": len(rows) - 2 * target_eval,
        "valid": target_eval,
        "test": target_eval,
    }
    if dict(counts) != expected:
        raise ValueError(
            "surviving assignment no longer satisfies exact 80/10/10 row "
            f"counts: observed={dict(counts)}, expected={expected}"
        )

    conditions_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        split = assignment[str(row["molecule_identity_key"])]
        conditions_by_split[split].add(str(row["condition_group"]))
    all_conditions = set.union(*conditions_by_split.values())
    missing = {
        split: sorted(all_conditions - conditions)
        for split, conditions in conditions_by_split.items()
        if conditions != all_conditions
    }
    if missing:
        raise ValueError(
            f"surviving assignment no longer covers every condition: {missing}"
        )

    singleton_counts = Counter(
        assignment[str(row["molecule_identity_key"])]
        for row in rows
        if int(row[VOTE_COUNT_FIELD]) == 1
    )
    return assignment, {
        "method": "preserved_existing_parent_assignment_after_deletion_only_repair",
        "target_fraction": {
            "train": 1.0 - 2 * EVAL_FRACTION,
            "valid": EVAL_FRACTION,
            "test": EVAL_FRACTION,
        },
        "target_rows": expected,
        "n_parent_groups": len(_parent_groups(rows)),
        "n_conditions": len(all_conditions),
        "n_reference_rows": len(reference_rows),
        "n_current_rows": len(rows),
        "n_removed_rows": len(reference_ids - current_ids),
        "n_added_rows": len(current_ids - reference_ids),
        "selected_valid_singletons": singleton_counts["valid"],
        "selected_test_singletons": singleton_counts["test"],
        "condition_coverage_constraint": "every condition has at least one parent in every split",
    }


def _content_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"split", "split_policy", "split_assignments"}
    }


def _union_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = sorted(
        (
            json.dumps(_content_row(row), sort_keys=True, separators=(",", ":"))
            for row in rows
        )
    )
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _minimal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in MINIMAL_FIELDS}


def _counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _pairwise_overlap(values: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        f"{left}__{right}": len(values[left] & values[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }


def _write_group_distribution(root: Path, rows_by_split: Mapping[str, list[dict[str, Any]]]) -> None:
    lines = ["condition_group,split,n_rows"]
    for split in SPLITS:
        for condition, count in _counts(rows_by_split[split], "condition_group").items():
            lines.append(f"{condition},{split},{count}")
    with atomic_output_path(root / "group_distribution.csv") as temporary:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_task(
    task: str,
    *,
    seed: int = DEFAULT_SEED,
    preserve_existing_split: bool = False,
    reference_root: Path = BENCHMARK_ROOT,
) -> dict[str, Any]:
    source_root = task_root(task, "scaffold")
    destination = task_root(task, "random")
    source_rows = [
        row
        for split in SPLITS
        for row in read_jsonl(source_root / f"{split}_molecule_condition_labels.jsonl")
    ]
    row_ids = [str(row["benchmark_row_id"]) for row in source_rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{task} benchmark_row_id values are not unique")
    if preserve_existing_split:
        reference_task_root = reference_root / TASK_DIRECTORIES[task] / "random"
        reference_rows = [
            row
            for split in SPLITS
            for row in read_jsonl(
                reference_task_root / f"{split}_molecule_condition_labels.jsonl"
            )
        ]
        assignment, optimizer = preserve_parent_assignment(
            source_rows, reference_rows
        )
    else:
        assignment, optimizer = allocate_parent_groups(
            source_rows,
            seed=seed,
            # ClinTox labels come from source-role construction, not assay-vote
            # aggregation; its source_record_count is therefore not a comparable
            # held-out label-quality measure.
            optimize_record_support=task != "clintox",
        )

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in source_rows:
        split = assignment[str(row["molecule_identity_key"])]
        output = dict(row)
        split_assignments = dict(output.get("split_assignments") or {})
        split_assignments["random"] = split
        output["split"] = split
        output["split_policy"] = RANDOM_SPLIT_CONTRACT
        output["split_assignments"] = split_assignments
        rows_by_split[split].append(output)

    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl_atomic(
            destination / f"{split}_molecule_condition_labels.jsonl",
            rows_by_split[split],
        )
        write_jsonl_atomic(
            destination / f"{split}.jsonl",
            (_minimal_row(row) for row in rows_by_split[split]),
        )
    write_jsonl_atomic(
        destination / "heldout_molecule_condition_labels.jsonl",
        rows_by_split["valid"] + rows_by_split["test"],
    )
    _write_group_distribution(destination, rows_by_split)

    identities = {
        split: {str(row["molecule_identity_key"]) for row in split_rows}
        for split, split_rows in rows_by_split.items()
    }
    scaffolds = {
        split: {
            str(row.get("bemis_murcko_scaffold") or "")
            for row in split_rows
            if str(row.get("bemis_murcko_scaffold") or "")
        }
        for split, split_rows in rows_by_split.items()
    }
    conditions = {
        split: {str(row["condition_group"]) for row in split_rows}
        for split, split_rows in rows_by_split.items()
    }
    identity_overlap = _pairwise_overlap(identities)
    if any(identity_overlap.values()):
        raise RuntimeError(f"{task} random split has parent identity leakage")
    all_conditions = set.union(*conditions.values())
    missing_conditions = {
        split: sorted(all_conditions - split_conditions)
        for split, split_conditions in conditions.items()
    }
    if any(missing_conditions.values()):
        raise RuntimeError(f"{task} condition coverage failed: {missing_conditions}")

    random_rows = [row for split in SPLITS for row in rows_by_split[split]]
    source_hash = _union_hash(source_rows)
    random_hash = _union_hash(random_rows)
    if source_hash != random_hash:
        raise RuntimeError(f"{task} row content changed during random splitting")
    summary = {
        "task": TASK_DIRECTORIES[task],
        "benchmark": "conditioned_benchmark",
        "contract": CONTRACT,
        "split_contract": RANDOM_SPLIT_CONTRACT,
        "source_scaffold_root": str(source_root),
        "root": str(destination),
        "source_union_sha256": source_hash,
        "random_union_sha256": random_hash,
        "same_molecule_condition_rows_and_labels": True,
        "optimizer": optimizer,
        "splits": {
            split: {
                "n": len(split_rows),
                "sha256": sha256_file(destination / f"{split}.jsonl"),
                "n_parents": len(identities[split]),
                "label_counts": _counts(split_rows, "Y"),
                "condition_counts": _counts(split_rows, "condition_group"),
            }
            for split, split_rows in rows_by_split.items()
        },
        "pairwise_identity_overlap": identity_overlap,
        "pairwise_scaffold_overlap": _pairwise_overlap(scaffolds),
        "conditions_present_in_all_splits": sorted(all_conditions),
        "missing_conditions_by_split": missing_conditions,
    }
    write_json_atomic(destination / "summary.json", summary)
    return summary


def _update_manifest(
    summaries: Mapping[str, Mapping[str, Any]], *, seed: int
) -> None:
    manifest_path = BENCHMARK_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_schemes"] = {
        "scaffold": {
            "default": True,
            "contract": CONTRACT,
            "parent_disjoint": True,
            "scaffold_disjoint": True,
        },
        "random": {
            "default": False,
            "contract": RANDOM_SPLIT_CONTRACT,
            "seed": seed,
            "parent_disjoint": True,
            "scaffold_disjoint": False,
            "condition_coverage": "all conditions in train, valid, and test",
            "record_support_priority": (
                "lexicographically minimize singleton-vote rows in valid+test "
                "for BBB, Bioavailability, and Skin; not applicable to ClinTox"
            ),
        },
    }
    for task, summary in summaries.items():
        task_manifest = manifest["tasks"][task]
        task_manifest["roots"] = {
            "scaffold": str(task_root(task, "scaffold")),
            "random": str(task_root(task, "random")),
        }
        task_manifest["split_counts_by_scheme"] = {
            "scaffold": dict(task_manifest["split_counts"]),
            "random": {
                split: int(summary["splits"][split]["n"])
                for split in SPLITS
            },
        }
    write_json_atomic(manifest_path, manifest)


def build_all(
    *,
    seed: int = DEFAULT_SEED,
    tasks: tuple[str, ...] | None = None,
    preserve_existing_split: bool = False,
    reference_root: Path = BENCHMARK_ROOT,
) -> dict[str, Any]:
    requested = tuple(TASK_DIRECTORIES) if tasks is None else tasks
    summaries: dict[str, dict[str, Any]] = {}
    receipt_path = BENCHMARK_ROOT / "random_split_receipt.json"
    if tasks is not None and receipt_path.exists():
        summaries.update(
            json.loads(receipt_path.read_text(encoding="utf-8")).get("tasks", {})
        )
    rebuilt: dict[str, dict[str, Any]] = {}
    for task in requested:
        print(f"[conditioned-random] solving {task}", flush=True)
        rebuilt[task] = build_task(
            task,
            seed=seed,
            preserve_existing_split=preserve_existing_split,
            reference_root=reference_root,
        )
        summaries[task] = rebuilt[task]
        print(f"[conditioned-random] completed {task}", flush=True)
    missing = sorted(set(TASK_DIRECTORIES) - set(summaries))
    if missing:
        raise FileNotFoundError(
            f"Partial random build lacks existing summaries for {missing}"
        )
    receipt = {
        "benchmark": "conditioned_benchmark",
        "split_contract": RANDOM_SPLIT_CONTRACT,
        "seed": seed,
        "source_scheme": "scaffold union",
        "policy": (
            "same molecule-condition rows and labels; parent-grouped 80/10/10 "
            "quality-stratified random assignment with three-way condition "
            "coverage; voter-based tasks minimize held-out singleton labels"
        ),
        "tasks": summaries,
    }
    write_json_atomic(receipt_path, receipt)
    # A partial rebuild still owns the suite-level manifest.  Refresh it from
    # the complete receipt (rebuilt tasks plus preserved task summaries), not
    # only from this invocation's subset, so every task retains both roots and
    # both split-count tables.
    _update_manifest(summaries, seed=seed)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_DIRECTORIES),
        help="Rebuild only selected tasks and preserve current summaries for others.",
    )
    parser.add_argument(
        "--preserve-existing-split",
        action="store_true",
        help=(
            "Reuse an existing parent assignment for a deletion-only repair; "
            "fail if exact counts or condition coverage no longer hold."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=BENCHMARK_ROOT,
        help="Benchmark root containing the reference <Task>/random split.",
    )
    args = parser.parse_args(argv)
    result = build_all(
        seed=args.seed,
        tasks=tuple(args.tasks) if args.tasks else None,
        preserve_existing_split=args.preserve_existing_split,
        reference_root=args.reference_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
