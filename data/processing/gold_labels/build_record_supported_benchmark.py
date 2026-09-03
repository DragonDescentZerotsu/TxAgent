"""Build the record-supported, scaffold-disjoint Starling benchmark lineage.

The held-out allocation is solved at Bemis--Murcko scaffold-group grain.  It
first minimizes singleton parents in valid+test, then balances record quality
and labels across the two splits, and finally prefers parents already present
in the frozen v1 valid split so compatible inference can be reused safely.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from tools.chembl_tool.common.json_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)


LINEAGE = "record_supported_v2"
PROTOCOL_VERSION = "starling_record_supported_benchmark.v2"
DEFAULT_SOURCE_ROOT = Path("data/gold_labels/legacy/processed_starling")
DEFAULT_OUTPUT_ROOT = Path("data/gold_labels/legacy/processed_starling_record_supported_v2")
TASKS = (
    "BBB_Martins",
    "Bioavailability_Ma",
    "Skin_Reaction",
)
DEFAULT_TASKS = ("BBB_Martins", "Bioavailability_Ma", "Skin_Reaction")
SEED = 20260807


@dataclass(frozen=True)
class ScaffoldGroup:
    scaffold: str
    rows: tuple[dict[str, Any], ...]
    stable_rank: int

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def singleton_count(self) -> int:
        return sum(int(row["source_record_count"]) == 1 for row in self.rows)

    @property
    def positive_count(self) -> int:
        return sum(int(row["Y"]) for row in self.rows)

    @property
    def prior_valid_count(self) -> int:
        return sum(bool(row.get("_prior_valid")) for row in self.rows)


def target_eval_size(task: str, n_parents: int) -> int:
    target = int(0.1 * n_parents)
    return min(500, target) if task == "BBB_Martins" else target


def allocate_scaffold_groups(
    rows: list[dict[str, Any]],
    *,
    target_size: int,
    seed: int = SEED,
    optimize_record_support: bool = True,
    exclude_empty_scaffold_from_heldout: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups = _scaffold_groups(rows, seed=seed)
    candidates = [
        group
        for group in groups
        if group.size <= target_size
        and not (exclude_empty_scaffold_from_heldout and not group.scaffold)
    ]
    if not candidates:
        raise ValueError("No scaffold group can enter the requested held-out split")

    size = np.asarray([group.size for group in candidates], dtype=float)
    singletons = np.asarray(
        [group.singleton_count for group in candidates], dtype=float
    )
    positives = np.asarray(
        [group.positive_count for group in candidates], dtype=float
    )
    prior_valid = np.asarray(
        [group.prior_valid_count for group in candidates], dtype=float
    )
    n_groups = len(candidates)
    binary_bounds = Bounds(np.zeros(2 * n_groups), np.ones(2 * n_groups))
    integrality = np.ones(2 * n_groups)
    base_rows, base_low, base_high = _base_constraints(size, target_size)

    constraints = list(base_rows)
    low = list(base_low)
    high = list(base_high)
    minimum_singletons: int | None = None
    singleton_imbalance: int | None = None
    if optimize_record_support:
        total_singletons = np.r_[singletons, singletons]
        stage1 = _solve(
            total_singletons,
            integrality,
            binary_bounds,
            base_rows,
            base_low,
            base_high,
        )
        minimum_singletons = int(round(total_singletons @ stage1))
        _append_equality(
            constraints, low, high, total_singletons, minimum_singletons
        )

        # With the minimum total frozen, minimize valid/test singleton imbalance.
        stage2 = _solve_with_absolute_difference(
            integrality,
            binary_bounds,
            constraints,
            low,
            high,
            np.r_[singletons, np.zeros(n_groups)],
            np.r_[np.zeros(n_groups), singletons],
        )
        valid_singletons = int(round(singletons @ stage2[:n_groups]))
        test_singletons = int(round(singletons @ stage2[n_groups:]))
        singleton_imbalance = abs(valid_singletons - test_singletons)
        _append_absolute_difference_bound(
            constraints,
            low,
            high,
            np.r_[singletons, np.zeros(n_groups)],
            np.r_[np.zeros(n_groups), singletons],
            singleton_imbalance,
        )

    desired_positives = round(
        target_size * sum(int(row["Y"]) for row in rows) / len(rows)
    )
    stage3, label_deviation = _solve_label_deviation(
        integrality,
        binary_bounds,
        constraints,
        low,
        high,
        positives,
        desired_positives,
    )
    _append_label_deviation_bound(
        constraints,
        low,
        high,
        positives,
        desired_positives,
        label_deviation,
    )

    # Prefer frozen-v1 valid parents only after data quality and label balance.
    reuse_objective = np.r_[-prior_valid, np.zeros(n_groups)]
    stage4 = _solve(
        reuse_objective,
        integrality,
        binary_bounds,
        constraints,
        low,
        high,
    )
    maximum_prior_valid = int(round(prior_valid @ stage4[:n_groups]))
    _append_equality(
        constraints,
        low,
        high,
        np.r_[prior_valid, np.zeros(n_groups)],
        maximum_prior_valid,
    )

    # Stable hash ranks remove solver-dependent ties and make the build replayable.
    ranks = np.asarray([group.stable_rank + 1 for group in candidates], dtype=float)
    tie_objective = np.r_[ranks, ranks[::-1]]
    selected = _solve(
        tie_objective,
        integrality,
        binary_bounds,
        constraints,
        low,
        high,
    )

    assignment = {group.scaffold: "train" for group in groups}
    for index, group in enumerate(candidates):
        if selected[index] > 0.5:
            assignment[group.scaffold] = "valid"
        elif selected[n_groups + index] > 0.5:
            assignment[group.scaffold] = "test"

    audit = {
        "record_support_objective_enabled": optimize_record_support,
        "empty_scaffold_heldout_eligible": not exclude_empty_scaffold_from_heldout,
        "minimum_singletons_in_heldout": minimum_singletons,
        "minimum_singleton_imbalance": singleton_imbalance,
        "minimum_label_deviation": label_deviation,
        "desired_positive_count_per_eval_split": desired_positives,
        "maximum_prior_valid_reuse": maximum_prior_valid,
        "n_scaffold_groups": len(groups),
        "n_candidate_scaffold_groups": len(candidates),
        "n_empty_scaffold_parents": sum(
            group.size for group in groups if not group.scaffold
        ),
    }
    return assignment, audit


def _scaffold_groups(
    rows: list[dict[str, Any]],
    *,
    seed: int = SEED,
) -> list[ScaffoldGroup]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("bemis_murcko_scaffold") or "")].append(row)
    ordered = sorted(
        grouped.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item[0]}".encode("utf-8")
        ).hexdigest(),
    )
    return [
        ScaffoldGroup(scaffold, tuple(group_rows), rank)
        for rank, (scaffold, group_rows) in enumerate(ordered)
    ]


def _base_constraints(
    sizes: np.ndarray,
    target_size: int,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    n_groups = len(sizes)
    rows: list[np.ndarray] = []
    low: list[float] = []
    high: list[float] = []
    for offset in (0, n_groups):
        row = np.zeros(2 * n_groups)
        row[offset : offset + n_groups] = sizes
        _append_equality(rows, low, high, row, target_size)
    for index in range(n_groups):
        row = np.zeros(2 * n_groups)
        row[index] = 1
        row[n_groups + index] = 1
        rows.append(row)
        low.append(-np.inf)
        high.append(1)
    return rows, low, high


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
        options={"time_limit": 300},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Scaffold allocation failed: {result.message}")
    return result.x


def _solve_with_absolute_difference(
    integrality: np.ndarray,
    bounds: Bounds,
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    n = len(integrality)
    objective = np.r_[np.zeros(n), 1.0]
    extended_rows = [np.r_[row, 0.0] for row in rows]
    extended_low = list(low)
    extended_high = list(high)
    for difference in (left - right, right - left):
        extended_rows.append(np.r_[difference, -1.0])
        extended_low.append(-np.inf)
        extended_high.append(0.0)
    solution = _solve(
        objective,
        np.r_[integrality, 0],
        Bounds(np.r_[bounds.lb, 0.0], np.r_[bounds.ub, np.inf]),
        extended_rows,
        extended_low,
        extended_high,
    )
    return solution[:n]


def _solve_label_deviation(
    integrality: np.ndarray,
    bounds: Bounds,
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    positives: np.ndarray,
    desired: int,
) -> tuple[np.ndarray, int]:
    n = len(integrality)
    n_groups = len(positives)
    objective = np.r_[np.zeros(n), 1.0, 1.0]
    extended_rows = [np.r_[row, 0.0, 0.0] for row in rows]
    extended_low = list(low)
    extended_high = list(high)
    for split_index, offset in enumerate((0, n_groups)):
        value = np.zeros(n)
        value[offset : offset + n_groups] = positives
        for sign in (1.0, -1.0):
            row = np.r_[sign * value, 0.0, 0.0]
            row[n + split_index] = -1.0
            extended_rows.append(row)
            extended_low.append(-np.inf)
            extended_high.append(sign * desired)
    solution = _solve(
        objective,
        np.r_[integrality, 0, 0],
        Bounds(np.r_[bounds.lb, 0.0, 0.0], np.r_[bounds.ub, np.inf, np.inf]),
        extended_rows,
        extended_low,
        extended_high,
    )
    return solution[:n], int(round(solution[n] + solution[n + 1]))


def _append_equality(
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    row: np.ndarray,
    value: float,
) -> None:
    rows.append(row)
    low.append(value)
    high.append(value)


def _append_absolute_difference_bound(
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    left: np.ndarray,
    right: np.ndarray,
    bound: int,
) -> None:
    for difference in (left - right, right - left):
        rows.append(difference)
        low.append(-np.inf)
        high.append(bound)


def _append_label_deviation_bound(
    rows: list[np.ndarray],
    low: list[float],
    high: list[float],
    positives: np.ndarray,
    desired: int,
    total_bound: int,
) -> None:
    # |v-desired| + |t-desired| <= total_bound is represented by its four
    # linear sign combinations.
    n_groups = len(positives)
    valid = np.r_[positives, np.zeros(n_groups)]
    test = np.r_[np.zeros(n_groups), positives]
    for valid_sign in (-1.0, 1.0):
        for test_sign in (-1.0, 1.0):
            rows.append(valid_sign * valid + test_sign * test)
            low.append(-np.inf)
            high.append(
                total_bound + (valid_sign + test_sign) * desired
            )


def build_task(
    task: str,
    *,
    source_root: Path,
    output_root: Path,
    lineage: str = LINEAGE,
    protocol_version: str = PROTOCOL_VERSION,
    seed: int = SEED,
) -> dict[str, Any]:
    source_task = source_root / task
    source_labels = _read_jsonl(source_task / "molecule_labels.jsonl")
    prior_valid_keys = {
        row["molecule_identity_key"]
        for row in _read_jsonl(source_task / "scaffold/valid_molecule_labels.jsonl")
    }
    rows = [
        {**row, "_prior_valid": row["molecule_identity_key"] in prior_valid_keys}
        for row in source_labels
    ]
    target = target_eval_size(task, len(rows))
    assignment, optimization = allocate_scaffold_groups(
        rows,
        target_size=target,
        seed=seed,
    )
    task_root = output_root / task
    split_root = task_root / "scaffold"

    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "valid", "test")}
    for row in rows:
        split = assignment[str(row.get("bemis_murcko_scaffold") or "")]
        clean = {key: value for key, value in row.items() if key != "_prior_valid"}
        clean.update(
            {
                "split": split,
                "split_policy": lineage,
            }
        )
        split_rows[split].append(clean)
    for values in split_rows.values():
        values.sort(key=lambda row: str(row["molecule_identity_key"]))

    for split, values in split_rows.items():
        write_jsonl_atomic(
            split_root / f"{split}.jsonl",
            [{"drug": row["drug"], "Y": int(row["Y"])} for row in values],
        )
        write_jsonl_atomic(split_root / f"{split}_molecule_labels.jsonl", values)
    heldout = split_rows["valid"] + split_rows["test"]
    write_jsonl_atomic(split_root / "heldout_molecule_labels.jsonl", heldout)

    source_summary = json.loads((source_task / "summary.json").read_text(encoding="utf-8"))
    split_summary = _split_summary(
        split_rows,
        target,
        optimization,
        split_root,
        lineage=lineage,
    )
    task_summary = {
        **source_summary,
        "protocol_version": protocol_version,
        "seed": seed,
        "record_support_policy": {
            "record_tier": "multi_record iff source_record_count >= 2",
            "heldout_objective": "maximize multi-record parents at scaffold-group grain",
            "bbb_eval_cap": 500,
            "bioavailability_and_skin_ratio": "approximately 8:1:1",
            "prior_valid_reuse_tiebreak": True,
        },
        "split_size_policy": {
            "formula": "BBB min(500, floor(0.1*n)); Bioavailability/Skin floor(0.1*n)",
            "target_valid_size": target,
            "target_test_size": target,
        },
        "splits": {"scaffold": split_summary},
        "paths": {
            **source_summary.get("paths", {}),
            "record_supported_scaffold_root": str(split_root),
        },
    }
    write_json_atomic(split_root / "summary.json", split_summary)
    write_json_atomic(task_root / "summary.json", task_summary)
    return task_summary


def build_task_preserving_split(
    task: str,
    *,
    source_root: Path,
    output_root: Path,
    reference_split_root: Path,
    lineage: str,
    protocol_version: str,
    seed: int = SEED,
    allow_new_parents: bool = False,
    new_parent_default_split: str = "train",
) -> dict[str, Any]:
    """Rebuild labels while preserving every surviving parent split assignment.

    When explicitly enabled, a new parent inherits the reference assignment of
    its scaffold. A genuinely new scaffold goes to train by default, so no
    existing held-out cohort or scaffold boundary is perturbed.
    """

    if new_parent_default_split not in {"train", "valid", "test"}:
        raise ValueError("new_parent_default_split must name a benchmark split")

    source_task = source_root / task
    source_labels = _read_jsonl(source_task / "molecule_labels.jsonl")
    reference_assignment: dict[str, str] = {}
    reference_scaffold_assignment: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for row in _read_jsonl(reference_split_root / f"{split}_molecule_labels.jsonl"):
            reference_assignment[str(row["molecule_identity_key"])] = split
            scaffold = str(row.get("bemis_murcko_scaffold") or "")
            previous = reference_scaffold_assignment.setdefault(scaffold, split)
            if previous != split:
                raise RuntimeError(
                    f"reference split violates scaffold disjointness for {scaffold!r}"
                )
    new_keys = {str(row["molecule_identity_key"]) for row in source_labels}
    added = sorted(new_keys - set(reference_assignment))
    if added and not allow_new_parents:
        raise ValueError(
            "split preservation requires no new parent identities; "
            f"found {len(added)}"
        )

    split_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "valid", "test")
    }
    scaffold_assignment: dict[str, str] = {}
    added_assignment_counts: Counter[str] = Counter()
    for row in source_labels:
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        identity_key = str(row["molecule_identity_key"])
        if identity_key in reference_assignment:
            split = reference_assignment[identity_key]
        else:
            split = reference_scaffold_assignment.get(
                scaffold, new_parent_default_split
            )
            added_assignment_counts[split] += 1
        prior = scaffold_assignment.setdefault(scaffold, split)
        if prior != split:
            raise RuntimeError(
                f"reference split violates scaffold disjointness for {scaffold!r}"
            )
        split_assignments = dict(row.get("split_assignments") or {})
        split_assignments["scaffold"] = split
        split_rows[split].append(
            {
                **row,
                "split": split,
                "split_policy": lineage,
                "split_assignments": split_assignments,
            }
        )
    for values in split_rows.values():
        values.sort(key=lambda row: str(row["molecule_identity_key"]))

    task_root = output_root / task
    split_root = task_root / "scaffold"
    for split, values in split_rows.items():
        write_jsonl_atomic(
            split_root / f"{split}.jsonl",
            [{"drug": row["drug"], "Y": int(row["Y"])} for row in values],
        )
        write_jsonl_atomic(split_root / f"{split}_molecule_labels.jsonl", values)
    write_jsonl_atomic(
        split_root / "heldout_molecule_labels.jsonl",
        split_rows["valid"] + split_rows["test"],
    )

    target = len(split_rows["valid"])
    reference_keys = set(reference_assignment)
    optimization = {
        "split_assignment_policy": (
            "preserve_reference_and_inherit_scaffold_for_new_parents"
            if added
            else "preserve_reference_for_surviving_parents"
        ),
        "reference_split_root": str(reference_split_root),
        "n_reference_parents": len(reference_keys),
        "n_surviving_reference_parents": len(new_keys & reference_keys),
        "n_output_parents": len(new_keys),
        "n_removed_parents": len(reference_keys - new_keys),
        "n_added_parents": len(added),
        "added_parent_assignment_counts": dict(sorted(added_assignment_counts.items())),
        "new_parent_default_split": new_parent_default_split,
    }
    split_summary = _split_summary(
        split_rows,
        target,
        optimization,
        split_root,
        lineage=lineage,
    )
    split_summary["method"] = (
        "preserved_scaffold_assignment_with_new_parent_inheritance_after_gold_revote"
        if added
        else "preserved_scaffold_assignment_after_gold_revote"
    )

    source_summary = json.loads((source_task / "summary.json").read_text())
    task_summary = {
        **source_summary,
        "protocol_version": protocol_version,
        "seed": seed,
        "record_support_policy": {
            "record_tier": "multi_record iff source_record_count >= 2",
            "heldout_objective": "preserve reference split for every surviving parent",
            "reference_split_root": str(reference_split_root),
        },
        "split_size_policy": {
            "formula": "preserve reference valid/test cohorts",
            "target_valid_size": len(split_rows["valid"]),
            "target_test_size": len(split_rows["test"]),
        },
        "splits": {"scaffold": split_summary},
        "paths": {
            **source_summary.get("paths", {}),
            "record_supported_scaffold_root": str(split_root),
        },
    }
    write_json_atomic(split_root / "summary.json", split_summary)
    write_json_atomic(task_root / "summary.json", task_summary)
    return task_summary


def _split_summary(
    split_rows: dict[str, list[dict[str, Any]]],
    target: int,
    optimization: dict[str, Any],
    split_root: Path,
    *,
    lineage: str = LINEAGE,
) -> dict[str, Any]:
    scaffolds = {
        split: {str(row.get("bemis_murcko_scaffold") or "") for row in rows}
        for split, rows in split_rows.items()
    }
    identities = {
        split: {str(row["molecule_identity_key"]) for row in rows}
        for split, rows in split_rows.items()
    }
    summary: dict[str, Any] = {
        "method": "quality_lexicographic_milp_scaffold_groups",
        "lineage": lineage,
        "target_valid_size": target,
        "target_test_size": target,
        "actual_valid_size": len(split_rows["valid"]),
        "actual_test_size": len(split_rows["test"]),
        "valid_size_shortfall": target - len(split_rows["valid"]),
        "test_size_shortfall": target - len(split_rows["test"]),
        "optimization": optimization,
    }
    for split, rows in split_rows.items():
        summary[f"n_{split}"] = len(rows)
        summary[f"{split}_label_counts"] = _counts(rows, lambda row: int(row["Y"]))
        summary[f"{split}_record_tier_counts"] = _counts(
            rows,
            lambda row: "singleton" if int(row["source_record_count"]) == 1 else "multi_record",
        )
        summary[f"n_{split}_scaffolds"] = len(scaffolds[split])
    summary["pairwise_scaffold_overlap"] = {
        "train_valid": len(scaffolds["train"] & scaffolds["valid"]),
        "train_test": len(scaffolds["train"] & scaffolds["test"]),
        "valid_test": len(scaffolds["valid"] & scaffolds["test"]),
    }
    summary["pairwise_identity_overlap"] = {
        "train_valid": len(identities["train"] & identities["valid"]),
        "train_test": len(identities["train"] & identities["test"]),
        "valid_test": len(identities["valid"] & identities["test"]),
    }
    for kind, values in summary["pairwise_scaffold_overlap"].items():
        summary[f"{kind}_scaffold_overlap"] = values
    for kind, values in summary["pairwise_identity_overlap"].items():
        summary[f"{kind}_identity_overlap"] = values
    summary["paths"] = {
        split: str(split_root / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    summary["paths"].update(
        {
            f"{split}_molecule_labels": str(split_root / f"{split}_molecule_labels.jsonl")
            for split in ("train", "valid", "test")
        }
    )
    summary["paths"]["heldout_molecule_labels"] = str(
        split_root / "heldout_molecule_labels.jsonl"
    )
    if any(summary["pairwise_scaffold_overlap"].values()) or any(
        summary["pairwise_identity_overlap"].values()
    ):
        raise RuntimeError("Generated split failed zero-overlap audit")
    return summary


def _counts(rows: list[dict[str, Any]], key) -> dict[str, int]:
    return {
        str(name): count
        for name, count in sorted(Counter(key(row) for row in rows).items())
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=list(DEFAULT_TASKS))
    args = parser.parse_args(argv)
    summaries = {
        task: build_task(task, source_root=args.source_root, output_root=args.output_root)
        for task in args.tasks
    }
    root_summary = {
        "schema_version": PROTOCOL_VERSION,
        "lineage": LINEAGE,
        "seed": SEED,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "tasks": {
            task: summary["splits"]["scaffold"] for task, summary in summaries.items()
        },
    }
    write_json_atomic(args.output_root / "summary.json", root_summary)
    print(json.dumps({"summary": str(args.output_root / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
