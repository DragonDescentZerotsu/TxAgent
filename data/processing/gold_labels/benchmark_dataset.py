"""Common machinery for building binary benchmark splits from Starling records."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


@dataclass(frozen=True)
class NumericInterval:
    """A reported numeric value represented without inventing a point estimate."""

    lower: float
    upper: float
    method: str
    normalized_text: str


@dataclass(frozen=True)
class LabeledSourceRecord:
    """One source extraction that can be mapped to the task's binary label."""

    smiles: str
    label: int
    source_id: str
    source_record_id: str = ""
    pmid: str = ""
    label_method: str = ""
    raw_value: str = ""
    context: str = ""
    source_row_uid: str = ""


@dataclass(frozen=True)
class LabelDecision:
    """Accepted label or an auditable reason why a source row was not used."""

    record: LabeledSourceRecord | None
    reason: str = ""
    example: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSplit:
    """One deterministic train/valid/test partition over accepted parents."""

    method: str
    train: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    test: list[dict[str, Any]]
    target_valid_size: int
    target_test_size: int


def accepted(record: LabeledSourceRecord) -> LabelDecision:
    return LabelDecision(record=record)


def rejected(reason: str, **example: Any) -> LabelDecision:
    return LabelDecision(record=None, reason=reason, example=example)


def parse_numeric_interval(
    value: Any,
    *,
    fraction_to_percent: bool = False,
) -> NumericInterval | None:
    """Parse the first reported value/range while preserving threshold ambiguity."""
    text = normalize_numeric_text(value)
    if not text:
        return None
    first_number = re.search(NUMBER_PATTERN, text)
    if first_number is None:
        return None

    prefix = text[max(0, first_number.start() - 24) : first_number.start()].lower()
    first_value = float(first_number.group(0))
    scale = _percent_scale(text, first_value, fraction_to_percent=fraction_to_percent)

    range_match = re.match(
        rf"\s*({NUMBER_PATTERN})\s*(?:-|to)\s*({NUMBER_PATTERN})",
        text[first_number.start() :],
        flags=re.IGNORECASE,
    )
    if range_match:
        left = float(range_match.group(1)) * scale
        right = float(range_match.group(2)) * scale
        return NumericInterval(min(left, right), max(left, right), "reported_range", text)

    relation = _relation_from_prefix(prefix, text[first_number.start() : first_number.start() + 2])
    value_scaled = first_value * scale
    if relation == "lower_bound":
        return NumericInterval(value_scaled, math.inf, relation, text)
    if relation == "upper_bound":
        return NumericInterval(-math.inf, value_scaled, relation, text)

    plus_minus = re.match(
        rf"\s*({NUMBER_PATTERN})\s*(?:±|\+/-|\+-|plus/minus)\s*({NUMBER_PATTERN})",
        text[first_number.start() :],
        flags=re.IGNORECASE,
    )
    if plus_minus:
        center = float(plus_minus.group(1)) * scale
        error = abs(float(plus_minus.group(2)) * scale)
        return NumericInterval(center - error, center + error, "reported_mean_plus_minus", text)
    return NumericInterval(value_scaled, value_scaled, "reported_point", text)


def classify_interval(interval: NumericInterval, *, threshold: float) -> int | None:
    """Apply a >= positive threshold only when the entire interval is one-sided."""
    if interval.lower >= threshold:
        return 1
    if interval.upper < threshold:
        return 0
    return None


def normalize_numeric_text(value: Any) -> str:
    text = str(value or "").strip()
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
        "％": "%",
        "﹪": "%",
        "per cent": "%",
        "percent": "%",
        "approximately": "about",
        "approx.": "about",
        "approx": "about",
    }
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def has_reported_text(value: Any) -> bool:
    """Return whether a nullable dataframe/dataset field contains real text."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "null", "n/a", "na"}


def build_benchmark_dataset(
    *,
    task_name: str,
    decisions: Iterable[LabelDecision],
    source_metadata: Mapping[str, Any],
    output_dir: str | Path,
    max_eval_size: int = 500,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    agreement_threshold: float = 0.70,
    seed: int = 20260723,
    max_rejection_examples: int = 20,
    voter_membership_path: str | Path | None = None,
    voter_membership_stage1_sha256: str | None = None,
    voter_membership_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply record-majority labels and write random/scaffold three-way splits."""
    if not 0.5 <= agreement_threshold <= 1.0:
        raise ValueError("agreement_threshold must be between 0.5 and 1.0")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped: dict[str, list[tuple[LabeledSourceRecord, dict[str, Any]]]] = defaultdict(list)

    for decision in decisions:
        source_counts["n_source_rows_considered"] += 1
        if decision.record is None:
            reason = decision.reason or "unspecified_rejection"
            rejection_counts[reason] += 1
            if len(rejection_examples[reason]) < max_rejection_examples:
                rejection_examples[reason].append(dict(decision.example))
            continue

        source_counts["n_source_rows_labeled"] += 1
        record = decision.record
        identity = normalize_molecule_identity(record.smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            rejection_counts["invalid_or_unresolved_smiles"] += 1
            if len(rejection_examples["invalid_or_unresolved_smiles"]) < max_rejection_examples:
                rejection_examples["invalid_or_unresolved_smiles"].append(
                    {"source_id": record.source_id, "source_record_id": record.source_record_id}
                )
            continue
        identity_key = identity.parent_inchi_key or identity.parent_smiles
        grouped[identity_key].append((record, identity.to_dict()))

    molecule_rows: list[dict[str, Any]] = []
    conflicting_rows: list[dict[str, Any]] = []
    rejected_parent_rows: list[dict[str, Any]] = []
    recovered_conflict_count = 0
    for identity_key, items in sorted(grouped.items()):
        labels = sorted({record.label for record, _ in items})
        label_counts = Counter(record.label for record, _ in items)
        total_records = len(items)
        majority_count = max(label_counts.values())
        minority_count = total_records - majority_count
        agreement_fraction = majority_count / total_records
        is_tie = len(labels) > 1 and label_counts[0] == label_counts[1]
        majority_label = 1 if label_counts[1] > label_counts[0] else 0
        identity = items[0][1]
        base = {
            "drug": identity["parent_smiles"],
            "molecule_identity_key": identity_key,
            "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
            "source_record_count": len(items),
            "source_ids": sorted({record.source_id for record, _ in items}),
            "label_methods": dict(Counter(record.label_method for record, _ in items)),
            "source_pmids": _unique_limited((record.pmid for record, _ in items), 50),
            "source_record_ids": _unique_limited((record.source_record_id for record, _ in items), 50),
            "raw_value_examples": _unique_limited((record.raw_value for record, _ in items), 20),
            "context_examples": _unique_limited((record.context for record, _ in items), 10),
            "molecule_identity": identity,
            "majority_label": majority_label,
            "majority_record_count": majority_count,
            "minority_record_count": minority_count,
            "agreement_fraction": agreement_fraction,
            "agreement_threshold": agreement_threshold,
            "vote_unit": "accepted_source_record",
        }
        if is_tie:
            rejected = {**base, "drop_reason": "parent_record_label_tie"}
            conflicting_rows.append({**rejected, "agreement_decision": "rejected"})
            rejected_parent_rows.append(rejected)
            continue
        if agreement_fraction < agreement_threshold:
            rejected = {
                **base,
                "drop_reason": "parent_record_agreement_below_threshold",
            }
            conflicting_rows.append({**rejected, "agreement_decision": "rejected"})
            rejected_parent_rows.append(rejected)
            continue
        if len(labels) > 1:
            recovered_conflict_count += 1
            conflicting_rows.append(
                {
                    **base,
                    "agreement_decision": "accepted_record_majority",
                    "assigned_label": majority_label,
                }
            )
        molecule_rows.append(
            {
                **base,
                "Y": majority_label,
                "label_decision": (
                    "unanimous" if len(labels) == 1 else "accepted_record_majority"
                ),
                "bemis_murcko_scaffold": bemis_murcko_scaffold(identity["parent_smiles"]),
            }
        )

    target_valid_size = calculate_eval_size(
        len(molecule_rows),
        max_eval_size=max_eval_size,
        eval_fraction=valid_fraction,
    )
    target_test_size = calculate_test_size(
        len(molecule_rows),
        max_test_size=max_eval_size,
        test_fraction=test_fraction,
    )
    random_train, random_valid, random_test = stratified_hash_three_way_split(
        molecule_rows,
        valid_size=target_valid_size,
        test_size=target_test_size,
        seed=seed,
    )
    scaffold_train, scaffold_valid, scaffold_test = scaffold_group_three_way_split(
        molecule_rows,
        valid_size=target_valid_size,
        test_size=target_test_size,
        seed=seed,
    )
    splits = {
        "random": DatasetSplit(
            method="label_stratified_stable_hash",
            train=random_train,
            valid=random_valid,
            test=random_test,
            target_valid_size=target_valid_size,
            target_test_size=target_test_size,
        ),
        "scaffold": DatasetSplit(
            method="bemis_murcko_scaffold_group_subset_sum",
            train=scaffold_train,
            valid=scaffold_valid,
            test=scaffold_test,
            target_valid_size=target_valid_size,
            target_test_size=target_test_size,
        ),
    }
    split_assignments = {
        name: {
            row["molecule_identity_key"]: subset
            for subset, rows in (
                ("train", split.train),
                ("valid", split.valid),
                ("test", split.test),
            )
            for row in rows
        }
        for name, split in splits.items()
    }
    labeled_rows = [
        {
            **row,
            "split_assignments": {
                name: split_assignments[name][row["molecule_identity_key"]]
                for name in splits
            },
        }
        for row in molecule_rows
    ]
    membership_manifest = None
    if voter_membership_path is not None:
        from data.processing.gold_labels.voter_membership import (
            NO_REPORTED_CONDITION,
            materialize_voter_membership,
            write_voter_membership,
        )

        vote_groups = {
            (identity_key, NO_REPORTED_CONDITION): [
                {
                    "source_row_uid": record.source_row_uid,
                    "source_id": record.source_id,
                    "source_record_id": record.source_record_id,
                    "molecule_identity_key": identity_key,
                    "condition_group": NO_REPORTED_CONDITION,
                    "Y": record.label,
                    "label_method": record.label_method,
                }
                for record, _ in items
            ]
            for identity_key, items in grouped.items()
        }
        scaffold_split = split_assignments["scaffold"]
        published_aggregates = [
            {
                **row,
                "condition_group": NO_REPORTED_CONDITION,
                "condition_atoms": [],
                "benchmark_row_id": _membership_row_id(
                    task_name, row["molecule_identity_key"]
                ),
                "split": scaffold_split[row["molecule_identity_key"]],
            }
            for row in molecule_rows
        ]
        rejected_aggregates = [
            {
                **row,
                "condition_group": NO_REPORTED_CONDITION,
                "condition_atoms": [],
            }
            for row in rejected_parent_rows
        ]
        membership_rows = materialize_voter_membership(
            task_name=task_name,
            vote_groups=vote_groups,
            published_aggregates=published_aggregates,
            rejected_parent_aggregates=rejected_aggregates,
        )
        membership_manifest = write_voter_membership(
            voter_membership_path,
            membership_rows,
            stage1_sha256=voter_membership_stage1_sha256,
            provenance=voter_membership_provenance,
        )
    _write_jsonl(output_path / "molecule_labels.jsonl", labeled_rows)
    _write_jsonl(output_path / "conflicting_molecules.jsonl", conflicting_rows)
    _write_jsonl(output_path / "rejected_parent_molecules.jsonl", rejected_parent_rows)
    _write_jsonl(
        output_path / "source_rejection_examples.jsonl",
        (
            {"reason": reason, "example": example}
            for reason in sorted(rejection_examples)
            for example in rejection_examples[reason]
        ),
    )

    split_summaries = {
        name: _write_split_artifacts(output_path, name, split)
        for name, split in splits.items()
    }
    summary = {
        "task": task_name,
        "protocol_version": "starling_binary_benchmark.v4",
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "seed": seed,
        "parent_label_policy": {
            "method": "record_weighted_majority",
            "agreement_formula": "max(n_label_0, n_label_1) / (n_label_0 + n_label_1)",
            "agreement_threshold": agreement_threshold,
            "vote_unit": "accepted_source_record",
            "tie_policy": "reject_exact_ties",
        },
        "split_size_policy": {
            "formula": "min(max_eval_size, floor(fraction * n_binary_molecules)) for valid and test",
            "max_eval_size": max_eval_size,
            "valid_fraction": valid_fraction,
            "test_fraction": test_fraction,
            "target_valid_size": target_valid_size,
            "target_test_size": target_test_size,
        },
        "n_source_rows_considered": source_counts["n_source_rows_considered"],
        "n_source_rows_labeled_before_structure_normalization": source_counts["n_source_rows_labeled"],
        "source_rejection_counts": dict(sorted(rejection_counts.items())),
        "n_parent_groups_with_any_label": len(grouped),
        "n_parent_groups_with_label_conflict": len(conflicting_rows),
        "n_parent_groups_recovered_by_majority": recovered_conflict_count,
        "n_rejected_parent_groups": len(rejected_parent_rows),
        "parent_rejection_counts": dict(
            sorted(Counter(row["drop_reason"] for row in rejected_parent_rows).items())
        ),
        "n_binary_molecules": len(molecule_rows),
        "all_label_counts": _label_counts(molecule_rows),
        "n_unique_bemis_murcko_scaffolds": len(
            {row["bemis_murcko_scaffold"] for row in molecule_rows}
        ),
        "n_acyclic_molecules": sum(
            not row["bemis_murcko_scaffold"] for row in molecule_rows
        ),
        "splits": split_summaries,
        "cross_method_eval_identity_overlap": {
            subset: len(
                {
                    row["molecule_identity_key"]
                    for row in getattr(splits["random"], subset)
                }
                & {
                    row["molecule_identity_key"]
                    for row in getattr(splits["scaffold"], subset)
                }
            )
            for subset in ("valid", "test")
        },
        "source_metadata": dict(source_metadata),
        "paths": {
            "molecule_labels": str(output_path / "molecule_labels.jsonl"),
            "conflicting_molecules": str(output_path / "conflicting_molecules.jsonl"),
            "rejected_parent_molecules": str(
                output_path / "rejected_parent_molecules.jsonl"
            ),
            "source_rejection_examples": str(output_path / "source_rejection_examples.jsonl"),
            "voter_membership": (
                str(voter_membership_path)
                if voter_membership_path is not None
                else None
            ),
        },
        "voter_membership": membership_manifest,
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_path / "report_zh.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _membership_row_id(task_name: str, parent: str) -> str:
    digest = hashlib.sha256(
        f"{task_name}\0{parent}\0no_reported_external_condition".encode("utf-8")
    ).hexdigest()[:20]
    return f"NULL_{digest.upper()}"


def calculate_test_size(
    n_molecules: int,
    *,
    max_test_size: int = 500,
    test_fraction: float = 0.2,
) -> int:
    """Backward-compatible wrapper for one evaluation subset size."""
    return calculate_eval_size(
        n_molecules,
        max_eval_size=max_test_size,
        eval_fraction=test_fraction,
    )


def calculate_eval_size(
    n_molecules: int,
    *,
    max_eval_size: int = 500,
    eval_fraction: float = 0.1,
) -> int:
    """Calculate floor(min(max_eval_size, fraction * accepted molecules))."""
    if n_molecules < 2:
        raise ValueError("at least two binary molecules are required")
    if max_eval_size <= 0:
        raise ValueError("max_eval_size must be positive")
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    eval_size = min(max_eval_size, math.floor(eval_fraction * n_molecules))
    if eval_size <= 0:
        raise ValueError(
            f"evaluation-size policy produced {eval_size} for {n_molecules} molecules"
        )
    return eval_size


def stratified_hash_three_way_split(
    rows: list[dict[str, Any]],
    *,
    valid_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact-size, label-stratified deterministic train/valid/test sets."""
    _validate_three_way_sizes(rows, valid_size=valid_size, test_size=test_size)
    remaining, test = _stratified_hash_take(
        rows,
        take_size=test_size,
        seed=seed,
        namespace="test",
    )
    train, valid = _stratified_hash_take(
        remaining,
        take_size=valid_size,
        seed=seed,
        namespace="valid",
    )
    return train, valid, test


def stratified_hash_split(
    rows: list[dict[str, Any]],
    *,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return an exact-size, label-stratified, deterministic molecule split."""
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if len(rows) < test_size:
        raise ValueError(f"need at least {test_size} binary molecules, found {len(rows)}")

    return _stratified_hash_take(
        rows,
        take_size=test_size,
        seed=seed,
        namespace="",
    )


def _stratified_hash_take(
    rows: list[dict[str, Any]],
    *,
    take_size: int,
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if take_size <= 0:
        raise ValueError("take_size must be positive")
    if len(rows) < take_size:
        raise ValueError(f"need at least {take_size} binary molecules, found {len(rows)}")

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["Y"])].append(row)
    if set(by_label) != {0, 1}:
        raise ValueError(f"both binary labels are required, found {sorted(by_label)}")

    positive_target = round(take_size * len(by_label[1]) / len(rows))
    positive_target = min(max(1, positive_target), len(by_label[1]), take_size - 1)
    negative_target = take_size - positive_target
    if negative_target > len(by_label[0]):
        shift = negative_target - len(by_label[0])
        negative_target -= shift
        positive_target += shift
    if positive_target > len(by_label[1]):
        shift = positive_target - len(by_label[1])
        positive_target -= shift
        negative_target += shift

    test_keys: set[str] = set()
    for label, target in ((0, negative_target), (1, positive_target)):
        ordered = sorted(
            by_label[label],
            key=lambda row: (
                _stable_hash(
                    f"{seed}\0{namespace}\0{row['molecule_identity_key']}"
                    if namespace
                    else f"{seed}\0{row['molecule_identity_key']}"
                ),
                row["molecule_identity_key"],
            ),
        )
        test_keys.update(row["molecule_identity_key"] for row in ordered[:target])

    train = sorted(
        (row for row in rows if row["molecule_identity_key"] not in test_keys),
        key=lambda row: row["molecule_identity_key"],
    )
    selected = sorted(
        (row for row in rows if row["molecule_identity_key"] in test_keys),
        key=lambda row: row["molecule_identity_key"],
    )
    if len(selected) != take_size:
        raise AssertionError(
            f"expected {take_size} selected molecules, found {len(selected)}"
        )
    return train, selected


def scaffold_group_split(
    rows: list[dict[str, Any]],
    *,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seed-order whole scaffolds and subset-sum toward the test target from below."""
    return _scaffold_group_take(
        rows,
        take_size=test_size,
        seed=seed,
        namespace="",
    )


def scaffold_group_three_way_split(
    rows: list[dict[str, Any]],
    *,
    valid_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select disjoint whole-scaffold test and valid sets from one parent pool."""
    _validate_three_way_sizes(rows, valid_size=valid_size, test_size=test_size)
    remaining, test = _scaffold_group_take(
        rows,
        take_size=test_size,
        seed=seed,
        namespace="test",
    )
    train, valid = _scaffold_group_take(
        remaining,
        take_size=valid_size,
        seed=seed,
        namespace="valid",
    )
    return train, valid, test


def _scaffold_group_take(
    rows: list[dict[str, Any]],
    *,
    take_size: int,
    seed: int,
    namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if take_size <= 0:
        raise ValueError("take_size must be positive")
    if len(rows) < take_size:
        raise ValueError(f"need at least {take_size} binary molecules, found {len(rows)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["bemis_murcko_scaffold"])].append(row)

    ordered_groups = sorted(
        (
            (scaffold, group)
            for scaffold, group in groups.items()
            if len(group) <= take_size
        ),
        key=lambda item: (
            _stable_hash(
                f"{seed}\0scaffold\0{namespace}\0{item[0]}"
                if namespace
                else f"{seed}\0scaffold\0{item[0]}"
            ),
            item[0],
        ),
    )
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for scaffold, group in ordered_groups:
        group_size = len(group)
        for current_size in sorted(tuple(reachable), reverse=True):
            new_size = current_size + group_size
            if new_size > take_size or new_size in reachable:
                continue
            reachable[new_size] = (*reachable[current_size], scaffold)
        if take_size in reachable:
            break

    actual_size = max(reachable)
    if actual_size <= 0:
        raise ValueError("no scaffold group fits within the requested evaluation size")
    selected_scaffolds = set(reachable[actual_size])
    train = sorted(
        (row for row in rows if row["bemis_murcko_scaffold"] not in selected_scaffolds),
        key=lambda row: row["molecule_identity_key"],
    )
    selected = sorted(
        (row for row in rows if row["bemis_murcko_scaffold"] in selected_scaffolds),
        key=lambda row: row["molecule_identity_key"],
    )
    train_scaffolds = {row["bemis_murcko_scaffold"] for row in train}
    observed_selected_scaffolds = {row["bemis_murcko_scaffold"] for row in selected}
    if train_scaffolds & observed_selected_scaffolds:
        raise AssertionError("scaffold leakage detected between retained and selected rows")
    if len(selected) != actual_size:
        raise AssertionError(
            f"expected {actual_size} scaffold-selected molecules, found {len(selected)}"
        )
    return train, selected


def _validate_three_way_sizes(
    rows: list[dict[str, Any]],
    *,
    valid_size: int,
    test_size: int,
) -> None:
    if valid_size <= 0 or test_size <= 0:
        raise ValueError("valid_size and test_size must be positive")
    if valid_size + test_size >= len(rows):
        raise ValueError(
            "valid_size + test_size must leave at least one training molecule"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relation_from_prefix(prefix: str, immediate: str) -> str:
    text = f"{prefix} {immediate}".lower()
    if re.search(r"(?:>=|≥|at least|not less than|greater than|more than|above)", text):
        return "lower_bound"
    if re.search(r"(?:<=|≤|at most|not more than|less than|lower than|below|up to)", text):
        return "upper_bound"
    return ""


def _percent_scale(text: str, first_value: float, *, fraction_to_percent: bool) -> float:
    if "%" in text:
        return 1.0
    if fraction_to_percent and 0.0 <= first_value <= 1.5:
        return 100.0
    return 1.0


def _unique_limited(values: Iterable[Any], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
            if len(output) >= limit:
                break
    return output


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _minimal_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {"drug": row["drug"], "Y": int(row["Y"])}


def _label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(int(row["Y"])) for row in rows)
    return {label: counts.get(label, 0) for label in ("0", "1")}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _write_split_artifacts(
    output_path: Path,
    name: str,
    split: DatasetSplit,
) -> dict[str, Any]:
    split_path = output_path / name
    split_path.mkdir(parents=True, exist_ok=True)
    subsets = {"train": split.train, "valid": split.valid, "test": split.test}
    key_sets = {
        subset: {row["molecule_identity_key"] for row in rows}
        for subset, rows in subsets.items()
    }
    scaffold_sets = {
        subset: {row["bemis_murcko_scaffold"] for row in rows}
        for subset, rows in subsets.items()
    }
    identity_overlaps = _pairwise_overlap_counts(key_sets)
    scaffold_overlaps = _pairwise_overlap_counts(scaffold_sets)
    if any(identity_overlaps.values()):
        raise AssertionError(f"{name} split has molecule-identity overlap")
    if name == "scaffold" and any(scaffold_overlaps.values()):
        raise AssertionError("scaffold split has train/valid/test scaffold overlap")

    _write_jsonl(split_path / "train.jsonl", _minimal_rows(split.train))
    _write_jsonl(split_path / "valid.jsonl", _minimal_rows(split.valid))
    _write_jsonl(split_path / "test.jsonl", _minimal_rows(split.test))
    for subset, rows in subsets.items():
        details = (
            {**row, "split": subset, "split_method": split.method}
            for row in rows
        )
        _write_jsonl(split_path / f"{subset}_molecule_labels.jsonl", details)
    heldout_rows = sorted(
        [*split.valid, *split.test],
        key=lambda row: row["molecule_identity_key"],
    )
    _write_jsonl(
        split_path / "heldout_molecule_labels.jsonl",
        (
            {
                **row,
                "split": (
                    "valid"
                    if row["molecule_identity_key"] in key_sets["valid"]
                    else "test"
                ),
                "split_method": split.method,
            }
            for row in heldout_rows
        ),
    )

    summary = {
        "method": split.method,
        "target_valid_size": split.target_valid_size,
        "target_test_size": split.target_test_size,
        "actual_valid_size": len(split.valid),
        "actual_test_size": len(split.test),
        "valid_size_shortfall": split.target_valid_size - len(split.valid),
        "test_size_shortfall": split.target_test_size - len(split.test),
        "n_train": len(split.train),
        "n_valid": len(split.valid),
        "n_test": len(split.test),
        "train_label_counts": _label_counts(split.train),
        "valid_label_counts": _label_counts(split.valid),
        "test_label_counts": _label_counts(split.test),
        "n_train_scaffolds": len(scaffold_sets["train"]),
        "n_valid_scaffolds": len(scaffold_sets["valid"]),
        "n_test_scaffolds": len(scaffold_sets["test"]),
        "pairwise_scaffold_overlap": scaffold_overlaps,
        "pairwise_identity_overlap": identity_overlaps,
        "train_valid_scaffold_overlap": scaffold_overlaps["train_valid"],
        "train_test_scaffold_overlap": scaffold_overlaps["train_test"],
        "valid_test_scaffold_overlap": scaffold_overlaps["valid_test"],
        "train_valid_identity_overlap": identity_overlaps["train_valid"],
        "train_test_identity_overlap": identity_overlaps["train_test"],
        "valid_test_identity_overlap": identity_overlaps["valid_test"],
        "paths": {
            "train": str(split_path / "train.jsonl"),
            "valid": str(split_path / "valid.jsonl"),
            "test": str(split_path / "test.jsonl"),
            "train_molecule_labels": str(split_path / "train_molecule_labels.jsonl"),
            "valid_molecule_labels": str(split_path / "valid_molecule_labels.jsonl"),
            "test_molecule_labels": str(split_path / "test_molecule_labels.jsonl"),
            "heldout_molecule_labels": str(
                split_path / "heldout_molecule_labels.jsonl"
            ),
        },
    }
    (split_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _pairwise_overlap_counts(values: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        "train_valid": len(values["train"] & values["valid"]),
        "train_test": len(values["train"] & values["test"]),
        "valid_test": len(values["valid"] & values["test"]),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['task']} Starling 二分类数据构建报告",
        "",
        f"- 协议：`{summary['protocol_version']}`",
        f"- 分子身份：`{summary['identity_normalizer_version']}`",
        f"- seed：{summary['seed']}",
        f"- 可用二分类 parent：{summary['n_binary_molecules']:,}",
        f"- record agreement threshold：{summary['parent_label_policy']['agreement_threshold']:.0%}",
        f"- valid / test target：{summary['split_size_policy']['target_valid_size']:,} / "
        f"{summary['split_size_policy']['target_test_size']:,}",
        f"- 原始 label-conflict parents：{summary['n_parent_groups_with_label_conflict']:,}",
        f"- majority 恢复：{summary['n_parent_groups_recovered_by_majority']:,}",
        f"- agreement/tie 拒绝：{summary['n_rejected_parent_groups']:,}",
        "",
        "## Splits",
        "",
        "| split | train | valid | test | valid Y=0 / Y=1 | test Y=0 / Y=1 | scaffold pairwise overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, split in summary["splits"].items():
        lines.append(
            f"| {name} | {split['n_train']:,} | {split['n_valid']:,} | {split['n_test']:,} | "
            f"{split['valid_label_counts']['0']:,} / {split['valid_label_counts']['1']:,} | "
            f"{split['test_label_counts']['0']:,} / {split['test_label_counts']['1']:,} | "
            f"{sum(split['pairwise_scaffold_overlap'].values()):,} |"
        )
    lines.extend(
        [
            "",
            "## Source-row rejection",
            "",
            "```json",
            json.dumps(summary["source_rejection_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "完整 provenance 见 `molecule_labels.jsonl`；所有原始 conflict parent、未达到 agreement 的拒绝",
            "以及 source-row rejection 示例分别见 `conflicting_molecules.jsonl`、",
            "`rejected_parent_molecules.jsonl` 与 `source_rejection_examples.jsonl`。",
            "",
            "每种构造方法的 `heldout_molecule_labels.jsonl` 是 valid+test union retrieval 泄漏隔离清单。",
            "现有 full-source Starling index 不能直接用于 valid 或 test。",
            "",
        ]
    )
    return "\n".join(lines)
