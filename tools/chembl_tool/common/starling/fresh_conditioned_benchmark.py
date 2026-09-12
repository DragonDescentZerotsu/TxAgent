"""Stage a fresh conditioned benchmark from task-accepted source-record votes.

No source semantics or review verdicts are inferred here. Task adapters supply
normalized parent/scaffold identities, exact condition groups, and provenance.
This builder neither extends a frozen split nor publishes canonical artifacts.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    SEED,
    allocate_scaffold_groups,
    resolve_conditioned_eval_size,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    BUILD_ROOT,
    CONTRACT,
    NO_REPORTED_CONDITION,
    TASK_DIRECTORIES,
)
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    SPLITS,
    _minimal_row,
    _row_id,
    _write_csv,
    aggregate_reviewed_votes,
    summarize_groups,
    validate_split_integrity,
)


def build_fresh_conditioned_benchmark(
    *,
    task: str,
    record_votes: Sequence[Mapping[str, Any]],
    output_root: Path | None = None,
    source_artifacts: Sequence[Path] = (),
    seed: int = SEED,
    target_size: int | None = None,
    required_labels: Sequence[int] = (),
    swap_evaluation_splits: bool = False,
) -> dict[str, Any]:
    """Write staging split/audit files and return their summary.

    ``swap_evaluation_splits`` exchanges valid/test names after allocation;
    train and cohort membership remain fixed. Optimizer diagnostics retain the
    original allocation names; output rows and group summaries use the new names.

    Votes require source_record_id, drug, molecule_identity_key, scaffold, Y,
    condition_group, condition_atoms, pmid, label_method, reviewer, raw_value,
    and condition_text. ``bemis_murcko_scaffold`` is also accepted as the
    scaffold field. Extra fields are retained verbatim in ``source_votes``.
    Each source_record_id must occur once. Reviewers/methods are caller claims,
    never evidence of manual review inferred by this builder.
    External condition groups must equal the sorted, unique atoms joined by
    '+'. Atoms are nonempty trimmed strings without the '+' separator or the
    reserved null-condition marker. The null group requires an empty atom list.
    Atom order is normalized for aggregation; original votes remain verbatim.

    Agreement is >=70% without a reported external condition and >=60% with
    an external condition, excluding ties. Groups need >=3 accepted parents and
    >=3 distinct nonempty scaffolds. Eligible groups must occur in every split;
    infeasible allocation fails before writing, without silently dropping them.
    Held-out size is measured in molecule-condition rows. When target_size is
    omitted, choose the smallest feasible equal valid/test size at or above
    max(number of eligible groups, floor(10% of rows)), then optimize quality
    at that fixed size. Explicit target_size remains strict. Whole scaffolds
    stay together and empty scaffolds are restricted to train.
    """
    directory = TASK_DIRECTORIES[task]
    root = (
        Path(output_root)
        if output_root is not None
        else BUILD_ROOT / directory / "scaffold"
    )
    votes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    originals: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    identities: dict[str, tuple[str, str]] = {}
    required = {
        "source_record_id",
        "drug",
        "molecule_identity_key",
        "Y",
        "condition_group",
        "condition_atoms",
        "pmid",
        "label_method",
        "reviewer",
        "raw_value",
        "condition_text",
    }
    for raw in record_votes:
        row = dict(raw)
        missing = required - row.keys()
        if missing or not ({"scaffold", "bemis_murcko_scaffold"} & row.keys()):
            raise ValueError(
                f"Incomplete accepted record vote: missing {sorted(missing)} or scaffold"
            )
        for field in (
            "source_record_id",
            "drug",
            "molecule_identity_key",
            "condition_group",
            "label_method",
            "reviewer",
        ):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Accepted record vote requires nonempty {field}")
        record_id = row["source_record_id"]
        if record_id in seen:
            raise ValueError(f"Duplicate source_record_id: {record_id}")
        seen.add(record_id)
        if row["Y"] not in (0, 1) or not isinstance(row["condition_atoms"], list):
            raise ValueError(f"Invalid binary label or condition_atoms: {record_id}")
        scaffold = row.get("scaffold", row.get("bemis_murcko_scaffold")) or ""
        if not isinstance(scaffold, str):
            raise ValueError(f"Invalid scaffold: {record_id}")
        if (
            "bemis_murcko_scaffold" in row
            and (row["bemis_murcko_scaffold"] or "") != scaffold
        ):
            raise ValueError(f"Conflicting scaffold fields: {record_id}")
        parent, group = row["molecule_identity_key"], row["condition_group"]
        identity = (row["drug"], scaffold)
        if identities.setdefault(parent, identity) != identity:
            raise ValueError(f"Inconsistent drug/scaffold for parent: {parent}")
        atoms = row["condition_atoms"]
        if group == NO_REPORTED_CONDITION:
            if atoms:
                raise ValueError("Unreported condition must have empty condition_atoms")
        elif (
            not atoms
            or any(
                not isinstance(atom, str)
                or not atom
                or atom != atom.strip()
                or "+" in atom
                or atom == NO_REPORTED_CONDITION
                for atom in atoms
            )
            or len(atoms) != len(set(atoms))
            or group != "+".join(sorted(atoms))
        ):
            raise ValueError(
                f"Accepted record vote has non-canonical condition signature: {record_id}"
            )
        originals[parent, group].append(row)
        votes[parent, group].append(
            {
                **row,
                "bemis_murcko_scaffold": scaffold,
                "condition_atoms": sorted(atoms),
            }
        )
    for grouped in (votes, originals):
        for rows in grouped.values():
            rows.sort(key=lambda row: row["source_record_id"])

    accepted, rejected = [], []
    for has_condition, threshold in ((False, 0.70), (True, 0.60)):
        scoped_votes = {
            key: rows for key, rows in votes.items()
            if (key[1] != NO_REPORTED_CONDITION) == has_condition
        }
        scoped_accepted, scoped_rejected = aggregate_reviewed_votes(
            scoped_votes, agreement_threshold=threshold
        )
        accepted.extend(scoped_accepted)
        rejected.extend(scoped_rejected)
    for rows in (accepted, rejected):
        rows.sort(key=lambda row: (row["molecule_identity_key"], row["condition_group"]))
    # The legacy aggregator caps example/id lists and uses reviewed/60% wording.
    # Keep its voting logic, but make the fresh adapter's provenance complete and
    # its policy truthful without changing historical builders or their bytes.
    for row in accepted + rejected:
        sources = originals[row["molecule_identity_key"], row["condition_group"]]
        row.update(
            {
                "source_votes": sources,
                "source_record_ids": [source["source_record_id"] for source in sources],
                "source_pmids": list(
                    dict.fromkeys(
                        str(source["pmid"])
                        for source in sources
                        if source["pmid"] is not None and str(source["pmid"])
                    )
                ),
                "raw_value_examples": [source["raw_value"] for source in sources],
                "condition_text_examples": [
                    source["condition_text"] for source in sources
                ],
                "vote_unit": "task_accepted_source_record",
                "condition_scope": "none_reported"
                if row["condition_group"] == NO_REPORTED_CONDITION
                else "external",
                "benchmark_row_id": _row_id(
                    directory.upper(),
                    row["molecule_identity_key"],
                    row["condition_group"],
                ),
            }
        )
        if "Y" in row:
            row["label_decision"] = (
                "unanimous"
                if not row["minority_record_count"]
                else f"accepted_record_majority_{round(100 * row['agreement_threshold'])}"
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        grouped[row["condition_group"]].append(row)
    group_gate = {}
    selected = []
    group_rejected = []
    for group, rows in sorted(grouped.items()):
        n_parents = len({row["molecule_identity_key"] for row in rows})
        n_scaffolds = len(
            {
                row["bemis_murcko_scaffold"]
                for row in rows
                if row["bemis_murcko_scaffold"]
            }
        )
        eligible = n_parents >= 3 and n_scaffolds >= 3
        group_gate[group] = {
            "n_parents": n_parents,
            "n_nonempty_scaffolds": n_scaffolds,
            "eligible": eligible,
        }
        if eligible:
            selected.extend(rows)
        else:
            group_rejected.extend(
                {
                    **row,
                    "drop_reason": "condition_group_requires_three_parents_and_nonempty_scaffolds",
                }
                for row in rows
            )
    if not selected:
        raise ValueError(
            "No condition group has three accepted parents and nonempty scaffolds"
        )
    required_groups = {row["condition_group"] for row in selected}
    target = (
        max(len(required_groups), len(selected) // 10)
        if target_size is None
        else target_size
    )
    if not isinstance(target, int) or target < 1:
        raise ValueError("target_size must be a positive integer row count")
    nominal_target = target
    if target_size is None:
        target = resolve_conditioned_eval_size(
            selected,
            nominal_target_size=nominal_target,
            required_condition_groups=required_groups,
            seed=seed,
            required_labels=required_labels,
        )
    assignment, optimization = allocate_scaffold_groups(
        selected,
        target_size=target,
        seed=seed,
        exclude_empty_scaffold_from_heldout=True,
        required_condition_groups=required_groups,
        required_labels=required_labels,
    )
    if swap_evaluation_splits:
        assignment = {
            scaffold: {"valid": "test", "test": "valid"}.get(split, split)
            for scaffold, split in assignment.items()
        }
    # The historical allocator consumes one row per parent. Fresh conditioned
    # cohorts may have several rows per parent; report both units accurately.
    empty_rows = [row for row in selected if not row["bemis_murcko_scaffold"]]
    optimization["n_empty_scaffold_rows"] = len(empty_rows)
    optimization["n_empty_scaffold_parents"] = len(
        {row["molecule_identity_key"] for row in empty_rows}
    )
    combined: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in selected:
        split = assignment[row["bemis_murcko_scaffold"]]
        output = {**row, "split": split, "split_policy": CONTRACT}
        if required_labels:
            output["split_required_labels"] = list(required_labels)
        combined[split].append(output)
    for rows in combined.values():
        rows.sort(
            key=lambda row: (row["molecule_identity_key"], row["condition_group"])
        )
    validate_split_integrity(combined)
    for split, rows in combined.items():
        if {row["condition_group"] for row in rows} != required_groups:
            raise RuntimeError(f"Condition coverage failed: {split}")
    distribution = summarize_groups(combined)
    source_digest = hashlib.sha256(
        json.dumps(
            sorted(
                (dict(row) for row in record_votes),
                key=lambda row: row["source_record_id"],
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    summary = {
        "task": directory,
        "contract": CONTRACT,
        "protocol_version": "fresh_conditioned_benchmark.v2",
        "seed": seed,
        "swap_evaluation_splits": swap_evaluation_splits,
        "source_votes_sha256": source_digest,
        "source_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in source_artifacts
        ],
        "n_source_records": len(record_votes),
        "n_accepted_parent_conditions": len(selected),
        "n_rejected_parent_conditions": len(rejected) + len(group_rejected),
        "label_policy": {
            "agreement_threshold_by_condition_scope": {
                "none_reported": 0.70,
                "external": 0.60,
            },
            "ties": "excluded",
            "vote_unit": "task_accepted_source_record",
            "review_policy": "caller-supplied label_method and reviewer retained verbatim; no manual review inferred",
        },
        "group_gate": group_gate,
        "optimization": optimization,
        "target_eval_rows": target,
        "nominal_target_eval_rows": nominal_target,
        "target_size_policy": (
            "minimum_feasible_equal_eval_size" if target_size is None else "explicit_strict"
        ),
        "target_expansion_reason": (
            "nominal_size_infeasible_with_whole_scaffold_three_way_condition_coverage"
            if target > nominal_target else None
        ),
        "group_distribution": distribution,
        "splits": {
            split: {
                "n": len(rows),
                "n_parents": len({row["molecule_identity_key"] for row in rows}),
            }
            for split, rows in combined.items()
        },
        "split_integrity_validated": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in combined.items():
        write_jsonl_atomic(root / f"{split}.jsonl", map(_minimal_row, rows))
        write_jsonl_atomic(root / f"{split}_molecule_condition_labels.jsonl", rows)
    write_jsonl_atomic(
        root / "heldout_molecule_condition_labels.jsonl",
        combined["valid"] + combined["test"],
    )
    write_jsonl_atomic(
        root / "accepted_record_votes.jsonl",
        sorted(
            (dict(row) for row in record_votes), key=lambda row: row["source_record_id"]
        ),
    )
    write_jsonl_atomic(
        root / "accepted_parent_conditions_before_group_gate.jsonl", accepted
    )
    write_jsonl_atomic(
        root / "accepted_parent_conditions.jsonl",
        [row for split in SPLITS for row in combined[split]],
    )
    write_jsonl_atomic(
        root / "rejected_parent_conditions.jsonl", rejected + group_rejected
    )
    _write_csv(root / "group_distribution.csv", distribution)
    write_json_atomic(root / "summary.json", summary)
    return summary
