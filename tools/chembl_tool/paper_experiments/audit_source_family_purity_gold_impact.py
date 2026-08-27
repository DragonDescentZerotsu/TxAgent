"""Audit whether retrieval-family purity candidates would change frozen gold votes.

This is a sensitivity audit only.  It applies each task's existing record-vote
eligibility and 70% parent agreement contract, writes explicit parent-level
diffs, and never rewrites a frozen benchmark artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.starling.benchmark_dataset import LabeledSourceRecord
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    DIRECT_REPORT_TYPES,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    has_reported_text,
    is_human_context,
    label_bioavailability_value,
    load_label_decisions as load_bio_decisions,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    normalized_direct_label,
)
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    load_label_decisions as load_skin_decisions,
)


VERSION = "source_family_purity_gold_vote_impact.v1"
PURITY_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "source_family_purity_v1"
)
DEFAULT_OUTPUT_ROOT = PURITY_ROOT / "gold_vote_impact"
AGREEMENT_THRESHOLD = 0.70


def _accepted_records(decisions: Iterable[Any]) -> list[LabeledSourceRecord]:
    return [decision.record for decision in decisions if decision.record is not None]


def _parent_key(smiles: str) -> str:
    identity = normalize_molecule_identity(smiles)
    return identity.parent_inchi_key or identity.parent_smiles


def _vote(records: Iterable[LabeledSourceRecord]) -> dict[str, Any]:
    counts = Counter(int(record.label) for record in records)
    total = sum(counts.values())
    if not total:
        return {"label": None, "status": "no_votes", "counts": {"0": 0, "1": 0}}
    if counts[0] == counts[1]:
        status, label = "exact_tie", None
    else:
        label = 1 if counts[1] > counts[0] else 0
        status = (
            "accepted"
            if counts[label] / total >= AGREEMENT_THRESHOLD
            else "below_agreement_threshold"
        )
        if status != "accepted":
            label = None
    return {
        "label": label,
        "status": status,
        "counts": {"0": counts[0], "1": counts[1]},
        "n_votes": total,
    }


def _votes_by_parent(records: Iterable[LabeledSourceRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[LabeledSourceRecord]] = defaultdict(list)
    for record in records:
        key = _parent_key(record.smiles)
        if key:
            grouped[key].append(record)
    return {key: _vote(rows) for key, rows in grouped.items()}


def _bio_candidates(path: Path) -> tuple[list[LabeledSourceRecord], Counter[str]]:
    columns = [
        "source_family_purity_reason",
        "canonical_bioavailability_report_type",
        "species_or_population",
        "qualifying_conditions",
        "canonical_measurement_text",
        "canonical_smiles",
        "canonical_record_id",
        "pmid",
    ]
    candidates: list[LabeledSourceRecord] = []
    audit: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=10_000):
        for row in batch.to_pylist():
            if not _text(row.get("source_family_purity_reason")):
                continue
            audit["n_purity_moved_rows"] += 1
            report_type = _text(
                row.get("canonical_bioavailability_report_type")
            ).lower()
            if report_type not in DIRECT_REPORT_TYPES:
                audit[f"reject_report_type:{report_type or '<missing>'}"] += 1
                continue
            if not is_human_context(row.get("species_or_population")):
                audit["reject_nonhuman_or_unresolved_population"] += 1
                continue
            if has_reported_text(row.get("qualifying_conditions")):
                audit["reject_qualifying_conditions"] += 1
                continue
            label, method = label_bioavailability_value(
                row.get("canonical_measurement_text")
            )
            if label is None:
                audit[f"reject_value:{method}"] += 1
                continue
            smiles = _text(row.get("canonical_smiles"))
            if not _parent_key(smiles):
                audit["reject_invalid_smiles"] += 1
                continue
            candidates.append(
                LabeledSourceRecord(
                    smiles=smiles,
                    label=label,
                    source_id="purity_candidate:bioavailability_ma",
                    source_record_id=_text(row.get("canonical_record_id")),
                    pmid=_text(row.get("pmid")),
                    label_method=f"purity_candidate:{method}",
                    raw_value=_text(row.get("canonical_measurement_text")),
                    context=_text(row.get("species_or_population")),
                )
            )
            audit["n_gold_eligible_candidate_votes"] += 1
    return candidates, audit


def _skin_existing_keys() -> set[tuple[str, str, str, int]]:
    source = Path("data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet")
    columns = ["SMILES", "pmid", "support_text", "outcome_label"]
    keys: set[tuple[str, str, str, int]] = set()
    for row in pq.read_table(source, columns=columns).to_pylist():
        label = _skin_binary_label(row.get("outcome_label"))
        key = _parent_key(_text(row.get("SMILES")))
        if label is None or not key:
            continue
        keys.add((key, _normalized(row.get("pmid")), _normalized(row.get("support_text")), label))
    return keys


def _skin_candidates(path: Path) -> tuple[list[LabeledSourceRecord], Counter[str]]:
    columns = [
        "source_family_purity_reason",
        "canonical_smiles",
        "canonical_record_id",
        "pmid",
        "support_text",
        "result_label",
        "outcome_label",
        "canonical_measurement_text",
        "canonical_assay_type",
        "canonical_assay_or_test",
        "assay_type",
        "assay_or_test",
    ]
    existing_keys = _skin_existing_keys()
    seen = set(existing_keys)
    candidates: list[LabeledSourceRecord] = []
    audit: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=10_000):
        for row in batch.to_pylist():
            if not _text(row.get("source_family_purity_reason")):
                continue
            audit["n_purity_moved_rows"] += 1
            label = _skin_binary_label(
                row.get("result_label")
                or row.get("outcome_label")
                or row.get("canonical_measurement_text")
            )
            if label is None:
                audit["reject_missing_or_ambiguous_label"] += 1
                continue
            smiles = _text(row.get("canonical_smiles"))
            parent = _parent_key(smiles)
            if not parent:
                audit["reject_invalid_smiles"] += 1
                continue
            key = (
                parent,
                _normalized(row.get("pmid")),
                _normalized(row.get("support_text")),
                label,
            )
            if key in seen:
                audit[
                    "reject_exact_duplicate_with_frozen_direct_or_prior_candidate"
                ] += 1
                continue
            seen.add(key)
            assay = _first_text(
                row,
                "canonical_assay_type",
                "canonical_assay_or_test",
                "assay_type",
                "assay_or_test",
            )
            candidates.append(
                LabeledSourceRecord(
                    smiles=smiles,
                    label=label,
                    source_id="purity_candidate:skin_reaction_aop_source",
                    source_record_id=_text(row.get("canonical_record_id")),
                    pmid=_text(row.get("pmid")),
                    label_method="purity_candidate:validated_final_assay",
                    raw_value=_first_text(
                        row, "result_label", "outcome_label", "canonical_measurement_text"
                    ),
                    context=assay,
                )
            )
            audit["n_gold_eligible_candidate_votes"] += 1
    return candidates, audit


def audit_task(task: str, output_root: Path) -> dict[str, Any]:
    if task == "bioavailability_ma":
        decisions, _ = load_bio_decisions()
        current = _accepted_records(decisions)
        source = PURITY_ROOT / task / "records.parquet"
        candidates, eligibility = _bio_candidates(source)
    elif task == "skin_reaction":
        decisions, _ = load_skin_decisions()
        current = _accepted_records(decisions)
        source = PURITY_ROOT / task / "records.parquet"
        candidates, eligibility = _skin_candidates(source)
    else:
        raise ValueError(task)

    current_votes = _votes_by_parent(current)
    augmented_votes = _votes_by_parent([*current, *candidates])
    diffs = []
    impact_counts: Counter[str] = Counter()
    candidate_parent_counts = Counter(_parent_key(row.smiles) for row in candidates)
    split_membership = _frozen_split_membership(task)
    conditioned_valid = _conditioned_valid_membership(task)
    for parent in sorted(set(current_votes) | set(augmented_votes)):
        before = current_votes.get(parent, _vote([]))
        after = augmented_votes.get(parent, _vote([]))
        if before["label"] == after["label"] and before["status"] == after["status"]:
            impact = "unchanged"
        elif before["label"] in (0, 1) and after["label"] in (0, 1):
            impact = "label_flip"
        elif before["label"] in (0, 1) and after["label"] is None:
            impact = "accepted_parent_becomes_rejected"
        elif before["label"] is None and after["label"] in (0, 1):
            impact = "new_or_recovered_accepted_parent"
        else:
            impact = "rejected_status_changed"
        impact_counts[impact] += 1
        if impact != "unchanged":
            diffs.append(
                {
                    "task": task,
                    "parent_identity_key": parent,
                    "n_candidate_votes": candidate_parent_counts[parent],
                    "frozen_split": (split_membership.get(parent) or {}).get("split", ""),
                    "frozen_label": (split_membership.get(parent) or {}).get("Y"),
                    "conditioned_valid_query_indices": conditioned_valid.get(parent, []),
                    "current": before,
                    "augmented": after,
                    "impact": impact,
                }
            )

    task_dir = output_root / task
    write_jsonl_atomic(task_dir / "parent_vote_changes.jsonl", diffs)
    changed_by_split = Counter(
        row["frozen_split"] or "not_in_frozen_binary_benchmark" for row in diffs
    )
    flips_by_split = Counter(
        row["frozen_split"] or "not_in_frozen_binary_benchmark"
        for row in diffs
        if row["impact"] == "label_flip"
    )
    affected_conditioned_valid = sorted(
        {
            index
            for row in diffs
            for index in row["conditioned_valid_query_indices"]
        }
    )
    flipped_conditioned_valid = sorted(
        {
            index
            for row in diffs
            if row["impact"] == "label_flip"
            for index in row["conditioned_valid_query_indices"]
        }
    )
    summary = {
        "version": VERSION,
        "task": task,
        "sensitivity_only": True,
        "frozen_gold_modified": False,
        "purity_records": str(source.resolve()),
        "purity_records_sha256": sha256_file(source),
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "n_current_accepted_source_votes": len(current),
        "n_candidate_votes_after_eligibility_and_exact_dedup": len(candidates),
        "n_candidate_parents": len(candidate_parent_counts),
        "eligibility_audit": dict(sorted(eligibility.items())),
        "parent_impact_counts": dict(sorted(impact_counts.items())),
        "n_parent_vote_changes": len(diffs),
        "n_label_flips": impact_counts["label_flip"],
        "changed_parents_by_frozen_split": dict(sorted(changed_by_split.items())),
        "label_flips_by_frozen_split": dict(sorted(flips_by_split.items())),
        "n_affected_conditioned_valid_queries": len(affected_conditioned_valid),
        "affected_conditioned_valid_query_indices": affected_conditioned_valid,
        "n_label_flipped_conditioned_valid_queries": len(flipped_conditioned_valid),
        "label_flipped_conditioned_valid_query_indices": flipped_conditioned_valid,
        "parent_vote_changes": str((task_dir / "parent_vote_changes.jsonl").resolve()),
    }
    write_json_atomic(task_dir / "summary.json", summary)
    return summary


def _first_text(record: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = _text(record.get(field))
        if value:
            return value
    return ""


def _frozen_split_membership(task: str) -> dict[str, dict[str, Any]]:
    dataset, task_dir = {
        "bioavailability_ma": (
            "processed_starling_record_supported_v2",
            "Bioavailability_Ma",
        ),
        "skin_reaction": ("processed_starling_record_supported_v2", "Skin_Reaction"),
    }[task]
    root = Path("data") / dataset / task_dir / "scaffold"
    membership: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        for row in read_jsonl(root / f"{split}_molecule_labels.jsonl"):
            key = _text(row.get("molecule_identity_key")) or _parent_key(
                _text(row.get("drug"))
            )
            if key:
                membership[key] = {"split": split, "Y": row.get("Y")}
    return membership


def _conditioned_valid_membership(task: str) -> dict[str, list[int]]:
    task_dir = {
        "bioavailability_ma": "Bioavailability_Ma",
        "skin_reaction": "Skin_Reaction",
    }[task]
    path = (
        Path("data/processed_starling_context_conditioned_selected_v1")
        / task_dir
        / "scaffold/valid.jsonl"
    )
    membership: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(read_jsonl(path)):
        key = _parent_key(_text(row.get("drug")))
        if key:
            membership[key].append(index)
    return dict(membership)


def _skin_binary_label(value: Any) -> int | None:
    label = normalized_direct_label(value)
    if label == "positive":
        return 1
    if label == "negative":
        return 0
    return None


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).lower()).strip()


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("bioavailability_ma", "skin_reaction"),
        default=["bioavailability_ma", "skin_reaction"],
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    summaries = [audit_task(task, Path(args.output_root)) for task in args.tasks]
    write_json_atomic(
        Path(args.output_root) / "summary.json",
        {"version": VERSION, "tasks": summaries},
    )
    for row in summaries:
        print(
            f"{row['task']}: candidates={row['n_candidate_votes_after_eligibility_and_exact_dedup']:,} "
            f"label_flips={row['n_label_flips']:,} changes={row['n_parent_vote_changes']:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
