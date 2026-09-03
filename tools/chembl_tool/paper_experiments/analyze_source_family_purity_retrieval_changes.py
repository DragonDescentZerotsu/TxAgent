"""Compare frozen progressive retrieval with the source-family-purity rebuild."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
    direct_like_bioavailability_reason,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.canonical_starling_source import (
    direct_outcome_reason,
)


VERSION = "source_family_purity_retrieval_diff.v2"
DEFAULT_OLD_ROOT = Path(
    "outputs/archive/conditioned_assay_superseded_20260827/progressive/"
    "starling_conditioned_assay_progressive_visible_v6/"
    "scaffold_valid_deepseek_v4_flash_0731"
)
DEFAULT_NEW_ROOT = Path(
    "outputs/archive/conditioned_assay_superseded_20260827/"
    "source_family_purity_v1/retrieval_diff_prepare_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/archive/conditioned_assay_superseded_20260827/"
    "source_family_purity_v1/retrieval_impact"
)
TASK_LEVELS = {"bbb_martins": 5, "bioavailability_ma": 6, "skin_reaction": 2}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepared(root: Path, task: str, query_index: int, level: int) -> dict[str, Any]:
    return _read(
        root
        / task
        / "queries"
        / f"query_idx{query_index:05d}"
        / "levels"
        / f"level_{level}"
        / "prepared.json"
    )


def _stable_query_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("molecule_identity_key") or ""),
        str(record.get("condition_group") or ""),
    )


def _query_rows(root: Path, task: str) -> list[dict[str, Any]]:
    manifest = _read(root / "experiment_manifest.json")
    input_path = Path(str(manifest["inputs"][task]["input_jsonl"]))
    return read_jsonl(input_path)


def _identity_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    output: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = _stable_query_key(row)
        if not key[0] or key in output:
            raise ValueError(f"invalid or duplicate stable query identity: {key}")
        output[key] = index
    return output


def _semantic_active(active: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for analog_id, analog_source in sorted(active.items()):
        analog = dict(analog_source)
        analog.pop("query_analog_tool_summaries", None)
        analog["cards"] = {
            card_id: dict(card)
            for card_id, card in sorted((analog.get("cards") or {}).items())
        }
        normalized[analog_id] = analog
    return normalized


def _card_locations(active: Mapping[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        str(card_id): (str(analog_id), dict(card))
        for analog_id, analog in active.items()
        for card_id, card in (analog.get("cards") or {}).items()
    }


def _looks_direct(task: str, card: Mapping[str, Any]) -> str:
    if task == "bbb_martins":
        return (
            "direct_family_card_first_introduced_after_l1"
            if card.get("evidence_family") == "direct_brain_exposure"
            else ""
        )
    if task == "bioavailability_ma":
        return direct_like_bioavailability_reason(
            {
                "group_id": "indirect-audit",
                "canonical_endpoint_name": card.get("endpoint"),
                "canonical_measurement_text": card.get("reported_value"),
                # Aggregated card units may summarize other records from the
                # same assay×molecule.  Only the card's own value/support can
                # establish that this representative record reports F.
                "canonical_unit_text": "",
                "support_text": card.get("support_text"),
            }
        )
    return direct_outcome_reason(
        {
            "canonical_assay_context": card.get("assay_context"),
            "canonical_measurement_text": card.get("reported_value"),
            "support_text": card.get("support_text"),
        }
    )


def analyze_task(
    task: str,
    old_root: Path,
    new_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    old_records = _query_rows(old_root, task)
    new_records = _query_rows(new_root, task)
    old_by_key = _identity_index(old_records)
    new_by_key = _identity_index(new_records)
    shared_keys = sorted(set(old_by_key) & set(new_by_key))
    rows = []
    earliest_counts: Counter[str] = Counter()
    changed_level_counts: Counter[str] = Counter()
    later_direct_leaks = []
    for key in shared_keys:
        old_query_index = old_by_key[key]
        query_index = new_by_key[key]
        level_rows = []
        earliest: int | None = None
        for level in range(1, TASK_LEVELS[task] + 1):
            old = _prepared(old_root, task, old_query_index, level)
            new = _prepared(new_root, task, query_index, level)
            old_active = _semantic_active(old.get("active_evidence") or {})
            new_active = _semantic_active(new.get("active_evidence") or {})
            old_cards = _card_locations(old_active)
            new_cards = _card_locations(new_active)
            old_card_ids = set(old_cards)
            new_card_ids = set(new_cards)
            changed_payload_ids = sorted(
                card_id
                for card_id in old_card_ids & new_card_ids
                if old_cards[card_id] != new_cards[card_id]
            )
            changed = (
                old_active != new_active
                or set(map(str, old.get("new_card_ids") or []))
                != set(map(str, new.get("new_card_ids") or []))
            )
            if changed and earliest is None:
                earliest = level
            if changed:
                changed_level_counts[str(level)] += 1
            new_this_level = set(map(str, new.get("new_card_ids") or []))
            if level > 1:
                for card_id in sorted(new_this_level):
                    location = new_cards.get(card_id)
                    if not location:
                        continue
                    analog_id, card = location
                    reason = _looks_direct(task, card)
                    if reason:
                        later_direct_leaks.append(
                            {
                                "task": task,
                                "query_index": query_index,
                                "level": level,
                                "analog_id": analog_id,
                                "card_id": card_id,
                                "reason": reason,
                                "card": card,
                            }
                        )
            level_rows.append(
                {
                    "level": level,
                    "changed": changed,
                    "old_molecules": len(old_active),
                    "new_molecules": len(new_active),
                    "old_cards": len(old_card_ids),
                    "new_cards": len(new_card_ids),
                    "added_molecule_ids": sorted(set(new_active) - set(old_active)),
                    "removed_molecule_ids": sorted(set(old_active) - set(new_active)),
                    "added_card_ids": sorted(new_card_ids - old_card_ids),
                    "removed_card_ids": sorted(old_card_ids - new_card_ids),
                    "changed_card_payload_ids": changed_payload_ids,
                    "old_new_card_ids": sorted(map(str, old.get("new_card_ids") or [])),
                    "new_new_card_ids": sorted(map(str, new.get("new_card_ids") or [])),
                }
            )
        earliest_key = str(earliest) if earliest is not None else "unchanged"
        earliest_counts[earliest_key] += 1
        rows.append(
            {
                "task": task,
                "query_index": query_index,
                "old_query_index": old_query_index,
                "benchmark_row_id": new_records[query_index].get("benchmark_row_id"),
                "molecule_identity_key": key[0],
                "condition_group": key[1],
                "earliest_changed_level": earliest,
                "requires_progressive_rerun": earliest is not None,
                "levels": level_rows,
            }
        )

    output_dir = output_root / task
    write_jsonl_atomic(output_dir / "query_retrieval_changes.jsonl", rows)
    write_jsonl_atomic(output_dir / "later_level_direct_leaks.jsonl", later_direct_leaks)
    summary = {
        "version": VERSION,
        "task": task,
        "old_root": str(old_root.resolve()),
        "new_root": str(new_root.resolve()),
        "n_queries": len(rows),
        "n_old_queries": len(old_records),
        "n_new_queries": len(new_records),
        "n_removed_query_identities": len(set(old_by_key) - set(new_by_key)),
        "n_added_query_identities": len(set(new_by_key) - set(old_by_key)),
        "n_queries_changed": sum(row["requires_progressive_rerun"] for row in rows),
        "n_queries_unchanged": sum(not row["requires_progressive_rerun"] for row in rows),
        "earliest_changed_level_counts": dict(sorted(earliest_counts.items())),
        "changed_query_counts_by_level": dict(sorted(changed_level_counts.items())),
        "n_direct_like_cards_first_introduced_after_l1": len(later_direct_leaks),
        "query_changes": str((output_dir / "query_retrieval_changes.jsonl").resolve()),
        "later_level_direct_leaks": str(
            (output_dir / "later_level_direct_leaks.jsonl").resolve()
        ),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", default=str(DEFAULT_OLD_ROOT))
    parser.add_argument("--new-root", default=str(DEFAULT_NEW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(TASK_LEVELS), default=sorted(TASK_LEVELS)
    )
    args = parser.parse_args(argv)
    summaries = [
        analyze_task(
            task,
            Path(args.old_root),
            Path(args.new_root),
            Path(args.output_root),
        )
        for task in args.tasks
    ]
    write_json_atomic(
        Path(args.output_root) / "summary.json",
        {"version": VERSION, "tasks": summaries},
    )
    for row in summaries:
        print(
            f"{row['task']}: changed={row['n_queries_changed']}/{row['n_queries']} "
            f"later_direct_leaks={row['n_direct_like_cards_first_introduced_after_l1']}"
        )
    return int(any(row["n_direct_like_cards_first_introduced_after_l1"] for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
