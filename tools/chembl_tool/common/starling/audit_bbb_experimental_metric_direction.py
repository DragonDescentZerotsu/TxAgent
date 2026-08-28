"""Audit conservative direction recovery on the frozen BBB source revision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.chembl_tool.common.json_utils import atomic_output_path, write_json_atomic
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    SOURCE_DATASET,
    SOURCE_REVISION,
    classify_scope,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v3 import (
    CONTRACT_VERSION as V3_CONTRACT_VERSION,
    label_record as label_record_v3,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_metric_direction_review import (
    MANUAL_SOURCE_EXCLUSIONS,
    REVIEW_VERSION,
    REVIEWED_PROPOSED_SOURCE_INDICES,
    adjudicate_missing_direction,
    review_missing_direction,
)


DEFAULT_OUTPUT_DIR = Path(
    "data/starling_data/bbb_martins/experimental_metric_direction_review_v1"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows, source_meta = _load_rows(args.source_arrow)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    candidate_rules: Counter[str] = Counter()
    candidate_labels: Counter[str] = Counter()
    proposed_rules: Counter[str] = Counter()
    proposed_labels: Counter[str] = Counter()
    candidate_endpoints: Counter[str] = Counter()
    proposed_source_indices: set[int] = set()
    validation: dict[str, Counter[str]] = defaultdict(Counter)

    for source_index, record in enumerate(rows):
        if args.max_rows and source_index >= args.max_rows:
            break
        v3_label, v3_reason = label_record_v3(record, source_index=source_index)
        if v3_reason == "no_explicit_binary_permeability_label":
            scope = classify_scope(record)
            proposed = review_missing_direction(record)
            review = adjudicate_missing_direction(record, source_index=source_index)
            if proposed.label is not None:
                proposed_source_indices.add(source_index)
            proposed_rules[proposed.rule_id] += 1
            proposed_labels[_label_key(proposed.label)] += 1
            candidate_rules[review.rule_id] += 1
            candidate_labels[_label_key(review.label)] += 1
            candidate_endpoints[str(scope.endpoint_family)] += 1
            candidate_rows.append(
                {
                    "source_index": source_index,
                    "source_record_id": f"row:{source_index}",
                    "v3_rejection_reason": v3_reason,
                    "endpoint_family": scope.endpoint_family,
                    "experimental_basis": scope.basis,
                    "proposed_label": proposed.label,
                    "proposed_rule_id": proposed.rule_id,
                    "proposed_matched_text": proposed.matched_text,
                    "review_label": review.label,
                    "review_rule_id": review.rule_id,
                    "matched_text": review.matched_text,
                    **dict(record),
                }
            )
            continue
        if v3_label is None:
            continue

        proposed = review_missing_direction(record)
        if proposed.label is None:
            continue
        outcome = "agree" if proposed.label == v3_label else "disagree"
        validation[proposed.rule_id][outcome] += 1
        validation[proposed.rule_id]["total"] += 1
        if outcome == "disagree":
            scope = classify_scope(record)
            validation_rows.append(
                {
                    "source_index": source_index,
                    "source_record_id": f"row:{source_index}",
                    "endpoint_family": scope.endpoint_family,
                    "source_label": v3_label,
                    "source_label_method": v3_reason,
                    "review_label": proposed.label,
                    "review_rule_id": proposed.rule_id,
                    "matched_text": proposed.matched_text,
                    **dict(record),
                }
            )

    if not args.max_rows and proposed_source_indices != REVIEWED_PROPOSED_SOURCE_INDICES:
        missing = sorted(REVIEWED_PROPOSED_SOURCE_INDICES - proposed_source_indices)
        added = sorted(proposed_source_indices - REVIEWED_PROPOSED_SOURCE_INDICES)
        raise RuntimeError(
            "frozen BBB manual review does not match proposed candidates: "
            f"missing={missing}, added={added}"
        )

    candidate_path = output_dir / "record_review_ledger.jsonl"
    disagreement_path = output_dir / "explicit_label_disagreements.jsonl"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(disagreement_path, validation_rows)
    validation_summary = {
        rule: {
            **dict(counts),
            "agreement": counts["agree"] / counts["total"],
        }
        for rule, counts in sorted(validation.items())
    }
    summary = {
        "review_version": REVIEW_VERSION,
        "source": source_meta,
        "v3_contract_version": V3_CONTRACT_VERSION,
        "n_source_rows_considered": min(
            len(rows), args.max_rows if args.max_rows else len(rows)
        ),
        "n_missing_direction_rows": len(candidate_rows),
        "proposed_review_label_counts": dict(sorted(proposed_labels.items())),
        "proposed_rule_counts": dict(sorted(proposed_rules.items())),
        "candidate_review_label_counts": dict(sorted(candidate_labels.items())),
        "candidate_rule_counts": dict(sorted(candidate_rules.items())),
        "candidate_endpoint_counts": dict(sorted(candidate_endpoints.items())),
        "explicit_label_rule_validation": validation_summary,
        "n_explicit_label_disagreements": len(validation_rows),
        "manual_review": {
            "n_proposed_rows_reviewed": len(REVIEWED_PROPOSED_SOURCE_INDICES),
            "n_manual_exclusions": len(MANUAL_SOURCE_EXCLUSIONS),
            "n_manual_approvals": (
                len(REVIEWED_PROPOSED_SOURCE_INDICES) - len(MANUAL_SOURCE_EXCLUSIONS)
            ),
            "manual_exclusion_reasons": dict(
                Counter(MANUAL_SOURCE_EXCLUSIONS.values())
            ),
        },
        "review_policy": {
            "recovered_scope": (
                "only v3-eligible experimental outcome rows rejected solely for "
                "missing bbb_permeability_label"
            ),
            "direction_policy": (
                "recover only an unambiguous qualitative direction stated in the "
                "row itself; do not threshold heterogeneous numeric endpoints"
            ),
            "unresolved_policy": "retain as non-voting evidence",
        },
        "artifacts": {
            "record_review_ledger": str(candidate_path),
            "record_review_ledger_sha256": _sha256(candidate_path),
            "explicit_label_disagreements": str(disagreement_path),
            "explicit_label_disagreements_sha256": _sha256(disagreement_path),
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _load_rows(source_arrow: str) -> tuple[Iterable[Mapping[str, Any]], dict[str, Any]]:
    if source_arrow:
        from datasets import Dataset

        path = Path(source_arrow)
        return Dataset.from_file(str(path)), {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "local_arrow": str(path.resolve()),
            "local_arrow_sha256": _sha256(path),
        }

    from datasets import load_dataset
    from huggingface_hub import HfApi

    rows = load_dataset(SOURCE_DATASET, split="train", revision=SOURCE_REVISION)
    resolved = HfApi().dataset_info(SOURCE_DATASET, revision=SOURCE_REVISION).sha
    return rows, {
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "resolved_revision": resolved,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _label_key(label: int | None) -> str:
    return "unresolved" if label is None else str(label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-arrow", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
