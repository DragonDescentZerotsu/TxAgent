"""Audit and sample the experimental meaningful-CNS-access BBB gold lineage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    CONTRACT_VERSION,
    SOURCE_DATASET,
    SOURCE_REVISION,
    label_record,
)


DEFAULT_DATA_ROOT = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v2"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.bbb_revision != SOURCE_REVISION:
        raise ValueError(
            "The source audit is bound to the frozen BBB revision "
            f"{SOURCE_REVISION}."
        )
    data_root = Path(args.data_root)
    task_root = data_root / "BBB_Martins"
    if args.mark_existing_review:
        return _mark_existing_review(task_root, args)
    summary = json.loads((task_root / "summary.json").read_text(encoding="utf-8"))
    dataset = load_dataset(
        SOURCE_DATASET,
        split="train",
        revision=args.bbb_revision,
    )
    accepted_counts: Counter[tuple[str, int]] = Counter()
    samples: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(dataset):
        label, method = label_record(row, source_index=index)
        if label is None:
            continue
        family = method.split(":", maxsplit=2)[1]
        accepted_counts[(family, label)] += 1
        sample = {
            "source_record_id": f"row:{index}",
            "pmid": str(row.get("pmid") or ""),
            "label": label,
            "label_method": method,
            "bbb_permeability_label": row.get("bbb_permeability_label"),
            "quant_metric": row.get("quant_metric"),
            "quant_value": row.get("quant_value"),
            "quant_units": row.get("quant_units"),
            "assay_model": row.get("assay_model"),
            "species": row.get("species"),
            "support_text": row.get("support_text"),
            "extra_details": row.get("extra_details"),
        }
        stable_rank = hashlib.sha256(
            f"{args.sample_seed}\0{index}".encode("utf-8")
        ).hexdigest()
        bucket = samples[(family, label)]
        bucket.append((stable_rank, sample))
        bucket.sort(key=lambda item: item[0])
        del bucket[args.samples_per_family_label :]

    accepted_total = sum(accepted_counts.values())
    expected_total = int(summary["n_source_rows_labeled_before_structure_normalization"])
    if accepted_total != expected_total:
        raise RuntimeError(
            f"accepted source-row mismatch: replay={accepted_total}, summary={expected_total}"
        )
    split = summary["splits"]["scaffold"]
    overlap_failures = sum(split["pairwise_identity_overlap"].values()) + sum(
        split["pairwise_scaffold_overlap"].values()
    )
    if overlap_failures:
        raise RuntimeError(f"split overlap audit failed: {overlap_failures}")

    sample_rows = [
        row
        for key in sorted(samples)
        for _, row in sorted(samples[key], key=lambda item: item[0])
    ]
    audit = {
        "schema_version": "bbb_experimental_meaningful_cns_access_gold_audit.v2",
        "gold_contract_version": CONTRACT_VERSION,
        "build_fingerprint": summary["build_fingerprint"],
        "source_dataset": SOURCE_DATASET,
        "source_revision": args.bbb_revision,
        "n_source_rows": len(dataset),
        "n_accepted_source_rows": accepted_total,
        "n_binary_parent_molecules": summary["n_binary_molecules"],
        "accepted_source_rows_by_family_and_label": {
            family: {
                "Y=0": accepted_counts[(family, 0)],
                "Y=1": accepted_counts[(family, 1)],
            }
            for family in sorted({family for family, _ in accepted_counts})
        },
        "split_gate": {
            "train": split["n_train"],
            "valid": split["n_valid"],
            "test": split["n_test"],
            "identity_overlap": split["pairwise_identity_overlap"],
            "scaffold_overlap": split["pairwise_scaffold_overlap"],
            "valid_record_tiers": split["valid_record_tier_counts"],
            "test_record_tiers": split["test_record_tier_counts"],
        },
        "qa_sample": {
            "seed": args.sample_seed,
            "samples_per_family_label": args.samples_per_family_label,
            "n_rows": len(sample_rows),
            "path": str(task_root / "audit" / "source_review_sample.jsonl"),
            "status": args.manual_review_status,
            "review_note": args.manual_review_note,
        },
    }
    audit_root = task_root / "audit"
    write_json_atomic(audit_root / "summary.json", audit)
    write_jsonl_atomic(audit_root / "source_review_sample.jsonl", sample_rows)
    (audit_root / "report.md").write_text(_render_report(audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def _mark_existing_review(task_root: Path, args: argparse.Namespace) -> int:
    if args.manual_review_status == "pending":
        raise ValueError("--mark-existing-review requires passed or failed status")
    audit_root = task_root / "audit"
    audit = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    build_summary = json.loads((task_root / "summary.json").read_text(encoding="utf-8"))
    if audit.get("build_fingerprint") != build_summary.get("build_fingerprint"):
        raise RuntimeError("existing QA sample belongs to a different build fingerprint")
    sample_path = audit_root / "source_review_sample.jsonl"
    sample_count = sum(bool(line.strip()) for line in sample_path.read_text().splitlines())
    if sample_count != int(audit["qa_sample"]["n_rows"]):
        raise RuntimeError("existing QA sample count does not match audit summary")
    audit["qa_sample"].update(
        {
            "status": args.manual_review_status,
            "review_note": args.manual_review_note,
        }
    )
    write_json_atomic(audit_root / "summary.json", audit)
    (audit_root / "report.md").write_text(_render_report(audit), encoding="utf-8")
    print(json.dumps(audit["qa_sample"], ensure_ascii=False, indent=2))
    return 0


def _render_report(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BBB experimental meaningful-CNS-access gold audit",
            "",
            f"- source rows: {audit['n_source_rows']:,}",
            f"- accepted experimental rows: {audit['n_accepted_source_rows']:,}",
            f"- binary parent molecules: {audit['n_binary_parent_molecules']:,}",
            "- identity/scaffold overlap: 0",
            f"- deterministic source-review sample: "
            f"{audit['qa_sample']['status']}",
            "",
            "The source-review sample is stratified by endpoint family and binary label. "
            "Review uses meaningful or adequate CNS access versus restricted or poor access; "
            "mere detectability is not sufficient for a positive label. Its status must be "
            "changed only after reading the stored source fields; automatic contract replay "
            "is not a substitute for manual semantic review.",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--bbb-revision", default=SOURCE_REVISION)
    parser.add_argument("--samples-per-family-label", type=int, default=25)
    parser.add_argument("--sample-seed", type=int, default=20260809)
    parser.add_argument(
        "--manual-review-status",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    parser.add_argument("--manual-review-note", default="")
    parser.add_argument("--mark-existing-review", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
