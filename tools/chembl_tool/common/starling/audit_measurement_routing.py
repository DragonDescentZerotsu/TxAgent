"""Report how deterministic routing partitions a task's cleaned rows.

Two questions this answers before any extraction spend:

1. **Where does every row go?**  The three buckets must sum to the row count, per
   source.  A drift from the reviewed numbers means the rule set changed.
2. **Which rows are discarded on the strength of a column?**  A declarative
   non-scalar rule that fires on a row *carrying a number* is the only place this
   pipeline drops a quantity without an extraction pass behind it.  Those rows are
   sampled here for manual review.  BBB declares no such rule, so the sample is
   normally empty; the no-digit rule needs no review because it is mechanically
   airtight.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    NO_DIGIT_RULE_ID,
    ROUTE_BUCKETS,
    SourceRoutingRules,
    route,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


AUDIT_VERSION = "starling_measurement_routing_audit.v1"
DEFAULT_REVIEW_SAMPLE = 200
#: Fixed so a re-run reproduces the same review sample.
REVIEW_SAMPLE_SEED = 20260820


def _review_columns(rules: SourceRoutingRules) -> list[str]:
    fields = [rules.measurement_field]
    if rules.unit_field:
        fields.append(rules.unit_field)
    fields.extend(rule.field for rule in rules.declarative_non_scalar)
    return fields


def audit_task(
    records_path: Path,
    rules_by_source: dict[str, SourceRoutingRules],
    *,
    task: str | None = None,
    review_sample: int = DEFAULT_REVIEW_SAMPLE,
    extra_review_fields: tuple[str, ...] = ("endpoint_name", "support_text"),
) -> dict[str, Any]:
    available = set(pq.read_schema(records_path).names)
    wanted = {"source_id", "cleaned_record_id"}
    for rules in rules_by_source.values():
        wanted.update(_review_columns(rules))
    wanted.update(extra_review_fields)
    columns = sorted(wanted & available)
    missing = sorted(wanted - available)

    buckets: dict[str, Counter] = {}
    rule_hits: dict[str, Counter] = {}
    # Reservoir per (source, declarative rule) so one pass suffices over a
    # multi-hundred-megabyte parquet.
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: Counter = Counter()
    rng = random.Random(REVIEW_SAMPLE_SEED)
    unknown_sources: Counter = Counter()

    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        for record in batch.to_pylist():
            source_id = str(record.get("source_id") or "")
            rules = rules_by_source.get(source_id)
            if rules is None:
                unknown_sources[source_id] += 1
                continue
            decision = route(record, rules, task=task)
            buckets.setdefault(source_id, Counter())[decision.bucket] += 1
            rule_hits.setdefault(source_id, Counter())[decision.rule_id or "unresolved"] += 1
            if decision.bucket != "reject" or decision.rule_id == NO_DIGIT_RULE_ID:
                continue
            key = (source_id, decision.rule_id or "")
            seen[key] += 1
            row = {field: record.get(field) for field in _review_columns(rules)}
            row.update(
                {
                    "cleaned_record_id": record.get("cleaned_record_id"),
                    "source_id": source_id,
                    "declarative_rule_id": decision.rule_id,
                    **{
                        field: record.get(field)
                        for field in extra_review_fields
                        if field in available
                    },
                }
            )
            pool = reservoirs.setdefault(key, [])
            if len(pool) < review_sample:
                pool.append(row)
            else:
                index = rng.randrange(seen[key])
                if index < review_sample:
                    pool[index] = row

    sources: dict[str, Any] = {}
    for source_id in sorted(buckets):
        counts = buckets[source_id]
        total = sum(counts.values())
        sources[source_id] = {
            "rows": total,
            "buckets": {bucket: counts.get(bucket, 0) for bucket in ROUTE_BUCKETS},
            "bucket_shares": {
                bucket: round(counts.get(bucket, 0) / total, 6) if total else 0.0
                for bucket in ROUTE_BUCKETS
            },
            "rule_hits": dict(rule_hits[source_id].most_common()),
            "declarative_rule_ids": [
                rule.rule_id for rule in rules_by_source[source_id].declarative_non_scalar
            ],
            "rows_discarded_by_declaration_despite_a_number": {
                rule_id: seen[(source_id, rule_id)]
                for rule_id in (
                    rule.rule_id
                    for rule in rules_by_source[source_id].declarative_non_scalar
                )
                if seen[(source_id, rule_id)]
            },
            "partition_is_exhaustive": total == sum(counts.values()),
        }
    llm_rows = sum(item["buckets"]["extract"] for item in sources.values())
    return {
        "audit_version": AUDIT_VERSION,
        "routing_version": MEASUREMENT_ROUTING_VERSION,
        "cleaned_records_path": str(records_path),
        "cleaned_records_sha256": file_sha256(records_path),
        "review_sample_seed": REVIEW_SAMPLE_SEED,
        "review_sample_size": review_sample,
        "columns_requested_but_absent": missing,
        "rows_from_undeclared_sources": dict(unknown_sources),
        "totals": {
            "rows": sum(item["rows"] for item in sources.values()),
            **{
                bucket: sum(item["buckets"][bucket] for item in sources.values())
                for bucket in ROUTE_BUCKETS
            },
            "extraction_rows": llm_rows,
        },
        "sources": sources,
        "rules": {
            source_id: rules.manifest()
            for source_id, rules in sorted(rules_by_source.items())
        },
        "_review_samples": {
            f"{source}|{rule_id}": rows for (source, rule_id), rows in reservoirs.items()
        },
    }


def _load_rules(task_id: str) -> tuple[dict[str, SourceRoutingRules], Path, int]:
    if task_id == "bbb_martins":
        from tools.chembl_tool.tasks.bbb_martins import starling_measurement_resolution as cfg

        return cfg.source_routing_rules(), cfg.DEFAULT_CLEANED_RECORDS, cfg.BATCH_SIZE
    raise ValueError(f"no measurement-routing rules declared for task {task_id!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["bbb_martins"])
    parser.add_argument("--cleaned-records", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--review-sample", type=int, default=DEFAULT_REVIEW_SAMPLE)
    parser.add_argument(
        "--review-out",
        type=Path,
        default=None,
        help="write the declarative-rule review rows as JSONL for manual audit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rules, default_records, batch_size = _load_rules(args.task)
    records_path = args.cleaned_records or default_records
    if not records_path.is_file():
        raise SystemExit(f"cleaned records not found: {records_path}")
    report = audit_task(
        records_path, rules, task=args.task, review_sample=args.review_sample
    )
    samples = report.pop("_review_samples")

    header = f"{'source':24}{'rows':>9}{'reject':>10}{'accept':>9}{'extract':>10}"
    print(header)
    for source_id, item in report["sources"].items():
        counts = item["buckets"]
        print(
            f"{source_id:24}{item['rows']:>9}"
            f"{counts['reject']:>10}{counts['accept']:>9}{counts['extract']:>10}"
        )
        for rule_id, hits in item[
            "rows_discarded_by_declaration_despite_a_number"
        ].items():
            print(f"    declaration over a number: {rule_id} -> {hits:,} rows")
    totals = report["totals"]
    print(
        f"{'TOTAL':24}{totals['rows']:>9}{totals['reject']:>10}"
        f"{totals['accept']:>9}{totals['extract']:>10}"
    )
    partition = totals["reject"] + totals["accept"] + totals["extract"]
    if partition != totals["rows"]:
        raise SystemExit(
            f"routing is not a partition: {partition} routed != {totals['rows']} rows"
        )
    if report["rows_from_undeclared_sources"]:
        raise SystemExit(
            f"undeclared sources present: {report['rows_from_undeclared_sources']}"
        )
    requests = -(-totals["extraction_rows"] // batch_size)
    print(
        f"\nextraction rows: {totals['extraction_rows']:,}"
        f"  ->  {requests:,} requests at batch_size={batch_size}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    if args.review_out:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        with args.review_out.open("w", encoding="utf-8") as handle:
            for key in sorted(samples):
                for row in samples[key]:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {args.review_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
