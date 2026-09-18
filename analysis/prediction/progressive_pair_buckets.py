"""Summarize pair buckets selected by cache-matched progressive retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any

import pyarrow.parquet as pq

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.retrieval.assay_reranking.cache_matched import load_cache_policy
from predict.retrieval.assay_reranking.cache_matched_v2 import load_candidates
from predict.utils.json import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
TASK_LEVELS = {"bbb_martins": 5, "bioavailability_ma": 6}
MODES = ("morgan", "assay-transfer")
POOLS = {"all": "all", "assay-transfer-trained": "tool-accepted"}
DEFAULT_CACHE_CONFIG = ROOT / "predict/retrieval/assay_reranking/cache_matched_retrieval_v2.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/analysis/prediction"
    / "reranked_progressive_v8_pair_bucket_distribution_records25_20260909"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _pair_bucket_lookup(task: str) -> tuple[dict[tuple[str, str], str], Path, Path]:
    stage = ROOT / f"data/evidence_libraries/{task}/v10/03_pair_buckets"
    records_path = stage / "records.parquet"
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["outputs"]["records.parquet"]
    actual = _sha256(records_path)
    if actual != expected:
        raise ValueError(f"V10 pair-bucket records differ from their manifest: {records_path}")

    table = pq.read_table(
        records_path,
        columns=["source_row_uid", "canonical_record_id", "pair_bucket_key"],
    )
    lookup: dict[tuple[str, str], str] = {}
    for row in table.to_pylist():
        key = (str(row["source_row_uid"]), str(row["canonical_record_id"]))
        bucket = str(row["pair_bucket_key"] or "")
        if not all(key) or not bucket or key in lookup:
            raise ValueError(f"Invalid or duplicate V10 record identity: {key}")
        lookup[key] = bucket
    return lookup, records_path, manifest_path


def _stage_ranker(mode: str, level: int) -> str:
    return "morgan" if mode == "morgan" or level == 5 else "assay_transfer"


def _effective_pool(requested_pool: str, level: int) -> str:
    return "fixed" if level in {1, 5} else POOLS[requested_pool]


def _selected_records(
    *, task: str, mode: str, requested_pool: str, cache_config: Path
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    query_path = split_path(task, "valid").with_name(
        "valid_molecule_condition_labels.jsonl"
    )
    queries = {
        str(row["benchmark_row_id"]): str(row["drug"])
        for row in read_jsonl(query_path)
    }
    policy = load_cache_policy(
        cache_config, task, "valid", mode, max_level=TASK_LEVELS[task]
    )
    molecules, later, audit = load_candidates(
        queries,
        task=task,
        subset="valid",
        policy=policy,
        molecule_limit=10,
        l1_limit=10,
        later_limit=25,
        tie_seed=0,
        cache_pool=POOLS[requested_pool],
    )
    selected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for query_id in queries:
        selected[query_id] = {
            "L1": [
                record
                for molecule in molecules[query_id]
                for record in molecule["l1_records"]
            ],
            **{
                level: list(bundle["records"])
                for level, bundle in later[query_id].items()
            },
        }
    return selected, {
        "query_path": str(query_path.resolve()),
        "query_sha256": _sha256(query_path),
        "cache_manifest": audit["cache_version"],
        "cache_content_id": audit["cache_content_id"],
    }


def analyze(cache_config: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_query: list[dict[str, Any]] = []
    frequencies: dict[tuple[str, str, str, int, str, str, str], Counter[str]] = defaultdict(Counter)
    provenance: dict[str, Any] = {}

    for task, last_level in TASK_LEVELS.items():
        bucket_lookup, records_path, stage_manifest = _pair_bucket_lookup(task)
        provenance[task] = {
            "pair_bucket_records": str(records_path.resolve()),
            "pair_bucket_records_sha256": _sha256(records_path),
            "pair_bucket_manifest": str(stage_manifest.resolve()),
            "pair_bucket_manifest_sha256": _sha256(stage_manifest),
            "conditions": {},
        }
        for mode in MODES:
            for requested_pool in POOLS:
                selected, inputs = _selected_records(
                    task=task,
                    mode=mode,
                    requested_pool=requested_pool,
                    cache_config=cache_config,
                )
                provenance[task]["conditions"][f"{mode}:{requested_pool}"] = inputs
                for query_id, levels in selected.items():
                    for level in range(1, last_level + 1):
                        records = levels[f"L{level}"]
                        buckets: list[str] = []
                        for record in records:
                            payload = record["payload"]
                            identity = (
                                str(payload.get("source_row_uid") or ""),
                                str(record["record_id"]),
                            )
                            try:
                                buckets.append(bucket_lookup[identity])
                            except KeyError as error:
                                raise ValueError(
                                    f"Selected record is absent from V10: {task}/{query_id}/L{level}/{identity}"
                                ) from error

                        ranker = _stage_ranker(mode, level)
                        effective_pool = _effective_pool(requested_pool, level)
                        per_query.append(
                            {
                                "task": task,
                                "requested_pool": requested_pool,
                                "effective_pool": effective_pool,
                                "run_mode": mode,
                                "stage_ranker": ranker,
                                "level": f"L{level}",
                                "benchmark_row_id": query_id,
                                "selected_records": len(records),
                                "unique_pair_buckets": len(set(buckets)),
                            }
                        )
                        group = (task, requested_pool, effective_pool, level, mode, ranker, query_id)
                        frequencies[group].update(buckets)

    grouped_queries: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        key = tuple(str(row[field]) for field in (
            "task", "requested_pool", "effective_pool", "run_mode", "stage_ranker", "level"
        ))
        grouped_queries[key].append(row)

    summary: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    ordered_groups = sorted(
        grouped_queries.items(),
        key=lambda item: (
            list(TASK_LEVELS).index(item[0][0]),
            list(POOLS).index(item[0][1]),
            int(item[0][5][1:]),
            list(MODES).index(item[0][3]),
        ),
    )
    for key, rows in ordered_groups:
        task, requested_pool, effective_pool, mode, ranker, level_name = key
        level = int(level_name[1:])
        records_per_query = [int(row["selected_records"]) for row in rows]
        buckets_per_query = [int(row["unique_pair_buckets"]) for row in rows]
        group_counts = Counter()
        queries_by_bucket: Counter[str] = Counter()
        for query_id in (str(row["benchmark_row_id"]) for row in rows):
            counts = frequencies[(task, requested_pool, effective_pool, level, mode, ranker, query_id)]
            group_counts.update(counts)
            queries_by_bucket.update(counts.keys())
        total_records = sum(group_counts.values())
        summary.append(
            {
                "task": task,
                "requested_pool": requested_pool,
                "effective_pool": effective_pool,
                "run_mode": mode,
                "stage_ranker": ranker,
                "level": level_name,
                "queries": len(rows),
                "selected_records_total": total_records,
                "selected_records_per_query_mean": round(fmean(records_per_query), 4),
                "selected_records_per_query_min": min(records_per_query),
                "selected_records_per_query_max": max(records_per_query),
                "unique_pair_buckets_global": len(group_counts),
                "unique_pair_buckets_per_query_mean": round(fmean(buckets_per_query), 4),
                "unique_pair_buckets_per_query_p25": round(_quantile(buckets_per_query, 0.25), 4),
                "unique_pair_buckets_per_query_median": round(median(buckets_per_query), 4),
                "unique_pair_buckets_per_query_p75": round(_quantile(buckets_per_query, 0.75), 4),
                "unique_pair_buckets_per_query_min": min(buckets_per_query),
                "unique_pair_buckets_per_query_max": max(buckets_per_query),
            }
        )
        for bucket, count in group_counts.most_common():
            identity = json.loads(bucket)
            if not isinstance(identity, list) or len(identity) < 3:
                raise ValueError(f"Invalid pair-bucket identity: {bucket}")
            frequency_rows.append(
                {
                    "task": task,
                    "requested_pool": requested_pool,
                    "effective_pool": effective_pool,
                    "run_mode": mode,
                    "stage_ranker": ranker,
                    "level": level_name,
                    "pair_bucket_key": bucket,
                    "source_id": identity[0],
                    "endpoint": identity[1],
                    "unit": identity[2],
                    "context_dimensions_json": json.dumps(identity[3:], separators=(",", ":")),
                    "selected_records": count,
                    "selected_record_fraction": round(count / total_records, 8),
                    "queries_retrieving_bucket": queries_by_bucket[bucket],
                    "query_fraction": round(queries_by_bucket[bucket] / len(rows), 8),
                }
            )
    return summary, per_query, frequency_rows, provenance


def _report(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Progressive retrieval pair-bucket distribution",
        "",
        "This analysis counts records introduced at each level independently. It compares the 25-record Morgan and assay-transfer runs on the validation split for the `all` and `assay-transfer-trained` pools.",
        "",
        "A pair bucket is a source-scoped experimental identity (source, endpoint, unit, and task-specific context), not necessarily a literal ChEMBL assay ID.",
        "",
        "## Mean unique pair buckets per query",
        "",
        "| Task | Pool | Mode | Ranker | Level | Mean | Median | Global unique |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {task} | {requested_pool} | {run_mode} | {stage_ranker} | {level} | "
            "{unique_pair_buckets_per_query_mean:.2f} | {unique_pair_buckets_per_query_median:.2f} | "
            "{unique_pair_buckets_global} |".format(**row)
        )
    lines.extend(
        [
            "",
            "L1 and L5 use the cache's fixed pool. L5 is Morgan-ranked in both run modes; duplicated rows are retained so the effective retrieval contract remains explicit.",
            "",
            "See `pair_bucket_frequency.tsv` for the full bucket distribution and `pair_bucket_per_query.tsv` for the underlying per-query counts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-config", type=Path, default=DEFAULT_CACHE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cache_config = args.cache_config.resolve()
    output = args.output_dir.resolve()
    summary, per_query, frequencies, provenance = analyze(cache_config)
    output.mkdir(parents=True, exist_ok=False)

    summary_path = output / "pair_bucket_summary.tsv"
    per_query_path = output / "pair_bucket_per_query.tsv"
    frequency_path = output / "pair_bucket_frequency.tsv"
    report_path = output / "REPORT.md"
    _write_tsv(summary_path, summary)
    _write_tsv(per_query_path, per_query)
    _write_tsv(frequency_path, frequencies)
    report_path.write_text(_report(summary), encoding="utf-8")

    manifest = {
        "version": "progressive_pair_bucket_distribution.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "tasks": list(TASK_LEVELS),
            "subset": "valid",
            "run_modes": list(MODES),
            "requested_pools": list(POOLS),
            "l1_molecules": 10,
            "l1_records_per_molecule": 10,
            "later_records_per_level": 25,
            "level_view": "records_introduced_at_each_level",
        },
        "cache_config": str(cache_config),
        "cache_config_sha256": _sha256(cache_config),
        "inputs": provenance,
        "outputs": {
            path.name: _sha256(path)
            for path in (summary_path, per_query_path, frequency_path, report_path)
        },
        "validation": {
            "selected_records_mapped_exactly_once": True,
            "frequency_counts_match_selected_totals": True,
            "status": "ok",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
