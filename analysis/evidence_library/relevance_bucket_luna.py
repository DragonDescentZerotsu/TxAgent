"""Rank every non-direct BBB and Skin progressive-V7 relevance bucket with Luna."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from analysis.evidence_library import relevance_bucket_diagnostic as calls
from analysis.evidence_library import relevance_bucket_rounds as ranking
from analysis.evidence_library import relevance_bucket_tournament as bbb
from analysis.evidence_library import skin_reaction_relevance_bucket_tournament as skin
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "non_direct_progressive_v7_relevance_bucket_luna.v1"
MODEL = "openai/gpt-5.6-luna"
BASE_URL = "https://openrouter.ai/api/v1"
REASONING_EFFORT = "medium"
DEGREE = 20
BATCH_SIZE = 5
INPUT_COST_PER_MILLION = 0.20
OUTPUT_COST_PER_MILLION = 1.20
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/analysis/evidence_library/relevance_bucket_luna_v7_progressive_v1"
)
PROVIDER_ROUTING = {
    "provider": {
        "only": ["openai"],
        "ignore": ["openai/fast"],
        "allow_fallbacks": True,
        "require_parameters": True,
    }
}
TASKS = {
    "bbb_martins": {
        "module": bbb,
        "input": Path(
            "data/evidence_libraries/bbb_martins/v7/03_pair_buckets/records.parquet"
        ),
        "sources": (
            "efflux_transport",
            "influx_transport",
            "passive_permeability",
        ),
        "template": Path(__file__).with_name("prompts")
        / "relevance_bucket_rounds_v4.jinja",
        "max_completion_tokens": 4_096,
    },
    "skin_reaction": {
        "module": skin,
        "input": Path(
            "data/evidence_libraries/skin_reaction/v7/03_pair_buckets/records.parquet"
        ),
        "sources": (
            "sensitization_aop",
            "phototoxicity_irritation_local_damage",
            "skin_exposure",
        ),
        "template": skin.TEMPLATE,
        "max_completion_tokens": 8_192,
    },
}


def _task(task: str) -> tuple[Any, tuple[str, ...], int]:
    config = TASKS[task]
    return (
        config["module"],
        tuple(config["sources"]),
        int(config["max_completion_tokens"]),
    )


def _complete_edges(keys: Sequence[str]) -> list[tuple[str, str]]:
    edges = []
    for left, a in enumerate(keys):
        for right in range(left + 1, len(keys)):
            b = keys[right]
            distance = right - left
            if distance < len(keys) / 2:
                edges.append((a, b))
            elif distance > len(keys) / 2:
                edges.append((b, a))
            else:
                edges.append((a, b) if left % 2 == 0 else (b, a))
    return edges


def comparison_edges(keys: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
    """Return deterministic source-local degree-20 graphs."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_source[json.loads(key)["source_id"]].append(key)
    output = {}
    for source, source_keys in sorted(by_source.items()):
        source_keys = sorted(source_keys)
        if len(source_keys) <= DEGREE + 1:
            output[source] = _complete_edges(source_keys)
            continue
        used: set[tuple[str, str]] = set()
        output[source] = [
            edge
            for cycle_index in range(DEGREE // 2)
            for edge in ranking._cycle(
                source_keys,
                f"luna-v1-{source}-{cycle_index}",
                used,
            )
        ]
    return output


def _insert_comparisons(
    connection: sqlite3.Connection,
    edges_by_source: Mapping[str, Sequence[tuple[str, str]]],
) -> None:
    for source, edges in edges_by_source.items():
        slug = source.replace("_", "-")
        for index, (a, b) in enumerate(edges):
            values = (
                f"full-{slug}-{index:06d}",
                "full",
                f"full-{slug}-batch-{index // BATCH_SIZE:06d}",
                a,
                b,
            )
            connection.execute(
                "INSERT OR IGNORE INTO comparisons VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                values,
            )
            stored = connection.execute(
                "SELECT comparison_id, phase, batch_id, bucket_a, bucket_b "
                "FROM comparisons WHERE comparison_id=?",
                (values[0],),
            ).fetchone()
            if tuple(stored) != values:
                raise ValueError(f"comparison schedule changed for {values[0]}")
    connection.commit()


def _graph_summary(
    edges_by_source: Mapping[str, Sequence[tuple[str, str]]],
) -> dict[str, Any]:
    output = {}
    for source, edges in edges_by_source.items():
        nodes = {key for edge in edges for key in edge}
        degree = Counter(key for edge in edges for key in edge)
        positions = Counter(a for a, _ in edges)
        output[source] = {
            "bucket_count": len(nodes),
            "edge_count": len(edges),
            "minimum_degree": min(degree.values()),
            "maximum_degree": max(degree.values()),
            "maximum_candidate_position_imbalance": max(
                abs(2 * positions[key] - degree[key]) for key in nodes
            ),
            "connected": bbb._connected([(a, b, "A") for a, b in edges]),
        }
    return output


def build(task: str, output_root: Path, *, rebuild: bool = False) -> dict[str, Any]:
    module, sources, max_completion_tokens = _task(task)
    input_path = Path(TASKS[task]["input"])
    template = Path(TASKS[task]["template"])
    for path in (input_path, template):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "requests.sqlite3"
    mapping_path = output_dir / "pair_bucket_relevance_map.parquet"
    manifest_path = output_dir / "manifest.json"
    input_hash = file_sha256(input_path)
    prompt_hash = file_sha256(template)
    if not rebuild and all(
        path.is_file() for path in (database_path, mapping_path, manifest_path)
    ):
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("version") == VERSION
            and prior.get("input_sha256") == input_hash
            and prior.get("prompt_sha256") == prompt_hash
        ):
            return prior
        raise RuntimeError(f"existing {task} Luna inventory targets different inputs")

    connection = bbb._connect(database_path)
    connection.execute("DELETE FROM buckets")
    connection.execute("DELETE FROM comparisons")
    connection.execute("DELETE FROM requests")
    available = set(pq.read_schema(input_path).names)
    columns = [column for column in module._input_columns() if column in available]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_bucket_edges: Counter[tuple[str, str, str]] = Counter()
    for batch in pq.ParquetFile(input_path).iter_batches(
        batch_size=20_000, columns=columns
    ):
        for record in batch.to_pylist():
            if (
                record.get("assay_transfer_eligible") is True
                and record.get("source_id") in sources
            ):
                key, _ = module.relevance_bucket(record)
                by_bucket[key].append(record)
                pair_bucket_edges[
                    (
                        str(record["source_id"]),
                        str(record["pair_bucket_key"]),
                        key,
                    )
                ] += 1

    mapping_rows = [
        {
            "source_id": source,
            "pair_bucket_key": pair_key,
            "relevance_bucket": key,
            "record_count": count,
        }
        for (source, pair_key, key), count in pair_bucket_edges.items()
    ]
    pair_variant_counts = Counter(
        (source, pair_key) for source, pair_key, _ in pair_bucket_edges
    )
    inconsistent_pairs = sum(count > 1 for count in pair_variant_counts.values())

    for key in sorted(by_bucket):
        records = by_bucket[key]
        _, identity = module.relevance_bucket(records[0])
        samples = module.select_diverse_records(records)
        if not samples:
            raise ValueError(f"{task} relevance bucket has no semantic sample: {key}")
        connection.execute(
            "INSERT INTO buckets VALUES (?, ?, ?, ?, ?, 1, NULL)",
            (
                key,
                identity["source_id"],
                key,
                bbb._canonical_json(samples),
                len(records),
            ),
        )
    connection.commit()
    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["source_id", "pair_bucket_key"]
    )
    mapping.to_parquet(mapping_path, index=False)
    keys = [
        row[0]
        for row in connection.execute(
            "SELECT relevance_bucket FROM buckets ORDER BY relevance_bucket"
        )
    ]
    edges = comparison_edges(keys)
    _insert_comparisons(connection, edges)
    calls._prepare_requests(
        connection, "full", template_path=template, model=MODEL
    )
    graph = _graph_summary(edges)
    for source, summary in graph.items():
        expected_degree = min(DEGREE, summary["bucket_count"] - 1)
        if (
            summary["minimum_degree"] != expected_degree
            or summary["maximum_degree"] != expected_degree
            or not summary["connected"]
            or summary["maximum_candidate_position_imbalance"] > 1
        ):
            raise ValueError(f"invalid {task} {source} comparison graph: {summary}")
    manifest = {
        "version": VERSION,
        "status": "initialized",
        "task": task,
        "ranking_universe": "progressive_v7_stage3_assay_transfer_eligible_non_direct",
        "input": str(input_path),
        "input_sha256": input_hash,
        "prompt": str(template),
        "prompt_sha256": prompt_hash,
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": max_completion_tokens,
        "batch_size_maximum": BATCH_SIZE,
        "target_degree": DEGREE,
        "sources": list(sources),
        "direct_sources_excluded": True,
        "winner_contract": "exact_bucket_id_strict_binary",
        "provider_routing": PROVIDER_ROUTING["provider"],
        "pricing_usd_per_million_tokens": {
            "input": INPUT_COST_PER_MILLION,
            "output": OUTPUT_COST_PER_MILLION,
        },
        "relevance_bucket_columns": {
            source: list(module.RELEVANCE_BUCKET_COLUMNS[source])
            for source in sources
        },
        "source_counts": {
            source: {
                "pair_buckets": int(
                    mapping.loc[
                        mapping.source_id == source, "pair_bucket_key"
                    ].nunique()
                ),
                "relevance_buckets": graph[source]["bucket_count"],
            }
            for source in sources
        },
        "relevance_bucket_count": len(keys),
        "pair_buckets_with_multiple_relevance_identities": inconsistent_pairs,
        "graph": graph,
        "artifacts": {
            "pair_bucket_relevance_map": str(mapping_path),
            "request_cache": str(database_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def _rankings(connection: sqlite3.Connection) -> pd.DataFrame:
    rows = []
    for source, keys in ranking._source_keys(
        [row[0] for row in connection.execute("SELECT relevance_bucket FROM buckets")]
    ).items():
        key_set = set(keys)
        outcomes = [
            (row[0], row[1], row[2])
            for row in connection.execute(
                "SELECT bucket_a, bucket_b, winner FROM comparisons "
                "WHERE phase='full' AND winner IS NOT NULL"
            )
            if row[0] in key_set
        ]
        scores = ranking.fit_bradley_terry(outcomes)
        ordered = sorted(keys, key=lambda key: (-scores[key], key))
        for index, key in enumerate(ordered):
            rows.append(
                {
                    "relevance_bucket": key,
                    "source_id": source,
                    "bradley_terry_score": scores[key],
                    "source_rank": index + 1,
                    "source_percentile": 100.0
                    * (len(ordered) - 1 - index)
                    / max(1, len(ordered) - 1),
                    "comparison_count": sum(key in outcome[:2] for outcome in outcomes),
                }
            )
    return pd.DataFrame(rows)


def _cost(connection: sqlite3.Connection) -> dict[str, Any]:
    input_tokens, output_tokens, completed, failed = connection.execute(
        "SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0), "
        "sum(status='complete'), sum(status='failed') FROM requests"
    ).fetchone()
    cost = (
        int(input_tokens) * INPUT_COST_PER_MILLION
        + int(output_tokens) * OUTPUT_COST_PER_MILLION
    ) / 1_000_000
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "actual_tokens": int(input_tokens + output_tokens),
        "completed_requests": int(completed),
        "failed_requests": int(failed),
        "reported_cost_usd": round(cost, 6),
    }


def run(
    tasks: Sequence[str],
    output_root: Path,
    *,
    parallelism: int,
    max_cost_usd: float,
    env_file: Path | None,
    credential_env: str,
    approved_prompt_hashes: Mapping[str, str],
) -> dict[str, Any]:
    results = {}
    for task in tasks:
        manifest = build(task, output_root)
        if approved_prompt_hashes.get(task) != manifest["prompt_sha256"]:
            raise ValueError(f"approved prompt hash does not match {task}")
        connection = bbb._connect(output_root / task / "requests.sqlite3")
        manifest["provider_routing"] = PROVIDER_ROUTING["provider"]
        manifest["accepted_reason_word_limit"] = 24
        write_json_atomic(output_root / task / "manifest.json", manifest)
        spent_on_other_tasks = sum(
            float(usage["reported_cost_usd"]) for usage in results.values()
        )
        remaining = max_cost_usd - spent_on_other_tasks
        if remaining <= 0:
            raise RuntimeError("combined Luna dollar budget is exhausted")
        _, _, max_completion_tokens = _task(task)
        try:
            calls.run_condition(
                connection,
                "full",
                parallelism=parallelism,
                token_budget=100_000_000,
                env_file=env_file,
                template_path=Path(manifest["prompt"]),
                allow_ties=False,
                credential_env=credential_env,
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                max_completion_tokens=max_completion_tokens,
                max_tokens_parameter="max_tokens",
                base_url=BASE_URL,
                provider="openrouter",
                request_extra_body=PROVIDER_ROUTING,
                max_cost_usd=remaining,
                input_cost_per_million=INPUT_COST_PER_MILLION,
                output_cost_per_million=OUTPUT_COST_PER_MILLION,
                max_reason_words=24,
            )
        except Exception:
            manifest.update({"status": "incomplete", "usage": _cost(connection)})
            write_json_atomic(output_root / task / "manifest.json", manifest)
            connection.close()
            raise
        incomplete = connection.execute(
            "SELECT count(*) FROM comparisons WHERE winner IS NULL"
        ).fetchone()[0]
        if incomplete:
            raise RuntimeError(f"{task} has {incomplete} incomplete comparisons")
        rankings = _rankings(connection)
        if len(rankings) != manifest["relevance_bucket_count"]:
            raise ValueError(f"{task} final ranking coverage mismatch")
        ranking_path = output_root / task / "relevance_bucket_rankings.parquet"
        rankings.to_parquet(ranking_path, index=False)
        usage = _cost(connection)
        manifest.update(
            {
                "status": "complete",
                "usage": usage,
                "served_models": {
                    row[0]: int(row[1])
                    for row in connection.execute(
                        "SELECT served_model, count(*) FROM requests "
                        "WHERE status='complete' GROUP BY served_model"
                    )
                },
                "ranking": str(ranking_path),
                "ranking_sha256": file_sha256(ranking_path),
            }
        )
        write_json_atomic(output_root / task / "manifest.json", manifest)
        connection.close()
        results[task] = usage
    total_cost = sum(
        float(usage["reported_cost_usd"]) for usage in results.values()
    )
    summary = {
        "version": VERSION,
        "status": "complete",
        "tasks": list(tasks),
        "max_cost_usd": max_cost_usd,
        "reported_cost_usd": round(total_cost, 6),
        "task_usage": results,
    }
    write_json_atomic(output_root / "manifest.json", summary)
    return summary


def show_prompt(task: str, output_root: Path) -> str:
    manifest = build(task, output_root)
    connection = bbb._connect(output_root / task / "requests.sqlite3")
    batch_id = connection.execute(
        "SELECT batch_id FROM requests ORDER BY batch_id LIMIT 1"
    ).fetchone()[0]
    prompt = calls.render_request(
        connection, batch_id, template_path=Path(manifest["prompt"])
    )
    connection.close()
    return prompt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "show-prompt", "run"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--max-cost-usd", type=float, default=25.0)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key-env", default="OPEN_ROUTER_KEY")
    parser.add_argument("--approved-bbb-prompt-sha256")
    parser.add_argument("--approved-skin-prompt-sha256")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.parallelism < 1 or args.max_cost_usd <= 0:
        raise ValueError("parallelism and max cost must be positive")
    if args.command == "show-prompt":
        if len(args.tasks) != 1:
            raise ValueError("show-prompt requires exactly one task")
        print(show_prompt(args.tasks[0], args.output_root))
        return
    if args.command == "build":
        result = {
            task: build(task, args.output_root, rebuild=args.rebuild)
            for task in args.tasks
        }
    else:
        approvals = {
            "bbb_martins": args.approved_bbb_prompt_sha256,
            "skin_reaction": args.approved_skin_prompt_sha256,
        }
        result = run(
            args.tasks,
            args.output_root,
            parallelism=args.parallelism,
            max_cost_usd=args.max_cost_usd,
            env_file=args.env_file,
            credential_env=args.api_key_env,
            approved_prompt_hashes=approvals,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
