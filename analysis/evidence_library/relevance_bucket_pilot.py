"""Run the outcome-blind, order-reconciled 512-bucket BBB ranking pilot."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from analysis.evidence_library import relevance_bucket_diagnostic as diagnostic
from analysis.evidence_library import relevance_bucket_tournament as base
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE
from tools.chembl_tool.common.json_utils import write_json_atomic

VERSION = "bbb_relevance_bucket_pilot.v3"
OUTPUT = Path("outputs/analysis/evidence_library/bbb_relevance_bucket_tournament_v3")
SOURCE = diagnostic.OUTPUT
TOTAL_TOKEN_BUDGET = 20_000_000
PRIMARY_BATCH_SIZE = 5


def comparison_edges(keys: list[str]) -> list[tuple[str, str]]:
    """Return a connected near-4-regular graph with 1,025 unique edges."""
    order = sorted(keys, key=lambda key: base._hash("v3-ring", key))
    edges = {
        tuple(sorted((order[index], order[(index + 1) % len(order)])))
        for index in range(len(order))
    }
    for matching_index in range(2):
        for attempt in range(10_000):
            shuffled = sorted(
                keys,
                key=lambda key: base._hash(
                    "v3-matching", str(matching_index), str(attempt), key
                ),
            )
            matching = {
                tuple(sorted((shuffled[index], shuffled[index + 1])))
                for index in range(0, len(shuffled), 2)
            }
            if len(matching) == len(keys) // 2 and not matching & edges:
                edges.update(matching)
                break
        else:
            raise RuntimeError("could not construct a non-overlapping matching")
    for a_index, a in enumerate(order):
        for b in order[a_index + 1 :]:
            edge = tuple(sorted((a, b)))
            if edge not in edges:
                edges.add(edge)
                break
        if len(edges) == 1_025:
            break
    degree = Counter(key for edge in edges for key in edge)
    if len(edges) != 1_025 or set(degree) != set(keys) or min(degree.values()) != 4:
        raise AssertionError("pilot graph does not satisfy its degree contract")
    return sorted(edges, key=lambda edge: base._hash("v3-edge", *edge))


def _insert_primary(
    connection: sqlite3.Connection, edges: list[tuple[str, str]], phase: str
) -> None:
    reverse = phase == "reverse"
    for index, edge in enumerate(edges):
        a, b = base._orient(*edge, f"v3-{index:04d}")
        if reverse:
            a, b = b, a
        connection.execute(
            "INSERT OR IGNORE INTO comparisons VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                f"{phase}-{index:04d}",
                phase,
                f"{phase}-batch-{index // PRIMARY_BATCH_SIZE:04d}",
                a,
                b,
                f"forward-{index:04d}" if reverse else None,
            ),
        )
    connection.commit()


def initialize(source_dir: Path, output_dir: Path, *, rebuild: bool = False) -> dict[str, Any]:
    source_database = source_dir / "requests.sqlite3"
    source_manifest = source_dir / "manifest.json"
    if not source_database.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("completed V2 diagnostic artifacts are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "requests.sqlite3"
    manifest_path = output_dir / "manifest.json"
    source_hash = file_sha256(source_manifest)
    if not rebuild and database.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("version") == VERSION and prior.get("source_manifest_sha256") == source_hash:
            return prior

    connection = base._connect(database)
    connection.execute("DELETE FROM comparisons")
    connection.execute("DELETE FROM requests")
    diagnostic._copy_buckets(base.OUTPUT / "requests.sqlite3", connection)
    keys = base._pilot_keys(connection)
    edges = comparison_edges(keys)
    _insert_primary(connection, edges, "forward")
    _insert_primary(connection, edges, "reverse")
    graph_outcomes = [(a, b, "tie") for a, b in edges]
    manifest = {
        "version": VERSION,
        "status": "initialized",
        "source_artifact": str(source_dir),
        "source_manifest_sha256": source_hash,
        "source_request_cache_sha256": file_sha256(source_database),
        "prompt": str(diagnostic.TEMPLATE),
        "prompt_sha256": file_sha256(diagnostic.TEMPLATE),
        "prompt_version": diagnostic.PROMPT_VERSION,
        "model": base.MODEL,
        "reasoning_effort": base.REASONING_EFFORT,
        "pilot_bucket_count": len(keys),
        "unique_edge_count": len(edges),
        "primary_batch_size": PRIMARY_BATCH_SIZE,
        "primary_judgments_per_edge": 2,
        "adjudication_batch_size": 1,
        "minimum_graph_degree": min(Counter(key for edge in edges for key in edge).values()),
        "maximum_graph_degree": max(Counter(key for edge in edges for key in edge).values()),
        "comparison_graph_connected": base._connected(graph_outcomes),
        "winner_contract": "exact_bucket_id_or_tie",
        "outcome_fields_withheld": sorted(diagnostic.OUTCOME_FIELDS),
        "full_run_started": False,
    }
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def _schedule_adjudication(connection: sqlite3.Connection) -> int:
    forward = {
        row["comparison_id"].rsplit("-", 1)[1]: row
        for row in connection.execute(
            "SELECT comparison_id, bucket_a, bucket_b, winner FROM comparisons "
            "WHERE phase='forward'"
        )
    }
    reverse = {
        row["comparison_id"].rsplit("-", 1)[1]: row
        for row in connection.execute(
            "SELECT comparison_id, winner FROM comparisons WHERE phase='reverse'"
        )
    }
    disagreements = [index for index in forward if forward[index]["winner"] != reverse[index]["winner"]]
    for index in disagreements:
        row = forward[index]
        a, b = base._orient(row["bucket_a"], row["bucket_b"], f"v3-adjudicate-{index}")
        connection.execute(
            "INSERT OR IGNORE INTO comparisons VALUES (?, 'adjudication', ?, ?, ?, ?, NULL, NULL)",
            (
                f"adjudication-{index}",
                f"adjudication-batch-{index}",
                a,
                b,
                f"forward-{index}",
            ),
        )
    connection.commit()
    return len(disagreements)


def _outcomes(rows: dict[str, sqlite3.Row]) -> list[tuple[str, str, str]]:
    output = []
    for row in rows.values():
        winner = "tie"
        if row["winner"] == row["bucket_a"]:
            winner = "A"
        elif row["winner"] == row["bucket_b"]:
            winner = "B"
        output.append((row["bucket_a"], row["bucket_b"], winner))
    return output


def summarize(connection: sqlite3.Connection, output_dir: Path) -> dict[str, Any]:
    phases = {}
    for phase in ("forward", "reverse", "adjudication"):
        phases[phase] = {
            row["comparison_id"].rsplit("-", 1)[1]: row
            for row in connection.execute(
                "SELECT comparison_id, bucket_a, bucket_b, winner, reason "
                "FROM comparisons WHERE phase=?",
                (phase,),
            )
        }
    agreement = {
        index: phases["forward"][index]["winner"] == phases["reverse"][index]["winner"]
        for index in phases["forward"]
    }
    resolved = {}
    rows = []
    for index, forward in phases["forward"].items():
        reverse = phases["reverse"][index]
        adjudication = phases["adjudication"].get(index)
        winner = forward["winner"] if agreement[index] else adjudication["winner"]
        resolved[index] = {
            "bucket_a": forward["bucket_a"],
            "bucket_b": forward["bucket_b"],
            "winner": winner,
        }
        rows.append(
            {
                "comparison_id": f"edge-{index}",
                "bucket_a": forward["bucket_a"],
                "bucket_b": forward["bucket_b"],
                "forward_winner": forward["winner"],
                "reverse_winner": reverse["winner"],
                "orders_agree": agreement[index],
                "adjudication_winner": adjudication["winner"] if adjudication else None,
                "resolved_winner": winner,
            }
        )
    pd.DataFrame(rows).to_parquet(output_dir / "resolved_comparisons.parquet", index=False)

    forward_scores = base.fit_davidson(_outcomes(phases["forward"]))
    reverse_scores = base.fit_davidson(_outcomes(phases["reverse"]))
    resolved_scores = base.fit_davidson(_outcomes(resolved))
    keys = sorted(resolved_scores)

    def ranks(scores: dict[str, float]) -> dict[str, int]:
        return {
            key: index
            for index, key in enumerate(sorted(keys, key=lambda item: scores[item]))
        }

    forward_rank, reverse_rank, resolved_rank = (
        ranks(forward_scores),
        ranks(reverse_scores),
        ranks(resolved_scores),
    )
    n = len(keys)
    ranking_rows = []
    within_one = []
    for key in keys:
        forward_decile = min(9, forward_rank[key] * 10 // n) + 1
        reverse_decile = min(9, reverse_rank[key] * 10 // n) + 1
        within_one.append(abs(forward_decile - reverse_decile) <= 1)
        ranking_rows.append(
            {
                "relevance_bucket": key,
                "source_id": json.loads(key)["source_id"],
                "davidson_score": resolved_scores[key],
                "rank": resolved_rank[key] + 1,
                "percentile": 100 * resolved_rank[key] / (n - 1),
                "decile": min(9, resolved_rank[key] * 10 // n) + 1,
                "forward_score": forward_scores[key],
                "reverse_score": reverse_scores[key],
                "forward_decile": forward_decile,
                "reverse_decile": reverse_decile,
            }
        )
    pd.DataFrame(ranking_rows).to_parquet(
        output_dir / "relevance_bucket_rankings.parquet", index=False
    )
    rho = float(
        spearmanr(
            [forward_rank[key] for key in keys],
            [reverse_rank[key] for key in keys],
        ).statistic
    )
    metrics = {
        "pilot_bucket_count": n,
        "unique_edge_count": len(resolved),
        "order_agreement": sum(agreement.values()) / len(agreement),
        "adjudicated_edge_count": sum(not value for value in agreement.values()),
        "forward_reverse_rank_spearman": rho,
        "forward_reverse_within_one_decile": sum(within_one) / len(within_one),
        "resolved_tie_rate": sum(row["winner"] == "tie" for row in resolved.values()) / len(resolved),
        "comparison_graph_connected": base._connected(_outcomes(resolved)),
    }
    metrics["automatic_gates_passed"] = (
        metrics["order_agreement"] >= 0.90
        and rho >= 0.90
        and metrics["forward_reverse_within_one_decile"] >= 0.95
        and metrics["comparison_graph_connected"]
    )
    return metrics


def run(
    source_dir: Path,
    output_dir: Path,
    *,
    parallelism: int,
    total_token_budget: int,
    env_file: Path | None,
) -> dict[str, Any]:
    if not (output_dir / "manifest.json").is_file():
        initialize(source_dir, output_dir)
    prior_manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in (base.OUTPUT, diagnostic.OUTPUT)
    ]
    prior_tokens = sum(
        int(manifest.get("actual_input_tokens", 0))
        + int(manifest.get("actual_output_tokens", 0))
        for manifest in prior_manifests
    )
    remaining_budget = total_token_budget - prior_tokens
    if remaining_budget <= 0:
        raise RuntimeError("the cumulative experiment token budget is exhausted")
    connection = base._connect(output_dir / "requests.sqlite3")
    diagnostic.run_condition(
        connection,
        "forward",
        parallelism=parallelism,
        token_budget=remaining_budget,
        env_file=env_file,
    )
    diagnostic.run_condition(
        connection,
        "reverse",
        parallelism=parallelism,
        token_budget=remaining_budget,
        env_file=env_file,
    )
    disagreement_count = _schedule_adjudication(connection)
    diagnostic.run_condition(
        connection,
        "adjudication",
        parallelism=parallelism,
        token_budget=remaining_budget,
        env_file=env_file,
    )
    metrics = summarize(connection, output_dir)
    tokens = connection.execute(
        "SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0) "
        "FROM requests WHERE status='complete'"
    ).fetchone()
    served_models = {
        row[0]: int(row[1])
        for row in connection.execute(
            "SELECT served_model, count(*) FROM requests WHERE status='complete' "
            "GROUP BY served_model"
        )
    }
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "pilot_complete_pending_review",
            "cumulative_token_budget": total_token_budget,
            "prior_experiment_tokens": prior_tokens,
            "actual_input_tokens": int(tokens[0]),
            "actual_output_tokens": int(tokens[1]),
            "cumulative_actual_tokens": prior_tokens + int(tokens[0]) + int(tokens[1]),
            "completed_request_count": sum(served_models.values()),
            "served_models": served_models,
            "scheduled_adjudication_count": disagreement_count,
            "metrics": metrics,
            "full_run_started": False,
        }
    )
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--total-token-budget", type=int, default=TOTAL_TOKEN_BUDGET)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = initialize(args.source_dir, args.output_dir, rebuild=args.rebuild)
    else:
        result = run(
            args.source_dir,
            args.output_dir,
            parallelism=args.parallelism,
            total_token_budget=args.total_token_budget,
            env_file=args.env_file,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
