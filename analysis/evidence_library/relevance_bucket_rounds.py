"""Run source-local comparison rounds for the 512-bucket BBB pilot."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from analysis.evidence_library import relevance_bucket_diagnostic as diagnostic
from analysis.evidence_library import relevance_bucket_pilot as prior_pilot
from analysis.evidence_library import relevance_bucket_tournament as base
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE
from tools.chembl_tool.common.json_utils import write_json_atomic

VERSION = "bbb_relevance_bucket_rounds.v4"
PROMPT_VERSION = "bbb_relevance_bucket_rounds_prompt.v4"
OUTPUT = Path("outputs/analysis/evidence_library/bbb_relevance_bucket_tournament_v4")
TEMPLATE = Path(__file__).with_name("prompts") / "relevance_bucket_rounds_v4.jinja"
TOTAL_TOKEN_BUDGET = 20_000_000
BATCH_SIZE = 5
L2_PENALTY = 0.3
ROUND_CYCLES = {
    "round1": 3,
    "round2": 2,
    "round3": 2,
    "round4": 2,
    "round5": 2,
    "round6": 2,
}


def _source_keys(keys: Sequence[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        output[json.loads(key)["source_id"]].append(key)
    return {source: sorted(values) for source, values in sorted(output.items())}


def _cycle(
    keys: Sequence[str], seed: str, used: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Find one deterministic Hamiltonian cycle with unused undirected edges."""
    for attempt in range(10_000):
        order = sorted(
            keys,
            key=lambda key: base._hash(seed, str(attempt), key),
        )
        directed = list(zip(order, order[1:] + order[:1], strict=True))
        undirected = {tuple(sorted(edge)) for edge in directed}
        if len(undirected) == len(keys) and not undirected & used:
            used.update(undirected)
            return directed
    for attempt in range(1_000):
        remaining = set(keys)
        order = [min(remaining, key=lambda key: base._hash(seed, str(attempt), key))]
        remaining.remove(order[0])
        while remaining:
            candidates = [
                key
                for key in remaining
                if tuple(sorted((order[-1], key))) not in used
            ]
            if not candidates:
                break
            following = min(
                candidates,
                key=lambda key: base._hash(
                    seed, str(attempt), str(len(order)), key
                ),
            )
            order.append(following)
            remaining.remove(following)
        if not remaining and tuple(sorted((order[-1], order[0]))) not in used:
            directed = list(zip(order, order[1:] + order[:1], strict=True))
            used.update(tuple(sorted(edge)) for edge in directed)
            return directed
    raise RuntimeError(f"could not construct an unused cycle for {seed}")


def comparison_rounds(
    keys: Sequence[str],
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Build source-local, edge-disjoint rounds with balanced candidate positions."""
    rounds: dict[str, dict[str, list[tuple[str, str]]]] = {
        phase: {} for phase in ROUND_CYCLES
    }
    for source, source_keys in _source_keys(keys).items():
        if source == "influx_transport":
            edges = []
            for i, a in enumerate(source_keys):
                for j in range(i + 1, len(source_keys)):
                    b = source_keys[j]
                    distance = j - i
                    if distance < len(source_keys) / 2:
                        edges.append((a, b))
                    elif distance > len(source_keys) / 2:
                        edges.append((b, a))
                    else:
                        edges.append((a, b) if i % 2 == 0 else (b, a))
            for phase in rounds:
                rounds[phase][source] = []
            rounds["round1"][source] = edges
            continue

        used: set[tuple[str, str]] = set()
        for phase, cycle_count in ROUND_CYCLES.items():
            rounds[phase][source] = []
            for cycle_index in range(cycle_count):
                rounds[phase][source].extend(
                    _cycle(
                        source_keys,
                        f"v4-{source}-{phase}-{cycle_index}",
                        used,
                    )
                )
    return rounds


def _insert_rounds(
    connection: sqlite3.Connection,
    rounds: dict[str, dict[str, list[tuple[str, str]]]],
) -> None:
    for phase, sources in rounds.items():
        for source, edges in sources.items():
            slug = source.replace("_", "-")
            for index, (a, b) in enumerate(edges):
                values = (
                    f"{phase}-{slug}-{index:04d}",
                    phase,
                    f"{phase}-{slug}-batch-{index // BATCH_SIZE:04d}",
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
    rounds: dict[str, dict[str, list[tuple[str, str]]]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    cumulative: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for phase, sources in rounds.items():
        summary[phase] = {}
        for source, edges in sources.items():
            cumulative[source].extend(edges)
            nodes = {
                key
                for edge in cumulative[source]
                for key in edge
            }
            degree = Counter(key for edge in cumulative[source] for key in edge)
            positions = Counter(a for a, _ in cumulative[source])
            summary[phase][source] = {
                "new_edges": len(edges),
                "cumulative_edges": len(cumulative[source]),
                "bucket_count": len(nodes),
                "minimum_cumulative_degree": min(degree.values()),
                "maximum_cumulative_degree": max(degree.values()),
                "maximum_candidate_position_imbalance": max(
                    abs(2 * positions[key] - degree[key]) for key in nodes
                ),
                "cumulative_graph_connected": base._connected(
                    [(a, b, "A") for a, b in cumulative[source]]
                ),
            }
    return summary


def initialize(output_dir: Path, *, rebuild: bool = False) -> dict[str, Any]:
    source_database = base.OUTPUT / "requests.sqlite3"
    source_manifest = prior_pilot.OUTPUT / "manifest.json"
    for path in (source_database, source_manifest, TEMPLATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "requests.sqlite3"
    manifest_path = output_dir / "manifest.json"
    prompt_hash = file_sha256(TEMPLATE)
    source_hash = file_sha256(source_manifest)
    if not rebuild and database.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("version") == VERSION
            and prior.get("prompt_sha256") == prompt_hash
            and prior.get("source_manifest_sha256") == source_hash
        ):
            return prior

    connection = base._connect(database)
    connection.execute("DELETE FROM comparisons")
    connection.execute("DELETE FROM requests")
    diagnostic._copy_buckets(source_database, connection)
    keys = base._pilot_keys(connection)
    rounds = comparison_rounds(keys)
    _insert_rounds(connection, rounds)
    graph = _graph_summary(rounds)
    manifest = {
        "version": VERSION,
        "status": "initialized",
        "source_artifact": str(prior_pilot.OUTPUT),
        "source_manifest_sha256": source_hash,
        "source_bucket_cache": str(source_database),
        "source_bucket_cache_sha256": file_sha256(source_database),
        "prompt": str(TEMPLATE),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "model": base.MODEL,
        "reasoning_effort": base.REASONING_EFFORT,
        "pilot_bucket_count": len(keys),
        "source_bucket_counts": {
            source: len(values) for source, values in _source_keys(keys).items()
        },
        "batch_size_maximum": BATCH_SIZE,
        "round_cycles": ROUND_CYCLES,
        "bradley_terry_l2_penalty": L2_PENALTY,
        "winner_contract": "exact_bucket_id_strict_binary",
        "comparison_policy": "one_shot_unique_source_local_edges",
        "outcome_fields_withheld": sorted(diagnostic.OUTCOME_FIELDS),
        "graph": graph,
        "full_run_started": False,
    }
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def fit_bradley_terry(
    outcomes: Sequence[tuple[str, str, str]], *, l2_penalty: float = L2_PENALTY
) -> dict[str, float]:
    """Fit an L2-regularized binary Bradley-Terry model."""
    keys = sorted({key for a, b, _ in outcomes for key in (a, b)})
    index = {key: i for i, key in enumerate(keys)}
    a_index = np.asarray([index[a] for a, _, _ in outcomes])
    b_index = np.asarray([index[b] for _, b, _ in outcomes])
    y = np.asarray([winner == a for a, _, winner in outcomes], dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        difference = theta[a_index] - theta[b_index]
        probability = 1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30)))
        loss = -float(
            (y * np.log(probability) + (1 - y) * np.log(1 - probability)).sum()
        ) + 0.5 * l2_penalty * float(theta @ theta)
        residual = probability - y
        gradient = l2_penalty * theta
        np.add.at(gradient, a_index, residual)
        np.add.at(gradient, b_index, -residual)
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(len(keys)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-20, 20)] * len(keys),
    )
    if not result.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {result.message}")
    scores = result.x - result.x.mean()
    return dict(zip(keys, scores.tolist(), strict=True))


def _outcomes(connection: sqlite3.Connection, phases: Sequence[str]) -> list[tuple[str, str, str]]:
    placeholders = ",".join("?" for _ in phases)
    rows = connection.execute(
        f"SELECT bucket_a, bucket_b, winner FROM comparisons "
        f"WHERE phase IN ({placeholders}) AND winner IS NOT NULL",
        tuple(phases),
    )
    return [(row[0], row[1], row[2]) for row in rows]


def _rankings(
    connection: sqlite3.Connection,
    phases: Sequence[str],
    output_path: Path | None,
) -> pd.DataFrame:
    rows = []
    for source, source_keys in _source_keys(base._pilot_keys(connection)).items():
        source_set = set(source_keys)
        outcomes = [
            outcome
            for outcome in _outcomes(connection, phases)
            if outcome[0] in source_set
        ]
        if not outcomes:
            continue
        scores = fit_bradley_terry(outcomes)
        ordered = sorted(source_keys, key=lambda key: (-scores[key], key))
        n = len(ordered)
        for index, key in enumerate(ordered):
            rows.append(
                {
                    "relevance_bucket": key,
                    "source_id": source,
                    "bradley_terry_score": scores[key],
                    "source_rank": index + 1,
                    "source_percentile": 100.0 * (n - 1 - index) / max(1, n - 1),
                    "source_decile": 10 - min(9, index * 10 // n),
                    "comparison_count": sum(key in outcome[:2] for outcome in outcomes),
                }
            )
    frame = pd.DataFrame(rows)
    if output_path is not None:
        frame.to_parquet(output_path, index=False)
    return frame


def _convergence(
    round1: pd.DataFrame,
    round2: pd.DataFrame,
    round2_only: pd.DataFrame,
) -> dict[str, Any]:
    merged = round1.merge(
        round2,
        on=["relevance_bucket", "source_id"],
        suffixes=("_round1", "_round2"),
        validate="one_to_one",
    )
    by_source = {}
    for source, rows in merged.groupby("source_id"):
        independent = rows[
            ["relevance_bucket", "source_rank_round1", "source_decile_round1"]
        ].merge(
            round2_only.loc[
                round2_only["source_id"] == source,
                ["relevance_bucket", "source_rank", "source_decile"],
            ],
            on="relevance_bucket",
            how="inner",
        )
        by_source[source] = {
            "bucket_count": len(rows),
            "rank_spearman": float(
                spearmanr(rows["source_rank_round1"], rows["source_rank_round2"]).statistic
            ),
            "within_one_decile": float(
                (
                    abs(rows["source_decile_round1"] - rows["source_decile_round2"])
                    <= 1
                ).mean()
            ),
            "top_decile_overlap": float(
                len(
                    set(rows.loc[rows["source_decile_round1"] == 10, "relevance_bucket"])
                    & set(rows.loc[rows["source_decile_round2"] == 10, "relevance_bucket"])
                )
                / max(1, int((rows["source_decile_round2"] == 10).sum()))
            ),
        }
        if len(independent) == len(rows):
            by_source[source].update(
                {
                    "independent_round_rank_spearman": float(
                        spearmanr(
                            independent["source_rank_round1"],
                            independent["source_rank"],
                        ).statistic
                    ),
                    "independent_round_within_one_decile": float(
                        (
                            abs(
                                independent["source_decile_round1"]
                                - independent["source_decile"]
                            )
                            <= 1
                        ).mean()
                    ),
                }
            )
    return {"by_source": by_source}


def _tokens(connection: sqlite3.Connection, phase: str | None = None) -> dict[str, int]:
    where = "status='complete'"
    parameters: tuple[str, ...] = ()
    if phase is not None:
        where += " AND phase=?"
        parameters = (phase,)
    row = connection.execute(
        "SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0), "
        f"count(*) FROM requests WHERE {where}",
        parameters,
    ).fetchone()
    return {
        "input_tokens": int(row[0]),
        "output_tokens": int(row[1]),
        "actual_tokens": int(row[0] + row[1]),
        "completed_requests": int(row[2]),
    }


def _prior_tokens() -> int:
    total = 0
    for path in (base.OUTPUT, diagnostic.OUTPUT, prior_pilot.OUTPUT):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        total += int(manifest.get("actual_input_tokens", 0))
        total += int(manifest.get("actual_output_tokens", 0))
    return total


def _legacy_retry_uncertainty(database: Path, phase: str | None = None) -> dict[str, int]:
    """Bound usage omitted by the pre-v4 retry ledger."""
    connection = sqlite3.connect(database)
    where = "status='complete' AND attempts>1"
    parameters: tuple[str, ...] = ()
    if phase is not None:
        where += " AND phase=?"
        parameters = (phase,)
    rows = connection.execute(
        f"SELECT attempts, input_tokens FROM requests WHERE {where}", parameters
    ).fetchall()
    connection.close()
    calls = sum(attempts - 1 for attempts, _ in rows)
    return {
        "unobserved_retry_calls": calls,
        "unobserved_retry_token_upper_bound": sum(
            (attempts - 1) * input_tokens for attempts, input_tokens in rows
        )
        + calls * base.MAX_COMPLETION_TOKENS,
    }


def _update_manifest(
    output_dir: Path,
    connection: sqlite3.Connection,
    *,
    status: str,
    prior_tokens: int,
    total_token_budget: int,
    convergence: dict[str, Any] | None = None,
    round3_convergence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = _tokens(connection)
    current_database = output_dir / "requests.sqlite3"
    prior_uncertainty = [
        _legacy_retry_uncertainty(path / "requests.sqlite3")
        for path in (base.OUTPUT, diagnostic.OUTPUT, prior_pilot.OUTPUT)
    ]
    current_uncertainties = [
        _legacy_retry_uncertainty(current_database, phase)
        for phase in ("round1", "round2")
    ]
    current_uncertainty = {
        key: sum(item[key] for item in current_uncertainties)
        for key in current_uncertainties[0]
    }
    retry_calls = current_uncertainty["unobserved_retry_calls"] + sum(
        item["unobserved_retry_calls"] for item in prior_uncertainty
    )
    retry_upper = current_uncertainty["unobserved_retry_token_upper_bound"] + sum(
        item["unobserved_retry_token_upper_bound"] for item in prior_uncertainty
    )
    served_models = {
        row[0]: int(row[1])
        for row in connection.execute(
            "SELECT served_model, count(*) FROM requests WHERE status='complete' "
            "GROUP BY served_model"
        )
    }
    manifest.update(
        {
            "status": status,
            "cumulative_token_budget": total_token_budget,
            "prior_experiment_tokens": prior_tokens,
            "actual_input_tokens": current["input_tokens"],
            "actual_output_tokens": current["output_tokens"],
            "cumulative_actual_tokens": prior_tokens + current["actual_tokens"],
            "token_ledger_scope": "reported usage from successful completions",
            "unobserved_retry_call_count": retry_calls,
            "unobserved_retry_token_upper_bound": retry_upper,
            "cumulative_actual_token_upper_bound": prior_tokens
            + current["actual_tokens"]
            + retry_upper,
            "remaining_actual_tokens": total_token_budget
            - prior_tokens
            - current["actual_tokens"],
            "remaining_actual_tokens_conservative": total_token_budget
            - prior_tokens
            - current["actual_tokens"]
            - retry_upper,
            "served_models": served_models,
            "convergence": convergence,
            "round3_convergence": round3_convergence,
            "full_run_started": False,
        }
    )
    for phase in ROUND_CYCLES:
        manifest[f"{phase}_token_ledger"] = _tokens(connection, phase)
        if phase in {"round1", "round2"}:
            manifest[f"{phase}_token_ledger"].update(
                _legacy_retry_uncertainty(current_database, phase)
            )
        else:
            manifest[f"{phase}_token_ledger"]["all_returned_completion_usage_accounted"] = True
    write_json_atomic(manifest_path, manifest)
    return manifest


def run(
    output_dir: Path,
    *,
    parallelism: int,
    total_token_budget: int,
    env_file: Path | None,
    credential_env: str | None,
) -> dict[str, Any]:
    if not (output_dir / "manifest.json").is_file():
        initialize(output_dir)
    prior_tokens = _prior_tokens()
    remaining = total_token_budget - prior_tokens
    if remaining <= 0:
        raise RuntimeError("the cumulative experiment token budget is exhausted")
    connection = base._connect(output_dir / "requests.sqlite3")
    rounds = comparison_rounds(base._pilot_keys(connection))
    _insert_rounds(connection, rounds)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "round_cycles": ROUND_CYCLES,
            "graph": _graph_summary(rounds),
            "credential_env_by_round": {
                **{f"round{number}": "OPENAI_API_KEY" for number in range(1, 6)},
                "round6": credential_env,
            },
        }
    )
    write_json_atomic(manifest_path, manifest)

    diagnostic.run_condition(
        connection,
        "round1",
        parallelism=parallelism,
        token_budget=remaining,
        env_file=env_file,
        template_path=TEMPLATE,
        allow_ties=False,
        credential_env=credential_env,
    )
    round1 = _rankings(
        connection,
        ("round1",),
        output_dir / "relevance_bucket_rankings_round1.parquet",
    )
    _update_manifest(
        output_dir,
        connection,
        status="round1_complete",
        prior_tokens=prior_tokens,
        total_token_budget=total_token_budget,
    )

    diagnostic.run_condition(
        connection,
        "round2",
        parallelism=parallelism,
        token_budget=remaining,
        env_file=env_file,
        template_path=TEMPLATE,
        allow_ties=False,
        credential_env=credential_env,
    )
    round2 = _rankings(
        connection,
        ("round1", "round2"),
        output_dir / "relevance_bucket_rankings_round2.parquet",
    )
    round2_only = _rankings(connection, ("round2",), None)
    convergence = _convergence(round1, round2, round2_only)
    _update_manifest(
        output_dir,
        connection,
        status="round2_complete_pending_review",
        prior_tokens=prior_tokens,
        total_token_budget=total_token_budget,
        convergence=convergence,
    )

    diagnostic.run_condition(
        connection,
        "round3",
        parallelism=parallelism,
        token_budget=remaining,
        env_file=env_file,
        template_path=TEMPLATE,
        allow_ties=False,
        credential_env=credential_env,
    )
    round3 = _rankings(
        connection,
        ("round1", "round2", "round3"),
        output_dir / "relevance_bucket_rankings_round3.parquet",
    )
    round3_only = _rankings(connection, ("round3",), None)
    manifest = _update_manifest(
        output_dir,
        connection,
        status="round3_complete_pending_review",
        prior_tokens=prior_tokens,
        total_token_budget=total_token_budget,
        convergence=convergence,
        round3_convergence=_convergence(round2, round3, round3_only),
    )
    previous = round3
    for number in (4, 5, 6):
        phase = f"round{number}"
        diagnostic.run_condition(
            connection,
            phase,
            parallelism=parallelism,
            token_budget=remaining,
            env_file=env_file,
            template_path=TEMPLATE,
            allow_ties=False,
            credential_env=credential_env,
        )
        cumulative = _rankings(
            connection,
            tuple(f"round{index}" for index in range(1, number + 1)),
            output_dir / f"relevance_bucket_rankings_{phase}.parquet",
        )
        independent = _rankings(connection, (phase,), None)
        manifest = _update_manifest(
            output_dir,
            connection,
            status=f"{phase}_complete_pending_review",
            prior_tokens=prior_tokens,
            total_token_budget=total_token_budget,
            convergence=convergence,
            round3_convergence=manifest["round3_convergence"],
        )
        manifest[f"{phase}_convergence"] = _convergence(
            previous, cumulative, independent
        )
        write_json_atomic(output_dir / "manifest.json", manifest)
        previous = cumulative
    connection.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--total-token-budget", type=int, default=TOTAL_TOKEN_BUDGET)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY_TWO")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = (
        initialize(args.output_dir, rebuild=args.rebuild)
        if args.command == "build"
        else run(
            args.output_dir,
            parallelism=args.parallelism,
            total_token_budget=args.total_token_budget,
            env_file=args.env_file,
            credential_env=args.api_key_env,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
