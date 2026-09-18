"""Build and run the source-local V9 Skin relevance-bucket pilot."""

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
from scipy.stats import spearmanr

from analysis.evidence_library import relevance_bucket_diagnostic as calls
from analysis.evidence_library import relevance_bucket_rounds as bbb_rounds
from analysis.evidence_library import relevance_bucket_tournament as storage
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE
from tools.chembl_tool.common.json_utils import write_json_atomic

VERSION = "skin_reaction_relevance_bucket_tournament.v1"
PROMPT_VERSION = "skin_reaction_relevance_bucket_prompt.v1"
MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "high"
MAX_COMPLETION_TOKENS = 8_192
TOTAL_TOKEN_BUDGET = 20_000_000
BATCH_SIZE = 5
PILOT_PER_SOURCE = 128
L2_PENALTY = 0.3
ROOT = Path("data/evidence_libraries/skin_reaction/v9")
INPUT = ROOT / "03_pair_buckets/records.parquet"
OUTPUT = Path("outputs/analysis/evidence_library/skin_relevance_bucket_tournament_v1")
TEMPLATE = Path(__file__).with_name("prompts") / "skin_reaction_relevance_bucket_v1.jinja"
BBB_LEDGER = Path("data/evidence_libraries/bbb_martins/v9/relevance_bucket_tournament_v4/manifest.json")

RELEVANCE_BUCKET_COLUMNS = {
    "direct_skin_reaction": (
        "canonical_endpoint_concept",
        "measurement_kind",
        "canonical_assay_or_test",
        "canonical_species_or_population",
    ),
    "sensitization_aop": (
        "canonical_endpoint_concept",
        "canonical_aop_event",
        "canonical_assay_type",
        "canonical_species_context",
    ),
    "phototoxicity_irritation_local_damage": (
        "canonical_endpoint_concept",
        "measurement_kind",
        "canonical_evidence_system",
    ),
    "skin_exposure": (
        "canonical_endpoint_concept",
        "canonical_study_design",
        "canonical_species_context",
    ),
}
EXPECTED_COUNTS = {
    "direct_skin_reaction": (310, 278),
    "sensitization_aop": (931, 760),
    "phototoxicity_irritation_local_damage": (25_908, 138),
    "skin_exposure": (6_456, 902),
}
BASE_CARD_COLUMNS = (
    "canonical_endpoint_concept",
    "measurement_kind",
    "canonical_unit_text",
)
SOURCE_CARD_COLUMNS = {
    "direct_skin_reaction": (
        "canonical_assay_or_test",
        "canonical_species_or_population",
    ),
    "sensitization_aop": (
        "canonical_aop_event",
        "canonical_assay_type",
        "canonical_species_context",
    ),
    "phototoxicity_irritation_local_damage": (
        "canonical_assay_method",
        "canonical_evidence_system",
    ),
    "skin_exposure": (
        "canonical_study_design",
        "canonical_species_context",
    ),
}
DIVERSITY_COLUMNS = (
    "pair_bucket_key",
    "canonical_unit_text",
    "canonical_assay_context",
    "canonical_species_context",
    "canonical_smiles",
)
ROUND_CYCLES = {
    "round1": 3,
    "round2": 2,
    "round3": 2,
    "round4": 2,
    "round5": 2,
}


def relevance_bucket(record: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    source = str(record.get("source_id") or "")
    if source not in RELEVANCE_BUCKET_COLUMNS:
        raise ValueError(f"unsupported Skin source_id: {source!r}")
    identity = {"source_id": source}
    identity.update(
        {
            column: str(storage._clean(record.get(column)) or storage.UNKNOWN)
            for column in RELEVANCE_BUCKET_COLUMNS[source]
        }
    )
    return storage._canonical_json(identity), identity


def visible_card(record: Mapping[str, Any]) -> dict[str, Any]:
    source = str(record.get("source_id") or "")
    columns = BASE_CARD_COLUMNS + SOURCE_CARD_COLUMNS[source]
    card = {
        column: storage._clean(record.get(column))
        for column in columns
        if storage._clean(record.get(column)) is not None
    }
    if any("reference" in column.lower() for column in card):
        raise AssertionError("reference semantics leaked into a sample card")
    return card


def select_diverse_records(
    records: Sequence[Mapping[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    candidates: list[tuple[Mapping[str, Any], dict[str, Any], str]] = []
    seen: set[str] = set()
    for record in sorted(records, key=lambda row: str(row.get("canonical_record_id") or "")):
        card = visible_card(record)
        encoded = storage._canonical_json(card)
        if card and encoded not in seen:
            candidates.append((record, card, encoded))
            seen.add(encoded)

    selected: list[tuple[Mapping[str, Any], dict[str, Any], str]] = []
    seen_values: dict[str, set[str]] = defaultdict(set)
    while candidates and len(selected) < limit:
        def score(item: tuple[Mapping[str, Any], dict[str, Any], str]) -> tuple[int, str]:
            record, _, encoded = item
            novelty = sum(
                str(storage._clean(record.get(column)) or storage.UNKNOWN)
                not in seen_values[column]
                for column in DIVERSITY_COLUMNS
            )
            return novelty, storage._hash(encoded)

        chosen = max(candidates, key=score)
        candidates.remove(chosen)
        selected.append(chosen)
        for column in DIVERSITY_COLUMNS:
            seen_values[column].add(
                str(storage._clean(chosen[0].get(column)) or storage.UNKNOWN)
            )
    return [card for _, card, _ in selected]


def _input_columns() -> list[str]:
    return sorted(
        {
            "source_id",
            "pair_bucket_key",
            "assay_transfer_eligible",
            "canonical_record_id",
            *BASE_CARD_COLUMNS,
            *DIVERSITY_COLUMNS,
            *(column for columns in RELEVANCE_BUCKET_COLUMNS.values() for column in columns),
            *(column for columns in SOURCE_CARD_COLUMNS.values() for column in columns),
        }
    )


def _pilot_keys(connection: sqlite3.Connection) -> list[str]:
    selected = []
    for source in RELEVANCE_BUCKET_COLUMNS:
        keys = [
            row[0]
            for row in connection.execute(
                "SELECT relevance_bucket FROM buckets WHERE source_id=? AND scorable=1",
                (source,),
            )
        ]
        ordered = sorted(keys, key=lambda key: storage._hash("skin-pilot", key))
        if len(ordered) < PILOT_PER_SOURCE:
            raise ValueError(f"not enough scorable {source} buckets")
        selected.extend(ordered[:PILOT_PER_SOURCE])
    return sorted(selected)


def _rounds(keys: Sequence[str]) -> dict[str, dict[str, list[tuple[str, str]]]]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_source[json.loads(key)["source_id"]].append(key)
    output = {phase: {} for phase in ROUND_CYCLES}
    for source, source_keys in sorted(by_source.items()):
        used: set[tuple[str, str]] = set()
        for phase, cycle_count in ROUND_CYCLES.items():
            edges = []
            for cycle_index in range(cycle_count):
                edges.extend(
                    bbb_rounds._cycle(
                        sorted(source_keys),
                        f"skin-v1-{source}-{phase}-{cycle_index}",
                        used,
                    )
                )
            output[phase][source] = edges
    return output


def _insert_rounds(
    connection: sqlite3.Connection,
    rounds: Mapping[str, Mapping[str, Sequence[tuple[str, str]]]],
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
    rounds: Mapping[str, Mapping[str, Sequence[tuple[str, str]]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    cumulative: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for phase, sources in rounds.items():
        summary[phase] = {}
        for source, edges in sources.items():
            cumulative[source].extend(edges)
            degree = Counter(key for edge in cumulative[source] for key in edge)
            positions = Counter(a for a, _ in cumulative[source])
            summary[phase][source] = {
                "new_edges": len(edges),
                "cumulative_edges": len(cumulative[source]),
                "bucket_count": len(degree),
                "minimum_cumulative_degree": min(degree.values()),
                "maximum_cumulative_degree": max(degree.values()),
                "maximum_candidate_position_imbalance": max(
                    abs(2 * positions[key] - degree[key]) for key in degree
                ),
                "cumulative_graph_connected": storage._connected(
                    [(a, b, "A") for a, b in cumulative[source]]
                ),
            }
    return summary


def build(output_dir: Path = OUTPUT, *, rebuild: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    database_path = output_dir / "requests.sqlite3"
    mapping_path = output_dir / "pair_bucket_relevance_map.parquet"
    input_hash = file_sha256(INPUT)
    prompt_hash = file_sha256(TEMPLATE)
    if not rebuild and all(path.is_file() for path in (manifest_path, database_path, mapping_path)):
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("version") == VERSION
            and prior.get("input_sha256") == input_hash
            and prior.get("prompt_sha256") == prompt_hash
        ):
            return prior

    connection = storage._connect(database_path)
    connection.execute("DELETE FROM buckets")
    connection.execute("DELETE FROM comparisons")
    connection.execute("DELETE FROM requests")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parquet = pq.ParquetFile(INPUT)
    available = set(parquet.schema_arrow.names)
    columns = [column for column in _input_columns() if column in available]
    for batch in parquet.iter_batches(batch_size=20_000, columns=columns):
        for record in batch.to_pylist():
            if record.get("assay_transfer_eligible") is True:
                by_pair[str(record["pair_bucket_key"])].append(record)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mapping_rows = []
    inconsistent_pairs = 0
    for pair_key, records in by_pair.items():
        representative = min(records, key=lambda row: str(row.get("canonical_record_id") or ""))
        key, _ = relevance_bucket(representative)
        inconsistent_pairs += len({relevance_bucket(record)[0] for record in records}) > 1
        by_bucket[key].extend(records)
        mapping_rows.append(
            {
                "source_id": representative["source_id"],
                "pair_bucket_key": pair_key,
                "relevance_bucket": key,
                "record_count": len(records),
            }
        )

    for key in sorted(by_bucket):
        records = by_bucket[key]
        _, identity = relevance_bucket(records[0])
        samples = select_diverse_records(records)
        connection.execute(
            "INSERT INTO buckets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                identity["source_id"],
                key,
                storage._canonical_json(samples),
                len(records),
                int(bool(samples)),
                None if samples else "no_semantic_sample_card",
            ),
        )
    connection.commit()
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_parquet(mapping_path, index=False)
    counts = {
        source: {
            "pair_buckets": int(
                mapping.loc[mapping.source_id == source, "pair_bucket_key"].nunique()
            ),
            "relevance_buckets": int(
                connection.execute(
                    "SELECT count(*) FROM buckets WHERE source_id=?", (source,)
                ).fetchone()[0]
            ),
        }
        for source in RELEVANCE_BUCKET_COLUMNS
    }
    observed = {
        source: (row["pair_buckets"], row["relevance_buckets"])
        for source, row in counts.items()
    }
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"V9 Skin relevance-bucket counts changed: {observed}")

    pilot_keys = _pilot_keys(connection)
    rounds = _rounds(pilot_keys)
    _insert_rounds(connection, rounds)
    for phase in ROUND_CYCLES:
        calls._prepare_requests(
            connection,
            phase,
            template_path=TEMPLATE,
            model=MODEL,
        )
    first_batch = connection.execute(
        "SELECT batch_id FROM requests ORDER BY phase, batch_id LIMIT 1"
    ).fetchone()[0]
    manifest = {
        "version": VERSION,
        "status": "prompt_pending_review",
        "input": str(INPUT),
        "input_sha256": input_hash,
        "prompt": str(TEMPLATE),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "first_review_batch_id": first_batch,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "batch_size_maximum": BATCH_SIZE,
        "bradley_terry_l2_penalty": L2_PENALTY,
        "assay_transfer_filter": {"field": "assay_transfer_eligible", "value": True},
        "relevance_bucket_columns": {
            source: list(columns) for source, columns in RELEVANCE_BUCKET_COLUMNS.items()
        },
        "source_counts": counts,
        "relevance_bucket_count": len(by_bucket),
        "pair_buckets_with_inconsistent_relevance_fields": inconsistent_pairs,
        "pilot_bucket_count": len(pilot_keys),
        "pilot_source_bucket_counts": {
            source: PILOT_PER_SOURCE for source in RELEVANCE_BUCKET_COLUMNS
        },
        "sample_card_limit": 5,
        "sample_card_columns": {
            "common": list(BASE_CARD_COLUMNS),
            **{source: list(columns) for source, columns in SOURCE_CARD_COLUMNS.items()},
        },
        "sample_fields_explicitly_excluded": [
            "support_text",
            "extra_details",
            "pmid",
            "canonical_smiles",
            "source_smiles",
            "outcomes_labels_values_and_directions",
            "all_reference_semantics_fields",
        ],
        "comparison_policy": "source_local_edge_disjoint_hamiltonian_cycles",
        "round_cycles": ROUND_CYCLES,
        "graph": _graph_summary(rounds),
        "full_run_started": False,
        "artifacts": {
            "pair_bucket_relevance_map": str(mapping_path),
            "request_cache": str(database_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def _source_keys(connection: sqlite3.Connection) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    for key in _pilot_keys(connection):
        output[json.loads(key)["source_id"]].append(key)
    return output


def _rankings(
    connection: sqlite3.Connection,
    phases: Sequence[str],
    output_path: Path | None,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in phases)
    outcomes = list(
        connection.execute(
            f"SELECT bucket_a, bucket_b, winner FROM comparisons "
            f"WHERE phase IN ({placeholders}) AND winner IS NOT NULL",
            tuple(phases),
        )
    )
    rows = []
    for source, keys in _source_keys(connection).items():
        key_set = set(keys)
        source_outcomes = [tuple(row) for row in outcomes if row[0] in key_set]
        scores = bbb_rounds.fit_bradley_terry(
            source_outcomes, l2_penalty=L2_PENALTY
        )
        ordered = sorted(keys, key=lambda key: (-scores[key], key))
        for index, key in enumerate(ordered):
            rows.append(
                {
                    "relevance_bucket": key,
                    "source_id": source,
                    "bradley_terry_score": scores[key],
                    "source_rank": index + 1,
                    "source_percentile": 100.0
                    * (len(ordered) - index - 1)
                    / max(1, len(ordered) - 1),
                    "source_decile": 10 - min(9, index * 10 // len(ordered)),
                    "comparison_count": sum(key in row[:2] for row in source_outcomes),
                }
            )
    frame = pd.DataFrame(rows)
    if output_path is not None:
        frame.to_parquet(output_path, index=False)
    return frame


def _convergence(
    first: pd.DataFrame, cumulative: pd.DataFrame, second_only: pd.DataFrame
) -> dict[str, Any]:
    merged = first.merge(
        cumulative,
        on=["relevance_bucket", "source_id"],
        suffixes=("_round1", "_round2"),
        validate="one_to_one",
    )
    by_source = {}
    for source, rows in merged.groupby("source_id"):
        independent = rows[
            ["relevance_bucket", "source_rank_round1", "source_decile_round1"]
        ].merge(
            second_only.loc[
                second_only.source_id == source,
                ["relevance_bucket", "source_rank", "source_decile"],
            ],
            on="relevance_bucket",
            validate="one_to_one",
        )
        top_first = set(
            rows.loc[rows.source_decile_round1 == 10, "relevance_bucket"]
        )
        top_second = set(
            rows.loc[rows.source_decile_round2 == 10, "relevance_bucket"]
        )
        by_source[source] = {
            "bucket_count": len(rows),
            "rank_spearman": float(
                spearmanr(rows.source_rank_round1, rows.source_rank_round2).statistic
            ),
            "within_one_decile": float(
                (abs(rows.source_decile_round1 - rows.source_decile_round2) <= 1).mean()
            ),
            "top_decile_overlap": len(top_first & top_second) / max(1, len(top_second)),
            "independent_round_rank_spearman": float(
                spearmanr(
                    independent.source_rank_round1, independent.source_rank
                ).statistic
            ),
            "independent_round_within_one_decile": float(
                (
                    abs(independent.source_decile_round1 - independent.source_decile)
                    <= 1
                ).mean()
            ),
        }
    return {"by_source": by_source}


def _token_ledger(connection: sqlite3.Connection) -> dict[str, Any]:
    prior = json.loads(BBB_LEDGER.read_text(encoding="utf-8"))
    row = connection.execute(
        "SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0) "
        "FROM requests"
    ).fetchone()
    local_input, local_output = int(row[0]), int(row[1])
    retried = connection.execute(
        "SELECT attempts, prompt FROM requests WHERE attempts>1"
    ).fetchall()
    retry_calls = sum(row[0] - 1 for row in retried)
    retry_upper = sum(
        (row[0] - 1) * (len(row[1].encode("utf-8")) + MAX_COMPLETION_TOKENS)
        for row in retried
    )
    actual = int(prior["cumulative_actual_tokens"]) + local_input + local_output
    conservative = (
        int(prior["cumulative_actual_token_upper_bound"])
        + local_input
        + local_output
        + retry_upper
    )
    return {
        "cumulative_token_budget": TOTAL_TOKEN_BUDGET,
        "prior_actual_tokens": int(prior["cumulative_actual_tokens"]),
        "prior_actual_token_upper_bound": int(
            prior["cumulative_actual_token_upper_bound"]
        ),
        "skin_input_tokens": local_input,
        "skin_output_tokens": local_output,
        "cumulative_actual_tokens": actual,
        "remaining_actual_tokens": TOTAL_TOKEN_BUDGET - actual,
        "unobserved_retry_call_count": retry_calls,
        "unobserved_retry_token_upper_bound": retry_upper,
        "cumulative_actual_token_upper_bound": conservative,
        "remaining_actual_tokens_conservative": TOTAL_TOKEN_BUDGET - conservative,
    }


def _require_complete(connection: sqlite3.Connection, phase: str) -> None:
    pending = connection.execute(
        "SELECT count(*) FROM requests WHERE phase=? AND status!='complete'", (phase,)
    ).fetchone()[0]
    if pending:
        raise RuntimeError(f"{phase} has {pending} incomplete request batches")


def run(
    output_dir: Path,
    *,
    approved_prompt_sha256: str,
    parallelism: int,
    env_file: Path | None,
    credential_env: str,
) -> dict[str, Any]:
    manifest = build(output_dir)
    if approved_prompt_sha256 != manifest["prompt_sha256"]:
        raise ValueError("approved prompt hash does not match the built prompt")
    connection = storage._connect(output_dir / "requests.sqlite3")
    prior = json.loads(BBB_LEDGER.read_text(encoding="utf-8"))
    remaining = TOTAL_TOKEN_BUDGET - int(prior["cumulative_actual_tokens"])
    if remaining <= 0:
        raise RuntimeError("the cumulative actual-token budget is exhausted")
    rounds = _rounds(_pilot_keys(connection))
    _insert_rounds(connection, rounds)
    for phase in ROUND_CYCLES:
        calls._prepare_requests(
            connection, phase, template_path=TEMPLATE, model=MODEL
        )

    cumulative_phases: list[str] = []
    previous: pd.DataFrame | None = None
    convergence = {}
    for phase in ROUND_CYCLES:
        calls.run_condition(
            connection,
            phase,
            parallelism=parallelism,
            token_budget=remaining,
            env_file=env_file,
            template_path=TEMPLATE,
            allow_ties=False,
            credential_env=credential_env,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        _require_complete(connection, phase)
        cumulative_phases.append(phase)
        current = _rankings(
            connection,
            tuple(cumulative_phases),
            output_dir / f"relevance_bucket_rankings_{phase}.parquet",
        )
        if previous is not None:
            convergence[phase] = _convergence(
                previous, current, _rankings(connection, (phase,), None)
            )
        previous = current

    last_phase = list(ROUND_CYCLES)[-1]
    manifest.update(
        {
            "status": f"{last_phase}_complete_pending_review",
            "approved_prompt_sha256": approved_prompt_sha256,
            "credential_env_by_round": {
                phase: credential_env for phase in ROUND_CYCLES
            },
            "round_cycles": ROUND_CYCLES,
            "graph": _graph_summary(rounds),
            "convergence": convergence["round2"],
            "token_ledger": _token_ledger(connection),
            "served_models": {
                row[0]: int(row[1])
                for row in connection.execute(
                    "SELECT served_model, count(*) FROM requests WHERE status='complete' "
                    "GROUP BY served_model"
                )
            },
        }
    )
    for phase in list(ROUND_CYCLES)[2:]:
        manifest[f"{phase}_convergence"] = convergence[phase]
    write_json_atomic(output_dir / "manifest.json", manifest)
    connection.close()
    return manifest


def show_prompt(output_dir: Path = OUTPUT) -> str:
    manifest = build(output_dir)
    connection = storage._connect(output_dir / "requests.sqlite3")
    prompt = connection.execute(
        "SELECT prompt FROM requests WHERE batch_id=?",
        (manifest["first_review_batch_id"],),
    ).fetchone()[0]
    connection.close()
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "show-prompt", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY_TWO")
    parser.add_argument("--approved-prompt-sha256")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result: Any = build(args.output_dir, rebuild=args.rebuild)
    elif args.command == "show-prompt":
        result = show_prompt(args.output_dir)
    else:
        if not args.approved_prompt_sha256:
            raise ValueError("--approved-prompt-sha256 is required for paid requests")
        result = run(
            args.output_dir,
            approved_prompt_sha256=args.approved_prompt_sha256,
            parallelism=args.parallelism,
            env_file=args.env_file,
            credential_env=args.api_key_env,
        )
    print(result if isinstance(result, str) else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
