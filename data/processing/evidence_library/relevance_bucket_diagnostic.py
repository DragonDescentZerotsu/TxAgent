"""Diagnose order, batch-context, and batch-size effects in BBB comparisons."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sqlite3
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library import relevance_bucket_tournament as base
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client
from tools.chembl_tool.common.json_utils import write_json_atomic

VERSION = "bbb_relevance_bucket_diagnostic.v2"
PROMPT_VERSION = "bbb_relevance_bucket_diagnostic_prompt.v2"
OUTPUT = base.ROOT / "relevance_bucket_tournament_v2"
TEMPLATE = Path(__file__).with_name("prompts") / "relevance_bucket_diagnostic_v2.jinja"
SOURCE = base.OUTPUT
PAIR_COUNT = 100
CONDITIONS = {
    "n10_base": 10,
    "n10_repeat": 10,
    "n10_reverse": 10,
    "n10_rebatch": 10,
    "n5_base": 5,
    "n5_repeat": 5,
    "n5_reverse": 5,
    "n5_rebatch": 5,
}
OUTCOME_FIELDS = {
    "canonical_measurement_text",
    "finite_scalar_value",
    "canonical_category_id",
    "bbb_permeability_label",
    "bbb_transport_label",
    "interaction_conclusion",
    "passive_bbb_interpretation",
    "metric_uncertainty",
}


def bucket_id(key: str) -> str:
    return f"RB_{base._hash('v2-bucket-id', key)[:16]}"


def semantic_samples(samples_json: str) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for sample in json.loads(samples_json):
        visible = {key: value for key, value in sample.items() if key not in OUTCOME_FIELDS}
        encoded = base._canonical_json(visible)
        if visible and encoded not in seen:
            output.append(visible)
            seen.add(encoded)
    return output[:5]


def _source_pairs(source_database: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(source_database)
    connection.row_factory = sqlite3.Row
    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    seen = set()
    rows = connection.execute(
        "SELECT bucket_a, bucket_b FROM comparisons WHERE audit_of IS NULL "
        "AND winner IS NOT NULL ORDER BY comparison_id"
    )
    for row in rows:
        pair = tuple(sorted((row["bucket_a"], row["bucket_b"])))
        if pair in seen:
            continue
        seen.add(pair)
        source_pair = tuple(
            sorted(
                (
                    json.loads(row["bucket_a"])["source_id"],
                    json.loads(row["bucket_b"])["source_id"],
                )
            )
        )
        strata[source_pair].append((row["bucket_a"], row["bucket_b"]))
    connection.close()
    if len(strata) != 10 or any(len(rows) < 10 for rows in strata.values()):
        raise ValueError("V1 pilot does not cover ten pairs in every source-pair stratum")
    selected = []
    for source_pair in sorted(strata):
        rows = sorted(
            strata[source_pair],
            key=lambda pair: base._hash("v2-diagnostic-pair", *pair),
        )[:10]
        selected.extend(rows)
    return selected


def _copy_buckets(source_database: Path, destination: sqlite3.Connection) -> None:
    source = sqlite3.connect(source_database)
    source.row_factory = sqlite3.Row
    destination.execute("DELETE FROM buckets")
    for row in source.execute("SELECT * FROM buckets ORDER BY relevance_bucket"):
        samples = semantic_samples(row["samples_json"])
        destination.execute(
            "INSERT INTO buckets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["relevance_bucket"],
                row["source_id"],
                row["identity_json"],
                base._canonical_json(samples),
                row["record_count"],
                int(bool(samples)),
                None if samples else "no_semantic_sample_card",
            ),
        )
    destination.commit()
    source.close()


def _condition_order(pairs: Sequence[tuple[str, str]], condition: str) -> list[int]:
    indices = list(range(len(pairs)))
    if condition.endswith("rebatch"):
        seed = "v2-rebatch" if condition == "n10_rebatch" else condition
        indices.sort(key=lambda index: base._hash(seed, str(index)))
    return indices


def _insert_condition(
    connection: sqlite3.Connection,
    pairs: Sequence[tuple[str, str]],
    condition: str,
) -> None:
    batch_size = CONDITIONS[condition]
    reverse = condition.endswith("reverse")
    for position, pair_index in enumerate(_condition_order(pairs, condition)):
        a, b = pairs[pair_index]
        if reverse:
            a, b = b, a
        comparison_id = f"{condition}-{pair_index:03d}"
        batch_id = f"{condition}-batch-{position // batch_size:03d}"
        reference = "n5_base" if condition.startswith("n5") else "n10_base"
        connection.execute(
            "INSERT OR IGNORE INTO comparisons VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                comparison_id,
                condition,
                batch_id,
                a,
                b,
                None if condition.endswith("base") else f"{reference}-{pair_index:03d}",
            ),
        )
    connection.commit()


def initialize(source_dir: Path, output_dir: Path, *, rebuild: bool = False) -> dict[str, Any]:
    source_database = source_dir / "requests.sqlite3"
    source_manifest = source_dir / "manifest.json"
    for path in (source_database, source_manifest, TEMPLATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "requests.sqlite3"
    manifest_path = output_dir / "manifest.json"
    source_manifest_hash = file_sha256(source_manifest)
    prompt_hash = file_sha256(TEMPLATE)
    if not rebuild and database.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("version") == VERSION
            and prior.get("source_manifest_sha256") == source_manifest_hash
            and prior.get("prompt_sha256") == prompt_hash
        ):
            return prior

    connection = base._connect(database)
    connection.execute("DELETE FROM comparisons")
    connection.execute("DELETE FROM requests")
    _copy_buckets(source_database, connection)
    pairs = _source_pairs(source_database)
    for condition in CONDITIONS:
        _insert_condition(connection, pairs, condition)
    manifest = {
        "version": VERSION,
        "status": "initialized",
        "source_artifact": str(source_dir),
        "source_manifest_sha256": source_manifest_hash,
        "source_request_cache_sha256": file_sha256(source_database),
        "prompt": str(TEMPLATE),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "pair_count": len(pairs),
        "source_pair_strata": 10,
        "pairs_per_source_pair_stratum": 10,
        "conditions": CONDITIONS,
        "outcome_fields_withheld": sorted(OUTCOME_FIELDS),
        "candidate_layout": "inline",
        "winner_contract": "exact_bucket_id_or_tie",
        "model": base.MODEL,
        "reasoning_effort": base.REASONING_EFFORT,
        "full_run_started": False,
    }
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def _template(template_path: Path = TEMPLATE) -> Any:
    environment = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
    )
    return environment.get_template(template_path.name)


def _display_id(comparison_id: str) -> str:
    return f"P{comparison_id.rsplit('-', 1)[1]}"


def _comparisons(connection: sqlite3.Connection, batch_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT comparison_id, bucket_a, bucket_b FROM comparisons "
        "WHERE batch_id=? ORDER BY comparison_id",
        (batch_id,),
    ).fetchall()


def render_request(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    template_path: Path = TEMPLATE,
) -> str:
    output = []
    for comparison in _comparisons(connection, batch_id):
        item = {"id": _display_id(comparison["comparison_id"])}
        for position, key in enumerate(
            (comparison["bucket_a"], comparison["bucket_b"]), start=1
        ):
            bucket = connection.execute(
                "SELECT source_id, identity_json, samples_json FROM buckets "
                "WHERE relevance_bucket=?",
                (key,),
            ).fetchone()
            item[f"candidate_{position}"] = {
                "bucket_id": bucket_id(key),
                "source": bucket["source_id"],
                "identity": json.loads(bucket["identity_json"]),
                "sample_records": json.loads(bucket["samples_json"]),
            }
        output.append(item)
    return _template(template_path).render(
        comparisons_json=json.dumps(output, ensure_ascii=False, indent=2)
    )


def response_format(
    comparisons: Sequence[sqlite3.Row], *, allow_ties: bool = True
) -> dict[str, Any]:
    ids = [_display_id(row["comparison_id"]) for row in comparisons]
    winners = {
        bucket_id(key)
        for row in comparisons
        for key in (row["bucket_a"], row["bucket_b"])
    }
    if allow_ties:
        winners.add("tie")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance_bucket_diagnostic",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["items"],
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": len(ids),
                        "maxItems": len(ids),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "winner_bucket_id", "reason"],
                            "properties": {
                                "id": {"type": "string", "enum": ids},
                                "winner_bucket_id": {
                                    "type": "string",
                                    "enum": sorted(winners),
                                },
                                "reason": {"type": "string", "minLength": 1, "maxLength": 160},
                            },
                        },
                    }
                },
            },
        },
    }


def validate_response(
    payload: Any,
    comparisons: Sequence[sqlite3.Row],
    *,
    allow_ties: bool = True,
    max_reason_words: int = 12,
) -> list[dict[str, str]]:
    if not isinstance(payload, Mapping) or set(payload) != {"items"}:
        raise ValueError("response must contain only items")
    expected = {_display_id(row["comparison_id"]): row for row in comparisons}
    output = {}
    for item in payload["items"] if isinstance(payload["items"], list) else ():
        if not isinstance(item, Mapping) or set(item) != {
            "id",
            "winner_bucket_id",
            "reason",
        }:
            raise ValueError("response item fields differ from contract")
        item_id = str(item["id"])
        winner = str(item["winner_bucket_id"])
        reason = str(item["reason"]).strip()
        if item_id not in expected or item_id in output:
            raise ValueError("invalid comparison ID")
        row = expected[item_id]
        allowed = {bucket_id(row["bucket_a"]), bucket_id(row["bucket_b"])}
        if allow_ties:
            allowed.add("tie")
        if winner not in allowed:
            raise ValueError("winner is not a candidate in this comparison")
        if not reason or len(reason.split()) > max_reason_words:
            raise ValueError(
                f"reason must contain 1-{max_reason_words} words"
            )
        winner_key = winner
        if winner == bucket_id(row["bucket_a"]):
            winner_key = row["bucket_a"]
        elif winner == bucket_id(row["bucket_b"]):
            winner_key = row["bucket_b"]
        output[item_id] = {
            "id": item_id,
            "winner_bucket_id": winner,
            "winner_key": winner_key,
            "reason": reason,
        }
    if set(output) != set(expected):
        raise ValueError("response omitted or duplicated comparison IDs")
    return [output[item_id] for item_id in expected]


def _prepare_requests(
    connection: sqlite3.Connection,
    condition: str,
    *,
    template_path: Path = TEMPLATE,
    model: str = base.MODEL,
) -> list[str]:
    batch_ids = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT batch_id FROM comparisons WHERE phase=? ORDER BY batch_id",
            (condition,),
        )
    ]
    for batch_id in batch_ids:
        prompt = render_request(connection, batch_id, template_path=template_path)
        connection.execute(
            "INSERT OR IGNORE INTO requests "
            "(batch_id, phase, prompt_sha256, prompt, status, requested_model) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (batch_id, condition, base._hash(prompt), prompt, model),
        )
        cached = connection.execute(
            "SELECT prompt_sha256 FROM requests WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        if cached != base._hash(prompt):
            raise ValueError(f"cached prompt changed for {batch_id}")
    connection.commit()
    return batch_ids


def run_condition(
    connection: sqlite3.Connection,
    condition: str,
    *,
    parallelism: int,
    token_budget: int,
    env_file: Path | None,
    template_path: Path = TEMPLATE,
    allow_ties: bool = True,
    credential_env: str | None = None,
    model: str = base.MODEL,
    reasoning_effort: str = base.REASONING_EFFORT,
    max_completion_tokens: int = base.MAX_COMPLETION_TOKENS,
    max_tokens_parameter: str = "max_completion_tokens",
    base_url: str = base.BASE_URL,
    provider: str = "openai",
    request_extra_body: Mapping[str, Any] | None = None,
    max_cost_usd: float | None = None,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
    max_reason_words: int = 12,
) -> None:
    if max_tokens_parameter not in {"max_completion_tokens", "max_tokens"}:
        raise ValueError(f"unsupported max-token parameter: {max_tokens_parameter}")
    batch_ids = _prepare_requests(
        connection, condition, template_path=template_path, model=model
    )
    pending = [
        batch_id
        for batch_id in batch_ids
        if connection.execute(
            "SELECT status FROM requests WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        != "complete"
    ]
    if not pending:
        return
    spent = connection.execute(
        "SELECT coalesce(sum(input_tokens + output_tokens), 0) FROM requests"
    ).fetchone()[0]
    if spent >= token_budget:
        raise RuntimeError("actual-token budget is exhausted")
    client, credential = openai_compatible_client(
        base_url=base_url,
        provider=provider,
        env_file=env_file,
        credential_env=credential_env,
        max_connections=parallelism,
        timeout_s=300,
    )
    jobs = {}
    for batch_id in pending:
        prompt = connection.execute(
            "SELECT prompt FROM requests WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        jobs[batch_id] = (prompt, _comparisons(connection, batch_id))

    def call(batch_id: str, prompt: str, comparisons: Sequence[sqlite3.Row]):
        last_error = None
        total_input_tokens = 0
        total_output_tokens = 0
        attempts = 0
        validation_failures = 0
        while attempts < 8:
            attempts += 1
            try:
                request: dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "reasoning_effort": reasoning_effort,
                    "response_format": response_format(
                        comparisons, allow_ties=allow_ties
                    ),
                }
                request[max_tokens_parameter] = max_completion_tokens
                if request_extra_body:
                    request["extra_body"] = dict(request_extra_body)
                completion = client.chat.completions.create(
                    **request,
                )
                usage = completion.usage
                total_input_tokens += int(
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                total_output_tokens += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
                items = validate_response(
                    json.loads(completion.choices[0].message.content),
                    comparisons,
                    allow_ties=allow_ties,
                    max_reason_words=max_reason_words,
                )
                return (
                    batch_id,
                    items,
                    str(completion.model),
                    total_input_tokens,
                    total_output_tokens,
                    attempts,
                    None,
                )
            except ValueError as error:
                last_error = f"{type(error).__name__}: {error}"
                validation_failures += 1
                if validation_failures >= 3:
                    break
                time.sleep(2 ** (validation_failures - 1))
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempts < 8:
                    time.sleep(min(30, 2 ** attempts))
        return (
            batch_id,
            None,
            None,
            total_input_tokens,
            total_output_tokens,
            attempts,
            last_error,
        )

    def cost(input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * input_cost_per_million
            + output_tokens * output_cost_per_million
        ) / 1_000_000

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        for offset in range(0, len(pending), parallelism):
            wave = pending[offset : offset + parallelism]
            if max_cost_usd is not None:
                reported = connection.execute(
                    "SELECT coalesce(sum(input_tokens),0), "
                    "coalesce(sum(output_tokens),0) FROM requests"
                ).fetchone()
                reservation = sum(
                    cost(
                        3 * len(jobs[batch_id][0].encode("utf-8")),
                        3 * max_completion_tokens,
                    )
                    for batch_id in wave
                )
                if cost(int(reported[0]), int(reported[1])) + reservation > max_cost_usd:
                    raise RuntimeError("conservative dollar budget is exhausted")
            futures = {
                pool.submit(call, batch_id, *jobs[batch_id]): batch_id
                for batch_id in wave
            }
            wave_successes = 0
            for future in concurrent.futures.as_completed(futures):
                batch_id, items, model, input_tokens, output_tokens, attempts, error = (
                    future.result()
                )
                if error:
                    connection.execute(
                        "UPDATE requests SET status='failed', attempts=attempts+?, "
                        "input_tokens=input_tokens+?, output_tokens=output_tokens+?, error=? "
                        "WHERE batch_id=?",
                        (attempts, input_tokens, output_tokens, error, batch_id),
                    )
                    connection.commit()
                    continue
                wave_successes += 1
                current = connection.execute(
                    "SELECT coalesce(sum(input_tokens + output_tokens), 0) FROM requests "
                ).fetchone()[0]
                if current + input_tokens + output_tokens > token_budget:
                    raise RuntimeError("actual-token budget exceeded")
                comparisons = {
                    _display_id(row["comparison_id"]): row
                    for row in _comparisons(connection, batch_id)
                }
                for item in items:
                    comparison_id = comparisons[item["id"]]["comparison_id"]
                    connection.execute(
                        "UPDATE comparisons SET winner=?, reason=? WHERE comparison_id=?",
                        (item["winner_key"], item["reason"], comparison_id),
                    )
                connection.execute(
                    "UPDATE requests SET status='complete', attempts=attempts+?, served_model=?, "
                    "input_tokens=input_tokens+?, output_tokens=output_tokens+?, response_json=?, error=NULL, "
                    "completed_at=? WHERE batch_id=?",
                    (
                        attempts,
                        model,
                        input_tokens,
                        output_tokens,
                        base._canonical_json(items),
                        time.time(),
                        batch_id,
                    ),
                )
                connection.commit()
            if not wave_successes:
                raise RuntimeError("every request in the current wave failed")
    failed = connection.execute(
        "SELECT count(*) FROM requests WHERE phase=? AND status!='complete'",
        (condition,),
    ).fetchone()[0]
    if failed:
        raise RuntimeError(f"{failed} {condition} request(s) failed")
    print(f"completed {condition} using credential {credential}")


def _results(connection: sqlite3.Connection, condition: str) -> dict[str, sqlite3.Row]:
    return {
        row["comparison_id"].rsplit("-", 1)[1]: row
        for row in connection.execute(
            "SELECT comparison_id, bucket_a, bucket_b, winner, reason "
            "FROM comparisons WHERE phase=?",
            (condition,),
        )
    }


def _agreement(
    left: Mapping[str, sqlite3.Row], right: Mapping[str, sqlite3.Row]
) -> float:
    if set(left) != set(right):
        raise ValueError("diagnostic conditions cover different pairs")
    return sum(left[key]["winner"] == right[key]["winner"] for key in left) / len(left)


def summarize(connection: sqlite3.Connection, output_dir: Path) -> dict[str, Any]:
    results = {condition: _results(connection, condition) for condition in CONDITIONS}
    contrasts = {
        "n10_exact_repeat": ("n10_base", "n10_repeat"),
        "n10_matched_reversal": ("n10_base", "n10_reverse"),
        "n10_rebatched_context": ("n10_base", "n10_rebatch"),
        "n5_exact_repeat": ("n5_base", "n5_repeat"),
        "n5_matched_reversal": ("n5_base", "n5_reverse"),
        "n5_rebatched_context": ("n5_base", "n5_rebatch"),
        "n10_vs_n5": ("n10_base", "n5_base"),
    }
    metrics = {
        name: _agreement(results[left], results[right])
        for name, (left, right) in contrasts.items()
    }
    metrics["tie_rates"] = {
        condition: sum(row["winner"] == "tie" for row in rows.values()) / len(rows)
        for condition, rows in results.items()
    }
    metrics["candidate_1_rates"] = {
        condition: sum(row["winner"] == row["bucket_a"] for row in rows.values())
        / len(rows)
        for condition, rows in results.items()
    }
    metrics["recommended_batch_size"] = (
        5
        if metrics["n5_matched_reversal"]
        >= metrics["n10_matched_reversal"] + 0.03
        else 10
    )
    metrics["diagnostic_passed"] = (
        metrics["n10_exact_repeat"] >= 0.90
        and max(
            metrics["n10_matched_reversal"], metrics["n5_matched_reversal"]
        )
        >= 0.90
    )

    review = []
    for contrast, (left, right) in contrasts.items():
        for pair_id in results[left]:
            first, second = results[left][pair_id], results[right][pair_id]
            if first["winner"] == second["winner"]:
                continue
            review.append(
                {
                    "contrast": contrast,
                    "pair_id": f"P{pair_id}",
                    "candidate_1": json.loads(first["bucket_a"]),
                    "candidate_2": json.loads(first["bucket_b"]),
                    "first": {"winner": first["winner"], "reason": first["reason"]},
                    "second": {
                        "winner": second["winner"],
                        "reason": second["reason"],
                    },
                }
            )
    write_json_atomic(output_dir / "diagnostic_disagreements.json", review)
    return metrics


def run(
    source_dir: Path,
    output_dir: Path,
    *,
    parallelism: int,
    token_budget: int,
    env_file: Path | None,
) -> dict[str, Any]:
    if not (output_dir / "manifest.json").is_file():
        initialize(source_dir, output_dir)
    connection = base._connect(output_dir / "requests.sqlite3")
    base_pairs = [
        (row["bucket_a"], row["bucket_b"])
        for row in connection.execute(
            "SELECT bucket_a, bucket_b FROM comparisons WHERE phase='n10_base' "
            "ORDER BY comparison_id"
        )
    ]
    for condition in CONDITIONS:
        _insert_condition(connection, base_pairs, condition)
    for condition in CONDITIONS:
        run_condition(
            connection,
            condition,
            parallelism=parallelism,
            token_budget=token_budget,
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
            "status": "diagnostic_complete_pending_review",
            "actual_token_budget": token_budget,
            "actual_input_tokens": int(tokens[0]),
            "actual_output_tokens": int(tokens[1]),
            "completed_request_count": sum(served_models.values()),
            "served_models": served_models,
            "conditions": CONDITIONS,
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
    parser.add_argument("--token-budget", type=int, default=base.TOKEN_BUDGET)
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
            token_budget=args.token_budget,
            env_file=args.env_file,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
