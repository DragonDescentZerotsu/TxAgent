"""Run degree-25 rankings over the reviewed Bioavailability V10 semantic map."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from os.path import commonprefix
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from jinja2 import Environment, StrictUndefined
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from semantic_buckets import bioavailability_semantic_readout_v1 as semantic
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "bioavailability_semantic_degree25.v1"
TASK_ID = "bioavailability_ma"
DEGREE = 25
RECORD_COUNT = 414_841
SEMANTIC_ROOT = semantic.DEFAULT_OUTPUT
INPUT_ROOT = SEMANTIC_ROOT
FINAL_MAP = SEMANTIC_ROOT / "semantic_bucket_map.parquet"
FINAL_MAP_MANIFEST = SEMANTIC_ROOT / "semantic_bucket_map_manifest.json"
DEFAULT_OUTPUT = SEMANTIC_ROOT / "degree25_rankings"
TEMPLATE = (
    Path(__file__).with_name("prompts")
    / "bioavailability_relevance_bucket_level_v2.jinja"
)
RANKING_PROMPT_APPROVAL_MANIFEST = semantic.V3_ROOT / "manifest.json"
RANKING_PROMPT_APPROVAL_FIELD = "prompt_sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(payload: dict[str, Any]) -> str:
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        TEMPLATE.read_text(encoding="utf-8")
    )
    return template.render(
        compact_payload_json=semantic._canonical_json(payload)
    ).strip()


def _fit_bradley_terry(
    outcomes: list[tuple[str, str, str]], l2_penalty: float = 0.3
) -> dict[str, float]:
    keys = sorted({key for left, right, _ in outcomes for key in (left, right)})
    index = {key: position for position, key in enumerate(keys)}
    left = np.asarray([index[a] for a, _, _ in outcomes])
    right = np.asarray([index[b] for _, b, _ in outcomes])
    observed = np.asarray(
        [0.5 if winner == "TIE" else winner == a for a, _, winner in outcomes],
        dtype=float,
    )

    def objective(scores: np.ndarray) -> tuple[float, np.ndarray]:
        difference = scores[left] - scores[right]
        probability = 1.0 / (1.0 + np.exp(-np.clip(difference, -30, 30)))
        loss = -float(
            (
                observed * np.log(probability)
                + (1 - observed) * np.log(1 - probability)
            ).sum()
        ) + 0.5 * l2_penalty * float(scores @ scores)
        residual = probability - observed
        gradient = l2_penalty * scores
        np.add.at(gradient, left, residual)
        np.add.at(gradient, right, -residual)
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
    centered = result.x - result.x.mean()
    return dict(zip(keys, centered.tolist(), strict=True))


def _summarize(values: set[str]) -> Any:
    ordered = sorted(values)
    if len(ordered) <= 12:
        return ordered
    return {"distinct_value_count": len(ordered), "examples": ordered[:12]}


def _bucket_payload(
    bucket_id: str,
    atom_ids: list[str],
    lookup: dict[str, dict[str, Any]],
    sample_cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_source: dict[str, list[str]] = {}
    for atom_id in atom_ids:
        by_source.setdefault(str(lookup[atom_id]["source_id"]), []).append(atom_id)
    components = []
    for source, members in sorted(by_source.items()):
        components.append(
            {
                "source_id": source,
                "canonical_dimensions": {
                    column: _summarize(
                        {str(lookup[atom]["values"][column]) for atom in members}
                    )
                    for column in semantic.PROMPT_DIMENSION_COLUMNS[source]
                },
            }
        )

    ordered = sorted(
        atom_ids,
        key=lambda atom: (semantic._stable_id("ranking-sample", bucket_id, atom), atom),
    )
    selected = []
    seen_sources: set[str] = set()
    for require_new_source in (True, False):
        for atom_id in ordered:
            source = str(lookup[atom_id]["source_id"])
            if atom_id in selected or (require_new_source and source in seen_sources):
                continue
            selected.append(atom_id)
            seen_sources.add(source)
            if len(selected) == semantic.SAMPLE_LIMIT:
                break
        if len(selected) == semantic.SAMPLE_LIMIT:
            break
    samples = []
    for atom_id in selected:
        source = str(lookup[atom_id]["source_id"])
        card = {
            column: sample_cards[atom_id][column]
            for column in semantic.PROMPT_DIMENSION_COLUMNS[source]
            if column in sample_cards[atom_id]
        }
        samples.append({"source_id": source, **card})
    if not samples:
        raise ValueError(f"semantic bucket has no sample records: {bucket_id}")
    return {
        "source": "+".join(sorted(by_source)),
        "identity": {"source_components": components},
        "sample_records": samples,
    }


def build(
    output: Path = DEFAULT_OUTPUT,
    *,
    final_map: Path = FINAL_MAP,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("final_map_sha256") != _sha256(final_map):
            raise ValueError("existing ranking build uses a different semantic map")
        return manifest
    map_manifest = json.loads(FINAL_MAP_MANIFEST.read_text())
    if map_manifest.get("status") != "complete_reviewed":
        raise ValueError("the agent-reviewed final semantic map is required")
    if map_manifest.get("semantic_bucket_map_sha256") != _sha256(final_map):
        raise ValueError("final semantic map hash does not match its manifest")
    approval = json.loads(RANKING_PROMPT_APPROVAL_MANIFEST.read_text())
    if approval.get(RANKING_PROMPT_APPROVAL_FIELD) != _sha256(TEMPLATE):
        raise ValueError("the reviewed ranking template changed")

    atoms = pd.read_parquet(INPUT_ROOT / "input_atoms.parquet")
    sample_cards = semantic._read_gzip_json(INPUT_ROOT / "sample_cards.json.gz")
    lookup = semantic._atom_lookup(atoms)
    mapping = pd.read_parquet(final_map)
    required = {"level", "semantic_bucket_id", "atom_id"}
    if not required <= set(mapping):
        raise ValueError(f"semantic map lacks columns: {sorted(required - set(mapping))}")
    if mapping["atom_id"].duplicated().any() or set(mapping["atom_id"]) != set(atoms["atom_id"]):
        raise ValueError("semantic map does not cover every atom exactly once")

    output.mkdir(parents=True, exist_ok=True)
    connection = semantic._request_database(output / "requests.sqlite3")
    schedule = []
    for level, level_rows in mapping.groupby("level", sort=True):
        buckets = {
            bucket: sorted(rows["atom_id"].tolist())
            for bucket, rows in level_rows.groupby("semantic_bucket_id", sort=True)
        }
        payloads = {
            bucket: _bucket_payload(bucket, members, lookup, sample_cards)
            for bucket, members in buckets.items()
        }
        for index, (left, right) in enumerate(
            semantic.degree_edges(list(buckets), level=str(level), degree=DEGREE)
        ):
            prompt = _render({"groups": {left: payloads[left], right: payloads[right]}})
            phase = f"degree25|{level}"
            request_id = semantic._request_id("ranking", phase, prompt)
            semantic._queue_request(
                connection,
                request_id=request_id,
                kind="ranking",
                phase=phase,
                prompt=prompt,
                reasoning_effort="low",
                max_tokens=semantic.LOW_MAX_TOKENS,
                validation={"candidate_bucket_ids": [left, right]},
                commit=False,
            )
            schedule.append(
                {
                    "comparison_id": f"{level}-{index:07d}",
                    "request_id": request_id,
                    "level": level,
                    "bucket_a_id": left,
                    "bucket_b_id": right,
                }
            )
        connection.commit()
    connection.close()
    schedule_frame = pd.DataFrame(schedule)
    schedule_frame.to_parquet(output / "comparison_schedule.parquet", index=False)
    counts = {}
    for level, rows in schedule_frame.groupby("level", sort=True):
        degrees = Counter(
            bucket
            for pair in rows[["bucket_a_id", "bucket_b_id"]].itertuples(index=False)
            for bucket in pair
        )
        counts[str(level)] = {
            "buckets": len(degrees),
            "comparisons": len(rows),
            "minimum_degree": min(degrees.values()),
            "maximum_degree": max(degrees.values()),
        }
    manifest = {
        "version": VERSION,
        "status": "prepared",
        "model": semantic.MODEL,
        "base_url": semantic.BASE_URL,
        "reasoning_effort": "low",
        "single_comparison_per_request": True,
        "sample_limit": semantic.SAMPLE_LIMIT,
        "samples_from_distinct_pair_buckets": True,
        "target_degree": DEGREE,
        "final_map": str(final_map),
        "final_map_sha256": _sha256(final_map),
        "final_map_manifest_sha256": _sha256(FINAL_MAP_MANIFEST),
        "prompt": str(TEMPLATE),
        "prompt_sha256": _sha256(TEMPLATE),
        "comparison_schedule_sha256": _sha256(output / "comparison_schedule.parquet"),
        "levels": counts,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _adjudicate_unique_prefix_failures(
    connection: sqlite3.Connection, request_ids: list[str]
) -> int:
    resolved = 0
    for request_id in dict.fromkeys(request_ids):
        request = connection.execute(
            "SELECT status,validation_json FROM requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if request is None or request["status"] != "failed":
            continue
        receipts = connection.execute(
            """
            SELECT receipt_json FROM request_attempt_receipts
            WHERE request_id=? ORDER BY attempt
            """,
            (request_id,),
        ).fetchall()
        candidates = json.loads(request["validation_json"])["candidate_bucket_ids"]
        selected = None
        for receipt in receipts:
            attempt = json.loads(receipt["receipt_json"])
            try:
                raw = json.loads(attempt["raw_response"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            value = raw.get("winner_bucket_id") if set(raw) == {"winner_bucket_id"} else None
            if attempt.get("model") != semantic.MODEL or not isinstance(value, str):
                continue
            matches = [
                candidate
                for candidate in candidates
                if (
                    abs(len(candidate) - len(value)) <= 2
                    and (candidate.startswith(value) or value.startswith(candidate))
                )
                or len(commonprefix((candidate, value))) >= 16
                or SequenceMatcher(None, candidate, value, autojunk=False).ratio() >= 0.9
            ]
            if len(matches) == 1:
                selected = attempt, value, matches[0]
                break
        if selected is None:
            continue
        attempt, value, winner = selected
        fallback = (
            "adjudication_fallback: unique_bucket_id_copy_error "
            f"raw={value} resolved={winner}"
        )
        connection.execute(
            """
            UPDATE requests SET status='complete',response_json=?,reasoning_content=?,
                served_model=?,provider_name=?,provider_base_url=?,error=? WHERE request_id=?
            """,
            (
                semantic._canonical_json({"winner_bucket_id": winner}),
                attempt.get("reasoning_content"),
                semantic.MODEL,
                attempt.get("provider_name"),
                attempt.get("provider_base_url"),
                fallback,
                request_id,
            ),
        )
        resolved += 1
    connection.commit()
    return resolved


def _completed_request_lookup(
    connection: sqlite3.Connection, request_ids: list[str]
) -> dict[str, sqlite3.Row]:
    requested = set(request_ids)
    rows = connection.execute(
        """
        SELECT request_id,response_json,served_model,provider_name,provider_base_url,
               attempts,input_tokens,output_tokens,error
        FROM requests WHERE status='complete'
        """
    )
    lookup = {row["request_id"]: row for row in rows if row["request_id"] in requested}
    missing = requested - lookup.keys()
    if missing:
        raise ValueError(f"ranking requests are not complete: {sorted(missing)[:5]}")
    return lookup


def run(output: Path = DEFAULT_OUTPUT, *, parallelism: int = 128) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("status") not in {"prepared", "running", "incomplete"}:
        return manifest
    maximum_context, endpoint_models = semantic._endpoint_contract()
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    schedule = pd.read_parquet(output / "comparison_schedule.parquet")
    connection = semantic._request_database(output / "requests.sqlite3")
    manifest["status"] = "running"
    write_json_atomic(output / "manifest.json", manifest)
    client, endpoint_receipts, total_parallelism = semantic._build_completion_pool(
        parallelism
    )
    try:
        request_ids = schedule["request_id"].tolist()
        try:
            semantic._run_pending(
                connection,
                request_ids,
                parallelism=total_parallelism,
                client=client,
                request_slots=threading.BoundedSemaphore(total_parallelism),
            )
        except RuntimeError:
            _adjudicate_unique_prefix_failures(connection, request_ids)
            remaining = connection.execute(
                "SELECT COUNT(*) FROM requests WHERE status!='complete'"
            ).fetchone()[0]
            if remaining:
                raise
        requests = _completed_request_lookup(connection, request_ids)
        comparisons = []
        for row in schedule.itertuples(index=False):
            request = requests[row.request_id]
            winner = json.loads(request["response_json"])["winner_bucket_id"]
            comparisons.append(
                {
                    **row._asdict(),
                    "winner_bucket_id": winner,
                    "served_model": request["served_model"],
                    "provider_name": request["provider_name"],
                    "provider_base_url": request["provider_base_url"],
                    "attempts": request["attempts"],
                    "input_tokens": request["input_tokens"],
                    "output_tokens": request["output_tokens"],
                    "adjudication_fallback": request["error"],
                }
            )
        comparison_frame = pd.DataFrame(comparisons)
        comparison_frame.to_parquet(output / "comparisons.parquet", index=False)

        ranking_rows = []
        for level, rows in comparison_frame.groupby("level", sort=True):
            outcomes = list(
                rows[["bucket_a_id", "bucket_b_id", "winner_bucket_id"]]
                .itertuples(index=False, name=None)
            )
            scores = _fit_bradley_terry(outcomes)
            counts = Counter(key for left, right, _ in outcomes for key in (left, right))
            ordered = sorted(scores, key=lambda key: (-scores[key], key))
            for index, bucket in enumerate(ordered):
                ranking_rows.append(
                    {
                        "task_id": TASK_ID,
                        "level": level,
                        "semantic_bucket_id": bucket,
                        "bradley_terry_score": scores[bucket],
                        "level_rank": index + 1,
                        "level_percentile": 100
                        * (len(ordered) - 1 - index)
                        / max(1, len(ordered) - 1),
                        "comparison_count": counts[bucket],
                    }
                )
        rankings = pd.DataFrame(ranking_rows)
        rankings.to_parquet(output / "semantic_bucket_rankings.parquet", index=False)

        atom_map = pd.read_parquet(FINAL_MAP)[["atom_id", "semantic_bucket_id"]]
        records = pd.read_parquet(semantic.RECORD_MAP)
        records["atom_id"] = [
            semantic._stable_id("atom", level, source, pair_bucket)
            for level, source, pair_bucket in records[
                ["level", "source_id", "pair_bucket_key"]
            ].itertuples(index=False, name=None)
        ]
        ranked_records = (
            records.merge(atom_map, on="atom_id", how="left", validate="many_to_one")
            .merge(
                rankings,
                on=["level", "semantic_bucket_id"],
                how="left",
                validate="many_to_one",
            )
        )
        if len(ranked_records) != RECORD_COUNT or ranked_records["level_rank"].isna().any():
            raise ValueError("degree-25 rankings do not cover every L2-L6 record")
        ranked_records.to_parquet(output / "record_relevance_rankings.parquet", index=False)

        manifest.update(
            {
                "status": "complete",
                "completed_at": time.time(),
                "endpoint_models_at_start": endpoint_models,
                "endpoint_pool_at_start": endpoint_receipts,
                "endpoint_pool_final_snapshot": client.snapshot(),
                "per_endpoint_parallelism": parallelism,
                "total_parallelism": total_parallelism,
                "max_model_len": maximum_context,
                "artifacts": {
                    "comparisons": {
                        "rows": len(comparison_frame),
                        "sha256": _sha256(output / "comparisons.parquet"),
                    },
                    "semantic_bucket_rankings": {
                        "rows": len(rankings),
                        "sha256": _sha256(output / "semantic_bucket_rankings.parquet"),
                    },
                    "record_relevance_rankings": {
                        "rows": len(ranked_records),
                        "sha256": _sha256(output / "record_relevance_rankings.parquet"),
                    },
                },
                "adjudication_fallback_count": int(
                    comparison_frame["adjudication_fallback"].notna().sum()
                ),
            }
        )
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        connection.close()
        write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "build-and-run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-map", type=Path, default=FINAL_MAP)
    parser.add_argument("--parallelism", type=int, default=128)
    args = parser.parse_args()
    if args.command in {"build", "build-and-run"}:
        result = build(args.output, final_map=args.final_map)
    if args.command in {"run", "build-and-run"}:
        result = run(args.output, parallelism=args.parallelism)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
