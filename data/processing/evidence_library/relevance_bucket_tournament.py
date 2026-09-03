"""Build and pilot a GPT pairwise ranking of V9 BBB relevance buckets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sqlite3
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from scipy.optimize import minimize
from scipy.stats import spearmanr

from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client
from tools.chembl_tool.common.json_utils import write_json_atomic

VERSION = "bbb_relevance_bucket_tournament.v1"
PROMPT_VERSION = "bbb_relevance_bucket_tournament_prompt.v1"
MODEL = "gpt-5.4-mini"
BASE_URL = "https://api.openai.com/v1"
REASONING_EFFORT = "medium"
TOKEN_BUDGET = 20_000_000
COMPARISONS_PER_REQUEST = 10
MAX_COMPLETION_TOKENS = 4_096
UNKNOWN = "__unknown__"
ROOT = Path("data/evidence_libraries/bbb_martins/v9")
INPUT = ROOT / "03_pair_buckets/records.parquet"
OUTPUT = ROOT / "relevance_bucket_tournament_v1"
TEMPLATE = Path(__file__).with_name("prompts") / "relevance_bucket_tournament_v1.jinja"

RELEVANCE_BUCKET_COLUMNS = {
    "direct_bbb": (
        "canonical_endpoint_name",
        "measurement_kind",
        "canonical_assay_context",
    ),
    "efflux_transport": (
        "canonical_endpoint_name",
        "measurement_kind",
        "canonical_transporter_identifier",
        "canonical_evidence_type",
    ),
    "influx_transport": (
        "canonical_endpoint_name",
        "measurement_kind",
        "canonical_transport_mechanism",
        "canonical_kinetic_symbol",
    ),
    "passive_permeability": (
        "canonical_endpoint_name",
        "measurement_kind",
        "canonical_assay_type",
    ),
}

BASE_CARD_COLUMNS = (
    "canonical_endpoint_name",
    "measurement_kind",
    "canonical_measurement_text",
    "canonical_unit_text",
    "finite_scalar_value",
    "canonical_category_id",
    "canonical_assay_context",
    "canonical_species_context",
)
SOURCE_CARD_COLUMNS = {
    "direct_bbb": (
        "bbb_permeability_label",
        "bbb_transport_label",
        "assay_model",
        "species",
        "qualifying_conditions",
    ),
    "efflux_transport": (
        "canonical_transporter_identifier",
        "canonical_evidence_type",
        "interaction_conclusion",
        "assay_system",
        "perturbation",
        "needs_more_context",
    ),
    "influx_transport": (
        "canonical_transport_mechanism",
        "canonical_kinetic_symbol",
        "mediator_name",
        "mediator_identifier",
        "evidence_basis",
        "transport_mechanism",
        "assay_model",
    ),
    "passive_permeability": (
        "canonical_assay_type",
        "assay_type",
        "biological_system",
        "passive_bbb_interpretation",
        "metric_uncertainty",
        "needs_more_context",
    ),
}
DIVERSITY_COLUMNS = (
    "pair_bucket_key",
    "canonical_unit_text",
    "canonical_species_context",
    "canonical_assay_context",
    "assay_system",
    "biological_system",
    "canonical_category_id",
    "interaction_conclusion",
    "perturbation",
    "canonical_smiles",
)
EXPECTED_COUNTS = {
    "direct_bbb": (16_890, 11_261),
    "efflux_transport": (42_389, 1_967),
    "influx_transport": (177, 12),
    "passive_permeability": (474, 97),
}
PILOT_COUNTS = {
    "direct_bbb": 202,
    "efflux_transport": 201,
    "influx_transport": 12,
    "passive_permeability": 97,
}


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def relevance_bucket(record: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    source = str(record.get("source_id") or "")
    if source not in RELEVANCE_BUCKET_COLUMNS:
        raise ValueError(f"unsupported BBB source_id: {source!r}")
    identity = {"source_id": source}
    identity.update(
        {
            column: str(_clean(record.get(column)) or UNKNOWN)
            for column in RELEVANCE_BUCKET_COLUMNS[source]
        }
    )
    # Dict insertion order is part of the human-readable identity contract.
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":")), identity


def visible_card(record: Mapping[str, Any]) -> dict[str, Any]:
    source = str(record.get("source_id") or "")
    columns = BASE_CARD_COLUMNS + SOURCE_CARD_COLUMNS[source]
    card = {column: _clean(record.get(column)) for column in columns}
    card = {key: value for key, value in card.items() if value is not None}
    if any("reference" in key.lower() for key in card):
        raise AssertionError("reference semantics leaked into a sample card")
    return card


def select_diverse_records(
    records: Sequence[Mapping[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    candidates: list[tuple[Mapping[str, Any], dict[str, Any], str]] = []
    seen_cards: set[str] = set()
    for record in sorted(records, key=lambda row: str(row.get("canonical_record_id") or "")):
        card = visible_card(record)
        encoded = _canonical_json(card)
        if card and encoded not in seen_cards:
            candidates.append((record, card, encoded))
            seen_cards.add(encoded)
    selected: list[tuple[Mapping[str, Any], dict[str, Any], str]] = []
    seen_values: dict[str, set[str]] = defaultdict(set)
    while candidates and len(selected) < limit:
        def score(item: tuple[Mapping[str, Any], dict[str, Any], str]) -> tuple[Any, ...]:
            record, _, encoded = item
            novelty = sum(
                str(_clean(record.get(column)) or UNKNOWN) not in seen_values[column]
                for column in DIVERSITY_COLUMNS
            )
            unit = str(_clean(record.get("canonical_unit_text")) or UNKNOWN)
            value = _clean(record.get("finite_scalar_value"))
            prior = [
                float(_clean(row.get("finite_scalar_value")))
                for row, _, _ in selected
                if _clean(row.get("finite_scalar_value")) is not None
                and str(_clean(row.get("canonical_unit_text")) or UNKNOWN) == unit
            ]
            spread = min((abs(float(value) - old) for old in prior), default=0.0) if value is not None else 0.0
            return novelty, spread, _hash(encoded)

        chosen = max(candidates, key=score)
        candidates.remove(chosen)
        selected.append(chosen)
        for column in DIVERSITY_COLUMNS:
            seen_values[column].add(str(_clean(chosen[0].get(column)) or UNKNOWN))
    return [card for _, card, _ in selected]


def _input_columns() -> list[str]:
    return sorted(
        {
            "source_id",
            "pair_bucket_key",
            "assay_transfer_eligible",
            "canonical_record_id",
            "canonical_smiles",
            *BASE_CARD_COLUMNS,
            *DIVERSITY_COLUMNS,
            *(column for columns in RELEVANCE_BUCKET_COLUMNS.values() for column in columns),
            *(column for columns in SOURCE_CARD_COLUMNS.values() for column in columns),
        }
    )


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS buckets (
          relevance_bucket TEXT PRIMARY KEY, source_id TEXT NOT NULL,
          identity_json TEXT NOT NULL, samples_json TEXT NOT NULL,
          record_count INTEGER NOT NULL, scorable INTEGER NOT NULL,
          unscorable_reason TEXT
        );
        CREATE TABLE IF NOT EXISTS comparisons (
          comparison_id TEXT PRIMARY KEY, phase TEXT NOT NULL,
          batch_id TEXT NOT NULL, bucket_a TEXT NOT NULL, bucket_b TEXT NOT NULL,
          audit_of TEXT, winner TEXT, reason TEXT
        );
        CREATE TABLE IF NOT EXISTS requests (
          batch_id TEXT PRIMARY KEY, phase TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
          prompt TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          requested_model TEXT NOT NULL, served_model TEXT,
          input_tokens INTEGER NOT NULL DEFAULT 0,
          output_tokens INTEGER NOT NULL DEFAULT 0, response_json TEXT, error TEXT,
          completed_at REAL
        );
        """
    )
    return connection


def build_inventory(
    input_path: Path,
    output_dir: Path,
    *,
    verify_v9: bool = True,
    rebuild: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "requests.sqlite3"
    manifest_path = output_dir / "manifest.json"
    mapping_path = output_dir / "pair_bucket_relevance_map.parquet"
    input_hash = file_sha256(input_path)
    prompt_hash = file_sha256(TEMPLATE)
    if not rebuild and manifest_path.is_file() and mapping_path.is_file() and database.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("version") == VERSION and prior.get("input_sha256") == input_hash and prior.get("prompt_sha256") == prompt_hash:
            return prior
    connection = _connect(database)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema_arrow.names)
    for batch in parquet.iter_batches(batch_size=20_000, columns=[c for c in _input_columns() if c in available]):
        for record in batch.to_pylist():
            if record.get("assay_transfer_eligible") is not True:
                continue
            by_pair[str(record["pair_bucket_key"])].append(record)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_to_bucket: dict[str, tuple[str, str, int]] = {}
    inconsistent_pairs = 0
    for pair_key, records in by_pair.items():
        representative = min(records, key=lambda row: str(row.get("canonical_record_id") or ""))
        key, _ = relevance_bucket(representative)
        variants = {relevance_bucket(record)[0] for record in records}
        inconsistent_pairs += len(variants) > 1
        pair_to_bucket[pair_key] = (str(representative["source_id"]), key, len(records))
        by_bucket[key].extend(records)

    rows = []
    connection.execute("DELETE FROM buckets")
    for key in sorted(by_bucket):
        records = by_bucket[key]
        _, identity = relevance_bucket(records[0])
        samples = select_diverse_records(records)
        scorable = bool(samples)
        connection.execute(
            "INSERT INTO buckets VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, identity["source_id"], key, _canonical_json(samples), len(records), int(scorable), None if scorable else "no_interpretable_structured_card"),
        )
    connection.commit()
    for pair_key, (source, key, record_count) in sorted(pair_to_bucket.items()):
        rows.append({"source_id": source, "pair_bucket_key": pair_key, "relevance_bucket": key, "record_count": record_count})
    mapping = pd.DataFrame(rows)
    mapping.to_parquet(mapping_path, index=False)

    counts = {}
    for source in RELEVANCE_BUCKET_COLUMNS:
        counts[source] = {
            "pair_buckets": int(mapping.loc[mapping.source_id == source, "pair_bucket_key"].nunique()),
            "relevance_buckets": sum(1 for records in by_bucket.values() if records[0]["source_id"] == source),
        }
    if verify_v9:
        observed = {source: (row["pair_buckets"], row["relevance_buckets"]) for source, row in counts.items()}
        if observed != EXPECTED_COUNTS:
            raise ValueError(f"V9 relevance-bucket counts changed: {observed}")
    manifest = {
        "version": VERSION,
        "status": "inventory_built",
        "input": str(input_path),
        "input_sha256": input_hash,
        "prompt": str(TEMPLATE),
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "relevance_bucket_columns": {key: list(value) for key, value in RELEVANCE_BUCKET_COLUMNS.items()},
        "missing_value": UNKNOWN,
        "assay_transfer_filter": {"field": "assay_transfer_eligible", "value": True},
        "sample_card_limit": 5,
        "sample_card_columns": {
            "common": list(BASE_CARD_COLUMNS),
            **{key: list(value) for key, value in SOURCE_CARD_COLUMNS.items()},
        },
        "sample_diversity_columns": list(DIVERSITY_COLUMNS),
        "sample_fields_explicitly_excluded": [
            "support_text",
            "extra_details",
            "molecule_name",
            "canonical_smiles",
            "source_smiles",
            "pmid",
            "all_reference_semantics_fields",
        ],
        "pair_bucket_count": len(pair_to_bucket),
        "pair_bucket_relevance_edge_count": len(pair_to_bucket),
        "pair_buckets_with_inconsistent_current_relevance_fields": inconsistent_pairs,
        "relevance_bucket_count": len(by_bucket),
        "unscorable_bucket_count": int(connection.execute("SELECT count(*) FROM buckets WHERE scorable=0").fetchone()[0]),
        "source_counts": counts,
        "artifacts": {
            "pair_bucket_relevance_map": str(mapping_path),
            "request_cache": str(database),
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    connection.close()
    return manifest


def _pilot_keys(connection: sqlite3.Connection) -> list[str]:
    selected = []
    for source, count in PILOT_COUNTS.items():
        rows = connection.execute(
            "SELECT relevance_bucket FROM buckets WHERE source_id=? AND scorable=1",
            (source,),
        ).fetchall()
        ordered = sorted((row[0] for row in rows), key=lambda key: _hash("pilot", key))
        if len(ordered) < count:
            raise ValueError(f"not enough scorable {source} buckets for pilot")
        selected.extend(ordered[:count])
    return sorted(selected)


def _orient(a: str, b: str, seed: str) -> tuple[str, str]:
    return (a, b) if int(_hash(seed, a, b), 16) % 2 == 0 else (b, a)


def _comparison(phase: str, index: int, a: str, b: str, *, audit_of: str | None = None) -> dict[str, Any]:
    comparison_id = f"{phase}-{index:05d}"
    if audit_of is None:
        a, b = _orient(a, b, comparison_id)
    return {"comparison_id": comparison_id, "phase": phase, "bucket_a": a, "bucket_b": b, "audit_of": audit_of}


def anchor_comparisons(pilot_keys: Sequence[str]) -> tuple[list[str], list[dict[str, Any]]]:
    candidates = []
    for source in RELEVANCE_BUCKET_COLUMNS:
        source_keys = [key for key in pilot_keys if json.loads(key)["source_id"] == source]
        candidates.extend(sorted(source_keys, key=lambda key: _hash("anchor", key))[:8])
    comparisons = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            comparisons.append(_comparison("anchor", len(comparisons), a, b))
    for original in sorted(comparisons, key=lambda row: _hash("anchor-audit", row["comparison_id"]))[:4]:
        comparisons.append(_comparison("anchor", len(comparisons), original["bucket_b"], original["bucket_a"], audit_of=original["comparison_id"]))
    assert len(candidates) == 32 and len(comparisons) == 500
    return candidates, comparisons


def choose_anchors(connection: sqlite3.Connection, candidates: Sequence[str]) -> list[str]:
    outcomes = _completed_outcomes(connection, phases=("anchor",), include_audits=False)
    if len(outcomes) != 496:
        raise RuntimeError("anchor round robin is not complete")
    ranked = fit_davidson(outcomes)
    order = [key for key, _ in sorted(ranked.items(), key=lambda item: item[1]) if key in candidates]
    return [order[round(position * (len(order) - 1))] for position in (0.10, 0.37, 0.63, 0.90)]


def main_comparisons(pilot_keys: Sequence[str], anchors: Sequence[str]) -> list[dict[str, Any]]:
    targets = [key for key in pilot_keys if key not in set(anchors)]
    targets.sort(key=lambda key: _hash("main", key))
    output = []
    for start in range(0, len(targets), 8):
        chunk = targets[start : start + 8]
        pairs = [(chunk[i], chunk[(i + 1) % len(chunk)]) for i in range(len(chunk))]
        needed = COMPARISONS_PER_REQUEST - len(pairs)
        additions = []
        for offset in range(needed):
            additions.append((chunk[offset % len(chunk)], anchors[(start + offset) % len(anchors)]))
        for a, b in pairs + additions:
            output.append(_comparison("main", len(output), a, b))
    assert len(output) % COMPARISONS_PER_REQUEST == 0
    return output


def audit_comparisons(base: Sequence[dict[str, Any]], anchor_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_audits = [row for row in anchor_rows if row["audit_of"]]
    unique_anchor = [row for row in anchor_rows if not row["audit_of"]]
    target_total = (len(unique_anchor) + len(base)) // 4
    needed = target_total - len(anchor_audits)
    chosen = sorted(base, key=lambda row: _hash("audit", row["comparison_id"]))[:needed]
    output = []
    for original in chosen:
        output.append(_comparison("audit", len(output), original["bucket_b"], original["bucket_a"], audit_of=original["comparison_id"]))
    assert (len(anchor_audits) + len(output)) * 4 == len(unique_anchor) + len(base)
    assert len(output) % COMPARISONS_PER_REQUEST == 0
    return output


def _insert_batches(connection: sqlite3.Connection, rows: Sequence[dict[str, Any]]) -> None:
    for start in range(0, len(rows), COMPARISONS_PER_REQUEST):
        chunk = rows[start : start + COMPARISONS_PER_REQUEST]
        if len(chunk) != COMPARISONS_PER_REQUEST:
            raise ValueError("every API batch must contain exactly 10 comparisons")
        batch_id = f"{chunk[0]['phase']}-batch-{start // COMPARISONS_PER_REQUEST:04d}"
        for row in chunk:
            connection.execute(
                "INSERT OR IGNORE INTO comparisons VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                (row["comparison_id"], row["phase"], batch_id, row["bucket_a"], row["bucket_b"], row["audit_of"]),
            )
    connection.commit()


def _template() -> Any:
    environment = Environment(loader=FileSystemLoader(str(TEMPLATE.parent)), undefined=StrictUndefined, autoescape=False)
    return environment.get_template(TEMPLATE.name)


def render_request(connection: sqlite3.Connection, batch_id: str) -> str:
    comparisons = connection.execute(
        "SELECT comparison_id, bucket_a, bucket_b FROM comparisons WHERE batch_id=? ORDER BY comparison_id",
        (batch_id,),
    ).fetchall()
    if len(comparisons) != COMPARISONS_PER_REQUEST:
        raise ValueError("request does not contain exactly 10 comparisons")
    keys = sorted({row["bucket_a"] for row in comparisons} | {row["bucket_b"] for row in comparisons})
    labels = {key: f"B{index:02d}" for index, key in enumerate(keys)}
    buckets = {}
    for key in keys:
        row = connection.execute("SELECT source_id, identity_json, samples_json FROM buckets WHERE relevance_bucket=?", (key,)).fetchone()
        buckets[labels[key]] = {"source": row["source_id"], "identity": json.loads(row["identity_json"]), "sample_records": json.loads(row["samples_json"])}
    payload = [
        {"id": row["comparison_id"], "bucket_a": labels[row["bucket_a"]], "bucket_b": labels[row["bucket_b"]]}
        for row in comparisons
    ]
    return _template().render(buckets_json=json.dumps(buckets, ensure_ascii=False, indent=2, sort_keys=True), comparisons_json=json.dumps(payload, ensure_ascii=False, indent=2))


def response_format(ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance_bucket_comparisons",
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
                            "required": ["id", "winner", "reason"],
                            "properties": {
                                "id": {"type": "string", "enum": list(ids)},
                                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                                "reason": {"type": "string", "minLength": 1, "maxLength": 160},
                            },
                        },
                    }
                },
            },
        },
    }


def validate_response(payload: Any, ids: Sequence[str]) -> list[dict[str, str]]:
    if not isinstance(payload, Mapping) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("response must contain only an items array")
    output = {}
    expected = set(ids)
    for item in payload["items"]:
        if not isinstance(item, Mapping) or set(item) != {"id", "winner", "reason"}:
            raise ValueError("response item fields differ from contract")
        item_id, winner, reason = str(item["id"]), str(item["winner"]), str(item["reason"]).strip()
        if item_id not in expected or item_id in output or winner not in {"A", "B", "tie"}:
            raise ValueError("invalid comparison result")
        if not reason or len(reason.split()) > 12:
            raise ValueError("reason must contain 1-12 words")
        output[item_id] = {"id": item_id, "winner": winner, "reason": reason}
    if set(output) != expected:
        raise ValueError("response omitted or duplicated comparison IDs")
    return [output[item_id] for item_id in ids]


def _prepare_requests(connection: sqlite3.Connection, phase: str) -> list[str]:
    batch_ids = [row[0] for row in connection.execute("SELECT DISTINCT batch_id FROM comparisons WHERE phase=? ORDER BY batch_id", (phase,))]
    for batch_id in batch_ids:
        prompt = render_request(connection, batch_id)
        connection.execute(
            "INSERT OR IGNORE INTO requests (batch_id, phase, prompt_sha256, prompt, status, requested_model) VALUES (?, ?, ?, ?, 'pending', ?)",
            (batch_id, phase, _hash(prompt), prompt, MODEL),
        )
        prior = connection.execute("SELECT prompt_sha256 FROM requests WHERE batch_id=?", (batch_id,)).fetchone()[0]
        if prior != _hash(prompt):
            raise ValueError(f"cached prompt changed for {batch_id}")
    connection.commit()
    return batch_ids


def run_phase(connection: sqlite3.Connection, phase: str, *, parallelism: int, token_budget: int, env_file: Path | None) -> None:
    batch_ids = _prepare_requests(connection, phase)
    pending = [batch_id for batch_id in batch_ids if connection.execute("SELECT status FROM requests WHERE batch_id=?", (batch_id,)).fetchone()[0] != "complete"]
    if not pending:
        return
    spent = sum(connection.execute("SELECT coalesce(sum(input_tokens + output_tokens), 0) FROM requests WHERE status='complete'").fetchone())
    if spent >= token_budget:
        raise RuntimeError("actual-token budget is exhausted")
    client, credential = openai_compatible_client(base_url=BASE_URL, provider="openai", env_file=env_file, max_connections=parallelism, timeout_s=300)
    jobs = {}
    for batch_id in pending:
        prompt = connection.execute("SELECT prompt FROM requests WHERE batch_id=?", (batch_id,)).fetchone()[0]
        ids = [r[0] for r in connection.execute("SELECT comparison_id FROM comparisons WHERE batch_id=? ORDER BY comparison_id", (batch_id,))]
        jobs[batch_id] = (prompt, ids)

    def call(batch_id: str, prompt: str, ids: Sequence[str]) -> tuple[str, Any, str | None, int, int, int, str | None]:
        last_error = None
        for attempt in range(1, 4):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort=REASONING_EFFORT,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    response_format=response_format(ids),
                )
                content = completion.choices[0].message.content
                parsed = validate_response(json.loads(content), ids)
                usage = completion.usage
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                return batch_id, parsed, str(completion.model), input_tokens, output_tokens, attempt, None
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        return batch_id, None, None, 0, 0, 3, last_error

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {
            pool.submit(call, batch_id, *jobs[batch_id]): batch_id
            for batch_id in pending
        }
        for future in concurrent.futures.as_completed(futures):
            batch_id, items, served_model, input_tokens, output_tokens, attempts, error = future.result()
            if error:
                connection.execute(
                    "UPDATE requests SET status='failed', attempts=?, error=? WHERE batch_id=?",
                    (attempts, error, batch_id),
                )
                connection.commit()
                continue
            current = connection.execute("SELECT coalesce(sum(input_tokens + output_tokens), 0) FROM requests WHERE status='complete'").fetchone()[0]
            if current + input_tokens + output_tokens > token_budget:
                raise RuntimeError("actual-token budget exceeded; results remain uncommitted")
            for item in items:
                comparison = connection.execute("SELECT audit_of FROM comparisons WHERE comparison_id=?", (item["id"],)).fetchone()
                winner = item["winner"]
                if comparison["audit_of"] and winner != "tie":
                    winner = "B" if winner == "A" else "A"
                connection.execute("UPDATE comparisons SET winner=?, reason=? WHERE comparison_id=?", (winner, item["reason"], item["id"]))
            connection.execute(
                "UPDATE requests SET status='complete', attempts=?, served_model=?, input_tokens=?, output_tokens=?, response_json=?, error=NULL, completed_at=? WHERE batch_id=?",
                (attempts, served_model, input_tokens, output_tokens, _canonical_json(items), time.time(), batch_id),
            )
            connection.commit()
    failed = connection.execute("SELECT count(*) FROM requests WHERE phase=? AND status!='complete'", (phase,)).fetchone()[0]
    if failed:
        raise RuntimeError(f"{failed} {phase} request(s) failed; completed requests are cached")
    if credential:
        print(f"completed {phase} using credential {credential}")


def _completed_outcomes(connection: sqlite3.Connection, *, phases: Sequence[str], include_audits: bool) -> list[tuple[str, str, str]]:
    placeholders = ",".join("?" for _ in phases)
    clause = "" if include_audits else " AND audit_of IS NULL"
    rows = connection.execute(
        f"SELECT bucket_a, bucket_b, winner FROM comparisons WHERE phase IN ({placeholders}) AND winner IS NOT NULL{clause}",
        tuple(phases),
    )
    return [(row[0], row[1], row[2]) for row in rows]


def fit_davidson(outcomes: Sequence[tuple[str, str, str]]) -> dict[str, float]:
    keys = sorted({key for a, b, _ in outcomes for key in (a, b)})
    index = {key: i for i, key in enumerate(keys)}
    if not keys:
        return {}
    a_index = np.asarray([index[a] for a, _, _ in outcomes])
    b_index = np.asarray([index[b] for _, b, _ in outcomes])
    result_code = np.asarray([{"A": 0, "B": 1, "tie": 2}[winner] for _, _, winner in outcomes])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.r_[parameters[: len(keys) - 1], 0.0]
        nu = math.exp(parameters[-1])
        ea, eb = np.exp(theta[a_index]), np.exp(theta[b_index])
        tie = 2 * nu * np.sqrt(ea * eb)
        denominator = ea + eb + tie
        probabilities = np.column_stack((ea, eb, tie)) / denominator[:, None]
        chosen = probabilities[np.arange(len(outcomes)), result_code]
        loss = -float(np.log(np.maximum(chosen, 1e-15)).sum()) + 1e-4 * float(theta @ theta)

        p_a, p_b, p_tie = probabilities.T
        y_a, y_b, y_tie = (result_code == value for value in range(3))
        gradient_theta = 2e-4 * theta
        np.add.at(gradient_theta, a_index, p_a + 0.5 * p_tie - y_a - 0.5 * y_tie)
        np.add.at(gradient_theta, b_index, p_b + 0.5 * p_tie - y_b - 0.5 * y_tie)
        gradient = np.r_[gradient_theta[:-1], float((p_tie - y_tie).sum())]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(len(keys)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-20, 20)] * (len(keys) - 1) + [(-10, 10)],
    )
    if not result.success:
        raise RuntimeError(f"Davidson fit failed: {result.message}")
    theta = np.r_[result.x[: len(keys) - 1], 0.0]
    theta -= theta.mean()
    return dict(zip(keys, theta.tolist(), strict=True))


def _connected(outcomes: Sequence[tuple[str, str, str]]) -> bool:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for a, b, _ in outcomes:
        neighbors[a].add(b)
        neighbors[b].add(a)
    if not neighbors:
        return False
    seen, stack = set(), [next(iter(neighbors))]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(neighbors[node] - seen)
    return len(seen) == len(neighbors)


def summarize_pilot(connection: sqlite3.Connection, output_dir: Path, pilot_keys: Sequence[str]) -> dict[str, Any]:
    base_rows = connection.execute("SELECT comparison_id, bucket_a, bucket_b, winner FROM comparisons WHERE audit_of IS NULL AND winner IS NOT NULL").fetchall()
    base = [(row["bucket_a"], row["bucket_b"], row["winner"]) for row in base_rows]
    base_by_id = {row["comparison_id"]: row for row in base_rows}
    audits = connection.execute("SELECT comparison_id, audit_of, winner, reason FROM comparisons WHERE audit_of IS NOT NULL AND winner IS NOT NULL").fetchall()
    agreements = [base_by_id[row["audit_of"]]["winner"] == row["winner"] for row in audits]
    replacement = {row["audit_of"]: row["winner"] for row in audits}
    alternate = [(row["bucket_a"], row["bucket_b"], replacement.get(row["comparison_id"], row["winner"])) for row in base_rows]
    scores, alternate_scores = fit_davidson(base), fit_davidson(alternate)
    ordered = sorted(pilot_keys, key=lambda key: scores[key])
    alternate_ordered = sorted(pilot_keys, key=lambda key: alternate_scores[key])
    rank = {key: i for i, key in enumerate(ordered)}
    alternate_rank = {key: i for i, key in enumerate(alternate_ordered)}
    n = len(ordered)
    rankings = []
    within_one = []
    for key in ordered:
        decile = min(9, rank[key] * 10 // n) + 1
        alternate_decile = min(9, alternate_rank[key] * 10 // n) + 1
        within_one.append(abs(decile - alternate_decile) <= 1)
        rankings.append({
            "relevance_bucket": key,
            "source_id": json.loads(key)["source_id"],
            "davidson_score": scores[key],
            "percentile": 100 * rank[key] / max(n - 1, 1),
            "decile": decile,
            "reversal_fit_score": alternate_scores[key],
            "reversal_fit_decile": alternate_decile,
        })
    ranking_path = output_dir / "relevance_bucket_rankings.parquet"
    pd.DataFrame(rankings).to_parquet(ranking_path, index=False)
    tie_rate = sum(winner == "tie" for _, _, winner in base) / len(base)
    rho = float(spearmanr([rank[key] for key in ordered], [alternate_rank[key] for key in ordered]).statistic)
    metrics = {
        "pilot_bucket_count": len(pilot_keys),
        "comparison_count": len(base),
        "reversal_audit_count": len(audits),
        "reversal_winner_agreement": sum(agreements) / len(agreements),
        "rank_spearman": rho,
        "within_one_decile": sum(within_one) / len(within_one),
        "tie_rate": tie_rate,
        "comparison_graph_connected": _connected(base),
    }
    metrics["automatic_gates_passed"] = (
        metrics["reversal_winner_agreement"] >= 0.90
        and rho >= 0.90
        and metrics["within_one_decile"] >= 0.95
        and tie_rate < 0.50
        and metrics["comparison_graph_connected"]
    )
    def review_row(row: sqlite3.Row, agreement: bool) -> dict[str, Any]:
        original = connection.execute(
            "SELECT bucket_a, bucket_b, winner, reason FROM comparisons WHERE comparison_id=?",
            (row["audit_of"],),
        ).fetchone()
        buckets = {}
        for label, key in (("A", original["bucket_a"]), ("B", original["bucket_b"])):
            bucket = connection.execute(
                "SELECT identity_json, samples_json FROM buckets WHERE relevance_bucket=?",
                (key,),
            ).fetchone()
            buckets[label] = {
                "identity": json.loads(bucket["identity_json"]),
                "sample_records": json.loads(bucket["samples_json"]),
            }
        return {
            "comparison_id": row["audit_of"],
            "agreement": agreement,
            "buckets": buckets,
            "original": {"winner": original["winner"], "reason": original["reason"]},
            "reversed_normalized": {"winner": row["winner"], "reason": row["reason"]},
        }

    review = []
    for row in audits:
        original = base_by_id[row["audit_of"]]
        if original["winner"] != row["winner"]:
            review.append(review_row(row, False))
    agreements_rows = [row for row in audits if base_by_id[row["audit_of"]]["winner"] == row["winner"]]
    strata: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in agreements_rows:
        original = base_by_id[row["audit_of"]]
        sources = tuple(sorted((json.loads(original["bucket_a"])["source_id"], json.loads(original["bucket_b"])["source_id"])))
        strata[sources].append(row)
    for rows in strata.values():
        rows.sort(key=lambda item: _hash("manual-review", item["audit_of"]))
    review_target = len(review) + 30
    while len(review) < review_target:
        progressed = False
        for source_pair in sorted(strata):
            if strata[source_pair] and len(review) < review_target:
                review.append(review_row(strata[source_pair].pop(), True))
                progressed = True
        if not progressed:
            break
    write_json_atomic(output_dir / "pilot_review_sample.json", {"review_comparisons": review})
    return metrics


def run_pilot(input_path: Path, output_dir: Path, *, parallelism: int, token_budget: int, env_file: Path | None) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        build_inventory(input_path, output_dir)
    connection = _connect(output_dir / "requests.sqlite3")
    pilot = _pilot_keys(connection)
    candidates, anchor_rows = anchor_comparisons(pilot)
    _insert_batches(connection, anchor_rows)
    run_phase(connection, "anchor", parallelism=parallelism, token_budget=token_budget, env_file=env_file)
    anchors = choose_anchors(connection, candidates)
    main_rows = main_comparisons(pilot, anchors)
    _insert_batches(connection, main_rows)
    run_phase(connection, "main", parallelism=parallelism, token_budget=token_budget, env_file=env_file)
    audit_rows = audit_comparisons(main_rows, anchor_rows)
    _insert_batches(connection, audit_rows)
    run_phase(connection, "audit", parallelism=parallelism, token_budget=token_budget, env_file=env_file)
    metrics = summarize_pilot(connection, output_dir, pilot)
    tokens = connection.execute("SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0) FROM requests WHERE status='complete'").fetchone()
    served_models = {
        row[0]: int(row[1])
        for row in connection.execute(
            "SELECT served_model, count(*) FROM requests WHERE status='complete' GROUP BY served_model"
        )
    }
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "pilot_complete_pending_manual_review",
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": REASONING_EFFORT,
        "comparisons_per_request": COMPARISONS_PER_REQUEST,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "actual_token_budget": token_budget,
        "actual_input_tokens": int(tokens[0]),
        "actual_output_tokens": int(tokens[1]),
        "completed_request_count": sum(served_models.values()),
        "served_models": served_models,
        "pilot_source_counts": PILOT_COUNTS,
        "anchor_candidate_count": len(candidates),
        "anchors": anchors,
        "pilot_metrics": metrics,
        "full_run_started": False,
    })
    write_json_atomic(manifest_path, manifest)
    connection.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "pilot"))
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--skip-v9-count-check", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = build_inventory(
            args.input,
            args.output_dir,
            verify_v9=not args.skip_v9_count_check,
            rebuild=args.rebuild,
        )
    else:
        result = run_pilot(args.input, args.output_dir, parallelism=args.parallelism, token_budget=args.token_budget, env_file=args.env_file)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
