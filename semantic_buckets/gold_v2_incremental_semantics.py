"""Incrementally align reviewed V10 semantics to Gold-v2 UID levels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any

from jinja2 import Environment, StrictUndefined
import pandas as pd
from scipy.optimize import minimize_scalar

from data.processing.gold_labels.level_mappings import level_mapping_path

from semantic_buckets import (
    bioavailability_semantic_readout_v1 as core,
)
from semantic_buckets import (
    bioavailability_semantic_readout_v3 as oral,
)
from semantic_buckets import bbb_semantic_readout_v3 as bbb
from semantic_buckets import bioavailability_semantic_degree25 as degree
from semantic_buckets import semantic_l2_retrieval_eligibility as eligibility
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "gold_v2_incremental_semantics.v1"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent / "history/pre_gold_v2"
LEVELS = ("L2", "L3", "L4")
ELIGIBILITY_LEVELS = frozenset({"L2", "L3", "L4"})
ENDPOINTS = (
    {"name": "dgx011", "base_url": "http://dgx011:50001/v1", "provider": "local",
     "credential_env": "", "max_inflight": 256},
    {"name": "dgx014", "base_url": "http://dgx014:50002/v1", "provider": "local",
     "credential_env": "", "max_inflight": 256},
)
TASKS = {
    "bbb_martins": {
        "stage3": ROOT / "data/evidence_libraries/bbb_martins/v10/03_pair_buckets/records.parquet",
        "level_map": level_mapping_path("bbb_martins", "v2"),
        "old_input": ARCHIVE / "bbb_martins/v10/semantic_buckets/generations/bbb_semantic_readout_v10_v3",
        "old_selected": ARCHIVE / "bbb_martins/v10/semantic_buckets/generations/bbb_semantic_readout_v10_v3",
    },
    "bioavailability_ma": {
        "stage3": ROOT / "data/evidence_libraries/bioavailability_ma/v10/03_pair_buckets/records.parquet",
        "level_map": level_mapping_path("bioavailability_ma", "v2"),
        "old_input": ARCHIVE / "bioavailability_ma/v10/semantic_buckets/generations/bioavailability_semantic_readout_v10_v3",
        "old_selected": ARCHIVE / "bioavailability_ma/v10/semantic_buckets/generations/bioavailability_semantic_readout_v10_v3_l4_refinement_v1",
    },
}
WEIGHTS = Path(__file__).resolve().parent / "policies/semantic_weighted_top10_v2.json"


def configure(task: str, record_map: Path, expected_records: int | None = None) -> None:
    """Configure the shared semantic engine for one Gold-v2 delta."""
    if task == "bbb_martins":
        bbb.configure_workflow()
        bbb.workflow.configure_core()
    elif task == "bioavailability_ma":
        oral.configure_core()
    else:
        raise ValueError(f"unsupported task: {task}")
    core.VERSION = f"{VERSION}.{task}"
    core.BATCH_SEED_VERSION = core.VERSION
    core.RECORD_MAP = record_map
    core.RECORDS = Path(TASKS[task]["stage3"])
    core.LEVELS = LEVELS
    core.PAIR_COLUMNS = {
        source: tuple(columns) for source, columns in core.REFINEMENT_COLUMNS.items()
    }
    core.PAIR_KEY_OPTIONAL_TRAILING_COLUMNS = {}
    core.EXPECTED_RECORD_COUNT = expected_records
    core.EXPECTED_ATOM_COUNT = None
    endpoints = tuple(TASKS[task].get("endpoints", ENDPOINTS))
    core.MODEL = str(TASKS[task].get("model", core.MODEL))
    core.BASE_URL = endpoints[0]["base_url"]
    core.ENDPOINTS = endpoints
    core.LOW_MAX_TOKENS = 65_536
    core.HIGH_MAX_TOKENS = 65_536
    core.MAX_ATTEMPTS = 8


def _record_map(task: str) -> pd.DataFrame:
    spec = TASKS[task]
    records = pd.read_parquet(
        spec["stage3"],
        columns=["canonical_record_id", "source_row_uid", "source_id", "pair_bucket_key"],
    )
    levels = pd.read_parquet(spec["level_map"], columns=["source_row_uid", "level"])
    if records.source_row_uid.duplicated().any() or levels.source_row_uid.duplicated().any():
        raise ValueError("current Stage-3 and level-map UIDs must be unique")
    merged = records.merge(levels, on="source_row_uid", validate="one_to_one")
    merged["level"] = merged["level"].map(lambda value: f"L{int(value)}")
    return _semantic_columns(merged[merged.level.isin(LEVELS)].copy())


def _semantic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    buckets, nodes = [], []
    for row in frame.itertuples(index=False):
        values = core.parse_pair_bucket(str(row.source_id), str(row.pair_bucket_key))
        identity = {"source_id": str(row.source_id)}
        identity.update({column: values[column] for column in core.INITIAL_COLUMNS[row.source_id]})
        bucket = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        buckets.append(bucket)
        nodes.append(core._canonical_json({"level": row.level, "relevance_bucket": bucket}))
    frame["relevance_bucket"] = buckets
    frame["node_key"] = nodes
    return frame


def _with_atom_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["atom_id"] = [
        core._stable_id("atom", level, source, pair)
        for level, source, pair in result[
            ["level", "source_id", "pair_bucket_key"]
        ].itertuples(index=False, name=None)
    ]
    return result


def _old_record_semantics(task: str) -> pd.DataFrame:
    spec = TASKS[task]
    rankings = spec["old_selected"] / "degree25_rankings/record_relevance_rankings.parquet"
    old = pd.read_parquet(
        rankings, columns=["source_row_uid", "level", "semantic_bucket_id"],
    )
    if old.source_row_uid.duplicated().any():
        raise ValueError("archived record semantics repeat source-row UIDs")
    return old


def _anchor_map(
    records: pd.DataFrame, old: pd.DataFrame, *, allow_conflicts: bool = False
) -> tuple[pd.DataFrame, set[str]]:
    joined = records.merge(
        old[["source_row_uid", "level", "semantic_bucket_id"]],
        on=["source_row_uid", "level"], how="left", validate="one_to_one",
    )
    counts = joined.groupby("atom_id").semantic_bucket_id.nunique(dropna=True)
    conflicting = set(counts[counts.gt(1)].index)
    if conflicting and not allow_conflicts:
        raise ValueError(f"current atoms have conflicting same-level anchors: {len(conflicting)}")
    anchored = joined[
        joined.semantic_bucket_id.notna() & ~joined.atom_id.isin(conflicting)
    ].drop_duplicates("atom_id")
    unanchored = set(counts[counts.eq(0)].index) | conflicting
    return anchored[["atom_id", "level", "source_id", "semantic_bucket_id"]], unanchored


def prepare(task: str, output: Path) -> dict[str, Any]:
    """Materialize current atoms and the unanchored Gold-v2 semantic delta."""
    if output.exists() and any(output.iterdir()):
        return _resume_prepare(task, output)
    output.mkdir(parents=True, exist_ok=True)
    full_map = output / "record_relevance_map.parquet"
    configure(task, full_map)
    records = _with_atom_ids(_record_map(task))
    records.drop(columns="atom_id").to_parquet(full_map, index=False)
    old = _old_record_semantics(task)
    anchors, unanchored = _anchor_map(
        records, old,
        allow_conflicts=bool(TASKS[task].get("allow_conflicting_anchors")),
    )
    delta = records[records.atom_id.isin(unanchored)].drop(columns="atom_id")
    delta_path = output / "delta_record_relevance_map.parquet"
    delta.to_parquet(delta_path, index=False)
    anchors.to_parquet(output / "anchored_atoms.parquet", index=False)
    configure(task, full_map, len(records))
    atoms, cards = core._load_atoms()
    atoms.to_parquet(output / "input_atoms.parquet", index=False)
    core._write_gzip_json(output / "sample_cards.json.gz", cards)
    manifest = _prepare_manifest(task, output, records, atoms, anchors, delta, unanchored)
    write_json_atomic(output / "manifest.json", manifest)
    _prepare_review(task, output, delta_path, len(delta))
    return manifest


def _prepare_review(task: str, output: Path, delta_path: Path, rows: int) -> None:
    configure(task, delta_path, rows)
    review = output / "prompt_review"
    review.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": f"{VERSION}.{task}.prompt_review.v1",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "model": core.MODEL,
        "base_url": core.BASE_URL,
        "source_record_map": {"path": str(delta_path), "sha256": core._file_sha256(delta_path)},
        "v10_records": {"path": str(core.RECORDS), "sha256": core._file_sha256(core.RECORDS)},
        "prompt_template_hashes": {
            path.name: core._file_sha256(path) for path in sorted(core.PROMPT_ROOT.glob("*.jinja"))
        },
        "settings": {
            "reasoning_effort": "high", "max_tokens": 65_536,
            "streaming": True,
            "per_endpoint_parallelism": max(
                int(endpoint["max_inflight"]) for endpoint in core.ENDPOINTS
            ),
        },
    }
    write_json_atomic(review / "manifest.json", manifest)


def _resume_prepare(task: str, output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"incremental output is incomplete: {output}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "prepared" or manifest.get("task_id") != task:
        raise FileExistsError(f"incremental output cannot be resumed: {output}")
    review = output / "prompt_review/manifest.json"
    if not review.is_file():
        delta = output / "delta_record_relevance_map.parquet"
        _prepare_review(task, output, delta, len(pd.read_parquet(delta)))
    return manifest


def _prepare_manifest(task: str, output: Path, records: pd.DataFrame, atoms: pd.DataFrame,
                      anchors: pd.DataFrame, delta: pd.DataFrame,
                      unanchored: set[str]) -> dict[str, Any]:
    return {
        "version": VERSION, "status": "prepared", "task_id": task,
        "levels": list(LEVELS), "endpoints": list(core.ENDPOINTS),
        "reasoning_effort": "high", "max_tokens": 65_536, "streaming": True,
        "current_records": len(records), "current_atoms": len(atoms),
        "anchored_atoms": len(anchors), "unanchored_atoms": len(unanchored),
        "delta_records": len(delta),
        "inputs": {
            str(Path(TASKS[task]["stage3"]).resolve()): core._file_sha256(TASKS[task]["stage3"]),
            str(Path(TASKS[task]["level_map"]).resolve()): core._file_sha256(TASKS[task]["level_map"]),
        },
        "artifacts": {name: core._file_sha256(output / name) for name in (
            "record_relevance_map.parquet", "delta_record_relevance_map.parquet",
            "anchored_atoms.parquet", "input_atoms.parquet", "sample_cards.json.gz",
        )},
    }


def run_delta(task: str, output: Path, parallelism: int = 256) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    delta_path = output / "delta_record_relevance_map.parquet"
    delta = pd.read_parquet(delta_path)
    configure(task, delta_path, len(delta))
    review = output / "prompt_review/manifest.json"
    run_root = output / "delta_semantic_run"
    result = core.run_semantic(
        run_root, review_manifest_path=review,
        approved_review_sha256=core._file_sha256(review),
        parallelism=parallelism,
    )
    manifest.update(status="delta_complete", delta_run=result)
    write_json_atomic(manifest_path, manifest)
    return manifest


def _bucket_payloads(atoms: pd.DataFrame, mapping: pd.DataFrame,
                     cards: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = core._atom_lookup(atoms)
    payloads = {}
    for bucket, rows in mapping.groupby("semantic_bucket_id", sort=True):
        members = sorted(rows.atom_id.astype(str))
        payloads[str(bucket)] = degree._bucket_payload(str(bucket), members, lookup, cards)
    return payloads


def _payload_tokens(payload: dict[str, Any]) -> set[str]:
    tokens = set()
    for component in payload["identity"]["source_components"]:
        tokens.add(f"source={component['source_id']}")
        for column, values in component["canonical_dimensions"].items():
            if isinstance(values, dict):
                values = values.get("examples", [])
            tokens.update(f"{column}={value}" for value in values if value != "__unknown__")
    return tokens


def _candidate_ids(anchor: dict[str, Any], old: dict[str, dict[str, Any]],
                   limit: int = 39) -> list[str]:
    anchor_sources = {row["source_id"] for row in anchor["identity"]["source_components"]}
    anchor_tokens = _payload_tokens(anchor)
    scored = []
    for bucket, payload in old.items():
        sources = {row["source_id"] for row in payload["identity"]["source_components"]}
        if sources != anchor_sources:
            continue
        tokens = _payload_tokens(payload)
        overlap = len(anchor_tokens & tokens)
        union = len(anchor_tokens | tokens) or 1
        scored.append((-overlap, -(overlap / union), bucket))
    return [bucket for _, _, bucket in sorted(scored)[:limit]]


def _alignment_inputs(task: str, output: Path) -> tuple[dict, dict, pd.DataFrame]:
    spec = TASKS[task]
    old_atoms = pd.read_parquet(spec["old_input"] / "input_atoms.parquet")
    old_cards = core._read_gzip_json(spec["old_input"] / "sample_cards.json.gz")
    old_map = pd.read_parquet(spec["old_selected"] / "semantic_bucket_map.parquet")
    old_map = old_map[old_map.atom_id.isin(set(old_atoms.atom_id))].copy()
    old_payloads = _bucket_payloads(old_atoms, old_map, old_cards)
    delta_root = output / "delta_semantic_run"
    delta_atoms = pd.read_parquet(delta_root / "input_atoms.parquet")
    delta_cards = core._read_gzip_json(delta_root / "sample_cards.json.gz")
    delta_map = pd.read_parquet(delta_root / "source_semantic_bucket_map.parquet")
    delta_map = delta_map.rename(columns={"source_semantic_bucket_id": "semantic_bucket_id"})
    return old_payloads, _bucket_payloads(delta_atoms, delta_map, delta_cards), delta_map


def prepare_alignment(task: str, output: Path) -> dict[str, Any]:
    configure(task, output / "delta_record_relevance_map.parquet")
    root = output / "alignment"
    root.mkdir(parents=True, exist_ok=True)
    old, delta, delta_map = _alignment_inputs(task, output)
    old_map = pd.read_parquet(TASKS[task]["old_selected"] / "semantic_bucket_map.parquet")
    old_levels = old_map.groupby("semantic_bucket_id")["level"].first().to_dict()
    delta_levels = delta_map.groupby("semantic_bucket_id")["level"].first().to_dict()
    connection = core._request_database(root / "requests.sqlite3")
    schedule = _queue_alignment_requests(connection, old, delta, old_levels, delta_levels)
    connection.commit()
    connection.close()
    pd.DataFrame(schedule).to_parquet(root / "schedule.parquet", index=False)
    manifest = {"version": VERSION, "status": "alignment_prepared", "task_id": task,
                "requests": len(schedule), "reasoning_effort": "high", "streaming": True,
                "per_endpoint_parallelism": 256,
                "schedule_sha256": core._file_sha256(root / "schedule.parquet")}
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _queue_alignment_requests(connection: sqlite3.Connection, old: dict, delta: dict,
                              old_levels: dict, delta_levels: dict) -> list[dict]:
    schedule = []
    for anchor, payload in sorted(delta.items()):
        level = str(delta_levels[anchor])
        scoped = {key: value for key, value in old.items() if old_levels[key] == level}
        candidates = _candidate_ids(payload, scoped)
        batch = {"task": core.TASK_NAME, "level": level, "anchor_bucket_id": anchor,
                 "buckets": [dict(payload, bucket_id=anchor)]
                 + [dict(scoped[key], bucket_id=key) for key in candidates]}
        prompt = core._render_cross_source(batch)
        request_id = core._queue_request(
            connection, request_id=core._request_id("merge_buckets", f"align|{level}", prompt),
            kind="merge_buckets", phase=f"align|{level}", prompt=prompt,
            reasoning_effort="high", max_tokens=65_536,
            validation={"valid_ids": [anchor, *candidates]}, commit=False,
        )
        schedule.append({"anchor_bucket_id": anchor, "level": level,
                         "request_id": request_id, "candidate_ids_json": json.dumps(candidates)})
    return schedule


def run_alignment(task: str, output: Path, parallelism: int = 256) -> dict[str, Any]:
    configure(task, output / "delta_record_relevance_map.parquet")
    root = output / "alignment"
    manifest = json.loads((root / "manifest.json").read_text())
    schedule = pd.read_parquet(root / "schedule.parquet")
    connection = core._request_database(root / "requests.sqlite3")
    client, receipts, total = core._build_completion_pool(parallelism)
    ids = schedule.request_id.tolist()
    core._run_pending(connection, ids, parallelism=total, client=client,
                      request_slots=threading.BoundedSemaphore(total))
    decisions = _alignment_decisions(connection, schedule)
    decisions.to_parquet(root / "decisions.parquet", index=False)
    connection.close()
    final = _write_final_map(task, output, decisions)
    manifest.update(status="complete", decisions=len(decisions), attached=int(decisions.attached.sum()),
                    endpoint_pool_at_start=receipts, endpoint_pool_final_snapshot=client.snapshot(),
                    decisions_sha256=core._file_sha256(root / "decisions.parquet"), **final)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _alignment_decisions(connection: sqlite3.Connection, schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in schedule.itertuples(index=False):
        response = core._response(connection, item.request_id)
        anchor = str(item.anchor_bucket_id)
        candidates = set(json.loads(item.candidate_ids_json))
        selected = []
        for entry in response["merge_sets"]:
            members = set(entry["member_ids"])
            if anchor in members:
                selected = sorted((members - {anchor}) & candidates)
        attached = len(selected) == 1
        rows.append({"level": item.level, "anchor_bucket_id": anchor,
                     "attached": attached,
                     "semantic_bucket_id": selected[0] if attached else anchor,
                     "selected_old_bucket_ids_json": json.dumps(selected),
                     "response_json": core._canonical_json(response)})
    return pd.DataFrame(rows).sort_values(["level", "anchor_bucket_id"])


def _write_final_map(task: str, output: Path, decisions: pd.DataFrame) -> dict[str, Any]:
    anchors = pd.read_parquet(output / "anchored_atoms.parquet")
    anchors["source_semantic_bucket_id"] = anchors["semantic_bucket_id"]
    delta = pd.read_parquet(output / "delta_semantic_run/source_semantic_bucket_map.parquet")
    choice = decisions.set_index("anchor_bucket_id").semantic_bucket_id.to_dict()
    delta["semantic_bucket_id"] = delta.source_semantic_bucket_id.map(choice)
    final = pd.concat([anchors, delta], ignore_index=True)
    final = final[["level", "source_id", "source_semantic_bucket_id",
                   "semantic_bucket_id", "atom_id"]].sort_values(
        ["level", "semantic_bucket_id", "source_id", "atom_id"]
    )
    expected = json.loads((output / "manifest.json").read_text())["current_atoms"]
    if len(final) != expected or final.atom_id.duplicated().any():
        raise ValueError("incremental final map does not cover every current atom")
    path = output / "semantic_bucket_map.parquet"
    final.to_parquet(path, index=False)
    return {"semantic_bucket_map_sha256": core._file_sha256(path),
            "semantic_bucket_count": int(final.semantic_bucket_id.nunique()),
            "new_semantic_bucket_count": int((~decisions.attached).sum())}


def _old_rankings(task: str) -> pd.DataFrame:
    path = TASKS[task]["old_selected"] / "degree25_rankings/semantic_bucket_rankings.parquet"
    return pd.read_parquet(path).query("level in @LEVELS").copy()


def _old_decisions(task: str) -> pd.DataFrame:
    path = TASKS[task].get("old_decisions")
    if path is None:
        path = (ARCHIVE / task / "v10/semantic_buckets/eligibility"
                / "semantic_indirect_retrieval_eligibility_v2/bucket_decisions.parquet")
    return pd.read_parquet(path).query("level in @LEVELS").copy()


def _current_payloads(output: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    atoms = pd.read_parquet(output / "input_atoms.parquet")
    cards = core._read_gzip_json(output / "sample_cards.json.gz")
    mapping = pd.read_parquet(output / "semantic_bucket_map.parquet")
    return mapping, _bucket_payloads(atoms, mapping, cards)


def _comparators(rankings: pd.DataFrame, level: str, count: int = 25) -> list[str]:
    ordered = rankings[rankings.level.eq(level)].sort_values("level_rank").semantic_bucket_id.tolist()
    if len(ordered) <= count:
        return ordered
    indices = sorted({round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)})
    return [ordered[index] for index in indices]


def prepare_final_reviews(task: str, output: Path) -> dict[str, Any]:
    configure(task, output / "record_relevance_map.parquet")
    root = output / "final_reviews"
    root.mkdir(parents=True, exist_ok=True)
    mapping, current = _current_payloads(output)
    historical_ranks = _old_rankings(task)
    old, _, _ = _alignment_inputs(task, output)
    comparator_ranks = historical_ranks[
        historical_ranks.semantic_bucket_id.isin(old)
    ].copy()
    old_ids = set(historical_ranks.semantic_bucket_id.astype(str))
    new_ids = sorted(set(mapping.semantic_bucket_id.astype(str)) - old_ids)
    connection = core._request_database(root / "requests.sqlite3")
    schedule = _queue_final_reviews(
        connection, task, new_ids, mapping, current, old, comparator_ranks
    )
    connection.commit()
    connection.close()
    pd.DataFrame(schedule).to_parquet(root / "schedule.parquet", index=False)
    manifest = {"version": VERSION, "status": "final_reviews_prepared", "task_id": task,
                "new_buckets": len(new_ids), "logical_requests": len(schedule),
                "reasoning_effort": "high", "streaming": True,
                "per_endpoint_parallelism": 256,
                "schedule_sha256": core._file_sha256(root / "schedule.parquet")}
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _queue_final_reviews(connection: sqlite3.Connection, task: str, new_ids: list[str],
                         mapping: pd.DataFrame, current: dict, old: dict,
                         rankings: pd.DataFrame) -> list[dict]:
    levels = mapping.groupby("semantic_bucket_id")["level"].first().to_dict()
    schedule = []
    for bucket in new_ids:
        level = str(levels[bucket])
        for comparator in _comparators(rankings, level):
            prompt = _ranking_prompt(task, {"groups": {
                bucket: current[bucket], comparator: old[comparator],
            }})
            request = core._request_id("ranking", f"incremental|{level}", prompt)
            core._queue_request(connection, request_id=request, kind="ranking",
                phase=f"incremental|{level}", prompt=prompt, reasoning_effort="high",
                max_tokens=65_536, validation={"candidate_bucket_ids": [bucket, comparator]},
                commit=False)
            schedule.append({"kind": "ranking", "level": level, "semantic_bucket_id": bucket,
                             "comparator_bucket_id": comparator, "request_id": request})
        if level in ELIGIBILITY_LEVELS:
            prompt = eligibility._prompt(task, level, current[bucket])
            request = core._request_id("retrieval_eligibility", f"incremental|{level}", prompt)
            core._queue_request(connection, request_id=request, kind="retrieval_eligibility",
                phase=f"incremental|{level}", prompt=prompt, reasoning_effort="high",
                max_tokens=65_536, validation={"reason_codes": list(eligibility.REASON_CODES)},
                commit=False)
            schedule.append({"kind": "eligibility", "level": level, "semantic_bucket_id": bucket,
                             "comparator_bucket_id": None, "request_id": request})
    return schedule


def _ranking_prompt(task: str, payload: dict[str, Any]) -> str:
    template = (
        Path(__file__).with_name("prompts") / "bbb_relevance_bucket_level_v3.jinja"
        if task == "bbb_martins" else degree.TEMPLATE
    )
    renderer = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        template.read_text(encoding="utf-8")
    )
    return renderer.render(compact_payload_json=core._canonical_json(payload)).strip()


def run_final_reviews(task: str, output: Path, parallelism: int = 256) -> dict[str, Any]:
    configure(task, output / "record_relevance_map.parquet")
    root = output / "final_reviews"
    manifest = json.loads((root / "manifest.json").read_text())
    schedule = pd.read_parquet(root / "schedule.parquet")
    connection = core._request_database(root / "requests.sqlite3")
    ids = schedule.request_id.tolist()
    pending = connection.execute(
        "SELECT COUNT(*) FROM requests WHERE status!='complete'"
    ).fetchone()[0]
    if pending:
        client, receipts, total = core._build_completion_pool(parallelism)
        core._run_pending(connection, ids, parallelism=total, client=client,
                          request_slots=threading.BoundedSemaphore(total))
        manifest.update(endpoint_pool_at_start=receipts,
                        endpoint_pool_final_snapshot=client.snapshot())
    outputs = _publish_rankings_and_eligibility(task, output, schedule, connection)
    request_summary = _request_summary(connection, ids)
    connection.close()
    manifest.update(status="complete", request_summary=request_summary, **outputs)
    write_json_atomic(root / "manifest.json", manifest)
    _complete_root_manifest(output, manifest)
    return manifest


def _request_summary(connection: sqlite3.Connection,
                     request_ids: list[str]) -> dict[str, int]:
    placeholders = ",".join("?" for _ in request_ids)
    row = connection.execute(
        """SELECT count(*) AS requests,sum(status='complete') AS complete,
                  sum(status='failed') AS failed,coalesce(sum(attempts),0) AS attempts,
                  coalesce(sum(max(attempts-1,0)),0) AS retries,
                  coalesce(sum(input_tokens),0) AS input_tokens,
                  coalesce(sum(output_tokens),0) AS output_tokens
           FROM requests WHERE request_id IN (""" + placeholders + ")",
        request_ids,
    ).fetchone()
    return dict(row)


def _complete_root_manifest(output: Path, reviews: dict[str, Any]) -> None:
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        status="complete",
        semantic_bucket_map_sha256=core._file_sha256(output / "semantic_bucket_map.parquet"),
        semantic_bucket_rankings_sha256=reviews["semantic_bucket_rankings_sha256"],
        bucket_decisions_sha256=reviews["bucket_decisions_sha256"],
        record_eligibility_sha256=reviews["record_eligibility_sha256"],
        semantic_bucket_count=reviews["ranked_buckets"],
        eligible_records=reviews["eligible_records"],
    )
    write_json_atomic(path, manifest)


def _new_scores(schedule: pd.DataFrame, connection: sqlite3.Connection,
                old_ranks: pd.DataFrame) -> dict[str, float]:
    old_scores = old_ranks.set_index("semantic_bucket_id").bradley_terry_score.to_dict()
    scores = {}
    for bucket, rows in schedule[schedule.kind.eq("ranking")].groupby("semantic_bucket_id"):
        outcomes = []
        for row in rows.itertuples(index=False):
            winner = core._response(connection, row.request_id)["winner_bucket_id"]
            outcomes.append((old_scores[row.comparator_bucket_id], winner == bucket))
        scores[str(bucket)] = _fit_incremental_score(outcomes)
    return scores


def _fit_incremental_score(outcomes: list[tuple[float, bool]]) -> float:
    def objective(value: float) -> float:
        loss = 0.15 * value * value
        for comparator, won in outcomes:
            difference = max(-30.0, min(30.0, value - comparator))
            probability = 1.0 / (1.0 + math.exp(-difference))
            loss -= math.log(probability if won else 1.0 - probability)
        return loss
    result = minimize_scalar(objective, bounds=(-20, 20), method="bounded")
    if not result.success:
        raise RuntimeError("incremental relevance-score fit failed")
    return float(result.x)


def _ranking_frame(task: str, mapping: pd.DataFrame,
                   new_scores: dict[str, float]) -> pd.DataFrame:
    current_ids = set(mapping.semantic_bucket_id.astype(str))
    levels = mapping.groupby("semantic_bucket_id")["level"].first().to_dict()
    old = _old_rankings(task)
    old = old[old.semantic_bucket_id.isin(current_ids)].copy()
    additions = [{"task_id": task, "level": levels[bucket], "semantic_bucket_id": bucket,
                  "bradley_terry_score": score, "comparison_count": 25}
                 for bucket, score in new_scores.items()]
    combined = pd.concat([old, pd.DataFrame(additions)], ignore_index=True)
    rows = []
    for level, frame in combined.groupby("level", sort=True):
        ordered = frame.sort_values(["bradley_terry_score", "semantic_bucket_id"],
                                    ascending=[False, True]).reset_index(drop=True)
        ordered["level_rank"] = ordered.index + 1
        ordered["level_percentile"] = 100 * (len(ordered) - 1 - ordered.index) / max(1, len(ordered) - 1)
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)


def _policy_weights(task: str) -> dict[str, float]:
    document = json.loads(WEIGHTS.read_text())
    output = {}
    for level in ELIGIBILITY_LEVELS:
        spec = document["tasks"][task]["levels"][level]
        output.update(zip(spec["bucket_ids"], map(float, spec["weights"]), strict=True))
    return output


def _decision_frame(task: str, current_ids: set[str], schedule: pd.DataFrame,
                    connection: sqlite3.Connection) -> pd.DataFrame:
    old = _old_decisions(task)
    old = old[old.semantic_bucket_id.isin(current_ids)].copy()
    rows = []
    for item in schedule[schedule.kind.eq("eligibility")].itertuples(index=False):
        response = core._response(connection, item.request_id)
        rows.append({"task_id": task, "level": item.level,
                     "semantic_bucket_id": item.semantic_bucket_id, **response})
    decisions = pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False)
    weights = _policy_weights(task)
    decisions["expert_weight"] = decisions.semantic_bucket_id.map(weights)
    default = decisions.decision.eq("include")
    decisions["retrieval_eligible"] = default.where(
        decisions.expert_weight.isna(), decisions.expert_weight.gt(0)
    )
    decisions["prior_zero_weight"] = decisions.expert_weight.eq(0)
    if set(decisions.semantic_bucket_id.astype(str)) != current_ids:
        raise ValueError("eligibility decisions do not cover current semantic buckets")
    return decisions.sort_values(["level", "semantic_bucket_id"])


def _record_eligibility(output: Path, mapping: pd.DataFrame, rankings: pd.DataFrame,
                        decisions: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_parquet(output / "record_relevance_map.parquet")
    records["atom_id"] = [
        core._stable_id("atom", level, source, pair)
        for level, source, pair in records[["level", "source_id", "pair_bucket_key"]]
        .itertuples(index=False, name=None)
    ]
    joined = records.merge(mapping[["atom_id", "semantic_bucket_id"]], on="atom_id",
                           validate="many_to_one")
    joined = joined.merge(rankings, on=["level", "semantic_bucket_id"],
                          validate="many_to_one")
    columns = ["level", "semantic_bucket_id", "decision", "reason_code",
               "retrieval_eligible", "expert_weight", "prior_zero_weight"]
    joined = joined.merge(decisions[columns], on=["level", "semantic_bucket_id"],
                          validate="many_to_one")
    expected = int(records.level.isin(ELIGIBILITY_LEVELS).sum())
    if len(joined) != expected or joined.source_row_uid.duplicated().any():
        raise ValueError("record eligibility does not cover current L2-L4 records")
    return joined


def _publish_rankings_and_eligibility(task: str, output: Path, schedule: pd.DataFrame,
                                      connection: sqlite3.Connection) -> dict[str, Any]:
    mapping = pd.read_parquet(output / "semantic_bucket_map.parquet")
    current_ids = set(mapping.semantic_bucket_id.astype(str))
    scores = _new_scores(schedule, connection, _old_rankings(task))
    rankings = _ranking_frame(task, mapping, scores)
    eligibility_ids = set(
        mapping[mapping.level.isin(ELIGIBILITY_LEVELS)].semantic_bucket_id.astype(str)
    )
    decisions = _decision_frame(task, eligibility_ids, schedule, connection)
    records = _record_eligibility(output, mapping, rankings, decisions)
    root = output / "final_reviews"
    rankings.to_parquet(root / "semantic_bucket_rankings.parquet", index=False)
    decisions.to_parquet(root / "bucket_decisions.parquet", index=False)
    records.to_parquet(root / "record_eligibility.parquet", index=False)
    return {"semantic_bucket_rankings_sha256": core._file_sha256(
                root / "semantic_bucket_rankings.parquet"),
            "bucket_decisions_sha256": core._file_sha256(root / "bucket_decisions.parquet"),
            "record_eligibility_sha256": core._file_sha256(root / "record_eligibility.parquet"),
            "ranked_buckets": len(rankings), "eligible_records": int(records.retrieval_eligible.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "run-delta", "prepare-and-run", "prepare-alignment", "run-alignment",
        "prepare-final-reviews", "run-final-reviews",
    ))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=256)
    args = parser.parse_args()
    if args.command in {"prepare", "prepare-and-run"}:
        result = prepare(args.task, args.output)
    if args.command in {"run-delta", "prepare-and-run"}:
        result = run_delta(args.task, args.output, args.parallelism)
    if args.command == "prepare-alignment":
        result = prepare_alignment(args.task, args.output)
    if args.command == "run-alignment":
        result = run_alignment(args.task, args.output, args.parallelism)
    if args.command == "prepare-final-reviews":
        result = prepare_final_reviews(args.task, args.output)
    if args.command == "run-final-reviews":
        result = run_final_reviews(args.task, args.output, args.parallelism)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
