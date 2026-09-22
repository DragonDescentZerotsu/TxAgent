"""Two-pass source-local review of small semantic buckets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Mapping, Sequence
import urllib.request

from jinja2 import Environment, StrictUndefined
import pandas as pd

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import official_two_pass_weights as cards_v6
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_small_bucket_merge.v1"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
MAX_TOKENS = 65_536
MAX_ATTEMPTS = 4
BATCH_SIZE = 40
SAMPLE_SIZE = 3
ENDPOINT_INFLIGHT = 400
ROOT = Path(__file__).resolve().parents[1]
PROMPT = Path(__file__).with_name("prompts") / VERSION.replace(".", "_") / "merge.jinja"
INVENTORY = ROOT / "predict/api_client/providers/current_endpoints.json"
DEFAULT_INPUT = Path("/local/jojolee/txagent_semantic_dili_carcinogens_finish_20260922_103428")
TASKS = ("dili", "carcinogens")
_WRITE_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable(*parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _endpoint(values_json: str) -> str:
    value = json.loads(values_json).get("canonical_endpoint_concept")
    return str(value or "__unknown__")


def _load_task(input_root: Path, task: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run = input_root / task / "semantic_run"
    atoms = pd.read_parquet(run / "input_atoms.parquet")
    mapping = pd.read_parquet(run / "source_semantic_bucket_map.parquet")
    with gzip.open(run / "sample_cards.json.gz", "rt", encoding="utf-8") as handle:
        samples = json.load(handle)
    for column in ("atom_id", "level", "source_id", "values_json"):
        atoms[column] = atoms[column].astype(str)
    for column in ("atom_id", "level", "source_id", "source_semantic_bucket_id"):
        mapping[column] = mapping[column].astype(str)
    if set(mapping.atom_id) != set(atoms.atom_id):
        raise ValueError(f"{task}: atom coverage mismatch")
    atoms = atoms.assign(canonical_endpoint=atoms.values_json.map(_endpoint))
    return atoms, mapping, samples


def _bucket_summary(atoms: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    joined = mapping.merge(
        atoms[["atom_id", "record_count", "canonical_endpoint"]], on="atom_id", validate="one_to_one"
    )
    grouped = joined.groupby("source_semantic_bucket_id", sort=True)
    summary = grouped.agg(
        level=("level", "first"), source_id=("source_id", "first"),
        record_count=("record_count", "sum"), endpoint_count=("canonical_endpoint", "nunique"),
        canonical_endpoint=("canonical_endpoint", "first"), atom_count=("atom_id", "size"),
    ).reset_index()
    return summary


def _payload(bucket: str, rows: pd.DataFrame, samples: Mapping[str, Any]) -> dict[str, Any]:
    source = str(rows.source_id.iloc[0])
    selected = cards_v6._sample_atoms(bucket, rows)[:SAMPLE_SIZE]
    records = []
    for atom in selected:
        sample = samples[atom]
        records.append({"source_id": source, **{
            key: value for key, value in sample.items()
            if key not in {"canonical_record_id", "source_id"}
        }})
    return {
        "identity": {"source_components": [cards_v6._component(source, rows)]},
        "sample_records": records,
    }


def _render(task: str, scope: tuple[str, str, str], buckets: Sequence[str],
            atom_lookup: Mapping[str, Mapping[str, str]], members: Mapping[str, Sequence[str]],
            samples: Mapping[str, Any]) -> str:
    level, source, endpoint = scope
    cards_v6.SOURCE_LABELS = {source: source.replace("_", " ")}
    rendered = []
    for local_id, bucket in enumerate(buckets):
        rows = pd.DataFrame([{"atom_id": atom, **atom_lookup[atom]} for atom in members[bucket]])
        rendered.append(cards_v6._card(f"ID {local_id}", _payload(bucket, rows, samples)))
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        PROMPT.read_text(encoding="utf-8")
    )
    return template.render(task=task, level=level, source_id=source,
                           canonical_endpoint=endpoint, last_id=len(buckets) - 1,
                           cards="\n\n".join(rendered)).strip()


def _schedule(task: str, pass_number: int, threshold: int, inclusive: bool,
              atoms: pd.DataFrame, mapping: pd.DataFrame, samples: Mapping[str, Any]) -> list[dict]:
    summary = _bucket_summary(atoms, mapping)
    atom_lookup = {row.atom_id: {"source_id": row.source_id, "values_json": row.values_json}
                   for row in atoms[["atom_id", "source_id", "values_json"]].itertuples(index=False)}
    members = mapping.groupby("source_semantic_bucket_id").atom_id.apply(list).to_dict()
    eligible = summary.record_count.le(threshold) if inclusive else summary.record_count.lt(threshold)
    selected = summary[eligible & summary.endpoint_count.eq(1)].copy()
    rows = []
    for key, group in selected.groupby(["level", "source_id", "canonical_endpoint"], sort=True):
        ordered = sorted(group.source_semantic_bucket_id.astype(str),
                         key=lambda value: (_stable(VERSION, pass_number, *key, value), value))
        for offset in range(0, len(ordered), BATCH_SIZE):
            buckets = ordered[offset:offset + BATCH_SIZE]
            if len(buckets) < 2:
                continue
            prompt = _render(task, key, buckets, atom_lookup, members, samples)
            rows.append({"request_id": _stable(VERSION, task, pass_number, *key, *buckets),
                         "task": task, "pass_number": pass_number, "level": key[0],
                         "source_id": key[1], "canonical_endpoint": key[2],
                         "bucket_ids": buckets, "prompt": prompt})
    return rows


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY, task TEXT NOT NULL, pass_number INTEGER NOT NULL,
        scope_json TEXT NOT NULL, bucket_ids_json TEXT NOT NULL, prompt TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
        response_json TEXT, reasoning TEXT, receipts_json TEXT NOT NULL DEFAULT '[]',
        error TEXT)""")
    connection.commit()
    return connection


def _insert(connection: sqlite3.Connection, schedule: Sequence[Mapping[str, Any]]) -> None:
    for row in schedule:
        scope = {key: row[key] for key in ("level", "source_id", "canonical_endpoint")}
        connection.execute("""INSERT OR IGNORE INTO requests
            (request_id,task,pass_number,scope_json,bucket_ids_json,prompt) VALUES (?,?,?,?,?,?)""",
            (row["request_id"], row["task"], row["pass_number"], json.dumps(scope),
             json.dumps(row["bucket_ids"]), row["prompt"]))
    connection.commit()


def _validate(raw: str, count: int) -> list[list[int]]:
    result = json.loads(raw)
    if not isinstance(result, list):
        raise ValueError("response root is not a list")
    seen: set[int] = set()
    for index, group in enumerate(result):
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError(f"merge group {index} is not a list with at least two IDs")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in group):
            raise ValueError(f"merge group {index} contains a non-integer ID")
        if len(set(group)) != len(group) or any(value < 0 or value >= count for value in group):
            raise ValueError(f"merge group {index} contains repeated or out-of-range IDs")
        if seen.intersection(group):
            raise ValueError(f"merge group {index} overlaps another group")
        seen.update(group)
    return result


def _call(client: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    receipts, last_error = json.loads(row["receipts_json"]), None
    attempts = int(row["attempts"])
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        started = time.time()
        try:
            completion = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": "Return only valid JSON."},
                                       {"role": "user", "content": row["prompt"]}],
                reasoning_effort="high", max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            )
            message = completion.choices[0].message
            parsed = _validate(message.content or "", len(json.loads(row["bucket_ids_json"])))
            receipts.append({"attempt": attempts, "elapsed_seconds": time.time() - started,
                             "generation_id": getattr(completion, "id", None),
                             "model": str(completion.model),
                             "provider_name": getattr(completion, "provider_name", None),
                             "provider_base_url": getattr(completion, "provider_base_url", None)})
            return {"status": "complete", "attempts": attempts, "response_json": json.dumps(parsed),
                    "reasoning": getattr(message, "reasoning_content", None),
                    "receipts_json": json.dumps(receipts), "error": None}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            receipts.append({"attempt": attempts, "elapsed_seconds": time.time() - started,
                             "error": last_error})
    return {"status": "failed", "attempts": attempts, "response_json": None,
            "reasoning": None, "receipts_json": json.dumps(receipts), "error": last_error}


def _execute(connection: sqlite3.Connection, client: Any, capacity: int, pass_number: int) -> None:
    connection.execute("UPDATE requests SET status='pending' WHERE status='in_flight'")
    connection.commit()
    pending = [dict(row) for row in connection.execute(
        "SELECT * FROM requests WHERE pass_number=? AND status!='complete'", (pass_number,))]
    pending.sort(key=lambda row: (_stable(VERSION, "dispatch", pass_number, row["request_id"]),
                                  row["request_id"]))
    if not pending:
        return
    def call(row: Mapping[str, Any]) -> dict[str, Any]:
        with _WRITE_LOCK:
            connection.execute("UPDATE requests SET status='in_flight' WHERE request_id=?",
                               (row["request_id"],))
            connection.commit()
        return _call(client, row)
    def save(request_id: str, result: Mapping[str, Any]) -> None:
        with _WRITE_LOCK:
            connection.execute("""UPDATE requests SET status=?,attempts=?,response_json=?,reasoning=?,
                receipts_json=?,error=? WHERE request_id=?""",
                (*[result[key] for key in ("status", "attempts", "response_json", "reasoning",
                                           "receipts_json", "error")], request_id))
            connection.commit()
    with ThreadPoolExecutor(max_workers=min(capacity, len(pending))) as executor:
        futures = {executor.submit(call, row): row["request_id"] for row in pending}
        for future in as_completed(futures):
            save(futures[future], future.result())
    failures = connection.execute(
        "SELECT COUNT(*) FROM requests WHERE pass_number=? AND status!='complete'", (pass_number,)).fetchone()[0]
    if failures:
        raise RuntimeError(f"pass {pass_number} has {failures} failed requests")


def _apply(connection: sqlite3.Connection, task: str, pass_number: int,
           mapping: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    replacements: dict[str, str] = {}
    merge_count = 0
    rows = connection.execute(
        "SELECT bucket_ids_json,response_json FROM requests WHERE task=? AND pass_number=? AND status='complete'",
        (task, pass_number)).fetchall()
    for row in rows:
        buckets, groups = json.loads(row[0]), json.loads(row[1])
        for group in groups:
            members = sorted(buckets[index] for index in group)
            successor = "sbm_" + _stable(VERSION, task, pass_number, *members)[:24]
            for bucket in members:
                if bucket in replacements:
                    raise ValueError(f"{task}: bucket merged twice in pass {pass_number}: {bucket}")
                replacements[bucket] = successor
            merge_count += 1
    result = mapping.copy()
    result["source_semantic_bucket_id"] = result.source_semantic_bucket_id.astype(str).replace(replacements)
    return result, merge_count


def _configured_endpoints() -> tuple[tuple[dict[str, Any], ...], str]:
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    selected = []
    for item in data["providers"]:
        if item.get("model") != MODEL:
            continue
        try:
            with urllib.request.urlopen(item["base_url"].rstrip("/") + "/models", timeout=10) as response:
                models = {str(row.get("id")) for row in json.load(response).get("data", [])}
        except Exception:
            continue
        if MODEL in models:
            selected.append({"name": item["name"], "base_url": item["base_url"], "provider": "local",
                             "credential_env": "", "max_inflight": ENDPOINT_INFLIGHT})
    if not selected:
        raise ValueError("current endpoint inventory contains no requested DeepSeek endpoints")
    return tuple(selected), _sha256(INVENTORY)


def _build_pool() -> tuple[Any, list[dict[str, Any]], int, str]:
    endpoints, inventory_hash = _configured_endpoints()
    core.MODEL, core.ENDPOINTS = MODEL, endpoints
    core.REQUEST_TIMEOUT_S, core.PROVIDER_POOL_CONFIG = 900, None
    client, receipts, capacity = core._build_completion_pool(ENDPOINT_INFLIGHT)
    return client, receipts, capacity, inventory_hash


def _write_manifest(output: Path, input_root: Path, receipts: Sequence[Mapping[str, Any]],
                    capacity: int, inventory_hash: str, state: str, counts: Mapping[str, Any]) -> None:
    inputs = {}
    for task in TASKS:
        run = input_root / task / "semantic_run"
        inputs[task] = {name: {"path": str(run / name), "sha256": _sha256(run / name)}
                        for name in ("input_atoms.parquet", "source_semantic_bucket_map.parquet",
                                     "sample_cards.json.gz")}
    write_json_atomic(output / "manifest.json", {
        "version": VERSION, "state": state, "input_root": str(input_root), "inputs": inputs,
        "selection": {"pass_1": "record_count <= 10", "pass_2": "record_count < 12",
                      "scope": ["task", "level", "source_id", "canonical_endpoint_concept"],
                      "batch_size": BATCH_SIZE, "representative_records": SAMPLE_SIZE},
        "inference": {"model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
                      "per_endpoint_max_inflight": ENDPOINT_INFLIGHT,
                      "aggregate_capacity": capacity, "inventory_sha256": inventory_hash,
                      "endpoints": list(receipts)}, "counts": dict(counts),
    })


def run(input_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    client, receipts, capacity, inventory_hash = _build_pool()
    connection = _database(output / "requests.sqlite3")
    task_data, maps = {}, {}
    for task in TASKS:
        atoms, mapping, samples = _load_task(input_root, task)
        task_data[task], maps[task] = (atoms, samples), mapping
    counts: dict[str, Any] = {}
    _write_manifest(output, input_root, receipts, capacity, inventory_hash, "running_pass_1", counts)
    for pass_number, threshold, inclusive in ((1, 10, True), (2, 12, False)):
        schedule = []
        for task in TASKS:
            atoms, samples = task_data[task]
            schedule.extend(_schedule(task, pass_number, threshold, inclusive,
                                      atoms, maps[task], samples))
        _insert(connection, schedule)
        counts[f"pass_{pass_number}_requests"] = len(schedule)
        _execute(connection, client, capacity, pass_number)
        for task in TASKS:
            maps[task], merged = _apply(connection, task, pass_number, maps[task])
            counts[f"{task}_pass_{pass_number}_merge_groups"] = merged
        _write_manifest(output, input_root, receipts, capacity, inventory_hash,
                        "running_pass_2" if pass_number == 1 else "finalizing", counts)
    for task, mapping in maps.items():
        target = output / task / "source_semantic_bucket_map.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_parquet(target, index=False)
        counts[f"{task}_final_buckets"] = int(mapping.source_semantic_bucket_id.nunique())
        counts[f"{task}_output_sha256"] = _sha256(target)
    _write_manifest(output, input_root, receipts, capacity, inventory_hash, "complete", counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
