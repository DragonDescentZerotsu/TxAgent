"""Review overmerged source-local semantic parents in two value-merge passes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

import pandas as pd

from semantic_buckets.overmerge_pass2_batches import batch_groups
from semantic_buckets import bioavailability_semantic_readout_v1 as completion_pool
from semantic_buckets import source_local_small_bucket_merge_v1 as prior
from semantic_buckets.materialize_overmerge_candidate import materialize as materialize_candidate
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_overmerge_reconcile.v1"
BATCH_SIZE = 40
MAX_TOKENS = 48_000
ENDPOINT_INFLIGHT = 512
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "semantic_buckets/provenance/source_local_small_bucket_merge_v1/20260922_163906/inputs"
MAPS = ROOT / "semantic_buckets/provenance/source_local_small_bucket_merge_v1/20260922_185048_le5"
CHOICES = ROOT / "semantic_buckets/provenance/semantic_overmerge_partition_v1/20260922_parent_column_draft/parent_columns.tsv"
CLUSTERS = {task: Path(f"/tmp/manual_semantic_partitions/{task}_one_column_clusters.jsonl")
            for task in ("dili", "carcinogens")}
PROMPT = ROOT / "semantic_buckets/prompts/source_local_overmerge_reconcile_v1/merge.md"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read_clusters(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verify_pins(manifest: dict[str, Any]) -> None:
    entries = [manifest["inputs"]["parent_columns"], manifest["prompt"]]
    entries.extend(entry for task in CLUSTERS for entry in manifest["inputs"][task].values())
    for entry in entries:
        if _hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"pinned semantic input changed: {entry['path']}")


def _values(row: dict[str, Any]) -> list[str]:
    return [str(item["value"] if isinstance(item, dict) else item) for item in row["labels"]]


def _validate_inputs(output: Path) -> dict[str, Any]:
    choices = pd.read_csv(CHOICES, sep="\t")
    high = choices[choices.review_mode.eq("tfidf_review") | choices.review_mode.eq("tfidf_review_clusters")]
    inputs: dict[str, Any] = {"parent_columns": {"path": str(CHOICES), "sha256": _hash(CHOICES)}}
    for task, source in CLUSTERS.items():
        target = output / "inputs" / f"{task}_clusters.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if source.exists() and _hash(source) != _hash(target):
            raise ValueError(f"{task}: input cluster snapshot differs from source")
        clusters = _read_clusters(target)
        expected = high[high.task.eq(task)]
        if set(expected.chosen_column) != {"canonical_assay_context"}:
            raise ValueError(f"{task}: unexpected selected column")
        if {row["parent_bucket_id"] for row in clusters} != set(expected.parent_id):
            raise ValueError(f"{task}: cluster parent coverage mismatch")
        atom_path = INPUT / task / "input_atoms.parquet"
        map_path = MAPS / task / "source_semantic_bucket_map.parquet"
        atoms = pd.read_parquet(atom_path, columns=["atom_id", "values_json"])
        mapping = pd.read_parquet(map_path, columns=["atom_id", "source_semantic_bucket_id"])
        selected = mapping[mapping.source_semantic_bucket_id.isin(expected.parent_id)]
        joined = selected.merge(atoms, on="atom_id", validate="one_to_one")
        seen: dict[str, set[str]] = {}
        for row in clusters:
            labels = _values(row)
            if len(labels) != len(set(labels)) or not 1 <= len(labels) <= 50:
                raise ValueError(f"{task}: malformed review cluster")
            values = seen.setdefault(row["parent_bucket_id"], set())
            if values.intersection(labels):
                raise ValueError(f"{task}: canonical value repeated across clusters")
            values.update(labels)
        for parent, rows in joined.groupby("source_semantic_bucket_id"):
            actual = {str(json.loads(value).get("canonical_assay_context") or "__unknown__")
                      for value in rows.values_json}
            if actual != seen.get(parent, set()):
                raise ValueError(f"{task}/{parent}: selected-column value coverage mismatch")
        inputs[task] = {name: {"path": str(path), "sha256": _hash(path)} for name, path in {
            "clusters": target, "atoms": atom_path, "semantic_map": map_path}.items()}
    return inputs


def _render(task: str, row: dict[str, Any], labels: list[str], phase: int) -> str:
    head = PROMPT.read_text(encoding="utf-8").strip()
    endpoints = set(row.get("canonical_endpoint_concepts", []))
    endpoints.update(value for label in row["labels"] if isinstance(label, dict)
                     for value in label.get("endpoint_context", []))
    lines = [head, "", f"Task: {task}", f"Level: {row['level']}",
             f"Source: {row['source_id']}", f"Original semantic parent: {row['parent_bucket_id']}",
             "Selected column: canonical_assay_context",
             "Contextual endpoint(s): " + ", ".join(sorted(endpoints)),
             f"Pass: {phase}", "", "Candidate IDs and canonical values:"]
    lines.extend(f"{index}: {json.dumps(value, ensure_ascii=False)}" for index, value in enumerate(labels))
    lines += ["", "Return only JSON:",
              '{"merge_sets":[{"member_ids":[0,1],"label":"short scientific identity",'
              '"rationale":"why every member is interchangeable evidence"}]}',
              "The response example specifies shape only; it is not a suggested decision."]
    return "\n".join(lines)


def _schedule(output: Path) -> list[dict[str, Any]]:
    requests = []
    parents: dict[tuple[str, str], dict[str, Any]] = {}
    for task in CLUSTERS:
        for row in _read_clusters(output / "inputs" / f"{task}_clusters.jsonl"):
            key = (task, row["parent_bucket_id"])
            parent = parents.setdefault(key, {**row, "labels": [], "canonical_endpoint_concepts": []})
            parent["labels"].extend(_values(row))
            parent["canonical_endpoint_concepts"].extend(row.get("canonical_endpoint_concepts", []))
            parent["canonical_endpoint_concepts"].extend(
                value for label in row["labels"] if isinstance(label, dict)
                for value in label.get("endpoint_context", []))
    for (task, parent_id), row in sorted(parents.items()):
        labels = row["labels"]
        row["canonical_endpoint_concepts"] = sorted(set(row["canonical_endpoint_concepts"]))
        for offset in range(0, len(labels), BATCH_SIZE):
            batch = labels[offset:offset + BATCH_SIZE]
            if len(batch) < 2:
                continue
            scope = {"parent_id": parent_id, "level": row["level"], "source_id": row["source_id"],
                     "canonical_endpoint": ", ".join(row["canonical_endpoint_concepts"])}
            requests.append({"request_id": _id(VERSION, 1, task, scope, batch), "task": task,
                             "pass_number": 1, "scope": scope, "items": batch,
                             "prompt": _render(task, row, batch, 1)})
    return requests


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY, task TEXT NOT NULL, pass_number INTEGER NOT NULL,
        scope_json TEXT NOT NULL, items_json TEXT NOT NULL, prompt TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
        response_json TEXT, reasoning TEXT, receipts_json TEXT NOT NULL DEFAULT '[]', error TEXT)""")
    connection.commit()
    return connection


def _insert(connection: sqlite3.Connection, requests: list[dict[str, Any]]) -> None:
    connection.executemany("""INSERT OR IGNORE INTO requests
        (request_id,task,pass_number,scope_json,items_json,prompt) VALUES (?,?,?,?,?,?)""",
        [(row["request_id"], row["task"], row["pass_number"], json.dumps(row["scope"]),
          json.dumps(row["items"], ensure_ascii=False), row["prompt"]) for row in requests])
    connection.commit()


def _validate_response(raw: str, count: int) -> dict[str, Any]:
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"merge_sets"}:
        raise ValueError("response must contain only merge_sets")
    if not isinstance(result["merge_sets"], list):
        raise ValueError("merge_sets must be a list")
    seen: set[int] = set()
    for group in result["merge_sets"]:
        if not isinstance(group, dict) or set(group) != {"member_ids", "label", "rationale"}:
            raise ValueError("merge group fields differ from approved schema")
        ids = group["member_ids"]
        if not isinstance(ids, list) or len(ids) < 2 or any(type(x) is not int for x in ids):
            raise ValueError("merge group needs two or more integer IDs")
        if (len(ids) != len(set(ids)) or any(x < 0 or x >= count for x in ids)
                or seen.intersection(ids)):
            raise ValueError("merge group has repeated or out-of-range IDs")
        if not all(isinstance(group[key], str) and group[key].strip()
                   for key in ("label", "rationale")):
            raise ValueError("merge group lacks scientific label or rationale")
        seen.update(ids)
    return result


def _call(client: Any, row: dict[str, Any]) -> dict[str, Any]:
    receipts = json.loads(row["receipts_json"])
    attempts = int(row["attempts"])
    error = None
    for _ in range(4):
        attempts += 1
        started = time.time()
        try:
            completion = client.chat.completions.create(
                model=prior.MODEL,
                messages=[{"role": "system", "content": "Return only valid JSON."},
                          {"role": "user", "content": row["prompt"]}],
                reasoning_effort="high", max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            )
            message = completion.choices[0].message
            parsed = _validate_response(message.content or "", len(json.loads(row["items_json"])))
            receipts.append({"attempt": attempts, "elapsed_seconds": time.time() - started,
                             "generation_id": getattr(completion, "id", None),
                             "transport": "nonstreaming",
                             "provider_name": getattr(completion, "provider_name", None),
                             "provider_base_url": getattr(completion, "provider_base_url", None)})
            return {"status": "complete", "attempts": attempts, "response_json": json.dumps(parsed),
                    "reasoning": getattr(message, "reasoning_content", None),
                    "receipts_json": json.dumps(receipts), "error": None}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            receipts.append({"attempt": attempts, "elapsed_seconds": time.time() - started,
                             "transport": "nonstreaming", "error": error})
    return {"status": "failed", "attempts": attempts, "response_json": None,
            "reasoning": None, "receipts_json": json.dumps(receipts), "error": error}


def _execute(connection: sqlite3.Connection, client: Any, capacity: int, phase: int) -> None:
    pending = [dict(row) for row in connection.execute(
        "SELECT * FROM requests WHERE pass_number=? AND status!='complete'", (phase,))]
    pending.sort(key=lambda row: _id("dispatch", row["request_id"]))
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=min(capacity, len(pending))) as executor:
        futures = {executor.submit(_call, client, row): row["request_id"] for row in pending}
        saved = 0
        for future in as_completed(futures):
            result = future.result()
            connection.execute("""UPDATE requests SET status=?,attempts=?,response_json=?,reasoning=?,
                receipts_json=?,error=? WHERE request_id=?""",
                (*[result[key] for key in ("status", "attempts", "response_json", "reasoning",
                                           "receipts_json", "error")], futures[future]))
            saved += 1
            if saved % 25 == 0:
                connection.commit()
        connection.commit()
    failed = connection.execute("SELECT COUNT(*) FROM requests WHERE pass_number=? AND status!='complete'",
                                (phase,)).fetchone()[0]
    if failed:
        raise RuntimeError(f"pass {phase} has {failed} failed requests; ledger retained for retry")


def _build_pool() -> tuple[Any, list[dict[str, Any]], int, str]:
    # Streaming thousands of high-reasoning tokens saturates one Python SSE parser.
    completion_pool._stream_completion = lambda client, kwargs: client.chat.completions.create(**kwargs)
    prior.ENDPOINT_INFLIGHT = ENDPOINT_INFLIGHT
    return prior._build_pool()


def _resolved(items: list[Any], response: dict[str, Any]) -> list[tuple[list[Any], str, str]]:
    groups = []
    used: set[int] = set()
    for decision in response["merge_sets"]:
        indices = decision["member_ids"]
        used.update(indices)
        groups.append(([items[index] for index in indices], decision["label"], decision["rationale"]))
    groups.extend(([item], "", "Unmerged singleton") for index, item in enumerate(items)
                  if index not in used)
    return groups


def _complete_rows(connection: sqlite3.Connection, phase: int,
                   expected_ids: set[str]) -> list[sqlite3.Row]:
    rows = connection.execute("SELECT * FROM requests WHERE pass_number=? ORDER BY request_id",
                              (phase,)).fetchall()
    if not expected_ids or {row["request_id"] for row in rows} != expected_ids:
        raise ValueError(f"pass {phase} request ledger does not match frozen schedule")
    if any(row["status"] != "complete" for row in rows):
        raise ValueError(f"pass {phase} is incomplete")
    return rows


def materialize_pass1(output: Path) -> list[dict[str, Any]]:
    connection = _database(output / "requests.sqlite3")
    groups: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    expected_ids = {row["request_id"] for row in _schedule(output)}
    for row in _complete_rows(connection, 1, expected_ids):
        scope, items = json.loads(row["scope_json"]), json.loads(row["items_json"])
        response = json.loads(row["response_json"])
        for values, label, rationale in _resolved(items, response):
            for value in values:
                key = (row["task"], scope["parent_id"], value)
                if key in seen:
                    raise ValueError(f"duplicate pass-1 value: {key}")
                seen.add(key)
            groups.append({"task": row["task"], "parent_id": scope["parent_id"],
                           "level": scope["level"], "source_id": scope["source_id"],
                           "group_id": "sbo1_" + _id(row["task"], scope["parent_id"], sorted(values))[:24],
                           "label": label or values[0], "values": sorted(values),
                           "rationale": rationale, "pass1_request_id": row["request_id"]})
    for task in CLUSTERS:
        for row in _read_clusters(output / "inputs" / f"{task}_clusters.jsonl"):
            for value in _values(row):
                key = (task, row["parent_bucket_id"], value)
                if key in seen:
                    continue
                seen.add(key)
                groups.append({"task": task, "parent_id": row["parent_bucket_id"],
                               "level": row["level"], "source_id": row["source_id"],
                               "group_id": "sbo1_" + _id(task, row["parent_bucket_id"], value)[:24],
                               "label": value, "values": [value],
                               "rationale": "Single-value review neighborhood", "pass1_request_id": None})
    expected = {(task, row["parent_bucket_id"], value) for task in CLUSTERS
                for row in _read_clusters(output / "inputs" / f"{task}_clusters.jsonl")
                for value in _values(row)}
    if seen != expected or len({g["group_id"] for g in groups}) != len(groups):
        raise ValueError("pass-1 groups do not cover canonical values exactly once")
    write_json_atomic(output / "pass1_groups.json", groups)
    return groups


def _render_pass2(batch: list[dict[str, Any]]) -> str:
    first = batch[0]
    head = PROMPT.read_text(encoding="utf-8").strip()
    lines = [head, "", f"Task: {first['task']}", f"Level: {first['level']}",
             f"Source: {first['source_id']}", f"Original semantic parent: {first['parent_id']}",
             "Selected column: canonical_assay_context", "Pass: 2",
             "These provisional groups may come from different Pass-1 batches. Apply the same scientific test to every underlying canonical value in each group. The provisional label alone is not evidence of equivalence.",
             "", "Candidate IDs and provisional groups:"]
    for index, group in enumerate(batch):
        lines.append(json.dumps({"id": index, "label": group["label"],
                                 "canonical_values": group["values"], "pass1_rationale": group["rationale"]},
                                ensure_ascii=False))
    lines += ["", "Return only JSON:",
              '{"merge_sets":[{"member_ids":[0,1],"label":"short shared scientific identity",'
              '"rationale":"why every underlying value is interchangeable evidence"}]}',
              "The response example specifies shape only; it is not a suggested decision."]
    return "\n".join(lines)


def prepare_pass2(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    _verify_pins(manifest)
    groups = materialize_pass1(output)
    schedule = []
    for batch in batch_groups(groups, batch_size=BATCH_SIZE):
        if len(batch) < 2:
            continue
        first = batch[0]
        scope = {key: first[key] for key in ("parent_id", "level", "source_id")}
        items = [group["group_id"] for group in batch]
        schedule.append({"request_id": _id(VERSION, 2, first["task"], scope, items),
                         "task": first["task"], "pass_number": 2, "scope": scope,
                         "items": items, "prompt": _render_pass2(batch)})
    connection = _database(output / "requests.sqlite3")
    _insert(connection, schedule)
    manifest.update(state="prepared_pass2", pass1_groups_sha256=_hash(output / "pass1_groups.json"),
                    pass1_groups=len(groups), pass2_requests=len(schedule),
                    exact_review_manifest_sha256=_hash(
                        ROOT / "semantic_buckets/provenance/semantic_overmerge_partition_v1/"
                        "20260922_exact_reviewed_v2/manifest.json"))
    write_json_atomic(output / "manifest.json", manifest)


def run_pass2(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["state"] == "pass1_complete":
        prepare_pass2(output)
    client, receipts, capacity, inventory_hash = _build_pool()
    connection = _database(output / "requests.sqlite3")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(state="running_pass2", endpoints=receipts, aggregate_capacity=capacity,
                    inventory_sha256=inventory_hash, transport="nonstreaming",
                    per_endpoint_max_inflight=ENDPOINT_INFLIGHT)
    write_json_atomic(output / "manifest.json", manifest)
    _execute(connection, client, capacity, 2)
    manifest["state"] = "pass2_complete"
    manifest["pass2_completed"] = connection.execute(
        "SELECT COUNT(*) FROM requests WHERE pass_number=2 AND status='complete'").fetchone()[0]
    write_json_atomic(output / "manifest.json", manifest)


def materialize_pass2(output: Path) -> list[dict[str, Any]]:
    provisional = json.loads((output / "pass1_groups.json").read_text())
    by_id = {group["group_id"]: group for group in provisional}
    if len(by_id) != len(provisional):
        raise ValueError("duplicate provisional group ID")
    connection = _database(output / "requests.sqlite3")
    final = []
    seen: set[str] = set()
    expected_ids = set()
    for batch in batch_groups(provisional, batch_size=BATCH_SIZE):
        if len(batch) < 2:
            continue
        first = batch[0]
        scope = {key: first[key] for key in ("parent_id", "level", "source_id")}
        expected_ids.add(_id(VERSION, 2, first["task"], scope,
                             [group["group_id"] for group in batch]))
    for row in _complete_rows(connection, 2, expected_ids):
        items = json.loads(row["items_json"])
        for members, label, rationale in _resolved(items, json.loads(row["response_json"])):
            if any(item in seen or item not in by_id for item in members):
                raise ValueError("pass-2 group repeated or unknown")
            seen.update(members)
            first = by_id[members[0]]
            if any(by_id[item]["parent_id"] != first["parent_id"] or
                   by_id[item]["task"] != first["task"] for item in members):
                raise ValueError("pass-2 merge crossed a task or original parent")
            values = sorted(value for item in members for value in by_id[item]["values"])
            final.append({"task": first["task"], "parent_id": first["parent_id"],
                          "level": first["level"], "source_id": first["source_id"],
                          "group_id": "sbo2_" + _id(first["task"], first["parent_id"], values)[:24],
                          "label": label or first["label"], "values": values,
                          "rationale": rationale, "pass1_group_ids": sorted(members),
                          "pass2_request_id": row["request_id"]})
    for group in provisional:
        if group["group_id"] not in seen:
            seen.add(group["group_id"])
            final.append({**group, "group_id": "sbo2_" + _id(group["task"], group["parent_id"],
                                                               group["values"])[:24],
                          "pass1_group_ids": [group["group_id"]], "pass2_request_id": None})
    if seen != set(by_id) or len({g["group_id"] for g in final}) != len(final):
        raise ValueError("pass-2 group coverage mismatch")
    original_values = {(g["task"], g["parent_id"], value) for g in provisional for value in g["values"]}
    final_values = [(g["task"], g["parent_id"], value) for g in final for value in g["values"]]
    if len(final_values) != len(set(final_values)) or set(final_values) != original_values:
        raise ValueError("pass-2 canonical value coverage mismatch")
    write_json_atomic(output / "final_groups.json", final)
    return final


def _decision_lookup(output: Path) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    exact_path = (ROOT / "semantic_buckets/provenance/semantic_overmerge_partition_v1/"
                  "20260922_exact_reviewed_v2/exact_value_decisions.tsv")
    exact = pd.read_csv(exact_path, sep="\t").fillna("")
    for row in exact.itertuples(index=False):
        key = (row.task, row.parent_id, row.value)
        if key in lookup:
            raise ValueError(f"duplicate exact decision: {key}")
        lookup[key] = "sbo_" + _id(VERSION, row.task, row.parent_id, row.column,
                                    row.final_child_label)[:24]
    for group in materialize_pass2(output):
        for value in group["values"]:
            key = (group["task"], group["parent_id"], value)
            if key in lookup:
                raise ValueError(f"duplicate cross-mode value: {key}")
            lookup[key] = group["group_id"]
    return lookup


def finalize(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["state"] != "pass2_complete":
        raise ValueError("Pass 2 must finish before candidate materialization")
    _verify_pins(manifest)
    exact_manifest = (ROOT / "semantic_buckets/provenance/semantic_overmerge_partition_v1/"
                      "20260922_exact_reviewed_v2/manifest.json")
    if _hash(exact_manifest) != manifest["exact_review_manifest_sha256"]:
        raise ValueError("reviewed exact-value manifest changed")
    exact_entry = json.loads(exact_manifest.read_text())["output"]
    if _hash(Path(exact_entry["path"])) != exact_entry["sha256"]:
        raise ValueError("reviewed exact-value decisions changed")
    if _hash(output / "pass1_groups.json") != manifest["pass1_groups_sha256"]:
        raise ValueError("provisional groups changed after pass-2 scheduling")
    lookup = _decision_lookup(output)
    choices = pd.read_csv(CHOICES, sep="\t")
    columns = choices.set_index(["task", "parent_id"]).chosen_column.to_dict()
    counts: dict[str, Any] = {}
    for task in CLUSTERS:
        atoms = pd.read_parquet(INPUT / task / "input_atoms.parquet",
                                columns=["atom_id", "values_json"])
        mapping = pd.read_parquet(MAPS / task / "source_semantic_bucket_map.parquet")
        selected = mapping[mapping.source_semantic_bucket_id.isin(
            choices[choices.task.eq(task)].parent_id)]
        joined = selected.merge(atoms, on="atom_id", validate="one_to_one")
        replacement: dict[str, str] = {}
        for row in joined.itertuples(index=False):
            parent = row.source_semantic_bucket_id
            column = columns[(task, parent)]
            value = str(json.loads(row.values_json).get(column) or "__unknown__")
            child = lookup.get((task, parent, value))
            if child is None:
                raise ValueError(f"{task}/{parent}: no reviewed child for {value!r}")
            replacement[row.atom_id] = child
        if len(replacement) != len(selected):
            raise ValueError(f"{task}: atom replacement coverage mismatch")
        result = mapping.copy()
        result["source_semantic_bucket_id"] = result.atom_id.map(replacement).fillna(
            result.source_semantic_bucket_id)
        if len(result) != len(mapping) or result.atom_id.tolist() != mapping.atom_id.tolist():
            raise ValueError(f"{task}: semantic map row identity changed")
        target = output / task / "source_semantic_bucket_map.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(target, index=False)
        counts[task] = {"reassigned_atoms": len(replacement),
                        "old_buckets": int(mapping.source_semantic_bucket_id.nunique()),
                        "candidate_buckets": int(result.source_semantic_bucket_id.nunique()),
                        "map_sha256": _hash(target)}
    for task in CLUSTERS:
        counts[task].update(materialize_candidate(task, output))
    exact_path = (ROOT / "semantic_buckets/provenance/semantic_overmerge_partition_v1/"
                  "20260922_exact_reviewed_v2/exact_value_decisions.tsv")
    manifest.update(state="candidate_unselected", exact_decisions_sha256=_hash(exact_path),
                    final_groups_sha256=_hash(output / "final_groups.json"), counts=counts)
    write_json_atomic(output / "manifest.json", manifest)


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    inputs = _validate_inputs(output)
    connection = _database(output / "requests.sqlite3")
    schedule = _schedule(output)
    _insert(connection, schedule)
    write_json_atomic(output / "manifest.json", {
        "version": VERSION, "state": "prepared_pass1", "batch_size": BATCH_SIZE,
        "inputs": inputs, "prompt": {"path": str(PROMPT), "sha256": _hash(PROMPT)},
        "pass1_requests": len(schedule), "model": prior.MODEL, "reasoning_effort": "high",
        "max_tokens": MAX_TOKENS,
    })


def run_pass1(output: Path) -> None:
    if not (output / "manifest.json").exists():
        prepare(output)
    client, receipts, capacity, inventory_hash = _build_pool()
    connection = _database(output / "requests.sqlite3")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(state="running_pass1", endpoints=receipts, aggregate_capacity=capacity,
                    inventory_sha256=inventory_hash, transport="nonstreaming",
                    per_endpoint_max_inflight=ENDPOINT_INFLIGHT)
    write_json_atomic(output / "manifest.json", manifest)
    _execute(connection, client, capacity, 1)
    manifest["state"] = "pass1_complete"
    manifest["pass1_completed"] = connection.execute(
        "SELECT COUNT(*) FROM requests WHERE pass_number=1 AND status='complete'").fetchone()[0]
    write_json_atomic(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run-pass1", "prepare-pass2", "run-pass2",
                                           "finalize"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve())
    elif args.action == "run-pass1":
        run_pass1(args.output.resolve())
    elif args.action == "prepare-pass2":
        prepare_pass2(args.output.resolve())
    elif args.action == "run-pass2":
        run_pass2(args.output.resolve())
    else:
        finalize(args.output.resolve())


if __name__ == "__main__":
    main()
