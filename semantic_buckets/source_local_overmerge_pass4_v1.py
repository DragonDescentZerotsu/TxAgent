"""One more parent-local TF-IDF merge pass over the completed Pass-3 candidate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_buckets import source_local_overmerge_pass3_v1 as prior
from semantic_buckets import source_local_overmerge_reconcile_v1 as ledger
from semantic_buckets.overmerge_pass2_batches import batch_groups
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_overmerge_pass4.v1"
SOURCE = prior.ROOT / ("semantic_buckets/provenance/source_local_overmerge_pass3_v1/"
                       "20260923_tfidf100_luna_reroute_v1")
PROMPT = prior.PROMPT
BATCH_SIZE = 100


def _groups() -> list[dict[str, Any]]:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "candidate_unselected":
        raise ValueError("Pass-3 candidate is not complete")
    path = SOURCE / "final_groups.json"
    if ledger._hash(path) != manifest["final_groups_sha256"]:
        raise ValueError("Pass-3 groups changed")
    for task in ("dili", "carcinogens"):
        target = SOURCE / task / "source_semantic_bucket_map.parquet"
        if ledger._hash(target) != manifest["counts"][task]["source_map_sha256"]:
            raise ValueError(f"{task}: Pass-3 source map changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for batch in batch_groups(groups, batch_size=BATCH_SIZE):
        if len(batch) < 2:
            continue
        first = batch[0]
        if any(any(row[key] != first[key] for key in ("task", "parent_id", "level", "source_id"))
               for row in batch):
            raise ValueError("Pass-4 TF-IDF batch crossed a scientific scope")
        scope = {key: first[key] for key in ("parent_id", "level", "source_id")}
        scope["provider_family"] = "luna"
        items = [row["group_id"] for row in batch]
        requests.append({"request_id": ledger._id(VERSION, first["task"], scope, items),
                         "task": first["task"], "pass_number": 4, "scope": scope,
                         "items": items, "prompt": prior._render(batch)})
    return requests


def _verify(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = {"source_manifest_sha256": SOURCE / "manifest.json",
                "source_groups_sha256": SOURCE / "final_groups.json",
                "prompt_sha256": PROMPT}
    if (manifest.get("version") != VERSION or manifest.get("model") != prior.LUNA_MODEL or
            manifest.get("service_tier") != "flex" or
            manifest.get("reasoning_effort") != "high" or
            manifest.get("max_tokens") != prior.MAX_TOKENS or
            manifest.get("max_inflight") != prior.PER_PROVIDER_INFLIGHT):
        raise ValueError("Pass-4 request configuration changed")
    if any(ledger._hash(path) != manifest[key] for key, path in expected.items()):
        raise ValueError("Pass-4 pinned input changed")
    schedule = _schedule(_groups())
    if prior._schedule_hash(schedule) != manifest["schedule_sha256"]:
        raise ValueError("Pass-4 schedule changed")
    connection = ledger._database(output / "requests.sqlite3")
    actual = {row["request_id"]: row for row in connection.execute("SELECT * FROM requests")}
    if set(actual) != {row["request_id"] for row in schedule}:
        raise ValueError("Pass-4 ledger differs from schedule")
    for row in schedule:
        stored = actual[row["request_id"]]
        if (stored["prompt"] != row["prompt"] or
                json.loads(stored["items_json"]) != row["items"] or
                json.loads(stored["scope_json"]) != row["scope"]):
            raise ValueError("Pass-4 request bytes changed")
    return manifest, schedule, connection


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    schedule = _schedule(_groups())
    if not schedule:
        raise ValueError("no Pass-4 requests")
    output.mkdir(parents=True)
    connection = ledger._database(output / "requests.sqlite3")
    ledger._insert(connection, schedule)
    connection.close()
    manifest = {"version": VERSION, "state": "prepared", "batch_size": BATCH_SIZE,
                "model": prior.LUNA_MODEL, "service_tier": "flex",
                "reasoning_effort": "high", "max_tokens": prior.MAX_TOKENS,
                "max_inflight": prior.PER_PROVIDER_INFLIGHT,
                "source_manifest_sha256": ledger._hash(SOURCE / "manifest.json"),
                "source_groups_sha256": ledger._hash(SOURCE / "final_groups.json"),
                "prompt_sha256": ledger._hash(PROMPT),
                "schedule_sha256": prior._schedule_hash(schedule),
                "requests": len(schedule),
                "max_prompt_chars": max(len(row["prompt"]) for row in schedule)}
    write_json_atomic(output / "manifest.json", manifest)
    _verify(output)[2].close()
    return manifest


def run(output: Path) -> None:
    manifest, schedule, connection = _verify(output)
    if manifest["state"] not in {"prepared", "running", "incomplete"}:
        raise ValueError("Pass 4 is not resumable from this state")
    pending = [dict(row) for row in connection.execute(
        "SELECT * FROM requests WHERE status!='complete' ORDER BY request_id")]
    clients, receipts = prior._clients({"luna"})
    manifest.update(state="running", endpoints=receipts,
                    aggregate_capacity=prior.PER_PROVIDER_INFLIGHT)
    write_json_atomic(output / "manifest.json", manifest)
    with ThreadPoolExecutor(max_workers=min(prior.PER_PROVIDER_INFLIGHT, len(pending) or 1)) as pool:
        futures = {pool.submit(prior._call, clients["luna"], row): row["request_id"]
                   for row in pending}
        for finished, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            connection.execute("""UPDATE requests SET status=?,attempts=?,response_json=?,reasoning=?,
                receipts_json=?,error=? WHERE request_id=?""",
                (*[result[key] for key in ("status", "attempts", "response_json", "reasoning",
                                            "receipts_json", "error")], futures[future]))
            connection.commit()
            if finished % 25 == 0 or finished == len(pending):
                print(f"Pass 4 saved {finished}/{len(pending)} resumed requests", flush=True)
    clients["luna"].close()
    complete = connection.execute("SELECT count(*) FROM requests WHERE status='complete'").fetchone()[0]
    manifest.update(state="complete" if complete == len(schedule) else "incomplete",
                    completed_requests=complete)
    write_json_atomic(output / "manifest.json", manifest)
    if complete != len(schedule):
        raise RuntimeError(f"Pass 4 has {len(schedule) - complete} failed requests")


def _decisions(groups: list[dict[str, Any]], connection: Any,
               schedule: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_id = {row["group_id"]: row for row in groups}
    merges, replacements = [], {}
    for row in ledger._complete_rows(connection, 4, {item["request_id"] for item in schedule}):
        items = json.loads(row["items_json"])
        for decision in json.loads(row["response_json"])["merge_sets"]:
            members = [items[index] for index in decision["member_ids"]]
            first = by_id[members[0]]
            if any(member in replacements or by_id[member]["parent_id"] != first["parent_id"]
                   or by_id[member]["task"] != first["task"] for member in members):
                raise ValueError("Pass-4 merge repeats or crosses a parent")
            values = sorted(value for member in members for value in by_id[member]["values"])
            if len(values) != len(set(values)):
                raise ValueError("Pass-4 merge repeats a canonical value")
            child = "sbo4_" + ledger._id(first["task"], first["parent_id"], values)[:24]
            replacements.update({member: child for member in members})
            merges.append({"task": first["task"], "parent_id": first["parent_id"],
                           "level": first["level"], "source_id": first["source_id"],
                           "group_id": child, "predecessor_group_ids": members,
                           "values": values, "label": decision["label"],
                           "rationale": decision["rationale"], "request_id": row["request_id"],
                           "provider_family": "luna"})
    return merges, replacements


def finalize(output: Path) -> None:
    manifest, schedule, connection = _verify(output)
    if manifest["state"] != "complete":
        raise ValueError("Pass 4 must complete before materialization")
    groups = _groups()
    merges, replacements = _decisions(groups, connection, schedule)
    successor = [row for row in groups if row["group_id"] not in replacements] + merges
    old_values = {(row["task"], row["parent_id"], value)
                  for row in groups for value in row["values"]}
    new_values = [(row["task"], row["parent_id"], value)
                  for row in successor for value in row["values"]]
    if len(new_values) != len(set(new_values)) or set(new_values) != old_values:
        raise ValueError("Pass-4 successor lost or duplicated a canonical value")
    write_json_atomic(output / "final_groups.json", successor)
    counts = {}
    for task in ("dili", "carcinogens"):
        original = pd.read_parquet(SOURCE / task / "source_semantic_bucket_map.parquet")
        updated = original.copy()
        updated["source_semantic_bucket_id"] = (
            original.source_semantic_bucket_id.map(replacements).fillna(
                original.source_semantic_bucket_id))
        target = output / task / "source_semantic_bucket_map.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(target, index=False)
        counts[task] = {"input_buckets": int(original.source_semantic_bucket_id.nunique()),
                        "candidate_buckets": int(updated.source_semantic_bucket_id.nunique()),
                        "merged_groups": sum(row["task"] == task for row in merges),
                        "source_map_sha256": ledger._hash(target)}
    manifest.update(state="candidate_for_small_bucket_pass",
                    final_groups_sha256=ledger._hash(output / "final_groups.json"), counts=counts)
    write_json_atomic(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(output), indent=2))
    elif args.action == "run":
        run(output)
    else:
        finalize(output)


if __name__ == "__main__":
    main()
