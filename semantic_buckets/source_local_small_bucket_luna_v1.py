"""Aggressively reconcile record-small semantic buckets in endpoint-local batches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_buckets import official_two_pass_weights as cards_v6
from semantic_buckets import source_local_overmerge_pass3_v1 as luna
from semantic_buckets import source_local_overmerge_pass4_v1 as prior
from semantic_buckets import source_local_overmerge_reconcile_v1 as ledger
from semantic_buckets import source_local_small_bucket_merge_v1 as small
from semantic_buckets.materialize_overmerge_candidate import materialize as materialize_candidate
from semantic_buckets.overmerge_pass2_batches import batch_groups
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_small_bucket_luna.v1"
SOURCE = luna.ROOT / ("semantic_buckets/provenance/source_local_overmerge_pass4_v1/"
                       "20260923_tfidf100_luna_flex_v1")
FROZEN = luna.ROOT / ("semantic_buckets/provenance/source_local_small_bucket_merge_v1/"
                       "20260922_163906/inputs")
PROMPT = luna.ROOT / "semantic_buckets/prompts/source_local_small_bucket_luna_v1/merge.md"
BATCH_SIZE = 100
MAX_RECORDS = 10
TASKS = ("dili", "carcinogens")


def _paths() -> dict[str, Path]:
    result = {"source_manifest": SOURCE / "manifest.json", "prompt": PROMPT}
    for task in TASKS:
        result[f"{task}_source_map"] = SOURCE / task / "source_semantic_bucket_map.parquet"
        result[f"{task}_atoms"] = FROZEN / task / "input_atoms.parquet"
        result[f"{task}_samples"] = FROZEN / task / "sample_cards.json.gz"
    return result


def _inputs(task: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("state") != "candidate_for_small_bucket_pass":
        raise ValueError("Pass-4 source candidate is not complete")
    mapping_path = SOURCE / task / "source_semantic_bucket_map.parquet"
    if ledger._hash(mapping_path) != source_manifest["counts"][task]["source_map_sha256"]:
        raise ValueError(f"{task}: Pass-4 source map changed")
    atoms = pd.read_parquet(FROZEN / task / "input_atoms.parquet")
    mapping = pd.read_parquet(mapping_path)
    if len(atoms) != len(mapping) or set(atoms.atom_id) != set(mapping.atom_id):
        raise ValueError(f"{task}: candidate changed the frozen atom universe")
    atoms = atoms.assign(canonical_endpoint=atoms.values_json.map(small._endpoint))
    with gzip.open(FROZEN / task / "sample_cards.json.gz", "rt", encoding="utf-8") as handle:
        samples = json.load(handle)
    return atoms, mapping, samples


def _render(batch: list[dict[str, Any]], lookup: dict[str, dict[str, str]],
            members: dict[str, list[str]], samples: dict[str, Any]) -> str:
    first = batch[0]
    cards_v6.SOURCE_LABELS = {first["source_id"]: first["source_id"].replace("_", " ")}
    lines = [PROMPT.read_text(encoding="utf-8").strip(), "",
             f"Task: {first['task']}", f"Evidence level: {first['level']}",
             f"Source: {first['source_id']}",
             f"Canonical endpoint: {first['canonical_endpoint']}", "",
             "Candidate IDs and three-record bucket cards:"]
    for index, group in enumerate(batch):
        bucket = group["group_id"]
        rows = pd.DataFrame([{"atom_id": atom, **lookup[atom]} for atom in members[bucket]])
        lines.append(cards_v6._card(f"ID {index}", small._payload(bucket, rows, samples)))
    lines += ["", "Return only JSON:",
              '{"merge_sets":[{"member_ids":[0,1],"label":"shared scientific identity",'
              '"rationale":"why every underlying bucket is interchangeable"}]}',
              "The example gives schema only, not a suggested decision."]
    return "\n\n".join(lines)


def _task_schedule(task: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    atoms, mapping, samples = _inputs(task)
    summary = small._bucket_summary(atoms, mapping)
    selected = summary[summary.record_count.le(MAX_RECORDS) & summary.endpoint_count.eq(1)]
    selected_ids = set(selected.source_semantic_bucket_id.astype(str))
    members = (mapping[mapping.source_semantic_bucket_id.isin(selected_ids)]
               .groupby("source_semantic_bucket_id").atom_id.apply(list).to_dict())
    lookup = {row.atom_id: {"source_id": row.source_id, "values_json": row.values_json}
              for row in atoms[["atom_id", "source_id", "values_json"]].itertuples(index=False)}
    groups = []
    for row in selected.itertuples(index=False):
        bucket = str(row.source_semantic_bucket_id)
        values = sorted({str(json.loads(lookup[atom]["values_json"]).get(
            "canonical_assay_context") or "__unknown__") for atom in members[bucket]})
        scope = [task, str(row.level), str(row.source_id), str(row.canonical_endpoint)]
        groups.append({"task": task, "level": scope[1], "source_id": scope[2],
                       "canonical_endpoint": scope[3], "parent_id": json.dumps(scope),
                       "group_id": bucket, "label": values[0], "values": values})
    schedule, covered = [], set()
    for batch in batch_groups(groups, batch_size=BATCH_SIZE):
        if len(batch) < 2:
            continue
        first = batch[0]
        if any(row["parent_id"] != first["parent_id"] for row in batch):
            raise ValueError("small-bucket batch crossed its endpoint-local scope")
        items = [row["group_id"] for row in batch]
        covered.update(items)
        scope = {key: first[key] for key in ("level", "source_id", "canonical_endpoint")}
        scope["provider_family"] = "luna"
        schedule.append({"request_id": ledger._id(VERSION, task, scope, items),
                         "task": task, "pass_number": 5, "scope": scope,
                         "items": items, "prompt": _render(batch, lookup, members, samples)})
    return schedule, {"eligible_buckets": len(selected), "scheduled_buckets": len(covered),
                      "isolated_buckets": len(selected) - len(covered),
                      "source_buckets": int(summary.source_semantic_bucket_id.nunique())}


def _verify(output: Path) -> tuple[dict[str, Any], Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION or manifest.get("model") != luna.LUNA_MODEL or
            manifest.get("service_tier") != "flex" or
            manifest.get("reasoning_effort") != "high" or
            manifest.get("max_tokens") != luna.MAX_TOKENS or
            manifest.get("max_inflight") != luna.PER_PROVIDER_INFLIGHT or
            manifest.get("batch_size") != BATCH_SIZE or
            manifest.get("max_records") != MAX_RECORDS):
        raise ValueError("small-bucket request configuration changed")
    if any(ledger._hash(path) != manifest["input_hashes"][key]
           for key, path in _paths().items()):
        raise ValueError("small-bucket pinned input changed")
    connection = ledger._database(output / "requests.sqlite3")
    schedule = [{"request_id": row["request_id"], "task": row["task"],
                 "pass_number": row["pass_number"], "scope": json.loads(row["scope_json"]),
                 "items": json.loads(row["items_json"]), "prompt": row["prompt"]}
                for row in connection.execute("SELECT * FROM requests ORDER BY request_id")]
    if (len(schedule) != manifest["requests"] or
            luna._schedule_hash(schedule) != manifest["schedule_sha256"] or
            any(row["pass_number"] != 5 or row["scope"]["provider_family"] != "luna"
                or not 1 < len(row["items"]) <= BATCH_SIZE for row in schedule)):
        raise ValueError("small-bucket frozen schedule changed")
    return manifest, connection


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    schedule, counts = [], {}
    for task in TASKS:
        rows, counts[task] = _task_schedule(task)
        schedule.extend(rows)
    if not schedule:
        raise ValueError("no eligible endpoint-local small-bucket comparisons")
    output.mkdir(parents=True)
    connection = ledger._database(output / "requests.sqlite3")
    ledger._insert(connection, schedule)
    connection.close()
    manifest = {"version": VERSION, "state": "prepared", "batch_size": BATCH_SIZE,
                "max_records": MAX_RECORDS, "representative_records": small.SAMPLE_SIZE,
                "scope": ["task", "level", "source_id", "canonical_endpoint_concept"],
                "model": luna.LUNA_MODEL, "service_tier": "flex",
                "reasoning_effort": "high", "max_tokens": luna.MAX_TOKENS,
                "max_inflight": luna.PER_PROVIDER_INFLIGHT,
                "input_hashes": {key: ledger._hash(path) for key, path in _paths().items()},
                "schedule_sha256": luna._schedule_hash(sorted(
                    schedule, key=lambda row: row["request_id"])),
                "requests": len(schedule), "selection_counts": counts,
                "max_prompt_chars": max(len(row["prompt"]) for row in schedule)}
    write_json_atomic(output / "manifest.json", manifest)
    _verify(output)[1].close()
    return manifest


def run(output: Path) -> None:
    manifest, connection = _verify(output)
    if manifest["state"] not in {"prepared", "running", "incomplete"}:
        raise ValueError("small-bucket pass is not resumable from this state")
    pending = [dict(row) for row in connection.execute(
        "SELECT * FROM requests WHERE status!='complete' ORDER BY request_id")]
    clients, receipts = luna._clients({"luna"})
    manifest.update(state="running", endpoints=receipts,
                    aggregate_capacity=luna.PER_PROVIDER_INFLIGHT)
    write_json_atomic(output / "manifest.json", manifest)
    with ThreadPoolExecutor(max_workers=min(luna.PER_PROVIDER_INFLIGHT, len(pending) or 1)) as pool:
        futures = {pool.submit(luna._call, clients["luna"], row): row["request_id"]
                   for row in pending}
        for finished, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            connection.execute("""UPDATE requests SET status=?,attempts=?,response_json=?,reasoning=?,
                receipts_json=?,error=? WHERE request_id=?""",
                (*[result[key] for key in ("status", "attempts", "response_json", "reasoning",
                                            "receipts_json", "error")], futures[future]))
            connection.commit()
            if finished % 25 == 0 or finished == len(pending):
                print(f"Small-bucket pass saved {finished}/{len(pending)} requests", flush=True)
    clients["luna"].close()
    complete = connection.execute("SELECT count(*) FROM requests WHERE status='complete'").fetchone()[0]
    manifest.update(state="complete" if complete == manifest["requests"] else "incomplete",
                    completed_requests=complete)
    write_json_atomic(output / "manifest.json", manifest)
    if complete != manifest["requests"]:
        raise RuntimeError(f"small-bucket pass has {manifest['requests'] - complete} failed requests")


def finalize(output: Path) -> None:
    manifest, connection = _verify(output)
    if manifest["state"] != "complete":
        raise ValueError("small-bucket pass must complete before materialization")
    replacements, decisions = {task: {} for task in TASKS}, {task: 0 for task in TASKS}
    for row in ledger._complete_rows(connection, 5, {
            item[0] for item in connection.execute("SELECT request_id FROM requests")}):
        items, scope = json.loads(row["items_json"]), json.loads(row["scope_json"])
        for decision in json.loads(row["response_json"])["merge_sets"]:
            members = sorted(items[index] for index in decision["member_ids"])
            task_replacements = replacements[row["task"]]
            if any(member in task_replacements for member in members):
                raise ValueError("small-bucket decision repeats a bucket")
            scientific_scope = {key: scope[key] for key in ("level", "source_id", "canonical_endpoint")}
            child = "sbsm_" + ledger._id(VERSION, row["task"], scientific_scope, members)[:24]
            task_replacements.update({member: child for member in members})
            decisions[row["task"]] += 1
    counts = {}
    for task in TASKS:
        original = pd.read_parquet(SOURCE / task / "source_semantic_bucket_map.parquet")
        if set(replacements[task]) - set(original.source_semantic_bucket_id):
            raise ValueError("small-bucket decision names an absent bucket")
        updated = original.copy()
        updated["source_semantic_bucket_id"] = (
            original.source_semantic_bucket_id.map(replacements[task]).fillna(
                original.source_semantic_bucket_id))
        target = output / task / "source_semantic_bucket_map.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(target, index=False)
        counts[task] = {"input_buckets": int(original.source_semantic_bucket_id.nunique()),
                        "candidate_buckets": int(updated.source_semantic_bucket_id.nunique()),
                        "merged_groups": decisions[task], "source_map_sha256": ledger._hash(target)}
        counts[task].update(materialize_candidate(
            task, output, version=f"{VERSION}.candidate.v1"))
    manifest.update(state="candidate_unselected", counts=counts)
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
