"""Reconcile provisional semantic children in new 100-bucket TF-IDF neighborhoods."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

import pandas as pd

from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client
from semantic_buckets import source_local_overmerge_reconcile_v1 as previous
from semantic_buckets.materialize_overmerge_candidate import materialize as materialize_candidate
from semantic_buckets.overmerge_pass2_batches import batch_groups
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_overmerge_pass3.v1"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "semantic_buckets/provenance/source_local_overmerge_reconcile_v1/"
          "20260922_two_pass_40_t48000")
PROMPT = ROOT / "semantic_buckets/prompts/source_local_overmerge_pass3_v1/merge.md"
INVENTORY = ROOT / "predict/api_client/providers/current_endpoints.json"
LOCAL_NAME = "dgx027_50001"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
LUNA_MODEL = "openai/gpt-6-luna"
LUNA_URL = "https://openrouter.ai/api/v1"
PER_PROVIDER_INFLIGHT = 512
BATCH_SIZE = 100
MAX_TOKENS = 128_000


def _groups() -> list[dict[str, Any]]:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "candidate_unselected":
        raise ValueError("predecessor is not a complete unselected candidate")
    path = SOURCE / "final_groups.json"
    if previous._hash(path) != manifest["final_groups_sha256"]:
        raise ValueError("predecessor final groups changed")
    if previous._hash(SOURCE / "pass1_groups.json") != manifest["pass1_groups_sha256"]:
        raise ValueError("predecessor first-pass rationales changed")
    groups = json.loads(path.read_text(encoding="utf-8"))
    if len({group["group_id"] for group in groups}) != len(groups):
        raise ValueError("predecessor contains duplicate group IDs")
    return groups


def _render(batch: list[dict[str, Any]]) -> str:
    first = batch[0]
    lines = [PROMPT.read_text(encoding="utf-8").strip(), "",
             f"Task: {first['task']}", f"Level: {first['level']}",
             f"Source: {first['source_id']}", f"Original parent: {first['parent_id']}",
             "", "Candidate IDs and provisional buckets:"]
    for index, group in enumerate(batch):
        lines.append(json.dumps({"id": index, "label": group["label"],
                                 "canonical_values": group["values"]}, ensure_ascii=False))
    lines += ["", "Return only JSON:",
              '{"merge_sets":[{"member_ids":[0,1],"label":"shared scientific identity",'
              '"rationale":"why every underlying value is interchangeable"}]}',
              "The example gives schema only, not a suggested decision."]
    return "\n".join(lines)


def _schedule(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches = [batch for batch in batch_groups(groups, batch_size=BATCH_SIZE) if len(batch) > 1]
    requests = []
    for index, batch in enumerate(batches):
        first = batch[0]
        if any(group["task"] != first["task"] or group["parent_id"] != first["parent_id"]
               or group["level"] != first["level"] or group["source_id"] != first["source_id"]
               for group in batch):
            raise ValueError("TF-IDF batch crossed a scientific scope")
        family = "luna" if index % 2 == 0 else "deepseek"
        scope = {key: first[key] for key in ("parent_id", "level", "source_id")}
        scope["provider_family"] = family
        items = [group["group_id"] for group in batch]
        requests.append({"request_id": previous._id(VERSION, first["task"], scope, items),
                         "task": first["task"], "pass_number": 3, "scope": scope,
                         "items": items, "prompt": _render(batch)})
    return requests


def _schedule_hash(schedule: list[dict[str, Any]]) -> str:
    payload = json.dumps(schedule, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rerouted_schedule(source: Path, schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    connection = previous._database(source / "requests.sqlite3")
    statuses = {row["request_id"]: row["status"] for row in connection.execute(
        "SELECT request_id,status FROM requests")}
    connection.close()
    if set(statuses) != {row["request_id"] for row in schedule}:
        raise ValueError("reroute source ledger differs from original schedule")
    result = []
    for row in schedule:
        if statuses[row["request_id"]] == "complete":
            result.append(row)
            continue
        scope = {**row["scope"], "provider_family": "luna"}
        result.append({**row, "scope": scope,
                       "request_id": previous._id(VERSION, row["task"], scope, row["items"])})
    if len({row["request_id"] for row in result}) != len(result):
        raise ValueError("rerouted requests collide")
    return result


def _verify(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], sqlite3.Connection]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION or manifest.get("models") != {
            "luna": LUNA_MODEL, "deepseek": DEEPSEEK_MODEL} or
            manifest.get("reasoning_effort") != "high" or
            manifest.get("max_tokens") != MAX_TOKENS or
            manifest.get("per_provider_max_inflight") != PER_PROVIDER_INFLIGHT):
        raise ValueError("Pass-3 request configuration changed")
    for key, path in (("source_manifest_sha256", SOURCE / "manifest.json"),
                      ("source_groups_sha256", SOURCE / "final_groups.json"),
                      ("prompt_sha256", PROMPT)):
        if previous._hash(path) != manifest[key]:
            raise ValueError(f"pinned Pass-3 input changed: {path}")
    schedule = _schedule(_groups())
    if manifest.get("rerouted_from"):
        source = ROOT / manifest["rerouted_from"]
        for key, path in (("reroute_manifest_sha256", source / "manifest.json"),
                          ("reroute_ledger_sha256", source / "requests.sqlite3")):
            if previous._hash(path) != manifest[key]:
                raise ValueError(f"reroute source changed: {path}")
        schedule = _rerouted_schedule(source, schedule)
    if _schedule_hash(schedule) != manifest["schedule_sha256"]:
        raise ValueError("Pass-3 schedule changed")
    connection = previous._database(output / "requests.sqlite3")
    actual = {row["request_id"]: row for row in connection.execute("SELECT * FROM requests")}
    if set(actual) != {row["request_id"] for row in schedule}:
        raise ValueError("Pass-3 ledger differs from pinned schedule")
    for row in schedule:
        stored = actual[row["request_id"]]
        if (stored["prompt"] != row["prompt"] or
                json.loads(stored["items_json"]) != row["items"] or
                json.loads(stored["scope_json"]) != row["scope"]):
            raise ValueError("Pass-3 request bytes changed")
    return manifest, schedule, connection


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    groups = _groups()
    schedule = _schedule(groups)
    if not schedule:
        raise ValueError("no Pass-3 requests")
    output.mkdir(parents=True)
    connection = previous._database(output / "requests.sqlite3")
    previous._insert(connection, schedule)
    manifest = {"version": VERSION, "state": "prepared", "batch_size": BATCH_SIZE,
                "max_tokens": MAX_TOKENS, "reasoning_effort": "high",
                "per_provider_max_inflight": PER_PROVIDER_INFLIGHT,
                "models": {"luna": LUNA_MODEL, "deepseek": DEEPSEEK_MODEL},
                "source_manifest_sha256": previous._hash(SOURCE / "manifest.json"),
                "source_groups_sha256": previous._hash(SOURCE / "final_groups.json"),
                "prompt_sha256": previous._hash(PROMPT),
                "schedule_sha256": _schedule_hash(schedule),
                "requests": len(schedule),
                "request_families": {family: sum(row["scope"]["provider_family"] == family
                                               for row in schedule)
                                     for family in ("luna", "deepseek")},
                "max_prompt_chars": max(len(row["prompt"]) for row in schedule)}
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def prepare_reroute(output: Path, source: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    original, old_schedule, old_connection = _verify(source)
    if original["state"] != "cancelled":
        raise ValueError("reroute source must be stopped and marked cancelled")
    schedule = _rerouted_schedule(source, old_schedule)
    output.mkdir(parents=True)
    target = previous._database(output / "requests.sqlite3")
    old_connection.backup(target)
    for old, new in zip(old_schedule, schedule):
        if old["request_id"] != new["request_id"]:
            target.execute("""UPDATE requests SET request_id=?,scope_json=?,status='pending',
                attempts=0,response_json=NULL,reasoning=NULL,receipts_json='[]',error=NULL
                WHERE request_id=?""", (new["request_id"], json.dumps(new["scope"]), old["request_id"]))
    target.commit()
    old_connection.close()
    target.close()
    manifest = {key: value for key, value in original.items()
                if key not in {"endpoints", "inventory_sha256", "aggregate_capacity"}}
    manifest.update(state="prepared", rerouted_from=str(source.relative_to(ROOT)),
                    reroute_manifest_sha256=previous._hash(source / "manifest.json"),
                    reroute_ledger_sha256=previous._hash(source / "requests.sqlite3"),
                    schedule_sha256=_schedule_hash(schedule),
                    request_families={family: sum(row["scope"]["provider_family"] == family
                                                  for row in schedule)
                                      for family in ("luna", "deepseek")},
                    rerouted_requests=sum(old["request_id"] != new["request_id"]
                                           for old, new in zip(old_schedule, schedule)))
    write_json_atomic(output / "manifest.json", manifest)
    _verify(output)[2].close()
    return manifest


def _clients(families: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs = []
    if "deepseek" in families:
        inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
        matches = [row for row in inventory["providers"] if row["name"] == LOCAL_NAME]
        if len(matches) != 1 or matches[0]["model"] != DEEPSEEK_MODEL:
            raise ValueError("selected local endpoint is absent or advertises another model")
        local = matches[0]
        if not local["base_url"].rstrip("/").endswith("dgx027:50001/v1"):
            raise ValueError("selected DeepSeek endpoint is not dgx027:50001")
        specs.append(("deepseek", local["base_url"], "local", None, DEEPSEEK_MODEL))
    if "luna" in families:
        specs.append(("luna", LUNA_URL, "openrouter", "OPEN_ROUTER_KEY_TWO", LUNA_MODEL))
    clients, receipts = {}, []
    for family, url, provider, credential_env, model in specs:
        client, selected = openai_compatible_client(
            base_url=url, provider=provider, credential_env=credential_env,
            env_file=None if provider == "local" else DEFAULT_ENV_FILE,
            max_connections=PER_PROVIDER_INFLIGHT, timeout_s=3_600, max_retries=0)
        models = {row.id for row in client.models.list().data}
        if model not in models:
            client.close()
            raise ValueError(f"{family} endpoint does not advertise {model}")
        clients[family] = client
        receipts.append({"provider_family": family, "base_url": url, "model": model,
                         "credential_env": selected, "max_inflight": PER_PROVIDER_INFLIGHT})
    return clients, receipts


def _call(client: Any, row: dict[str, Any]) -> dict[str, Any]:
    family = json.loads(row["scope_json"])["provider_family"]
    model = LUNA_MODEL if family == "luna" else DEEPSEEK_MODEL
    attempts, receipts, error = int(row["attempts"]), json.loads(row["receipts_json"]), None
    for _ in range(2):
        attempts += 1
        started = time.time()
        try:
            kwargs = {"model": model, "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": row["prompt"]}],
                "reasoning_effort": "high"}
            if family == "luna":
                kwargs.update(max_completion_tokens=MAX_TOKENS,
                              extra_body={"service_tier": "flex"},
                              response_format={"type": "json_object"})
            else:
                kwargs.update(max_tokens=MAX_TOKENS, extra_body={
                    "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})
            completion = client.chat.completions.create(**kwargs)
            if completion.model != model:
                raise ValueError(f"served model {completion.model!r} differs from {model!r}")
            message = completion.choices[0].message
            parsed = previous._validate_response(message.content or "", len(json.loads(row["items_json"])))
            usage = completion.usage.model_dump() if completion.usage else {}
            receipts.append({"attempt": attempts, "elapsed_seconds": time.time() - started,
                             "generation_id": completion.id, "model": completion.model,
                             "provider_family": family, "usage": usage})
            reasoning = (getattr(message, "reasoning_content", None) or
                         getattr(message, "reasoning", None))
            if reasoning is not None and not isinstance(reasoning, str):
                reasoning = json.dumps(reasoning, ensure_ascii=False)
            return {"status": "complete", "attempts": attempts,
                    "response_json": json.dumps(parsed), "reasoning": reasoning,
                    "receipts_json": json.dumps(receipts), "error": None}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            receipts.append({"attempt": attempts, "elapsed_seconds": time.time() - started,
                             "provider_family": family, "error": error[:500]})
    return {"status": "failed", "attempts": attempts, "response_json": None,
            "reasoning": None, "receipts_json": json.dumps(receipts), "error": error}


def run(output: Path) -> None:
    manifest, schedule, connection = _verify(output)
    if manifest["state"] not in {"prepared", "running", "incomplete"}:
        raise ValueError("Pass 3 is not resumable from this state")
    pending = [dict(row) for row in connection.execute(
        "SELECT * FROM requests WHERE status!='complete' ORDER BY request_id")]
    families = {json.loads(row["scope_json"])["provider_family"] for row in pending}
    clients, receipts = _clients(families)
    manifest.update(state="running", endpoints=receipts,
                    aggregate_capacity=PER_PROVIDER_INFLIGHT * len(clients))
    if "deepseek" in families:
        manifest["inventory_sha256"] = previous._hash(INVENTORY)
    write_json_atomic(output / "manifest.json", manifest)
    limits = {family: threading.BoundedSemaphore(PER_PROVIDER_INFLIGHT)
              for family in clients}

    def work(row: dict[str, Any]) -> dict[str, Any]:
        family = json.loads(row["scope_json"])["provider_family"]
        with limits[family]:
            return _call(clients[family], row)

    with ThreadPoolExecutor(max_workers=min(manifest["aggregate_capacity"], len(pending) or 1)) as executor:
        futures = {executor.submit(work, row): row["request_id"] for row in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            connection.execute("""UPDATE requests SET status=?,attempts=?,response_json=?,reasoning=?,
                receipts_json=?,error=? WHERE request_id=?""",
                (*[result[key] for key in ("status", "attempts", "response_json", "reasoning",
                                           "receipts_json", "error")], futures[future]))
            if completed % 25 == 0:
                connection.commit()
                print(f"Pass 3 saved {completed}/{len(pending)} resumed requests", flush=True)
        connection.commit()
    for client in clients.values():
        client.close()
    complete = connection.execute("SELECT count(*) FROM requests WHERE status='complete'").fetchone()[0]
    manifest.update(state="complete" if complete == len(schedule) else "incomplete",
                    completed_requests=complete)
    write_json_atomic(output / "manifest.json", manifest)
    if complete != len(schedule):
        raise RuntimeError(f"Pass 3 has {len(schedule) - complete} failed requests; resume same ledger")


def _decisions(output: Path, groups: list[dict[str, Any]], connection: sqlite3.Connection,
               schedule: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_id = {group["group_id"]: group for group in groups}
    merges, replacements = [], {}
    for row in previous._complete_rows(connection, 3, {item["request_id"] for item in schedule}):
        items = json.loads(row["items_json"])
        family = json.loads(row["scope_json"])["provider_family"]
        for decision in json.loads(row["response_json"])["merge_sets"]:
            members = [items[index] for index in decision["member_ids"]]
            first = by_id[members[0]]
            if any(member in replacements or by_id[member]["parent_id"] != first["parent_id"]
                   or by_id[member]["task"] != first["task"] for member in members):
                raise ValueError("Pass-3 decision repeats or crosses a parent")
            values = sorted(value for member in members for value in by_id[member]["values"])
            if len(values) != len(set(values)):
                raise ValueError("Pass-3 merge repeats a canonical value")
            child = "sbo3_" + previous._id(first["task"], first["parent_id"], values)[:24]
            replacements.update({member: child for member in members})
            merges.append({"task": first["task"], "parent_id": first["parent_id"],
                           "level": first["level"], "source_id": first["source_id"],
                           "group_id": child, "predecessor_group_ids": members,
                           "values": values, "label": decision["label"],
                           "rationale": decision["rationale"], "request_id": row["request_id"],
                           "provider_family": family})
    return merges, replacements


def _successor_groups(groups: list[dict[str, Any]], merges: list[dict[str, Any]],
                      replacements: dict[str, str]) -> list[dict[str, Any]]:
    pass1 = {group["group_id"]: group for group in json.loads(
        (SOURCE / "pass1_groups.json").read_text(encoding="utf-8"))}
    result = []
    for group in groups:
        if group["group_id"] in replacements:
            continue
        row = dict(group)
        if len(row["values"]) > 1 and row["rationale"] == "Unmerged singleton":
            predecessors = row["pass1_group_ids"]
            if len(predecessors) != 1 or pass1[predecessors[0]]["values"] != row["values"]:
                raise ValueError("cannot recover first-pass merge rationale")
            row["rationale"] = pass1[predecessors[0]]["rationale"]
        result.append(row)
    result.extend(merges)
    old_values = {(row["task"], row["parent_id"], value)
                  for row in groups for value in row["values"]}
    new_values = [(row["task"], row["parent_id"], value)
                  for row in result for value in row["values"]]
    if (len(new_values) != len(set(new_values)) or set(new_values) != old_values or
            len({row["group_id"] for row in result}) != len(result)):
        raise ValueError("Pass-3 successor lost or duplicated a group/value")
    return result


def finalize(output: Path) -> None:
    manifest, schedule, connection = _verify(output)
    if manifest["state"] != "complete":
        raise ValueError("Pass 3 must complete before materialization")
    groups = _groups()
    merges, replacements = _decisions(output, groups, connection, schedule)
    write_json_atomic(output / "final_groups.json", _successor_groups(
        groups, merges, replacements))
    counts = {}
    for task in ("dili", "carcinogens"):
        old = pd.read_parquet(SOURCE / task / "source_semantic_bucket_map.parquet")
        updated = old.copy()
        updated["source_semantic_bucket_id"] = (
            old.source_semantic_bucket_id.map(replacements).fillna(old.source_semantic_bucket_id))
        if len(updated) != len(old) or updated.atom_id.tolist() != old.atom_id.tolist():
            raise ValueError(f"{task}: Pass 3 changed atom identity")
        target = output / task / "source_semantic_bucket_map.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        updated.to_parquet(target, index=False)
        counts[task] = {"old_buckets": int(old.source_semantic_bucket_id.nunique()),
                        "candidate_buckets": int(updated.source_semantic_bucket_id.nunique()),
                        "merged_groups": sum(group["task"] == task for group in merges),
                        "source_map_sha256": previous._hash(target)}
        counts[task].update(materialize_candidate(
            task, output, version=f"{VERSION}.candidate.v1"))
    manifest.update(state="candidate_unselected", final_groups_sha256=previous._hash(
        output / "final_groups.json"), counts=counts)
    write_json_atomic(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-reroute", "run", "finalize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(output), indent=2))
    elif args.action == "prepare-reroute":
        if args.source_run is None:
            parser.error("prepare-reroute requires --source-run")
        print(json.dumps(prepare_reroute(output, args.source_run.resolve()), indent=2))
    elif args.action == "run":
        run(output)
    else:
        finalize(output)


if __name__ == "__main__":
    main()
