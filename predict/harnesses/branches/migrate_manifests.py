"""Stream the metadata-only conversion of saved prediction batches."""

from __future__ import annotations

import argparse
import fcntl
import csv
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from predict.harnesses.branches.manifests import (
    ROOT, SCHEMA, FINAL_SCHEMA, SCHEMAS, canonical_location, compact_record,
    read_record, selection_parameters, named_artifacts, scientific_settings, write_batch,
)
from predict.utils.json import write_json_atomic


def discover(roots):
    """Visit containers, never enumerate request/trace payload directories."""
    for root in roots:
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = [name for name in children if name not in {
                "git_bundle", "logs", "inputs", "queries", "requests", "responses", "traces", "live", "samples", "__pycache__"}]
            path = Path(directory)
            if "runs" in children:
                yield path
                children.remove("runs")
            if "manifest.json" in files:
                # Only these headers define prediction preparation/overlay records.
                document = json.loads((path / "manifest.json").read_text())
                if (document.get("schema_version") == "branch_query_prior_overlay.v1"
                        or "harness_version" in document and "input_jsonl" in document):
                    yield path


def _query(path):
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    match = re.search(r"(\d+)$", path.parent.name)
    index = document.get("query_index", document.get("canonical_split_index"))
    if index is None and match:
        index = int(match.group(1))
    return path, index, document, scientific_settings(document)


def materialize_overlay(batch: Path, document: dict):
    """Keep reuse inputs local; hardlinks avoid copying immutable payload bytes."""
    for entry in document.get("sources") or []:
        index = int(entry["target_index"])
        source = canonical_location(entry["run_dir"])
        target = batch / "runs" / f"{batch.name}_idx{index:05d}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("retrieval.json", "single_molecule_reasoning_output.json", "trace_messages.jsonl"):
            original, destination = source / name, target / name
            if original.is_file() and not destination.exists():
                try:
                    os.link(original, destination)
                except OSError:
                    shutil.copyfile(original, destination)


def convert_batch(batch: Path, *, apply=False, run_uid=None, source_uids=None) -> dict:
    path = batch / "manifest.json"
    original = json.loads(path.read_text()) if path.exists() else {}
    if original.get("schema_version") in SCHEMAS:
        remaining = list((batch / "runs").glob("*/manifest.json"))
        if apply:
            for query in remaining:
                query.unlink()
        return {"batch": str(batch), "status": "already_converted", "deleted_queries": 0}
    run_root = batch / "runs"
    query_paths = []
    if run_root.exists() and batch.name != "cache_matched_retrieval":
        query_paths = [Path(child.path) / "manifest.json" for child in os.scandir(run_root) if child.is_dir()]
    with ThreadPoolExecutor(max_workers=16) as pool:
        queries = [query for query in pool.map(_query, query_paths) if query is not None]
    is_overlay = original.get("schema_version") == "branch_query_prior_overlay.v1"
    if not (is_overlay or "batch_id" in original or "harness_version" in original
            or original.get("schema_version", "").startswith("joseph_final_arm")
            or any("query_index" in query[2] or "benchmark_row_id" in query[2] for query in queries)):
        return {"batch": str(batch), "status": "not_prediction", "deleted_queries": 0}
    schema = FINAL_SCHEMA if "/final/" in str(batch) else SCHEMA
    result = compact_record(original, schema=schema, run_uid=run_uid)
    result["batch_id"] = original.get("batch_id") or batch.name
    # Use the exact saved source headers, never current releases or recipe-name parsing.
    source_records = []
    for entry in original.get("source_batches") or []:
        if entry.get("path"):
            source_path = canonical_location(entry["path"]) / "manifest.json"
        elif entry.get("copied_manifest"):
            source_path = batch / entry["copied_manifest"]
        else:
            continue
        if source_path.is_file():
            source = read_record(source_path)
            source_dir = (source.get("paths") or {}).get("batch_dir")
            if source_dir and not source.get("run_uid"):
                source["run_uid"] = (source_uids or {}).get(str(canonical_location(source_dir)), "")
            source_records.append(source)
    if source_records:
        common = scientific_settings(source_records[0])
        for source in source_records[1:]:
            settings = scientific_settings(source)
            common = {key: value for key, value in common.items() if key in settings and settings[key] == value}
        result["settings"] = dict(common, **result["settings"])
        for source in source_records:
            for role, artifact in named_artifacts(source).items():
                result["artifacts"].setdefault(role, artifact)
    for role in ("selection_manifest", "flat_selection_manifest"):
        entry = original.get(role)
        location = entry.get("path") if isinstance(entry, dict) else entry
        if location and canonical_location(location).is_file():
            selected = read_record(canonical_location(location))
            parameters = selected.get("selection_parameters") or selection_parameters(selected)
            if parameters:
                result["settings"]["selection_parameters"] = dict(result["settings"].get("selection_parameters") or {}, **parameters)
            for key, artifact in named_artifacts(selected).items():
                result["artifacts"].setdefault(key, artifact)
    input_path = original.get("input_jsonl")
    rows = []
    if input_path and canonical_location(input_path).is_file():
        rows = [json.loads(line) for line in canonical_location(input_path).read_text().splitlines() if line.strip()]
    if not input_path:
        for _, _, query, _ in queries:
            if query.get("input_jsonl"):
                input_path = query["input_jsonl"]
                if canonical_location(input_path).is_file():
                    rows = [json.loads(line) for line in canonical_location(input_path).read_text().splitlines() if line.strip()]
                break
    # Lift values that actually were common to all queries; keep only genuine differences.
    if queries:
        common = dict(queries[0][3])
        for _, _, _, settings in queries[1:]:
            common = {key: value for key, value in common.items() if key in settings and settings[key] == value}
        result["settings"].update(common)
        for _, _, query, _ in queries:
            for role, artifact in named_artifacts(query).items():
                result["artifacts"].setdefault(role, artifact)
    changes = []
    for query_path, index, query, settings in queries:
        if index is None:
            raise ValueError(f"Query index cannot be recovered: {query_path}")
        uid = query.get("benchmark_row_id") or (rows[index].get("benchmark_row_id", "") if index < len(rows) else "")
        result["queries"][str(index)] = uid
        overrides = {key: value for key, value in settings.items() if result["settings"].get(key) != value}
        artifact_overrides = {role: artifact for role, artifact in named_artifacts(query).items()
                              if result["artifacts"].get(role) != artifact}
        if artifact_overrides:
            overrides["artifacts"] = dict(result["artifacts"], **artifact_overrides)
        if overrides:
            result["query_overrides"][str(index)] = overrides
        state = {key: query[key] for key in ("stage_pool", "n_groups_with_neighbors", "expected_group_ids") if key in query}
        if not rows and "query_label_for_eval_only" in query:
            state["query_label_for_eval_only"] = query["query_label_for_eval_only"]
        changes.append((query_path, state))
    if not queries:
        indices = original.get("indices", range(len(rows)))
        result["queries"] = {str(i): rows[i].get("benchmark_row_id", "") if i < len(rows) else "" for i in indices}
    if is_overlay:
        result["queries"] = {str(entry["target_index"]): (entry.get("benchmark_row_id") or (rows[int(entry["target_index"])].get("benchmark_row_id", "")
                              if int(entry["target_index"]) < len(rows) else ""))
                             for entry in original.get("sources") or []}
        result["settings"]["experiment_mode"] = "none"
    order = list(original.get("indices") or [])
    order.extend(sorted((int(key) for key in result["queries"] if int(key) not in set(order))))
    if order:
        result["queries"] = {str(index): result["queries"].get(str(index)) or
            (rows[index].get("benchmark_row_id", "") if index < len(rows) else "") for index in order}
    source_ids = [source["run_uid"] for source in source_records if source.get("run_uid")]
    source_paths = [entry.get("path") for entry in original.get("source_batches") or []]
    source_paths += [str(canonical_location(entry["run_dir"]).parent.parent)
                     for entry in original.get("sources") or [] if entry.get("run_dir")]
    for source in dict.fromkeys(path for path in source_paths if path):
        source_manifest = canonical_location(source) / "manifest.json"
        if source_manifest.is_file():
            identity = json.loads(source_manifest.read_text()).get("run_uid", "")
            identity = identity or (source_uids or {}).get(str(source_manifest.parent), "")
            if identity:
                source_ids.append(identity)
    if source_ids:
        result["source_run_uids"] = list(dict.fromkeys(source_ids))
    missing = sum(not uid for uid in result["queries"].values())
    summary = {"batch": str(batch), "status": "converted" if apply else "planned",
               "deleted_queries": len(changes), "missing_query_uids": missing,
               "run_uid": result["run_uid"]}
    if not apply:
        return summary
    if is_overlay:
        materialize_overlay(batch, original)
    if input_path:
        # This preserves the original input; no current release is substituted.
        result["artifacts"].setdefault("input_jsonl", compact_record({"input_jsonl": input_path})["artifacts"]["input_jsonl"])
        if canonical_location(input_path).is_file():
            local = batch / "inputs/input_jsonl.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                try:
                    os.link(canonical_location(input_path), local)
                except OSError:
                    shutil.copyfile(canonical_location(input_path), local)
            result["artifacts"]["input_jsonl"]["local_format"] = ".jsonl"
    # Keep repeated checkpoint configuration once; real query events stay in checkpoints.
    defaults = {"n_groups_with_neighbors": 0, "expected_group_ids": []}
    pools = [state.get("stage_pool") or {} for _, state in changes]
    if pools:
        common_pool = {key: value for key, value in pools[0].items() if key != "events"
                       and all(pool.get(key) == value for pool in pools)}
        if common_pool:
            defaults["stage_pool"] = common_pool
    result["checkpoint_defaults"] = defaults
    sparse_changes = []
    for query_path, state in changes:
        sparse = {key: value for key, value in state.items() if defaults.get(key) != value}
        if "stage_pool" in sparse:
            pool = {key: value for key, value in sparse["stage_pool"].items()
                    if (key != "events" or value) and defaults.get("stage_pool", {}).get(key) != value}
            if pool:
                sparse["stage_pool"] = pool
            else:
                sparse.pop("stage_pool")
        sparse_changes.append((query_path, sparse))
    # Checkpoint writes precede the shared record and query removal; interrupted runs can resume.
    def checkpoint(change):
        query_path, state = change
        if not state:
            return
        checkpoint_path = query_path.with_name("stage_resume.json")
        lock_path = query_path.parent / ".run.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            existing = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
            merged = dict(state, **existing)
            if state and merged != existing:
                write_json_atomic(checkpoint_path, merged)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(checkpoint, sparse_changes))
    summary["query_checkpoints_needed"] = sum(bool(state) for _, state in sparse_changes)
    write_json_atomic(path, result)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda change: change[0].unlink(missing_ok=True), changes))
    return summary



def refresh_indexes(batches, inventory=None):
    """Replace metadata pins with run IDs; retain score and payload audit columns."""
    identities = {}
    index_paths = set()
    for batch in batches:
        path = batch / "manifest.json"
        if path.is_file():
            document = json.loads(path.read_text())
            if document.get("schema_version") in SCHEMAS:
                identities[os.path.abspath(batch)] = document["run_uid"]
        for parent in (batch, *batch.parents):
            if parent == ROOT:
                break
            index_paths.add(parent / "matrix.json")
            index_paths.add(parent / "collection.json")
    if inventory:
        index_paths.update(canonical_location(name) for name in inventory.read_text().splitlines()
                           if Path(name).name in {"matrix.json", "collection.json"}
                           and "/git_bundle/" not in name)
    changed = 0
    def update(value, base):
        if isinstance(value, list):
            return [update(item, base) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: update(item, base) for key, item in value.items()
                  if key not in {"manifest_sha256", "retrieval_manifest_sha256", "selection_contract_sha256"}}
        locator = value.get("path")
        if isinstance(locator, str):
            candidates = [canonical_location(locator), base / locator]
            for location in candidates:
                container = location.parent if location.name == "manifest.json" else location
                identity = identities.get(os.path.abspath(container))
                if identity:
                    result.pop("sha256", None)
                    result["run_uid"] = identity
                    break
        return result
    with ThreadPoolExecutor(max_workers=32) as pool:
        existing_indexes = [path for path, exists in zip(sorted(index_paths), pool.map(Path.is_file, sorted(index_paths))) if exists]
    for path in existing_indexes:
        original = json.loads(path.read_text())
        result = update(original, path.parent)
        if result != original:
            write_json_atomic(path, result)
            changed += 1
    final = ROOT / "outputs/paper/assay_transfer_harness/joseph/final"
    tables = list(final.glob("*.tsv"))
    tables.extend(batch / "files.tsv" for batch in batches if batch.is_relative_to(final))
    tables.extend(batch / "query_provenance.tsv" for batch in batches if batch.is_relative_to(final))
    for path in tables:
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            rows = list(reader)
        kept = [field for field in fields if "manifest_sha256" not in field]
        result = []
        for row in rows:
            if path.name == "files.tsv" and (row.get("final_file", "").endswith("/manifest.json") or "/source_manifests/" in row.get("final_file", "")):
                continue
            row = {key: row.get(key, "") for key in kept}
            for field in list(kept):
                if field.endswith("_final_arm"):
                    uid_field = field.removesuffix("_final_arm") + "_run_uid"
                    row[uid_field] = identities.get(os.path.abspath(final / row[field]), "")
                    if uid_field not in kept:
                        kept.append(uid_field)
            result.append(row)
        output = io.StringIO()
        writer = csv.DictWriter(output, kept, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(result)
        if output.getvalue() != path.read_text():
            path.write_text(output.getvalue())
            changed += 1
    return changed

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[ROOT / "outputs", ROOT / "archive"])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--containers", type=Path, help="Completed batch-container inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-index-refresh", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.containers:
        batches = {canonical_location(name) for name in args.containers.read_text().splitlines() if name}
    elif args.inventory:
        batches = {canonical_location(name).parent.parent.parent for name in args.inventory.read_text().splitlines()
                   if name.endswith("/manifest.json") and Path(name).parent.parent.name == "runs"
                   and "/git_bundle/" not in name}
        batches.update(canonical_location(name).parent for name in args.inventory.read_text().splitlines()
                       if name.endswith("/manifest.json") and "cache_matched_retrieval" in name)
    else:
        batches = set()
    if not args.containers:
        batches.update(discover(args.roots))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    identities = {}
    if args.report.exists():
        for line in args.report.read_text().splitlines():
            row = json.loads(line)
            if row.get("run_uid"):
                identities[row["batch"]] = row["run_uid"]
            if row.get("status") in {"converted", "already_converted"} or (not args.apply and row.get("status") == "planned"):
                completed[row["batch"]] = row
    for batch in batches:
        identities.setdefault(str(batch), str(uuid.uuid4()))
    counts = Counter()
    pending = sorted(batch for batch in batches if str(batch) not in completed)
    def convert(batch):
        try:
            return convert_batch(batch, apply=args.apply, run_uid=identities[str(batch)], source_uids=identities)
        except (OSError, ValueError, TypeError, KeyError) as error:
            return {"batch": str(batch), "status": "error", "error": str(error)}
    with args.report.open("a") as report, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, row in enumerate(pool.map(convert, pending), 1):
            report.write(json.dumps(row) + "\n")
            report.flush()
            counts[row["status"]] += 1
            if number % 25 == 0:
                print(number, "/", len(pending), dict(counts), flush=True)
    print(dict(counts), flush=True)
    if args.apply and not args.no_index_refresh and not counts["error"]:
        print("Updated operational indexes:", refresh_indexes(batches, args.inventory), flush=True)


if __name__ == "__main__":
    main()
