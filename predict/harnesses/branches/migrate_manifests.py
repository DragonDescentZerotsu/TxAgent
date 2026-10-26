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
import subprocess
import uuid

from predict.harnesses.branches.manifests import (
    ROOT, SCHEMA, FINAL_SCHEMA, SCHEMAS, canonical_location, compact_record,
    read_record, selection_parameters, named_artifacts, scientific_settings, write_batch,
    PREDICTION_ROOT, prediction_run_root, prediction_batch_directory,
)
from predict.utils.json import write_json_atomic, read_json, sha256_file, atomic_output_path


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
                if ((path / "inputs/assay_transfer_cache.json").is_file()
                        or document.get("schema_version") == "branch_query_prior_overlay.v1"
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
    previous = read_record(batch / "manifest.json") if (batch / "manifest.json").exists() else {}
    label = document.get("batch_id") or previous.get("batch_id") or batch.name
    for entry in document.get("sources") or []:
        index = int(entry["target_index"])
        source = canonical_location(entry["run_dir"])
        target = batch / "runs" / f"{label}_idx{index:05d}"
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
            or original.get("schema_version", "").startswith("evidence_final_arm")
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
    final = ROOT / "outputs/paper/assay_transfer_harness/evidence/final"
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

def migrate_cache_catalog(batch: Path, catalog: Path, *, apply=False) -> dict:
    """Migrate prepared cache bindings only after proving identical frozen inputs."""
    from predict.retrieval.assay_reranking.artifact_bundle import cache_task
    from predict.harnesses.branches.manifests import resolve_artifact, require_query_ids

    path, snapshot = batch / "manifest.json", batch / "inputs/assay_transfer_cache.json"
    result = {"batch": str(batch), "status": "unrelated"}
    if not snapshot.is_file():
        return result
    record = json.loads(path.read_text())
    parent_path = batch.parent / "manifest.json"
    parent = json.loads(parent_path.read_text()) if parent_path.is_file() else {}
    if (record.get("status") in {"complete", "completed", "done"}
            or parent.get("status") in {"complete", "completed", "done"}
            or any((root / name).exists() for root in (batch, batch.parent)
                   for name in ("metrics.json", "completion.json", "predictions.jsonl"))):
        return dict(result, status="completed_preserved")
    if not record.get("run_uid") or not record.get("queries") or not all(record["queries"].values()):
        raise ValueError("Historical query identities are unknown; cannot migrate cache bindings")
    effective = read_record(path)
    source = resolve_artifact(record, "input_jsonl", batch)
    if source is None or not source.is_file():
        raise ValueError("Frozen query input is unavailable; cannot migrate cache bindings")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    require_query_ids(effective, rows, effective["indices"])
    query_hash = sha256_file(source)
    history = batch / "inputs/historical_tdc"
    old = read_json(history / "catalog.json" if (history / "catalog.json").exists() else snapshot)
    current = read_json(catalog)
    if (old.get("schema_version"), old.get("name"), old.get("release")) != (
            current.get("schema_version"), current.get("name"), current.get("release")):
        raise ValueError("Cache set identity differs; use a new run identity")
    task, subset = effective["task"], effective["evaluation_subset"]
    benchmark = effective.get("benchmark", "gold")
    selected = cache_task(catalog, task, benchmark)
    candidates = [tasks[task] for scope, tasks in old["benchmarks"].items()
                  if task in tasks and (scope == selected["benchmark"] or benchmark == "tdc" and scope.startswith("tdc_"))]
    if len(candidates) != 1:
        raise ValueError("Frozen cache benchmark binding is unknown or ambiguous")
    for role in ("L1", "later"):
        indexes = [canonical_location(spec["retrieval"][role]) for spec in (candidates[0], selected)]
        headers = [read_json(index) for index in indexes]
        if any(header.get("task_id") != task or header.get("status") != "complete" for header in headers):
            raise ValueError("Cache task or completion differs; cannot migrate cache bindings")
        coverage = [header["splits"][subset]["levels"] for header in headers]
        if coverage[0].keys() != coverage[1].keys():
            raise ValueError("Cache level coverage differs; use a new run identity")
        for level in coverage[0]:
            paths = [index.parent / levels[level]["manifest"] for index, levels in zip(indexes, coverage)]
            if any(sha256_file(path) != levels[level].get("manifest_sha256") for path, levels in zip(paths, coverage)):
                raise ValueError("Cache manifest pin differs; cannot migrate cache bindings")
            manifests = [read_json(path) for path in paths]
            identities = [(m.get("content_id"), m.get("database_sha256"), m.get("inputs", {}).get("query_sha256")) for m in manifests]
            # L1 predecessors may pin compact queries or only their membership database.
            if (not all(identities[0][:2]) or identities[0] != identities[1]
                    or level != "L1" and identities[0][2] != query_hash):
                raise ValueError("Cache payload or query identity differs; use a new run identity")
    def canonical(value, key=""):
        if isinstance(value, dict):
            return {name: canonical(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [canonical(item, key) for item in value]
        return "tdc" if key in {"benchmark", "benchmark_release"} and value in ("tdc_v1", "tdc_v2") else value
    migrated = canonical(record)
    parent_matches = (parent.get("run_uid") and parent.get("queries") == record["queries"]
                      and parent.get("settings", {}).get("cache_set") == selected["cache_set"])
    if (migrated == record and snapshot.read_bytes() == catalog.read_bytes()
            and not (parent_matches and canonical(parent) != parent)):
        return dict(result, status="already_converted", run_uid=record["run_uid"])
    if apply:
        history.mkdir(parents=True, exist_ok=True)
        if not (history / "manifest.json").exists():
            shutil.copyfile(path, history / "manifest.json")
        if not (history / "catalog.json").exists():
            shutil.copyfile(snapshot, history / "catalog.json")
        write_json_atomic(path, migrated)
        with atomic_output_path(snapshot) as temporary:
            shutil.copyfile(catalog, temporary)
        if parent_matches:
            parent_history = batch.parent / "inputs/historical_tdc/manifest.json"
            parent_history.parent.mkdir(parents=True, exist_ok=True)
            if not parent_history.exists():
                shutil.copyfile(parent_path, parent_history)
            write_json_atomic(parent_path, canonical(parent))
    return dict(result, status="converted" if apply else "planned", run_uid=record["run_uid"])


def plan_output_relocations(roots: list[Path], output_root: Path, classifications: dict) -> dict:
    """Inventory batch containers and indexes without walking query payloads."""
    from data.processing.paths import task_id
    containers, indexes = {}, []
    for root in roots:
        root = root.resolve()
        for directory, children, files in os.walk(root, followlinks=False):
            path = Path(directory)
            children[:] = [name for name in children if not name.startswith(".") and name not in {
                "final", "git_bundle", "logs", "inputs", "queries", "requests", "responses", "traces", "samples", "__pycache__"}]
            if "runs" in children or "cache_matched_retrieval" in children:
                containers[path] = root
                children[:] = [name for name in children if name not in {"runs", "cache_matched_retrieval"}]
            for name in ("matrix.json", "execution.json"):
                if name in files:
                    indexes.append((path / name, root))
    headers = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        paths = [path / "manifest.json" if (path / "manifest.json").exists()
                 else path / "cache_matched_retrieval/manifest.json" for path in containers]
        for path, content in zip(paths, pool.map(lambda p: p.read_text() if p.is_file() else "{}", paths)):
            try:
                headers[path] = json.loads(content)
            except ValueError:
                headers[path] = {}
    index_headers = {}
    for path, root in indexes:
        try:
            index_headers[path] = json.loads(path.read_text())
        except ValueError:
            index_headers[path] = {}
    moves, manual = [], []
    for source, root in containers.items():
        header_path = source / "manifest.json"
        if header_path not in headers:
            header_path = source / "cache_matched_retrieval/manifest.json"
        header = headers[header_path]
        settings = header.get("settings") or header
        context = {}
        for parent in (source, *source.parents):
            if not parent.is_relative_to(root):
                break
            for name in ("matrix.json", "execution.json"):
                index_path = parent / name
                if index_path not in index_headers:
                    relocated = canonical_location(index_path)
                    index_headers[index_path] = json.loads(relocated.read_text()) if relocated.is_file() else {}
                if index_headers[index_path]:
                    context = index_headers[index_path]
                    break
            if context:
                break
        artifact = (header.get("artifacts") or {}).get("input_jsonl") or {}
        prior = settings.get("experiment_mode") == "none" or header.get("schema_version") == "branch_query_prior_overlay.v1"
        identity = {
            "harness": ("query-prior" if prior else settings.get("harness_version") or settings.get("organization")
                        or context.get("harness_version") or context.get("organization")),
            "prompt": settings.get("prompt_version") or settings.get("flat_prompt_version")
                      or context.get("prompt_version") or (settings.get("task_prompt_profile") if prior else None),
            "task": settings.get("task") or settings.get("task_id") or artifact.get("task"),
            "benchmark": settings.get("benchmark") or context.get("benchmark")
                         or ("tdc" if artifact.get("benchmark") == "TDC" else "gold_v1" if artifact.get("kind") == "benchmark" else None),
            "subset": settings.get("evaluation_subset") or settings.get("subset") or context.get("subset"),
            "run_name": source.relative_to(root).parts[0],
            "profile": "query-prior" if prior else header.get("batch_id") or source.name,
        }
        if identity["run_name"] in {"query_priors", "legacy", "_batches"} and len(source.relative_to(root).parts) > 1:
            identity["run_name"] = source.relative_to(root).parts[1]
        if not identity["subset"]:
            for split in ("valid_small", "valid", "test", "train", "fresh"):
                if artifact.get("name", "") in {split, split + "_molecule_condition_labels"}:
                    identity["subset"] = split
                    break
        identity.update(classifications.get(str(source.relative_to(ROOT)), classifications.get(str(source), {})))
        if identity["task"]:
            identity["task"] = task_id(identity["task"])
        missing = [key for key, value in identity.items() if not value]
        if missing:
            manual.append({"source": str(source), "missing": missing, "identity": identity})
            continue
        try:
            campaign = prediction_run_root(output_root, identity["harness"], identity["prompt"], identity["run_name"])
            target = prediction_batch_directory(campaign, identity["benchmark"], identity["task"], identity["subset"], identity["profile"])
        except ValueError as error:
            manual.append({"source": str(source), "reason": str(error), "identity": identity})
            continue
        stat = header_path.stat()
        legacy_run_id = next((context[key].get("run_id") for key in
                              ("preselected_direct_grid_manifest", "preselected_grid_manifest")
                              if isinstance(context.get(key), dict) and context[key].get("run_id")), None)
        if not legacy_run_id:
            for marker in ("with_query_prior", "no_query_prior"):
                if marker in source.parts and source.parts.index(marker) + 1 < len(source.parts) - 2:
                    legacy_run_id = source.parts[source.parts.index(marker) + 1]
                    break
        moves.append({"source": str(source), "destination": str(target), "kind": "batch", "identity": identity,
                      "legacy_run_id": legacy_run_id,
                      "run_uid": header.get("run_uid"), "header": str(header_path.relative_to(source)),
                      "header_stat": [stat.st_ino, stat.st_size, stat.st_mtime_ns], "status": "planned"})
    batch_sources = {Path(move["source"]) for move in moves}
    for source, root in indexes:
        if any(parent in batch_sources for parent in source.parents):
            continue
        header = index_headers[source]
        harness = header.get("harness_version") or header.get("organization")
        if source.name == "execution.json" and header.get("prompt_version"):
            harness = "query-prior"
        prompt = header.get("prompt_version")
        if not harness or not prompt:
            continue
        relative = source.relative_to(root)
        name = relative.parts[0]
        if name in {"query_priors", "_batches", "legacy"} and len(relative.parts) > 2:
            name = relative.parts[1]
        try:
            campaign = prediction_run_root(output_root, harness, prompt, name)
        except ValueError as error:
            manual.append({"source": str(source), "kind": "index", "reason": str(error)})
            continue
        target = campaign / "matrices" / relative
        moves.append({"source": str(source), "destination": str(target), "kind": "index", "status": "planned"})
    destinations = Counter(move["destination"] for move in moves)
    for move in moves:
        if destinations[move["destination"]] > 1 and move.get("legacy_run_id"):
            identity = move["identity"]
            identity["run_name"] += "__" + move["legacy_run_id"]
            campaign = prediction_run_root(output_root, identity["harness"], identity["prompt"], identity["run_name"])
            move["destination"] = str(prediction_batch_directory(campaign, identity["benchmark"], identity["task"], identity["subset"], identity["profile"]))
    destinations = Counter(move["destination"] for move in moves)
    for move in moves:
        if destinations[move["destination"]] > 1 or Path(move["destination"]).exists():
            move["status"] = "collision"
            manual.append({**move, "reason": "Destination collision; choose a distinct semantic run name"})
    unresolved_parents = {parent for row in manual for parent in Path(row["source"]).parents}
    for move in moves:
        if move["kind"] == "batch" and move["status"] == "planned" and Path(move["source"]) in unresolved_parents:
            move["status"] = "manual"
            manual.append({**move, "reason": "Contains an unclassified nested run; leave the parent in place"})
    return {"schema_version": "prediction_output_relocation.v1", "roots": [str(root.resolve()) for root in roots],
            "moves": sorted((move for move in moves if move["status"] == "planned"),
                            key=lambda move: len(Path(move["source"]).parts), reverse=True),
            "manual_classification": manual}


def apply_output_relocations(plan: dict, report: Path, registry: Path) -> None:
    """Rename frozen containers, preserving bytes and recording resumable locators."""
    if any(move["status"] == "collision" for move in plan["moves"]):
        raise ValueError(f"Output relocation has collisions; review {report}")
    processes = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], check=True,
                               capture_output=True, text=True).stdout.splitlines()
    try:
        jobs = subprocess.run(["squeue", "--me", "--json"], check=True, capture_output=True,
                              text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        jobs = ""
    mapping = json.loads(registry.read_text()) if registry.exists() else {}
    registered = False
    added = set()
    for ordinal, move in enumerate(plan["moves"]):
        source, target = Path(move["source"]), Path(move["destination"])
        if move["status"] == "moved":
            if not target.exists() or source.exists():
                raise ValueError(f"Relocation receipt disagrees with the filesystem: {source}")
            continue
        if not source.exists() and target.exists() and mapping.get(str(source.relative_to(ROOT))) == str(target.relative_to(ROOT)):
            move["status"] = "moved"
            continue
        if not source.exists() or target.exists():
            raise ValueError(f"Relocation source is missing or destination exists: {source} -> {target}")
        locations = [str(source)]
        for root in map(Path, plan.get("roots", [])):
            locations.extend(str(parent) for parent in source.parents if parent != root and parent.is_relative_to(root))
        active = any(any(location in line for location in locations) and "migrate_manifests" not in line for line in processes)
        if any(location in jobs for location in locations):
            active = True
        locks = []
        try:
            for parent in (source, *source.parents):
                if not parent.is_relative_to(ROOT / "outputs"):
                    break
                lock_path = parent / "launcher.lock"
                if lock_path.is_file():
                    handle = lock_path.open("a")
                    locks.append(handle)
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        active = True
            if active:
                move["status"] = "active_deferred"
                continue
            move["status"] = "planned"
            if move.get("header"):
                stat = (source / move["header"]).stat()
                if [stat.st_ino, stat.st_size, stat.st_mtime_ns] != move["header_stat"]:
                    raise ValueError(f"Batch metadata changed after inventory; regenerate the plan: {source}")
            source_stat = source.stat()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not registered:
                for candidate in plan["moves"]:
                    if candidate["status"] == "planned":
                        key = str(Path(candidate["source"]).relative_to(ROOT))
                        mapping[key] = str(Path(candidate["destination"]).relative_to(ROOT))
                        added.add(key)
                write_json_atomic(registry, mapping)
                registered = True
            source.rename(target)
            if (target.stat().st_dev, target.stat().st_ino) != (source_stat.st_dev, source_stat.st_ino):
                raise ValueError(f"Relocation did not preserve the original filesystem object: {target}")
            move["status"] = "moved"
        finally:
            for handle in locks:
                handle.close()
            if move["status"] != "moved" or ordinal % 25 == 0:
                write_json_atomic(report, plan)
    for key in added:
        if (ROOT / key).exists():
            mapping.pop(key, None)
    if registered:
        write_json_atomic(registry, mapping)
    write_json_atomic(report, plan)
    parents = {parent for move in plan["moves"] if move["status"] == "moved"
               for parent in Path(move["source"]).parents
               if any(parent != Path(root) and parent.is_relative_to(root) for root in plan.get("roots", []))}
    for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[ROOT / "outputs", ROOT / "archive"])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--containers", type=Path, help="Completed batch-container inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--relocate-runs", action="store_true")
    parser.add_argument("--output-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--classifications", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-index-refresh", action="store_true")
    parser.add_argument("--cache-catalog", type=Path, help="Migrate prepared bindings to a payload-identical named cache catalog")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.relocate_runs:
        if args.apply and args.classifications and args.report.exists():
            parser.error("Generate a new dry-run report with --classifications before applying it")
        roots = args.roots if args.roots != [ROOT / "outputs", ROOT / "archive"] else [ROOT / "outputs/paper/assay_transfer_harness/evidence"]
        classifications = json.loads(args.classifications.read_text()) if args.classifications else {}
        plan = (json.loads(args.report.read_text()) if args.apply and args.report.exists()
                else plan_output_relocations(roots, args.output_root, classifications))
        if plan.get("schema_version") != "prediction_output_relocation.v1":
            parser.error("Report is not a prediction-output relocation plan")
        write_json_atomic(args.report, plan)
        if args.apply:
            apply_output_relocations(plan, args.report, PREDICTION_ROOT / "relocations.json")
        print(json.dumps({"statuses": dict(Counter(move["status"] for move in plan["moves"])),
                          "manual_classification": len(plan["manual_classification"]), "report": str(args.report)}))
        return
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
            if args.cache_catalog:
                return migrate_cache_catalog(batch, args.cache_catalog, apply=args.apply)
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
    if args.cache_catalog and counts["error"]:
        raise SystemExit(1)
    if args.apply and not args.cache_catalog and not args.no_index_refresh and not counts["error"]:
        print("Updated operational indexes:", refresh_indexes(batches, args.inventory), flush=True)


if __name__ == "__main__":
    main()
