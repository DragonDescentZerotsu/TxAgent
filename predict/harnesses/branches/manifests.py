"""Shared scientific batch records; query state belongs in checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import uuid

from predict.utils.json import write_json_atomic
from data.processing.paths import resolve_legacy_path

ROOT = Path(__file__).resolve().parents[3]
PREDICTION_ROOT = ROOT / "outputs/predictions"
SCHEMA = "prediction_batch.v2"
FINAL_SCHEMA = "evidence_final_arm.v3"
SCHEMAS = {SCHEMA, FINAL_SCHEMA, "prediction_selection.v2"}
RUNTIME = {
    "paths", "indices", "n_items", "n_queries", "query_index", "run_id", "run_name",
    "batch_id", "schema_version", "status", "started_at", "finished_at",
    "parallelism", "requested_parallelism", "group_workers", "scheduler",
    "save_trace", "combine_traces", "stream_logs", "execution_mode",
    "transport_max_retries", "max_inflight", "base_url", "tool_service_url",
    "api_key_env", "provider_pool_config", "rerank_cache_preflight", "rerank_preflight_reused",
    "stage_pool", "expected_group_ids", "n_groups_with_neighbors",
    "query_label_for_eval_only", "sources", "source_batches", "files",
    "source_analysis", "source_kind_counts", "prompt_assets", "metrics",
    "selector_code", "cache_reader_code", "cache_reader_codes", "inputs",
    "artifacts", "queries", "query_overrides", "run_uid",
    "benchmark_row_id", "canonical_split_index", "query_uid", "input", "reused", "fresh",
    "source_run_uids", "checkpoint_defaults", "selection_manifest", "fingerprint", "fingerprints", "content_id", "cache_content_id",
}
PATH_FIELDS = {
    "input_jsonl", "evidence_library", "level_mapper", "assay_transfer_cache",
    "gold_context_mapping", "flat_selection_manifest", "selection_manifest", "flat_preselected_uids",
    "flat_preselected_contexts", "flat_mixed_selection", "artifact_manifest",
    "artifact_receipt", "neighbor_index", "index", "chembl_sqlite",
    "rerank_catalog", "rerank_cache", "rerank_candidate_manifest",
    "rerank_cache_version_manifest", "tier1_replacement_index",
}


def canonical_location(value: str | Path) -> Path:
    """Interpret old locators without creating a filesystem or import alias."""
    value = str(value)
    value = value.replace(str(ROOT / "semantic_buckets"), str(ROOT / "optimization/semantic_buckets"))
    if value.startswith("semantic_buckets/"):
        value = "optimization/" + value
    path = Path(value)
    return resolve_legacy_path(path if path.is_absolute() else ROOT / path)


def prediction_run_root(output_root: Path, harness: str, prompt: str, run_name: str) -> Path:
    for value in (harness, prompt, run_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value or "") or len(value.encode()) > 255:
            raise ValueError(f"Expected a readable single-component run name or identity, got {value!r}")
    return canonical_location(output_root) / harness / prompt / run_name


def prediction_batch_directory(run_root: Path, benchmark: str, task: str, subset: str,
                               profile: str = "default") -> Path:
    parts = ("gold_v1" if benchmark == "gold" else benchmark, task, subset, profile or "default")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts):
        raise ValueError(f"Invalid prediction output identity: {parts}")
    return run_root.joinpath(*parts)


def query_prior_batch(root: Path, task: str, subset: str, benchmark: str) -> Path:
    root = canonical_location(root)
    candidates = (root, root / task / f"{task}__none",
                  root / subset / task / f"{task}__none",
                  prediction_batch_directory(root, benchmark, task, subset, "query-prior"))
    for candidate in candidates:
        candidate = canonical_location(candidate)
        path = candidate / "manifest.json"
        if path.is_file():
            header = json.loads(path.read_text())
            settings = header.get("settings") or header
            if settings.get("experiment_mode") == "none" or header.get("schema_version") == "branch_query_prior_overlay.v1":
                return candidate
    return canonical_location(root / task / f"{task}__none")


def artifact_name(value: str | Path) -> dict:
    path = canonical_location(value)
    if path.suffix == ".json" and path.is_file():
        try:
            document = json.loads(path.read_text())
        except ValueError:
            document = {}
        if isinstance(document, dict) and document.get("schema_version") == "retrieval_cache_set.v1":
            return {"kind": "retrieval", "format": "cache_set", "name": document["name"], "version": document["release"]}
    recorded = Path(value)
    if "TDC" in recorded.parts and {"v1", "v2"}.intersection(recorded.parts):
        path = recorded
    parts = path.parts
    result = {"name": path.stem, "version": ""}
    for owner, kind in (("semantic_buckets", "semantic"),
                        ("evidence_libraries", "evidence"), ("gold_labels", "benchmark")):
        if owner not in parts:
            continue
        tail = parts[parts.index(owner) + 1:]
        if tail and tail[0] == "releases":
            tail = tail[1:]
        if kind == "benchmark" and tail and tail[0] == "TDC":
            result["benchmark"] = "TDC"
            tail = tail[1:]
        result["kind"] = kind
        if kind == "benchmark" and result.get("benchmark") == "TDC" and len(tail) == 2 and tail[1].endswith(".jsonl"):
            result.update(task=tail[0], cohort="tdc", partition="")
            result.pop("version", None)
            return result
        if kind == "semantic" and tail and tail[0] not in {"bbb_martins", "bioavailability_ma", "skin_reaction", "ames", "dili", "carcinogens"}:
            return result
        if len(tail) > 1:
            result.update(task=tail[0], version="" if tail[1] == "CURRENT" else tail[1])
        if kind == "benchmark":
            result["partition"] = next((part for part in tail if part in {"original", "scaffold"}), "")
        for section in ("generations", "weighting", "eligibility"):
            if section in tail and tail.index(section) + 1 < len(tail):
                result.update(operation=section, generation=tail[tail.index(section) + 1])
        return result
    if "assay_reranking" in parts and "active" in parts:
        tail = parts[parts.index("active") + 1:]
        result["kind"] = "retrieval"
        if len(tail) >= 4 and tail[:2] == ("flat_v5", "tdc"):
            result.update(benchmark="tdc", task=tail[2], role=tail[3])
            if len(tail) >= 7 and tail[4] not in {"scaffold", "original", "evidence"}:
                result.update(method=tail[4], version=tail[5], variant=tail[6])
            else:
                result.pop("version", None)
        elif len(tail) >= 7 and tail[0] == "flat_v5":
            result.update(benchmark=tail[1], task=tail[2], role=tail[3], method=tail[4],
                          version=tail[5], variant=tail[6])
        elif len(tail) >= 2:
            result.update(version=tail[0], task=tail[1])
        if any(part in {"original", "scaffold"} for part in tail):
            result["partition"] = next(part for part in tail if part in {"original", "scaffold"})
        for part in tail:
            if part in {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}:
                result["level"] = part
    return result


def scientific_settings(value, key=""):
    if (key in RUNTIME or key in PATH_FIELDS or "sha256" in key.lower()
            or "hash" in key.lower() or key.endswith(("_sha", "_digest", "_path", "_dir", "_file", "_batch", "_root"))):
        return None
    if isinstance(value, dict):
        return {k: clean for k, v in value.items()
                if (clean := scientific_settings(v, k)) is not None}
    if isinstance(value, list):
        return [clean for item in value if (clean := scientific_settings(item)) is not None]
    if isinstance(value, str) and (value.startswith(("/", "./", "../", "data/", "outputs/",
                                                   "predict/", "optimization/", "archive/", "semantic_buckets/"))
                                    or value.endswith((".jsonl", ".json", ".tsv", ".csv", ".sqlite", ".yaml", ".parquet"))):
        return None
    return value


def named_artifacts(document: dict) -> dict:
    artifacts = dict(document.get("artifacts") or {})
    for key, value in document.items():
        if key in PATH_FIELDS and value and key not in artifacts:
            location = value.get("path") if isinstance(value, dict) else value
            if isinstance(location, (str, Path)) and str(location) not in {"/dev/null", "null"}:
                artifacts[key] = artifact_name(location)
    inputs = (document.get("selection_audit") or {}).get("inputs") or {}
    for role, fragment in (("semantics", "/semantic_buckets/"),
                           ("retrieval", "/assay_reranking/active/")):
        items = [artifact_name(path) for path in inputs if isinstance(path, str) and fragment in path]
        if items:
            artifacts[role] = list({json.dumps(item, sort_keys=True): item for item in items}.values())
    return artifacts


def selection_parameters(record: dict) -> dict:
    keys = {"objective", "optimizer", "selection_contract", "selection_policy", "semantic_policy",
            "score_policy", "semantic_relevance_formula", "direct_profile", "indirect_profile",
            "direct_budget", "indirect_budget", "joint_uid_budget", "budget", "budgets",
            "l1_molecules", "l1_records_per_molecule", "records_per_level", "record_limits_by_level",
            "max_level", "l1_min_contrast", "ranking_tie_seed", "manual_uids", "manual_uid_level"}
    keys.add("bbb_evidence_policy")
    result = scientific_settings({key: value for key, value in record.items() if key in keys})
    audit = record.get("selection_audit") or {}
    if audit:
        result.update(scientific_settings({key: audit[key] for key in
            ("contract", "selection_policy", "semantic_policy") if key in audit}))
    return result


def compact_record(document: dict, *, schema=SCHEMA, run_uid=None) -> dict:
    if document.get("schema_version") in SCHEMAS and "settings" in document:
        return document
    artifacts = named_artifacts(document)
    audit = document.get("selection_audit") or {}
    settings = scientific_settings(document)
    provider = document.get("provider_pool_config")
    if provider:
        location = provider.get("path") if isinstance(provider, dict) else provider
        if location:
            settings["provider_profile"] = Path(location).stem
    if audit:
        settings["selection_audit"] = scientific_settings({
            key: audit[key] for key in ("contract", "selection_policy", "semantic_policy") if key in audit})
    result = {"schema_version": schema, "run_uid": run_uid or str(uuid.uuid4()),
            "batch_id": document.get("batch_id", ""), "settings": settings,
            "artifacts": artifacts, "queries": {}, "query_overrides": {}}
    if document.get("run_name"):
        result["run_name"] = document["run_name"]
    if "status" in document:
        result["status"] = document["status"]
    if document.get("source_run_uids"):
        result["source_run_uids"] = document["source_run_uids"]
    return result


def resolve_artifact(record: dict, role: str, batch: Path) -> Path | None:
    entry = record.get("artifacts", {}).get(role) or {}
    if not isinstance(entry, dict):
        return None
    if entry.get("local_format"):
        return batch / "inputs" / (role + entry["local_format"])
    kind, task, version = entry.get("kind"), entry.get("task"), entry.get("version")
    if kind == "retrieval" and entry.get("format") == "cache_set" and version:
        return ROOT / "predict/retrieval/assay_reranking/flat_v5_manifests" / (entry["name"] + "_" + version) / "catalog.json"
    if kind == "benchmark" and entry.get("cohort") == "tdc":
        return ROOT / "data/gold_labels/TDC" / task / (entry["name"] + ".jsonl")
    if kind == "benchmark" and version:
        base = ROOT / "data/gold_labels"
        base = base / "TDC" if entry.get("benchmark") == "TDC" else base
        parent = base / task / version / entry.get("partition", "")
        return resolve_legacy_path(parent / (entry["name"] + ".jsonl"))
    if kind == "evidence" and version:
        parent = ROOT / "data/evidence_libraries" / task / version
        candidates = list(parent.glob(entry["name"] + ".*"))
        return candidates[0] if len(candidates) == 1 else None
    return None


def write_batch(path: Path, document: dict, *, records=None) -> dict:
    old = json.loads(path.read_text()) if path.exists() else {}
    result = compact_record(document, schema=old.get("schema_version", SCHEMA)
                            if old.get("schema_version") in SCHEMAS else SCHEMA,
                            run_uid=old.get("run_uid"))
    result["batch_id"] = result.get("batch_id") or path.parent.name
    if old.get("schema_version") in SCHEMAS:
        for key in ("queries", "query_overrides", "checkpoint_defaults"):
            if key in old:
                result[key] = old[key]
    if records is not None:
        result["queries"] = {str(index): row.get("benchmark_row_id", "") for index, row in records}
    if records is not None and old.get("schema_version") in SCHEMAS:
        if any(not uid for uid in old.get("queries", {}).values()):
            raise ValueError("Historical query identities are unknown; use a new run identity")
        if list(old.get("queries", {}).items()) != list(result["queries"].items()):
            raise ValueError("Batch query membership differs; use a new run identity")
        changed = [key for key, value in result["settings"].items()
                   if key in old["settings"] and old["settings"][key] != value]
        if changed:
            raise ValueError(f"Batch scientific settings differ ({changed}); use a new run identity")
        require_artifact_versions(old, {key: value.get("path") if isinstance(value, dict) else value
                                      for key, value in document.items() if key in PATH_FIELDS and value})
    input_path = document.get("input_jsonl")
    cache = document.get("assay_transfer_cache")
    cache_identity = result["artifacts"].get("assay_transfer_cache", {})
    if cache and cache_identity.get("format") == "cache_set":
        target = path.parent / "inputs/assay_transfer_cache.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(canonical_location(cache), target)
        cache_identity["local_format"] = ".json"
    if input_path and canonical_location(input_path).is_file():
        if not result["queries"]:
            rows = [json.loads(line) for line in canonical_location(input_path).read_text().splitlines() if line.strip()]
            indices = document.get("indices", range(len(rows)))
            result["queries"] = {str(i): rows[i].get("benchmark_row_id", "") for i in indices}
        target = path.parent / "inputs/input_jsonl.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(canonical_location(input_path), target)
        result["artifacts"]["input_jsonl"].update(local_format=".jsonl")
    write_json_atomic(path, result)
    return result


def read_record(path: Path) -> dict:
    path = canonical_location(path)
    if not path.exists() and path.parent.parent.name == "runs":
        return read_run(path.parent)
    document = json.loads(path.read_text())
    if document.get("schema_version") not in SCHEMAS:
        return document
    result = dict(document["settings"])
    if document.get("run_name"):
        result["run_name"] = document["run_name"]
    result.update(schema_version=document["schema_version"], run_uid=document["run_uid"],
                  batch_id=document.get("batch_id", ""), indices=[int(i) for i in document["queries"]],
                  n_items=len(document["queries"]), n_queries=len(document["queries"]),
                  artifacts=document.get("artifacts", {}), queries=document["queries"],
                  query_overrides=document.get("query_overrides", {}))
    if "status" in document:
        result["status"] = document["status"]
    result["checkpoint_defaults"] = document.get("checkpoint_defaults", {})
    if document.get("source_run_uids"):
        result["source_run_uids"] = document["source_run_uids"]
    for role in document.get("artifacts", {}):
        location = resolve_artifact(document, role, path.parent)
        if location is not None:
            result[role] = str(location)
    return result


def read_run(run_dir: Path, *, batch=None) -> dict:
    run_dir = canonical_location(run_dir)
    parent = run_dir.parent.parent
    if not batch and not (parent / "manifest.json").exists():
        return json.loads((run_dir / "manifest.json").read_text())
    record = batch if batch and ("queries" in batch or not (parent / "manifest.json").exists()) else read_record(parent / "manifest.json")
    index = int(re.search(r"(\d+)$", run_dir.name).group(1))
    result = dict(record)
    result.update(record.get("checkpoint_defaults", {}))
    legacy = run_dir / "manifest.json"
    if record.get("schema_version") not in SCHEMAS and legacy.exists():
        result.update(json.loads(legacy.read_text()))
    override = record.get("query_overrides", {}).get(str(index), {})
    result.update(override)
    if "artifacts" in override:
        result["artifacts"] = dict(override["artifacts"])
        for role, entry in result["artifacts"].items():
            inherited = record.get("artifacts", {}).get(role) or {}
            # A query's unchanged input identity keeps the batch's local snapshot.
            if (isinstance(inherited, dict) and inherited.get("local_format")
                    and {k: v for k, v in inherited.items() if k != "local_format"} == entry):
                result["artifacts"][role] = dict(entry, local_format=inherited["local_format"])
    for role in result.get("artifacts", {}):
        location = resolve_artifact(result, role, parent)
        if location is not None:
            result[role] = str(location)
    checkpoint = run_dir / "stage_resume.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if "stage_pool" in state:
            state["stage_pool"] = dict(result.get("stage_pool") or {}, **state["stage_pool"])
        result.update(state)
    result.update(query_index=index, benchmark_row_id=record.get("queries", {}).get(str(index), result.get("benchmark_row_id", "")))
    if "flat_one_call" not in result:
        result["flat_one_call"] = result.get("experiment_mode") == "full_flat" and bool(result.get("flat_layout"))
    result.setdefault("query_prior_only", result.get("experiment_mode") == "none")
    result.setdefault("n_groups_with_neighbors", 0)
    result.setdefault("expected_group_ids", [])
    return result


def require_query_ids(record: dict, records: list[dict], indices: list[int]) -> None:
    declared = record.get("queries") or {}
    for index in indices:
        uid = declared.get(str(index))
        if not uid or uid != records[index].get("benchmark_row_id"):
            raise ValueError(f"Missing or incompatible query identity at index {index}")


def require_artifact_versions(record: dict, inputs: dict) -> None:
    for role, location in inputs.items():
        if not location:
            continue
        observed = record.get("artifacts", {}).get(role) or {}
        expected = artifact_name(location)
        for key in ("name", "kind", "task", "version", "cohort", "generation", "operation",
                    "partition", "benchmark", "role", "method", "variant", "level"):
            if observed.get(key) != expected.get(key):
                raise ValueError(f"Incompatible {role} {key}: {observed.get(key)!r} != {expected.get(key)!r}")


def run_directory(batch: Path, index: int) -> Path:
    batch = canonical_location(batch)
    record = read_record(batch / "manifest.json") if (batch / "manifest.json").exists() else {}
    label = record.get("batch_id") or batch.name
    raw = batch / "runs" / f"{label}_idx{index:05d}"
    return raw if raw.exists() else batch / "runs" / f"{index:05d}"


def prior_sources(batch: Path) -> list[dict]:
    record = read_record(batch / "manifest.json")
    if record.get("schema_version") == "branch_query_prior_overlay.v1":
        return record.get("sources") or []
    input_path = record.get("input_jsonl")
    if not input_path:
        raise ValueError(f"Query-prior identities unavailable: {batch}")
    rows = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    require_query_ids(record, rows, record["indices"])
    return [dict(rows[index], target_index=index, run_dir=str(run_directory(batch, index)))
            for index in record["indices"]]


def write_prior_batch(batch: Path, document: dict):
    from predict.harnesses.branches.migrate_manifests import materialize_overlay
    materialize_overlay(batch, document)
    old = read_record(batch / "manifest.json") if (batch / "manifest.json").exists() else {}
    document = dict(old, **document)
    document.update(experiment_mode="none", batch_id=old.get("batch_id") or document.get("batch_id") or batch.name)
    source_ids = [entry.get("source_run_uid") for entry in document.get("sources") or []]
    document["source_run_uids"] = list(dict.fromkeys(uid for uid in source_ids if uid))
    return write_batch(batch / "manifest.json", document)
