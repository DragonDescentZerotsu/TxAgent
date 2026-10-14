"""Ordered physical evidence selections shared by optimizers and predictors."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from predict.retrieval.policies import decide_candidate, normalize_molecule_identity
from predict.utils.json import sha256_file, write_json_atomic

SCHEMA = "evidence_uid_selection.v1"
FIELDS = ("benchmark_row_id", "level", "selection_rank", "source_row_uid",
          "context_id", "context_rank", "within_context_rank")


def selection_query_ids(path: Path) -> list[str]:
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if document.get("uid_selection"):
            return selection_query_ids(path.parent / document["uid_selection"]["path"])
        ids = list(map(str, document["benchmark_row_ids"]))
    else:
        with path.open(newline="") as handle:
            ids = list(dict.fromkeys(row["benchmark_row_id"] for row in csv.DictReader(handle, delimiter="\t")))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("UID selection must declare unique query identities")
    return ids


def assert_disjoint(query: str, parent_smiles: str, policy: str) -> None:
    identity = normalize_molecule_identity(query)
    candidate = normalize_molecule_identity(parent_smiles)
    if identity.status != "ok" or candidate.status != "ok":
        raise ValueError("Cannot verify disjointness for an unresolved molecule")
    decision = decide_candidate(identity, {"canonical_smiles": parent_smiles}, policy)
    if decision.excluded:
        raise ValueError(f"Selected evidence violates disjointness: {decision.reason}")


def read_selection(path: Path, *, task: str, subset: str,
                   queries: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict]:
    """Read a manual TSV or a pinned optimizer manifest, without trusting payloads."""
    path = path.resolve()
    receipt = {"path": str(path), "sha256": sha256_file(path)}
    source = path
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if document.get("uid_selection"):
            if document.get("schema_version") == "context_selection.v1":
                from optimization.selection.search import validate_selection_manifest
                validate_selection_manifest(path)
            pin = document["uid_selection"]
            child = (path.parent / pin["path"]).resolve()
            if not child.is_relative_to(path.parent) or sha256_file(child) != pin["sha256"]:
                raise ValueError("Optimizer UID export differs from its manifest")
            rows, receipt = read_selection(child, task=task, subset=subset, queries=queries)
            receipt["optimizer_manifest"] = dict(path=str(path), sha256=sha256_file(path))
            return rows, receipt
        if (document.get("schema_version") != SCHEMA or document.get("status") != "complete"
                or document.get("task_id") != task or document.get("subset") != subset
                or set(document.get("benchmark_row_ids", [])) != set(queries)):
            raise ValueError("UID selection identity differs from the requested benchmark")
        source = (path.parent / document["records"]["path"]).resolve()
        if not source.is_relative_to(path.parent) or sha256_file(source) != document["records"]["sha256"]:
            raise ValueError("UID selection records differ from their manifest")
        for pin in document["cache_manifests"].values():
            if sha256_file(Path(pin["path"])) != pin["sha256"]:
                raise ValueError("UID selection cache identity changed")
        receipt["manifest"] = document
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"benchmark_row_id", "level", "source_row_uid"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("UID selection needs query, level and source_row_uid columns")
        rows = list(reader)
    if path.suffix == ".json" and len(rows) != document["records"]["row_count"]:
        raise ValueError("UID selection row count differs from its manifest")
    receipt.update(records_path=str(source), records_sha256=sha256_file(source))
    return rows, receipt


def resolve_selection(rows: Sequence[Mapping[str, Any]], *, task: str, subset: str,
                      queries: Mapping[str, str], cache_manifests: Mapping[str, Path],
                      molecule_limit: int = 10, l1_limit: int = 10,
                      limits: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """Bind selections to indexed candidates, frozen L1 members and exclusions."""
    from predict.retrieval.assay_reranking.ranked_uid_retrieval import _manifest, _open

    if not rows or {str(row["benchmark_row_id"]) for row in rows} != set(queries):
        raise ValueError("UID selections must supply evidence for every requested query")
    limits = limits or {level: 50 for level in cache_manifests if level != "L1"}
    output = []
    ranks = defaultdict(int)
    seen = set()
    contexts = defaultdict(list)
    for level in dict.fromkeys(str(row["level"]) for row in rows):
        if level not in cache_manifests:
            raise ValueError(f"UID selection names an unavailable level: {level}")
        path = Path(cache_manifests[level]).resolve()
        document = _manifest(path, task, subset, level)
        policy = document.get("neighbor_identity_policy", "parent_disjoint")
        if policy not in {"parent_disjoint", "scaffold_disjoint"}:
            raise ValueError("UID selection requires an explicit disjointness policy")
        with _open(path.with_name(document["database"])) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if metadata.get("content_id") != document["content_id"]:
                raise ValueError("UID selection cache and manifest differ")
            for selected in (row for row in rows if str(row["level"]) == level):
                query_id = str(selected["benchmark_row_id"])
                if query_id not in queries:
                    raise ValueError(f"UID selection names a foreign query: {query_id}")
                registered = connection.execute(
                    "SELECT drug FROM queries WHERE benchmark_row_id=?", (query_id,)
                ).fetchone()
                if registered is None or registered["drug"] != queries[query_id]:
                    raise ValueError(f"UID selection query differs from its cache: {query_id}")
                uid, context = str(selected.get("source_row_uid") or ""), str(selected.get("context_id") or "")
                if level == "L1":
                    clause = "cr.source_row_uid=?" if uid else "cr.context_id=?"
                    candidates = connection.execute(
                        "SELECT r.parent_smiles,cr.* FROM rankings r JOIN context_records cr "
                        "ON cr.context_id=r.morgan_context_id OR cr.context_id=r.assay_context_id "
                        f"WHERE r.benchmark_row_id=? AND {clause} ORDER BY cr.within_context_rank",
                        (query_id, uid or context),
                    ).fetchall()
                    if context:
                        candidates = [row for row in candidates if row["context_id"] == context]
                    identities = {row["context_id"] for row in candidates}
                    if len(identities) != 1:
                        raise ValueError(f"Unknown or ambiguous L1 membership: {query_id}/{uid or context}")
                    context = str(next(iter(identities)))
                    if context not in contexts[query_id]:
                        contexts[query_id].append(context)
                    if len(contexts[query_id]) > molecule_limit:
                        raise ValueError("L1 selection exceeds its context budget")
                    if not uid:
                        candidates = candidates[:l1_limit]
                else:
                    candidates = connection.execute(
                        "SELECT parent_smiles,item_id AS source_row_uid FROM rankings "
                        "WHERE benchmark_row_id=? AND item_id=?", (query_id, uid),
                    ).fetchall()
                    if len(candidates) != 1:
                        raise ValueError(f"UID outside frozen candidates: {query_id}/{level}/{uid}")
                for candidate in candidates:
                    physical = str(candidate["source_row_uid"])
                    key = (query_id, physical)
                    if key in seen:
                        raise ValueError(f"Repeated evidence UID: {key}")
                    assert_disjoint(queries[query_id], candidate["parent_smiles"], policy)
                    seen.add(key)
                    ranks[(query_id, level)] += 1
                    supplied_rank = selected.get("selection_rank")
                    if uid and supplied_rank and int(supplied_rank) != ranks[(query_id, level)]:
                        raise ValueError("UID selection ranks must preserve file order")
                    output.append(dict(benchmark_row_id=query_id, level=level,
                                       selection_rank=ranks[(query_id, level)], source_row_uid=physical,
                                       context_id=context if level == "L1" else "",
                                       context_rank=contexts[query_id].index(context) + 1 if level == "L1" else "",
                                       within_context_rank=candidate["within_context_rank"] if level == "L1" else ""))
                    if level != "L1" and ranks[(query_id, level)] > limits[level]:
                        raise ValueError(f"UID selection exceeds its {level} record budget")
    for query_id, ordered in contexts.items():
        for context in ordered:
            members = [row for row in output if row["benchmark_row_id"] == query_id and row["context_id"] == context]
            if len(members) > l1_limit:
                raise ValueError("L1 selection exceeds its per-context record budget")
        grouped = [row["context_id"] for row in output if row["benchmark_row_id"] == query_id and row["level"] == "L1"]
        if grouped != sorted(grouped, key=ordered.index):
            raise ValueError("Interleaved L1 contexts would change the requested UID order")
    return output


def export_selection(output: Path, rows: Sequence[Mapping[str, Any]], *, task: str,
                     subset: str, queries: Mapping[str, str],
                     cache_manifests: Mapping[str, Path], molecule_limit: int = 10,
                     l1_limit: int = 10, limits: Mapping[str, int] | None = None,
                     provenance: Mapping[str, Any] | None = None) -> Path:
    ordered = [{key: value for key, value in row.items() if key != "selection_rank"} for row in rows]
    resolved = resolve_selection(ordered, task=task, subset=subset, queries=queries,
                                 cache_manifests=cache_manifests, molecule_limit=molecule_limit,
                                 l1_limit=l1_limit, limits=limits)
    output.mkdir(parents=True, exist_ok=False)
    records = output / "selected_uids.tsv"
    with records.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(resolved)
    manifest = output / "manifest.json"
    write_json_atomic(manifest, dict(schema_version=SCHEMA, status="complete", task_id=task,
                                    subset=subset, benchmark_row_ids=list(queries),
                                    records=dict(path=records.name, sha256=sha256_file(records), row_count=len(resolved)),
                                    cache_manifests={level: dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))
                                                     for level, path in cache_manifests.items()},
                                    budgets=dict(contexts=molecule_limit, records_per_context=l1_limit,
                                                 records_per_level=dict(limits or {})),
                                    provenance=dict(provenance or {})))
    return manifest


def indexed_manifests(index: Path, subset: str, levels: Sequence[str]) -> dict[str, Path]:
    document = json.loads(index.read_text())
    output = {}
    for level in levels:
        entry = document["splits"][subset]["levels"][level]
        path = (index.parent / entry["manifest"]).resolve()
        if sha256_file(path) != entry["manifest_sha256"]:
            raise ValueError("Candidate manifest differs from its release index")
        manifest = json.loads(path.read_text())
        if manifest.get("neighbor_identity_policy", "parent_disjoint") != document["neighbor_identity_policy_by_level"][level]:
            raise ValueError("Candidate disjointness policy differs from its release index")
        output[level] = path
    return output


def attach_uid_selection(document: dict, root: Path, rows: Sequence[Mapping[str, Any]], *,
                         queries: Mapping[str, str], cache_manifests: Mapping[str, Path],
                         l1_limit: int = 10) -> None:
    """Add physical UIDs without discarding an optimizer's scientific provenance."""
    snapshot = root / "optimizer_snapshot.json"
    write_json_atomic(snapshot, document)
    subset = "valid" if document["subset"] == "valid_small" else document["subset"]
    levels = set(str(row["level"]) for row in rows)
    limit = int(document.get("indirect_budget", document.get("budget", 50)))
    limits = document.get("k_per_level") or {level: limit for level in levels if level != "L1"}
    manifest = export_selection(root / "uid_selection", rows, task=document["task_id"],
                                subset=subset, queries=queries, cache_manifests=cache_manifests,
                                molecule_limit=int(document.get("direct_budget", document.get("budget", 10))),
                                l1_limit=l1_limit, limits=limits,
                                provenance=dict(selection_cohort=document["subset"],
                                                optimizer_snapshot=dict(path="../optimizer_snapshot.json",
                                                                        sha256=sha256_file(snapshot))))
    document["uid_selection"] = dict(path=str(manifest.relative_to(root)), sha256=sha256_file(manifest))
