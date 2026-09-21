"""Build Gold-v1 direct and joint-indirect submodular selections."""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import platform
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
from rdkit import rdBase

from data.processing.gold_labels.conditioned_benchmark import split_path
from optimization.select_records import (
    _default_release_index,
    _fingerprint,
    _semantic_rows,
)
from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_ranked_universe
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic


DIRECT_OBJECTIVE_VERSION = "gold_direct_normalized_gated_bit_log_label.v1"
INDIRECT_OBJECTIVE_VERSION = "gold_joint_indirect_bit_log_semantic_level.v1"
DIRECT_SCHEMA = "gold_direct_context_selection.v1"
INDIRECT_SCHEMA = "gold_joint_indirect_uid_selection.v1"
MIXED_SCHEMA = "gold_mixed_selection.v1"
GRID_SCHEMA = "gold_submodular_selection_grid.v1"
DIRECT_BUDGET = 10
INDIRECT_BUDGET = 50
LEVEL_DIVERSITY_VALUES = (0.25, 0.5, 0.75)
TASKS = {
    "bbb_martins": {
        "gold": "BBB_Martins",
        "levels": ("L2", "L3", "L4", "L5"),
    },
    "bioavailability_ma": {
        "gold": "Bioavailability_Ma",
        "levels": ("L2", "L3", "L4", "L5", "L6"),
    },
    "skin_reaction": {
        "gold": "Skin_Reaction",
        "levels": ("L2", "L3"),
    },
}
PROFILE_SCREENS = (
    Path("outputs/analysis/record_selection/")
    / "morgan_normalized_gated_feature_semantic_completed_top75_top16_small_valid_v1/screen_manifest.json",
    Path("outputs/analysis/record_selection/")
    / "morgan_normalized_gated_feature_semantic_grid_v1/screen_manifest.json",
    Path("outputs/analysis/record_selection/")
    / "morgan_normalized_gated_feature_semantic_high_grid_v1/screen_manifest.json",
)
SKIN_DIRECT_ROOT = Path(
    "predict/retrieval/cache/assay_reranking/active/"
    "v9_skin_gold_v1_scaffold_morgan100_v1/skin_reaction/scaffold"
)
SKIN_SEMANTIC_ROOT = Path(
    "semantic_buckets/releases/skin_reaction/v10_main_universe_v5"
)


@dataclass(frozen=True)
class DirectProfile:
    name: str
    gated_assay: float
    molecular: float
    label: float


@dataclass(frozen=True)
class IndirectProfile:
    name: str
    gated_assay: float
    molecular: float
    semantic_relevance: float
    semantic_diversity: float
    level_diversity: float
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class PreparedCandidates:
    rows: tuple[dict[str, Any], ...]
    bits: tuple[frozenset[int], ...]
    universe: frozenset[int]
    molecular_coverage_ceiling: float


def _code(value: float) -> str:
    return f"{round(value * 100):03d}"


def direct_profiles() -> list[DirectProfile]:
    crossed = [
        DirectProfile(
            f"ga{_code(ga)}_mc{_code(mc)}_label{_code(label)}", ga, mc, label
        )
        for ga, mc, label in itertools.product(
            (0.75, 1.0, 1.25, 1.5), (0.25, 0.5, 0.75), (0.25, 0.5)
        )
    ]
    controls = [
        DirectProfile(f"ga{_code(ga)}_mc000_label000", ga, 0.0, 0.0)
        for ga in (0.75, 1.0, 1.25, 1.5)
    ]
    ablations = [DirectProfile("ga000_mc025_label025", 0.0, 0.25, 0.25)]
    return [*crossed, *controls, *ablations]


def indirect_profiles(
    screens: Sequence[Path] = PROFILE_SCREENS,
) -> tuple[list[IndirectProfile], list[dict[str, Any]]]:
    unique: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    source_entries = []
    for path in screens:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") != "complete":
            raise ValueError(f"Profile screen is not complete: {path}")
        family = path.parent.name
        for row in document.get("selected_profiles") or []:
            lambdas = row.get("lambdas") or {}
            key = tuple(float(lambdas[name]) for name in (
                "assay_gated", "molecular_coverage",
                "semantic_relevance", "semantic_coverage",
            ))
            alias = f"{family}/{row['name']}"
            source_entries.append({"alias": alias, "lambdas": key})
            unique.setdefault(key, {"name": row["name"], "aliases": []})["aliases"].append(alias)
    if len(source_entries) != 94 or len(unique) != 75:
        raise ValueError(
            f"Expected 94 source profiles and 75 unique objectives, got "
            f"{len(source_entries)} and {len(unique)}"
        )
    profiles = []
    for key, row in sorted(unique.items(), key=lambda item: item[1]["name"]):
        ga, mc, sr, sd = key
        for level in LEVEL_DIVERSITY_VALUES:
            profiles.append(IndirectProfile(
                f"{row['name']}_ld{_code(level)}", ga, mc, sr, sd, level,
                tuple(sorted(row["aliases"])),
            ))
    return profiles, source_entries


def _write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _gold_queries(task: str, subset: str = "valid_small") -> tuple[dict[str, str], Path]:
    path = split_path(task, subset, version="v1")
    rows = read_jsonl(path)
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in rows}
    if not rows or len(queries) != len(rows):
        raise ValueError(f"Gold split must contain unique queries: {path}")
    return queries, path


def _log_ceiling(capacities: Mapping[str, int], budget: int) -> float:
    if budget < 1 or sum(capacities.values()) < budget:
        raise ValueError("Group capacities cannot fill the requested budget")
    counts = {group: 0 for group in capacities}
    heap = [(-math.log(2.0), group) for group, cap in capacities.items() if cap]
    heapq.heapify(heap)
    total = 0.0
    for _ in range(budget):
        if not heap:
            raise ValueError("Group capacities cannot fill the requested budget")
        negative_gain, group = heapq.heappop(heap)
        total -= negative_gain
        counts[group] += 1
        if counts[group] < capacities[group]:
            gain = math.log(counts[group] + 2) - math.log(counts[group] + 1)
            heapq.heappush(heap, (-gain, group))
    return total


def normalized_log_diversity(
    selected_groups: Iterable[str], capacities: Mapping[str, int], budget: int
) -> float:
    counts = Counter(map(str, selected_groups))
    if any(counts[group] > capacities.get(group, 0) for group in counts):
        raise ValueError("Selected group count exceeds candidate capacity")
    denominator = _log_ceiling(capacities, budget)
    value = sum(math.log1p(count) for count in counts.values()) / denominator
    if not -1e-12 <= value <= 1 + 1e-12:
        raise AssertionError(f"Normalized log diversity escaped [0, 1]: {value}")
    return min(1.0, max(0.0, value))


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, direct: bool) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    ids = [str(row["item_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate identifiers must be unique")
    for row in output:
        for field in ("morgan_similarity", "assay_transfer_score"):
            value = row.get(field)
            if value is None and field == "assay_transfer_score" and not direct:
                continue
            number = float(value)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"{field} must be finite and within [0, 1]")
            row[field] = number
        if direct and int(row["gold_label"]) not in (0, 1):
            raise ValueError("Direct Gold labels must be binary")
        if not direct:
            if not str(row.get("level") or "") or not str(row.get("semantic_bucket_id") or ""):
                raise ValueError("Indirect candidates require level and semantic bucket")
            weight = row.get("semantic_weight")
            if weight is not None and (not math.isfinite(float(weight)) or not 0 <= float(weight) <= 1):
                raise ValueError("Semantic weights must be within [0, 1]")
    return output


def _prepare_candidates(
    candidates: Sequence[Mapping[str, Any]], *, direct: bool, budget: int
) -> PreparedCandidates:
    rows = _validate_rows(candidates, direct=direct)
    if budget < 1 or budget > len(rows):
        raise ValueError("Budget must be positive and no greater than the candidate count")
    bits = tuple(
        frozenset(_fingerprint(str(row["parent_smiles"])).GetOnBits())
        for row in rows
    )
    universe = frozenset(bit for current in bits for bit in current)
    return PreparedCandidates(tuple(rows), bits, universe, 1.0)


def _lazy_select(
    prepared: PreparedCandidates, *, budget: int, base_scores: Sequence[float],
    molecular_lambda: float, group_terms: Sequence[tuple[float, Sequence[str], float]],
) -> tuple[list[int], list[float], float, float]:
    rows, bits, universe = prepared.rows, prepared.bits, prepared.universe
    raw_ceiling = prepared.molecular_coverage_ceiling
    coverage_ceiling = raw_ceiling or 1.0
    counts = [Counter() for _ in group_terms]
    covered: set[int] = set()
    chosen: list[int] = []
    gains: list[float] = []
    remaining = set(range(len(rows)))

    def marginal(index: int) -> float:
        gain = float(base_scores[index]) / budget
        if molecular_lambda:
            gain += molecular_lambda * (
                len(bits[index] - covered) / len(universe) / coverage_ceiling
                if universe else 0.0
            )
        for term_index, (weight, groups, denominator) in enumerate(group_terms):
            if weight:
                count = counts[term_index][groups[index]]
                gain += weight * (
                    math.log(count + 2) - math.log(count + 1)
                ) / denominator
        return gain

    heap = [(-marginal(index), str(rows[index]["item_id"]), index) for index in remaining]
    heapq.heapify(heap)
    while len(chosen) < budget:
        _, tie, index = heapq.heappop(heap)
        if index not in remaining:
            continue
        gain = marginal(index)
        next_bound = -heap[0][0] if heap else -math.inf
        if gain + 1e-15 < next_bound:
            heapq.heappush(heap, (-gain, tie, index))
            continue
        chosen.append(index)
        gains.append(gain)
        remaining.remove(index)
        covered.update(bits[index])
        for term_index, (_, groups, _) in enumerate(group_terms):
            counts[term_index][groups[index]] += 1
    coverage = (
        min(1.0, len(covered) / len(universe) / coverage_ceiling)
        if universe else 0.0
    )
    return chosen, gains, coverage, raw_ceiling


def select_direct_contexts(
    candidates: Sequence[Mapping[str, Any]], *, profile: DirectProfile,
    budget: int = DIRECT_BUDGET, prepared: PreparedCandidates | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepared or _prepare_candidates(candidates, direct=True, budget=budget)
    rows = prepared.rows
    labels = [str(int(row["gold_label"])) for row in rows]
    capacities = Counter(labels)
    label_ceiling = _log_ceiling(capacities, budget)
    base = [
        (row["morgan_similarity"] + profile.gated_assay * row["morgan_similarity"]
         * row["assay_transfer_score"]) / (1 + profile.gated_assay)
        for row in rows
    ]
    ranking, gains, molecular, coverage_ceiling = _lazy_select(
        prepared, budget=budget, base_scores=base, molecular_lambda=profile.molecular,
        group_terms=((profile.label, labels, label_ceiling),),
    )
    selected = [
        {**rows[index], "selection_rank": rank, "marginal_gain": gains[rank - 1]}
        for rank, index in enumerate(ranking, start=1)
    ]
    return selected, {
        "normalized_relevance": sum(base[index] for index in ranking) / budget,
        "molecular_coverage_normalized": molecular,
        "molecular_coverage_ceiling": coverage_ceiling,
        "label_diversity": normalized_log_diversity(
            (labels[index] for index in ranking), capacities, budget
        ),
        "label_0": sum(labels[index] == "0" for index in ranking),
        "label_1": sum(labels[index] == "1" for index in ranking),
        "candidate_contexts": len(rows),
        "selected_contexts": budget,
        "objective_score": sum(gains),
    }


def select_joint_indirect(
    candidates: Sequence[Mapping[str, Any]], *, profile: IndirectProfile,
    budget: int = INDIRECT_BUDGET, prepared: PreparedCandidates | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepared or _prepare_candidates(candidates, direct=False, budget=budget)
    rows = prepared.rows
    levels = [str(row["level"]) for row in rows]
    semantics = [str(row["semantic_bucket_id"]) for row in rows]
    level_capacities, semantic_capacities = Counter(levels), Counter(semantics)
    level_ceiling = _log_ceiling(level_capacities, budget)
    semantic_ceiling = _log_ceiling(semantic_capacities, budget)
    base, semantic_weights = [], []
    for row in rows:
        similarity = float(row["morgan_similarity"])
        assay = row.get("assay_transfer_score")
        base.append(
            similarity if assay is None else
            (similarity + profile.gated_assay * similarity * float(assay))
            / (1 + profile.gated_assay)
        )
        semantic_weights.append(float(row.get("semantic_weight") or 0.0))
    semantic_available = all(row.get("semantic_weight") is not None for row in rows)
    sr_effective = profile.semantic_relevance if semantic_available else 0.0
    modular = [
        base[index] + sr_effective * semantic_weights[index]
        for index in range(len(rows))
    ]
    ranking, gains, molecular, coverage_ceiling = _lazy_select(
        prepared, budget=budget, base_scores=modular,
        molecular_lambda=profile.molecular,
        group_terms=(
            (profile.semantic_diversity, semantics, semantic_ceiling),
            (profile.level_diversity, levels, level_ceiling),
        ),
    )
    selected = [
        {**rows[index], "selection_rank": rank, "marginal_gain": gains[rank - 1]}
        for rank, index in enumerate(ranking, start=1)
    ]
    selected_levels = [levels[index] for index in ranking]
    return selected, {
        "normalized_relevance": sum(base[index] for index in ranking) / budget,
        "molecular_coverage_normalized": molecular,
        "molecular_coverage_ceiling": coverage_ceiling,
        "semantic_relevance": sum(semantic_weights[index] for index in ranking) / budget,
        "semantic_relevance_available": semantic_available,
        "semantic_relevance_lambda_effective": sr_effective,
        "semantic_diversity": normalized_log_diversity(
            (semantics[index] for index in ranking), semantic_capacities, budget
        ),
        "level_diversity": normalized_log_diversity(
            selected_levels, level_capacities, budget
        ),
        "level_counts": dict(sorted(Counter(selected_levels).items())),
        "candidate_records": len(rows),
        "selected_records": budget,
        "objective_score": sum(gains),
    }


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _load_ranked_direct(
    task: str, queries: Mapping[str, str], subset: str = "valid_small"
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    index_path = _default_release_index(task).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cache_subset = "valid" if subset == "valid_small" else subset
    entry = index["splits"][cache_subset]["levels"]["L1"]
    manifest_path = (index_path.parent / entry["manifest"]).resolve()
    if sha256_file(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("L1 manifest differs from the release index")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    database = manifest_path.with_name(manifest["database"])
    output = {}
    with _open_database(database) as connection:
        for query_id, drug in queries.items():
            registered = connection.execute(
                "SELECT drug FROM queries WHERE benchmark_row_id=?", (query_id,)
            ).fetchone()
            if registered is None or str(registered["drug"]) != drug:
                raise ValueError(f"Direct cache query mismatch: {query_id}")
            ranked = connection.execute(
                "SELECT r.*,c.gold_label FROM rankings r JOIN contexts c "
                "ON c.context_id=r.assay_context_id WHERE r.benchmark_row_id=? "
                "ORDER BY r.assay_rank,r.item_id", (query_id,),
            ).fetchall()
            output[query_id] = [{
                "item_id": str(row["assay_context_id"]),
                "parent_id": str(row["parent_id"]),
                "parent_smiles": str(row["parent_smiles"]),
                "morgan_similarity": float(row["morgan_similarity"]),
                "assay_transfer_score": float(row["assay_transfer_score"]),
                "gold_label": int(row["gold_label"]),
            } for row in ranked]
    return output, {
        "profile": index["profile"],
        "gold_release": index["gold_release"],
        "release_index": str(index_path),
        "release_index_sha256": sha256_file(index_path),
        "l1_manifest": str(manifest_path),
        "l1_manifest_sha256": sha256_file(manifest_path),
        "database": str(database),
        "database_sha256": sha256_file(database),
    }


def _load_skin_direct(
    queries: Mapping[str, str], subset: str = "valid_small"
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cache_subset = "valid" if subset == "valid_small" else subset
    manifest_path = (SKIN_DIRECT_ROOT / cache_subset / "VERSION.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest.get("model") or {}
    if (
        manifest.get("status") != "complete"
        or manifest.get("task_id") != "skin_reaction"
        or model.get("model")
        != "jiosephlee/assay-transfer-tool-soft-v9.0.2-skin-reaction-mixed-continuous"
    ):
        raise ValueError("Skin Gold-v1 v9.0.2 ranking manifest is incompatible")
    rankings_path = manifest_path.with_name(manifest["rankings"])
    if sha256_file(rankings_path) != manifest["rankings_sha256"]:
        raise ValueError("Skin direct rankings differ from their manifest")
    table = pq.read_table(rankings_path, filters=[("query_record_id", "in", list(queries))])
    output = {query_id: [] for query_id in queries}
    for row in table.to_pylist():
        query_id = str(row["query_record_id"])
        if str(row["query_smiles"]) != queries[query_id]:
            raise ValueError(f"Skin direct cache query mismatch: {query_id}")
        output[query_id].append({
            "item_id": str(row["retrieval_record_id"]),
            "parent_id": str(row["retrieval_molecule_identity_key"]),
            "parent_smiles": str(row["retrieval_smiles"]),
            "morgan_similarity": float(row["morgan_tanimoto_similarity"]),
            "assay_transfer_score": float(row["prob_transfer"]),
            "gold_label": int(row["retrieval_gold_Y"]),
        })
    if any(not rows for rows in output.values()):
        raise ValueError(f"Skin direct rankings omit {subset} queries")
    return output, {
        "profile": "v9_skin_gold_v1_scaffold_morgan100_v1",
        "gold_release": "v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rankings": str(rankings_path),
        "rankings_sha256": sha256_file(rankings_path),
        "model": model,
    }


def _load_skin_semantics() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    manifest_path = (SKIN_SEMANTIC_ROOT / "manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["outputs"]["record_readout_bucket_map"]
    mapping_path = manifest_path.parent / entry["path"]
    if (
        manifest.get("task") != "skin_reaction"
        or manifest.get("evidence_library_version") != "v10_main_universe_v5"
        or sha256_file(mapping_path) != entry["sha256"]
    ):
        raise ValueError("Skin semantic-bucket release is incompatible")
    table = pq.read_table(
        mapping_path, columns=["source_row_uid", "level", "semantic_bucket_id"]
    )
    rows = {
        (str(row["source_row_uid"]), f"L{int(row['level'])}"): {
            "semantic_bucket_id": str(row["semantic_bucket_id"]),
            "semantic_weight": None,
        }
        for row in table.to_pylist()
    }
    return rows, {
        "release": "v10_main_universe_v5",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record_map": str(mapping_path),
        "record_map_sha256": sha256_file(mapping_path),
        "semantic_relevance": "unavailable_effective_zero",
    }


def _load_indirect(
    task: str, queries: Mapping[str, str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    levels = TASKS[task]["levels"]
    release_index = _default_release_index(task)
    ranked, _, cache_audit = load_ranked_universe(
        release_index, task=task, subset="valid", levels=levels, queries=queries
    )
    semantics, semantic_audit = (
        _load_skin_semantics() if task == "skin_reaction" else _semantic_rows(task)
    )
    output = {query_id: [] for query_id in queries}
    missing = []
    for level in levels:
        for query_id, rows in ranked[level].items():
            for row in rows:
                uid = str(row["item_id"])
                semantic = semantics.get((uid, level))
                if semantic is None:
                    missing.append((uid, level))
                    continue
                output[query_id].append({**row, **semantic, "level": level})
    if missing:
        raise ValueError(f"Semantic assignments omit indirect candidates: {missing[:5]}")
    for query_id, rows in output.items():
        uids = [str(row["item_id"]) for row in rows]
        if len(uids) != len(set(uids)):
            raise ValueError(f"Joint L2+ universe repeats UIDs: {query_id}")
    return output, {"cache": cache_audit, "semantics": semantic_audit}


def _profile_root(output_root: Path, kind: str, profile: str, task: str) -> Path:
    path = output_root / kind / profile / task
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_direct(
    output_root: Path, tasks: Sequence[str], *, subset: str = "valid_small",
    profile_names: Sequence[str] | None = None,
) -> Path:
    profiles = direct_profiles()
    if profile_names:
        requested = set(profile_names)
        profiles = [profile for profile in profiles if profile.name in requested]
        missing = requested - {profile.name for profile in profiles}
        if missing:
            raise ValueError(f"Unknown direct profiles: {sorted(missing)}")
    manifests = {}
    for task in tasks:
        queries, query_path = _gold_queries(task, subset)
        universes, cache_audit = (
            _load_skin_direct(queries, subset) if task == "skin_reaction"
            else _load_ranked_direct(task, queries, subset)
        )
        prepared_universes = {
            query_id: _prepare_candidates(rows, direct=True, budget=DIRECT_BUDGET)
            for query_id, rows in universes.items()
        }
        for profile in profiles:
            root = _profile_root(output_root, "direct", profile.name, task)
            selected_rows, diagnostics = [], []
            for query_id in queries:
                selected, summary = select_direct_contexts(
                    universes[query_id], profile=profile,
                    prepared=prepared_universes[query_id],
                )
                selected_rows.extend({
                    "benchmark_row_id": query_id,
                    "selection_rank": row["selection_rank"],
                    "context_id": row["item_id"],
                    "parent_id": row["parent_id"],
                    "gold_label": row["gold_label"],
                } for row in selected)
                diagnostics.append({
                    "benchmark_row_id": query_id,
                    **{**summary, "level_counts": json.dumps(summary.get("level_counts", {}), sort_keys=True)},
                })
            selected_path = root / "selected_contexts.tsv"
            selected_count = _write_tsv(selected_path, selected_rows, (
                "benchmark_row_id", "selection_rank", "context_id", "parent_id", "gold_label",
            ))
            diagnostics_path = root / "selection_diagnostics.tsv"
            _write_tsv(diagnostics_path, diagnostics, tuple(diagnostics[0]))
            manifest = {
                "schema_version": DIRECT_SCHEMA,
                "status": "complete",
                "task_id": task,
                "subset": subset,
                "benchmark_row_ids": list(queries),
                "budget": DIRECT_BUDGET,
                "profile": profile.name,
                "objective": {
                    "version": DIRECT_OBJECTIVE_VERSION,
                    "gated_assay_lambda": profile.gated_assay,
                    "molecular_coverage_lambda": profile.molecular,
                    "label_diversity_lambda": profile.label,
                    "diversity_function": "capacity_normalized_log1p.v1",
                    "optimizer": "deterministic_lazy_greedy.v1",
                },
                "inputs": {
                    "queries": str(query_path.resolve()),
                    "queries_sha256": sha256_file(query_path),
                    "direct_cache": cache_audit,
                },
                "records": {
                    "path": selected_path.name,
                    "sha256": sha256_file(selected_path),
                    "row_count": selected_count,
                },
                "outputs": {diagnostics_path.name: sha256_file(diagnostics_path)},
                "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
            }
            manifest_path = root / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            manifests.setdefault(profile.name, {})[task] = {
                "path": str(manifest_path.relative_to(output_root)),
                "sha256": sha256_file(manifest_path),
            }
    grid_path = output_root / "direct_grid_manifest.json"
    write_json_atomic(grid_path, {
        "schema_version": GRID_SCHEMA,
        "status": "complete",
        "kind": "direct",
        "subset": subset,
        "objective_version": DIRECT_OBJECTIVE_VERSION,
        "profile_count": len(profiles),
        "tasks": list(tasks),
        "profiles": [{
            "name": profile.name,
            "gated_assay_lambda": profile.gated_assay,
            "molecular_coverage_lambda": profile.molecular,
            "label_diversity_lambda": profile.label,
            "task_manifests": manifests[profile.name],
        } for profile in profiles],
    })
    return grid_path


def _select_indirect_query(
    item: tuple[str, list[dict[str, Any]], list[IndirectProfile]],
) -> tuple[str, list[tuple[str, list[dict[str, Any]], dict[str, Any]]]]:
    query_id, rows, profiles = item
    prepared = _prepare_candidates(rows, direct=False, budget=INDIRECT_BUDGET)
    results = []
    for profile in profiles:
        selected, summary = select_joint_indirect(
            rows, profile=profile, prepared=prepared
        )
        results.append((profile.name, selected, summary))
    return query_id, results


def build_indirect(
    output_root: Path, tasks: Sequence[str], *, workers: int = 1
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    profiles, aliases = indirect_profiles()
    manifests = {}
    for task in tasks:
        queries, query_path = _gold_queries(task)
        universes, input_audit = _load_indirect(task, queries)
        selected_by_profile = {profile.name: [] for profile in profiles}
        diagnostics_by_profile = {profile.name: [] for profile in profiles}
        work = ((query_id, universes[query_id], profiles) for query_id in queries)
        if workers == 1:
            query_results = map(_select_indirect_query, work)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            query_results = executor.map(_select_indirect_query, work, chunksize=1)
        try:
            for query_id, results in query_results:
                for profile_name, selected, summary in results:
                    selected_by_profile[profile_name].extend({
                        "benchmark_row_id": query_id,
                        "selection_rank": row["selection_rank"],
                        "level": row["level"],
                        "source_row_uid": row["item_id"],
                    } for row in selected)
                    diagnostics_by_profile[profile_name].append({
                        "benchmark_row_id": query_id,
                        **{
                            **summary,
                            "level_counts": json.dumps(
                                summary["level_counts"], sort_keys=True
                            ),
                        },
                    })
        finally:
            if workers != 1:
                executor.shutdown()
        for profile in profiles:
            root = _profile_root(output_root, "indirect", profile.name, task)
            selected_path = root / "selected_uids.tsv"
            selected_count = _write_tsv(selected_path, selected_by_profile[profile.name], (
                "benchmark_row_id", "selection_rank", "level", "source_row_uid",
            ))
            diagnostics_path = root / "selection_diagnostics.tsv"
            diagnostics = diagnostics_by_profile[profile.name]
            _write_tsv(diagnostics_path, diagnostics, tuple(diagnostics[0]))
            manifest = {
                "schema_version": INDIRECT_SCHEMA,
                "status": "complete",
                "task_id": task,
                "subset": "valid_small",
                "benchmark_row_ids": list(queries),
                "budget": INDIRECT_BUDGET,
                "profile": profile.name,
                "objective": {
                    "version": INDIRECT_OBJECTIVE_VERSION,
                    "gated_assay_lambda": profile.gated_assay,
                    "molecular_coverage_lambda": profile.molecular,
                    "semantic_relevance_lambda": profile.semantic_relevance,
                    "semantic_diversity_lambda": profile.semantic_diversity,
                    "level_diversity_lambda": profile.level_diversity,
                    "diversity_function": "capacity_normalized_log1p.v1",
                    "optimizer": "deterministic_lazy_greedy.v1",
                },
                "source_profile_aliases": list(profile.aliases),
                "inputs": {
                    "queries": str(query_path.resolve()),
                    "queries_sha256": sha256_file(query_path),
                    **input_audit,
                },
                "records": {
                    "path": selected_path.name,
                    "sha256": sha256_file(selected_path),
                    "row_count": selected_count,
                },
                "outputs": {diagnostics_path.name: sha256_file(diagnostics_path)},
                "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
            }
            manifest_path = root / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            manifests.setdefault(profile.name, {})[task] = {
                "path": str(manifest_path.relative_to(output_root)),
                "sha256": sha256_file(manifest_path),
            }
    grid_path = output_root / "indirect_grid_manifest.json"
    write_json_atomic(grid_path, {
        "schema_version": GRID_SCHEMA,
        "status": "complete",
        "kind": "indirect",
        "objective_version": INDIRECT_OBJECTIVE_VERSION,
        "source_profile_count": len(aliases),
        "unique_source_profile_count": len(profiles) // len(LEVEL_DIVERSITY_VALUES),
        "level_diversity_values": list(LEVEL_DIVERSITY_VALUES),
        "profile_count": len(profiles),
        "workers": workers,
        "tasks": list(tasks),
        "source_aliases": aliases,
        "profiles": [{
            "name": profile.name,
            "source_profile_aliases": list(profile.aliases),
            "task_manifests": manifests[profile.name],
        } for profile in profiles],
    })
    return grid_path


def compose_mixed(direct_manifest: Path, indirect_manifest: Path, output: Path) -> Path:
    direct_manifest, indirect_manifest = direct_manifest.resolve(), indirect_manifest.resolve()
    direct = validate_selection_manifest(direct_manifest, DIRECT_SCHEMA)
    indirect = validate_selection_manifest(indirect_manifest, INDIRECT_SCHEMA)
    for field in ("task_id", "subset", "benchmark_row_ids"):
        if direct.get(field) != indirect.get(field):
            raise ValueError(f"Direct and indirect selections disagree on {field}")
    document = {
        "schema_version": MIXED_SCHEMA,
        "status": "complete",
        "task_id": direct["task_id"],
        "subset": direct["subset"],
        "benchmark_row_ids": direct["benchmark_row_ids"],
        "direct_budget": direct["budget"],
        "indirect_budget": indirect["budget"],
        "direct": {"path": str(direct_manifest), "sha256": sha256_file(direct_manifest)},
        "indirect": {"path": str(indirect_manifest), "sha256": sha256_file(indirect_manifest)},
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, document)
    return output


def validate_selection_manifest(path: Path, expected_schema: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    schema = document.get("schema_version")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(f"Expected {expected_schema}, found {schema}")
    if schema not in {DIRECT_SCHEMA, INDIRECT_SCHEMA} or document.get("status") != "complete":
        raise ValueError(f"Selection manifest is not complete and compatible: {path}")
    records = document.get("records") or {}
    relative = Path(str(records.get("path") or ""))
    records_path = (path.parent / relative).resolve()
    if not relative.name or not records_path.is_relative_to(path.parent):
        raise ValueError("Selection records path escapes its manifest directory")
    if sha256_file(records_path) != records.get("sha256"):
        raise ValueError("Selection records hash mismatch")
    with records_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    expected_fields = (
        ["benchmark_row_id", "selection_rank", "context_id", "parent_id", "gold_label"]
        if schema == DIRECT_SCHEMA else
        ["benchmark_row_id", "selection_rank", "level", "source_row_uid"]
    )
    if reader.fieldnames != expected_fields or len(rows) != int(records.get("row_count", -1)):
        raise ValueError("Selection records schema or row count is incompatible")
    query_ids = [str(value) for value in document.get("benchmark_row_ids") or []]
    budget = int(document.get("budget", 0))
    grouped = {query_id: [] for query_id in query_ids}
    identity_field = "context_id" if schema == DIRECT_SCHEMA else "source_row_uid"
    for row in rows:
        query_id = row["benchmark_row_id"]
        if query_id not in grouped:
            raise ValueError(f"Selection contains an unexpected query: {query_id}")
        if int(row["selection_rank"]) != len(grouped[query_id]) + 1:
            raise ValueError(f"Selection ranks are not contiguous: {query_id}")
        grouped[query_id].append(row[identity_field])
    if any(len(values) != budget or len(values) != len(set(values)) for values in grouped.values()):
        raise ValueError("Selection budget or identity uniqueness is invalid")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("direct", "indirect"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output-root", type=Path, required=True)
        subparser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
        if command == "indirect":
            subparser.add_argument("--workers", type=int, default=1)
        else:
            subparser.add_argument(
                "--subset", choices=("valid_small", "valid", "test"),
                default="valid_small",
            )
            subparser.add_argument("--profiles", nargs="+")
    compose = subparsers.add_parser("compose")
    compose.add_argument("--direct-manifest", type=Path, required=True)
    compose.add_argument("--indirect-manifest", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "direct":
        print(build_direct(
            args.output_root.resolve(), args.tasks,
            subset=args.subset, profile_names=args.profiles,
        ))
    elif args.command == "indirect":
        print(build_indirect(args.output_root.resolve(), args.tasks, workers=args.workers))
    elif args.command == "compose":
        print(compose_mixed(args.direct_manifest, args.indirect_manifest, args.output))
    else:
        document = validate_selection_manifest(args.manifest)
        print(f"{args.manifest.resolve()}\t{document['schema_version']}\tcomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
