"""Build Gold-v1 direct and joint-indirect submodular selections."""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
import platform
import re
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import yaml
from rdkit import rdBase

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path
from optimization.select_records import (
    _default_release_index,
    _fingerprint,
    _semantic_rows,
)
from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_ranked_universe
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from semantic_buckets.artifacts import load_reviewed_record_weights, resolve_semantic_bucket_artifacts


DIRECT_OBJECTIVE_VERSION = "direct_normalized_gated_bit_log_label.v2"
INDIRECT_OBJECTIVE_VERSION = "gold_joint_indirect_bit_log_semantic_level.v1"
SQRT_INDIRECT_OBJECTIVE_VERSION = "gold_joint_indirect_bit_sqrt_semantic_level.v1"
DIRECT_SCHEMA = "direct_context_selection.v2"
LEGACY_DIRECT_SCHEMA = "gold_direct_context_selection.v1"
INDIRECT_SCHEMA = "gold_joint_indirect_uid_selection.v1"
MIXED_SCHEMA = "gold_mixed_selection.v1"
GRID_SCHEMA = "gold_submodular_selection_grid.v1"
DIRECT_GRID_SCHEMA = "direct_context_selection_grid.v2"
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
    "ames": {"gold": "Ames", "levels": ("L2", "L3", "L4", "L5")},
    "dili": {"gold": "DILI", "levels": ("L2", "L3", "L4", "L5", "L6", "L7")},
    "carcinogens": {"gold": "Carcinogens", "levels": ("L2", "L3", "L4", "L5", "L6", "L7")},
}
DIRECT_TASKS = {
    "bbb_martins": "ranked_level_retrieval_v4",
    "bioavailability_ma": "ranked_level_retrieval_v4",
    "skin_reaction": "ranked_level_retrieval_skin_gold_v1_l1_adapter_v2",
    "ames": "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
    "dili": "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
    "carcinogens": "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
}
TDC_DIRECT_TASKS = {
    task: "ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1"
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction")
}
TDC_DIRECT_TASKS.update({
    task: f"flat_v5/tdc_v1/{task}/l1/assay_transfer/v10_3/tdc_pinned_v1"
    for task in ("ames", "dili", "carcinogens")
})
PROFILE_SCREENS = (
    Path("outputs/analysis/record_selection/")
    / "morgan_normalized_gated_feature_semantic_completed_top75_top16_small_valid_v1/screen_manifest.json",
    Path("outputs/analysis/record_selection/")
    / "morgan_normalized_gated_feature_semantic_grid_v1/screen_manifest.json",
    Path("outputs/analysis/record_selection/")
    / "morgan_normalized_gated_feature_semantic_high_grid_v1/screen_manifest.json",
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
    return [
        DirectProfile(
            f"ga{_code(ga)}_mc{_code(mc)}_label{_code(label)}", ga, mc, label
        )
        for ga, mc, label in itertools.chain(
            itertools.product(
                (0.75, 1.0, 1.25), (0.0, 0.1, 0.25), (0.0, 0.1, 0.25)
            ),
            itertools.product((0.25, 0.5), (0.0, 0.1), (0.1, 0.25, 0.35)),
        )
    ]


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


def sqrt_indirect_profiles() -> list[IndirectProfile]:
    return [
        IndirectProfile(
            f"ga{_code(ga)}_mc{_code(mc)}_sr{_code(sr)}_sd{_code(sd)}_ld{_code(ld)}",
            ga, mc, sr, sd, ld, (),
        )
        for ga, sr, mc, sd, ld in itertools.product(
            (0.75, 1.0), (0.1, 0.25, 0.5), (0.1, 0.25),
            (0.1, 0.25), (0.1, 0.25),
        )
    ]


def _write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _direct_queries(
    benchmark: str, task: str, subset: str,
) -> tuple[dict[str, str], Path]:
    path = (
        split_path(task, subset, version="v1")
        if benchmark == "gold_v1"
        else tdc_split_path(task, subset, version="v1")
    )
    rows = read_jsonl(path)
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in rows}
    if not rows or len(queries) != len(rows):
        raise ValueError(f"Gold split must contain unique queries: {path}")
    return queries, path


def _log_ceiling(capacities: Mapping[str, int], budget: int) -> float:
    return _group_ceiling(capacities, budget, "log")


def _group_gain(count: int, function: str) -> float:
    if function == "sqrt":
        return math.sqrt(count + 1) - math.sqrt(count)
    if function == "log":
        return math.log(count + 2) - math.log(count + 1)
    raise ValueError(f"Unknown diversity function: {function}")


def _group_ceiling(capacities: Mapping[str, int], budget: int, function: str) -> float:
    if budget < 1 or sum(capacities.values()) < budget:
        raise ValueError("Group capacities cannot fill the requested budget")
    counts = {group: 0 for group in capacities}
    heap = [(-_group_gain(0, function), group) for group, cap in capacities.items() if cap]
    heapq.heapify(heap)
    total = 0.0
    for _ in range(budget):
        if not heap:
            raise ValueError("Group capacities cannot fill the requested budget")
        negative_gain, group = heapq.heappop(heap)
        total -= negative_gain
        counts[group] += 1
        if counts[group] < capacities[group]:
            gain = _group_gain(counts[group], function)
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


def normalized_sqrt_diversity(
    selected_groups: Iterable[str], capacities: Mapping[str, int], budget: int
) -> float:
    counts = Counter(map(str, selected_groups))
    if any(counts[group] > capacities.get(group, 0) for group in counts):
        raise ValueError("Selected group count exceeds candidate capacity")
    value = sum(math.sqrt(count) for count in counts.values()) / _group_ceiling(
        capacities, budget, "sqrt"
    )
    if not -1e-12 <= value <= 1 + 1e-12:
        raise AssertionError(f"Normalized sqrt diversity escaped [0, 1]: {value}")
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
    diversity_function: str = "log",
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
                gain += weight * _group_gain(count, diversity_function) / denominator
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
    diversity_function: str = "log",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepared or _prepare_candidates(candidates, direct=False, budget=budget)
    rows = prepared.rows
    levels = [str(row["level"]) for row in rows]
    semantics = [str(row["semantic_bucket_id"]) for row in rows]
    level_capacities, semantic_capacities = Counter(levels), Counter(semantics)
    level_ceiling = _group_ceiling(level_capacities, budget, diversity_function)
    semantic_ceiling = _group_ceiling(semantic_capacities, budget, diversity_function)
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
        diversity_function=diversity_function,
    )
    selected = [
        {**rows[index], "selection_rank": rank, "marginal_gain": gains[rank - 1]}
        for rank, index in enumerate(ranking, start=1)
    ]
    selected_levels = [levels[index] for index in ranking]
    normalize = normalized_sqrt_diversity if diversity_function == "sqrt" else normalized_log_diversity
    return selected, {
        "normalized_relevance": sum(base[index] for index in ranking) / budget,
        "molecular_coverage_normalized": molecular,
        "molecular_coverage_ceiling": coverage_ceiling,
        "semantic_relevance": sum(semantic_weights[index] for index in ranking) / budget,
        "semantic_relevance_available": semantic_available,
        "semantic_relevance_lambda_effective": sr_effective,
        "semantic_diversity": normalize(
            (semantics[index] for index in ranking), semantic_capacities, budget
        ),
        "level_diversity": normalize(
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
    task: str, queries: Mapping[str, str], release_index: Path,
    subset: str = "valid_small",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    index_path = release_index.resolve()
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
        "label_release": index.get("gold_release") or index.get("label_release"),
        "release_index": str(index_path),
        "release_index_sha256": sha256_file(index_path),
        "l1_manifest": str(manifest_path),
        "l1_manifest_sha256": sha256_file(manifest_path),
        "database": str(database),
        "database_sha256": sha256_file(database),
    }


def _load_skin_semantics(
    release: str = "v10_main_universe_v5",
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    if release == "v10_main_universe_v6":
        artifacts = resolve_semantic_bucket_artifacts("skin_reaction", release)
        manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete_reviewed"
            or manifest.get("weight_scope") != "morgan_top100_valid_test_l2_l3"
        ):
            raise ValueError("Skin v6 semantic weights are not reviewed for this universe")
        sources = {
            key: getattr(artifacts, key) for key in (
                "record_semantic_bucket_map", "record_relevance_rankings",
                "semantic_bucket_weights", "semantic_bucket_rankings",
            )
        }
        for key, path in sources.items():
            if sha256_file(path) != manifest["files"][key]["sha256"]:
                raise ValueError(f"Skin v6 {key} differs from reviewed release")
        binding_path = artifacts.root / "provenance/evidence_input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        evidence_root = Path("data/evidence_libraries/skin_reaction")
        if (
            binding.get("v5_release_manifest_sha256")
            != sha256_file(evidence_root / "v10_main_universe_v5/manifest.json")
            or binding.get("v6_release_manifest_sha256")
            != sha256_file(evidence_root / "v10_main_universe_v6/manifest.json")
            or any(pair["v5"] != pair["v6"] for pair in binding["scientific_inputs"].values())
        ):
            raise ValueError("Skin v6 weights are not bound to the v5 cache evidence")
        weights = {}
        for row in pq.read_table(
            sources["semantic_bucket_weights"],
            columns=["level", "semantic_bucket_id", "weight"],
        ).to_pylist():
            key = (str(row["level"]), str(row["semantic_bucket_id"]))
            if key in weights or not 0 <= float(row["weight"]) <= 1:
                raise ValueError(f"Invalid Skin v6 bucket weight: {key}")
            weights[key] = float(row["weight"])
        rankings = pq.read_table(
            sources["semantic_bucket_rankings"],
            columns=["level", "semantic_bucket_id", "weight"],
        ).to_pylist()
        if len(rankings) != len(weights) or any(
            weights.get((str(row["level"]), str(row["semantic_bucket_id"])))
            != float(row["weight"]) for row in rankings
        ):
            raise ValueError("Skin v6 bucket rankings disagree with reviewed weights")
        rows = {}
        for row in pq.read_table(
            sources["record_semantic_bucket_map"],
            columns=["source_row_uid", "level", "semantic_bucket_id"],
        ).to_pylist():
            key = (str(row["source_row_uid"]), str(row["level"]))
            if key in rows:
                raise ValueError(f"Skin v6 mapping repeats {key}")
            rows[key] = {
                "semantic_bucket_id": str(row["semantic_bucket_id"]),
                "semantic_weight": None,
            }
        for row in pq.read_table(
            sources["record_relevance_rankings"],
            columns=["source_row_uid", "level", "semantic_bucket_id", "weight"],
        ).to_pylist():
            key = (str(row["source_row_uid"]), str(row["level"]))
            bucket = (str(row["level"]), str(row["semantic_bucket_id"]))
            if (
                key not in rows
                or rows[key]["semantic_bucket_id"] != bucket[1]
                or rows[key]["semantic_weight"] is not None
            ):
                raise ValueError(f"Skin v6 ranked record has no matching bucket: {key}")
            weight = weights.get(bucket)
            if weight is None or weight != float(row["weight"]):
                raise ValueError(f"Skin v6 ranked record disagrees with bucket weight: {key}")
            rows[key]["semantic_weight"] = weight
        return rows, {
            "release": release,
            "release_manifest": str(artifacts.manifest),
            "release_manifest_sha256": sha256_file(artifacts.manifest),
            "evidence_input_binding": str(binding_path),
            "evidence_input_binding_sha256": sha256_file(binding_path),
            **{key: str(path) for key, path in sources.items()},
            **{f"{key}_sha256": manifest["files"][key]["sha256"] for key in sources},
            "weighted_records": sum(row["semantic_weight"] is not None for row in rows.values()),
            "unweighted_records": sum(row["semantic_weight"] is None for row in rows.values()),
        }
    if release != "v10_main_universe_v5":
        raise ValueError(f"Unsupported Skin semantic release: {release}")
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


def _load_reviewed_semantics(task: str, release: str) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    return load_reviewed_record_weights(task, release)
def _load_indirect(
    task: str, queries: Mapping[str, str], *, benchmark: str = "gold_v1",
    subset: str = "valid_small", historical: bool = False,
    skin_semantic_release: str = "v10_main_universe_v5",
    cache_bundle: Path | None = None,
    budget: int = INDIRECT_BUDGET,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    levels = TASKS[task]["levels"]
    if historical and cache_bundle is not None:
        raise ValueError("Historical indirect selection cannot override its cache")
    if historical:
        release_index = _default_release_index(task)
        config = None
    elif cache_bundle is not None:
        config = cache_bundle.resolve()
    elif benchmark == "gold_v1":
        config = Path("predict/retrieval/assay_reranking/ranked_level_retrieval_gold_v1_all_tasks_v1.yaml")
    elif benchmark == "tdc_v1":
        name = ("ranked_level_retrieval_tdc_v1_v27_successors_v1.yaml"
                if task in {"ames", "dili", "carcinogens"}
                else "ranked_level_retrieval_tdc_v1_task_best_l1_v1.yaml")
        config = Path("predict/retrieval/assay_reranking") / name
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if config is not None:
        bundle = yaml.safe_load(config.read_text(encoding="utf-8"))
        release_index = (config.parent / bundle["caches"][task]["later"]).resolve()
    ranked, _, cache_audit = load_ranked_universe(
        release_index, task=task,
        subset="valid" if subset == "valid_small" else subset,
        levels=levels, queries=queries,
    )
    partial_snapshot = False
    if not historical:
        index = json.loads(release_index.read_text(encoding="utf-8"))
        evidence_manifest = str(index["evidence"]["manifest"])
        expected_release = "v10_main_universe_v5" if task == "skin_reaction" else "v10_main_universe_v3"
        if f"/{expected_release}/" not in evidence_manifest:
            raise ValueError(f"Unexpected evidence release for semantic assignments: {evidence_manifest}")
        partial_snapshot = index.get("score_coverage") == "partial_snapshot"
    if task == "skin_reaction":
        semantics, semantic_audit = _load_skin_semantics(skin_semantic_release)
    elif historical:
        semantics, semantic_audit = _semantic_rows(task)
    else:
        semantics, semantic_audit = _load_reviewed_semantics(task, "v10_main_universe_v3")
    output = {query_id: [] for query_id in queries}
    missing, unscored = [], 0
    for level in levels:
        for query_id, rows in ranked[level].items():
            for row in rows:
                if partial_snapshot and row.get("assay_transfer_score") is None:
                    unscored += 1
                    continue
                uid = str(row["item_id"])
                semantic = semantics.get((uid, level))
                if semantic is None:
                    missing.append((uid, level))
                    continue
                output[query_id].append({**row, **semantic, "level": level})
    if missing and (cache_bundle is None or task not in {"ames", "dili", "carcinogens"}):
        raise ValueError(f"Semantic assignments omit indirect candidates: {missing[:5]}")
    for query_id, rows in output.items():
        if len(rows) < budget:
            raise ValueError(f"{query_id} has only {len(rows)} eligible L2+ records")
        uids = [str(row["item_id"]) for row in rows]
        if len(uids) != len(set(uids)):
            raise ValueError(f"Joint L2+ universe repeats UIDs: {query_id}")
    return output, {
        **({"cache_bundle": str(config.resolve()), "cache_bundle_sha256": sha256_file(config)}
           if config is not None else {}),
        "later_release_index": str(release_index),
        "later_release_index_sha256": sha256_file(release_index),
        "excluded_unscored_candidates": unscored,
        "excluded_unreviewed_candidates": len(missing),
        "cache": cache_audit, "semantics": semantic_audit,
    }


def _profile_root(output_root: Path, kind: str, profile: str, task: str) -> Path:
    path = output_root / kind / profile / task
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_direct(
    output_root: Path, tasks: Sequence[str], *, subset: str = "valid_small",
    profile_names: Sequence[str] | None = None, benchmark: str = "gold_v1",
) -> Path:
    profiles = direct_profiles()
    if profile_names:
        available = {profile.name: profile for profile in profiles}
        for name in profile_names:
            if name not in available:
                match = re.fullmatch(r"ga(\d{3})_mc(\d{3})_label(\d{3})", name)
                if match is None:
                    raise ValueError(f"Unknown direct profile: {name}")
                available[name] = DirectProfile(name, *(int(value) / 100 for value in match.groups()))
        profiles = [available[name] for name in dict.fromkeys(profile_names)]
    manifests = {}
    profiles_by_task = DIRECT_TASKS if benchmark == "gold_v1" else TDC_DIRECT_TASKS
    for task in tasks:
        queries, query_path = _direct_queries(benchmark, task, subset)
        release_index = cache_profile_root(profiles_by_task[task]) / task / "RELEASE_INDEX.json"
        universes, cache_audit = _load_ranked_direct(
            task, queries, release_index, subset
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
                "benchmark": benchmark,
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
        "schema_version": DIRECT_GRID_SCHEMA,
        "status": "complete",
        "kind": "direct",
        "benchmark": benchmark,
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
    item: tuple[str, list[dict[str, Any]], list[IndirectProfile], str, int],
) -> tuple[str, list[tuple[str, list[dict[str, Any]], dict[str, Any]]]]:
    query_id, rows, profiles, diversity_function, budget = item
    prepared = _prepare_candidates(rows, direct=False, budget=budget)
    results = []
    for profile in profiles:
        selected, summary = select_joint_indirect(
            rows, profile=profile, budget=budget, prepared=prepared,
            diversity_function=diversity_function,
        )
        results.append((profile.name, selected, summary))
    return query_id, results


def build_indirect(
    output_root: Path, tasks: Sequence[str], *, workers: int = 1,
    benchmark: str = "gold_v1", subset: str = "valid_small", grid: str = "historical",
    skin_semantic_release: str = "v10_main_universe_v5",
    profile_names: Sequence[str] | None = None,
    cache_bundle: Path | None = None,
    budget: int = INDIRECT_BUDGET,
) -> Path:
    if workers < 1 or budget < 1:
        raise ValueError("Workers and budget must be positive")
    if grid == "historical":
        if (benchmark, subset) != ("gold_v1", "valid_small"):
            raise ValueError("Historical grid is Gold valid_small only")
        profiles, aliases = indirect_profiles()
        function, objective_version = "log", INDIRECT_OBJECTIVE_VERSION
    elif grid == "sqrt48":
        if (benchmark, subset) not in {
            ("gold_v1", "valid_small"), ("gold_v1", "test"),
            ("tdc_v1", "valid"), ("tdc_v1", "test"),
        }:
            raise ValueError("Sqrt48 grid requires Gold valid_small/test or TDC valid/test")
        profiles, aliases = sqrt_indirect_profiles(), []
        function, objective_version = "sqrt", SQRT_INDIRECT_OBJECTIVE_VERSION
    else:
        raise ValueError(f"Unknown indirect grid: {grid}")
    if profile_names:
        available = {profile.name: profile for profile in profiles}
        for name in profile_names:
            if name not in available:
                match = re.fullmatch(
                    r"ga(\d{3})_mc(\d{3})_sr(\d{3})_sd(\d{3})_ld(\d{3})", name
                ) if grid == "sqrt48" else None
                if match is None:
                    raise ValueError(f"Unknown indirect profile: {name}")
                available[name] = IndirectProfile(
                    name, *(int(value) / 100 for value in match.groups()), ()
                )
        profiles = [available[name] for name in dict.fromkeys(profile_names)]
    if skin_semantic_release == "v10_main_universe_v6" and (grid != "sqrt48" or tuple(tasks) != ("skin_reaction",)):
        raise ValueError("Skin v6 weights require the sqrt48 grid and Skin-only task")
    manifests = {}
    for task in tasks:
        queries, query_path = _direct_queries(benchmark, task, subset)
        universes, input_audit = _load_indirect(
            task, queries, benchmark=benchmark, subset=subset,
            historical=grid == "historical",
            skin_semantic_release=skin_semantic_release,
            cache_bundle=cache_bundle, budget=budget,
        )
        selected_by_profile = {profile.name: [] for profile in profiles}
        diagnostics_by_profile = {profile.name: [] for profile in profiles}
        work = ((query_id, universes[query_id], profiles, function, budget) for query_id in queries)
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
                "benchmark": benchmark,
                "task_id": task,
                "subset": subset,
                "benchmark_row_ids": list(queries),
                "budget": budget,
                "profile": profile.name,
                "objective": {
                    "version": objective_version,
                    "gated_assay_lambda": profile.gated_assay,
                    "molecular_coverage_lambda": profile.molecular,
                    "semantic_relevance_lambda": profile.semantic_relevance,
                    "semantic_diversity_lambda": profile.semantic_diversity,
                    "level_diversity_lambda": profile.level_diversity,
                    "diversity_function": f"capacity_normalized_{function}.v1",
                    "molecular_diversity_function": "linear_morgan_bit_coverage.v1",
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
        "benchmark": benchmark,
        "subset": subset,
        "grid": grid,
        "skin_semantic_release": skin_semantic_release if "skin_reaction" in tasks else None,
        "objective_version": objective_version,
        "source_profile_count": len(aliases),
        "unique_source_profile_count": len(profiles) // (len(LEVEL_DIVERSITY_VALUES) if grid == "historical" else 1),
        "level_diversity_values": list(LEVEL_DIVERSITY_VALUES if grid == "historical" else (0.1, 0.25)),
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
    direct = validate_selection_manifest(direct_manifest)
    if direct["schema_version"] not in {DIRECT_SCHEMA, LEGACY_DIRECT_SCHEMA}:
        raise ValueError("Mixed selection requires a direct context manifest")
    indirect = validate_selection_manifest(indirect_manifest, INDIRECT_SCHEMA)
    for field in ("benchmark", "task_id", "subset", "benchmark_row_ids"):
        if direct.get(field) != indirect.get(field):
            raise ValueError(f"Direct and indirect selections disagree on {field}")
    document = {
        "schema_version": MIXED_SCHEMA,
        "status": "complete",
        "benchmark": direct.get("benchmark"),
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
    if schema == MIXED_SCHEMA:
        if document.get("status") != "complete":
            raise ValueError(f"Mixed selection is incomplete: {path}")
        for kind in ("direct", "indirect"):
            entry = document[kind]
            source = Path(entry["path"])
            if sha256_file(source) != entry["sha256"]:
                raise ValueError(f"Mixed {kind} manifest hash mismatch")
            child = validate_selection_manifest(source)
            for field in ("benchmark", "task_id", "subset", "benchmark_row_ids"):
                if child.get(field) != document.get(field):
                    raise ValueError(f"Mixed {kind} disagrees on {field}")
            if child["budget"] != document[f"{kind}_budget"]:
                raise ValueError(f"Mixed {kind} budget mismatch")
        return document
    direct_schemas = {DIRECT_SCHEMA, LEGACY_DIRECT_SCHEMA}
    if schema not in {*direct_schemas, INDIRECT_SCHEMA} or document.get("status") != "complete":
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
        if schema in direct_schemas else
        ["benchmark_row_id", "selection_rank", "level", "source_row_uid"]
    )
    if reader.fieldnames != expected_fields or len(rows) != int(records.get("row_count", -1)):
        raise ValueError("Selection records schema or row count is incompatible")
    query_ids = [str(value) for value in document.get("benchmark_row_ids") or []]
    budget = int(document.get("budget", 0))
    grouped = {query_id: [] for query_id in query_ids}
    identity_field = "context_id" if schema in direct_schemas else "source_row_uid"
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
        if command == "indirect":
            subparser.add_argument("--tasks", nargs="+", choices=tuple(TASKS),
                                   default=["bbb_martins", "bioavailability_ma", "skin_reaction"])
            subparser.add_argument("--workers", type=int, default=1)
            subparser.add_argument("--benchmark", choices=("gold_v1", "tdc_v1"), default="gold_v1")
            subparser.add_argument("--subset", choices=("valid_small", "valid", "test"), default="valid_small")
            subparser.add_argument("--grid", choices=("historical", "sqrt48"), default="historical")
            subparser.add_argument("--profiles", nargs="+")
            subparser.add_argument("--budget", type=int, default=INDIRECT_BUDGET)
            subparser.add_argument("--assay-transfer-cache", type=Path)
            subparser.add_argument(
                "--skin-semantic-release",
                choices=("v10_main_universe_v5", "v10_main_universe_v6"),
                default="v10_main_universe_v5",
            )
        else:
            subparser.add_argument(
                "--benchmark", choices=("gold_v1", "tdc_v1"), default="gold_v1",
            )
            subparser.add_argument("--tasks", nargs="+")
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
        available = DIRECT_TASKS if args.benchmark == "gold_v1" else TDC_DIRECT_TASKS
        tasks = args.tasks or list(available)
        unknown = set(tasks) - set(available)
        if unknown:
            parser.error(f"unsupported {args.benchmark} direct tasks: {sorted(unknown)}")
        print(build_direct(
            args.output_root.resolve(), tasks, benchmark=args.benchmark,
            subset=args.subset, profile_names=args.profiles,
        ))
    elif args.command == "indirect":
        print(build_indirect(
            args.output_root.resolve(), args.tasks, workers=args.workers,
            benchmark=args.benchmark, subset=args.subset, grid=args.grid,
            skin_semantic_release=args.skin_semantic_release,
            profile_names=args.profiles,
            cache_bundle=args.assay_transfer_cache,
            budget=args.budget,
        ))
    elif args.command == "compose":
        print(compose_mixed(args.direct_manifest, args.indirect_manifest, args.output))
    else:
        document = validate_selection_manifest(args.manifest)
        print(f"{args.manifest.resolve()}\t{document['schema_version']}\tcomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
