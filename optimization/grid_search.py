"""Select and screen the full record-selection lambda grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from rdkit import rdBase
from scipy.optimize import Bounds, LinearConstraint, milp

from optimization.select_records import (
    GATED_OBJECTIVE_VERSION,
    GOLD_TASKS,
    NORMALIZED_GATED_OBJECTIVE_VERSION,
    OBJECTIVE_VERSION,
    TASK_LEVELS,
    UID_MANIFEST_SCHEMA,
    Profile,
    _attach_semantics,
    _default_release_index,
    _semantic_rows,
    select_records,
)
from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_ranked_universe
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic


GRID_SCHEMA = "record_selection_lambda_grid.v1"
SCREEN_SCHEMA = "record_selection_lambda_screen.v1"
ASSAY_VALUES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
MOLECULAR_VALUES = (0.0, 0.05, 0.1, 0.25)
SEMANTIC_RELEVANCE_VALUES = (0.0, 0.1, 0.25, 0.5)
SEMANTIC_DIVERSITY_VALUES = (0.0, 0.05, 0.1, 0.25)
GATED_ASSAY_VALUES = (0.0, 0.1, 0.25, 0.5)
GATED_MOLECULAR_VALUES = (0.0, 0.15, 0.25)
GATED_SEMANTIC_VALUES = (0.0, 0.15, 0.25)
FIXED_GATED_VALUES = (0.0, 0.25, 0.5)
FIXED_GATED_ASSAY = 1.0
FIXED_GATED_OVERLAP = 0.8
NORMALIZED_GATED_ASSAY_VALUES = (0.75, 1.0, 1.25)
NORMALIZED_MOLECULAR_VALUES = (0.25, 0.5, 0.75)
NORMALIZED_SEMANTIC_VALUES = (0.05, 0.1)
NORMALIZED_HIGH_SEMANTIC_VALUES = (0.25, 0.5)
COMPONENTS = (
    "morgan",
    "assay",
    "molecular_coverage",
    "semantic_relevance",
    "semantic_coverage",
)
GATED_COMPONENTS = ("morgan", "assay", "assay_gated", *COMPONENTS[2:])
NORMALIZED_COMPONENTS = (
    "normalized_relevance",
    "molecular_coverage_normalized",
    "semantic_relevance",
    "semantic_coverage",
)


def _code(value: float) -> str:
    return f"{round(value * 100):03d}"


def profile_name(assay: float, molecular: float, relevance: float, diversity: float) -> str:
    return f"a{_code(assay)}_mc{_code(molecular)}_sr{_code(relevance)}_sd{_code(diversity)}"


def grid_profiles() -> list[Profile]:
    return [
        Profile(profile_name(a, mc, sr, sd), a, mc, sr, sd)
        for a, mc, sr, sd in itertools.product(
            ASSAY_VALUES,
            MOLECULAR_VALUES,
            SEMANTIC_RELEVANCE_VALUES,
            SEMANTIC_DIVERSITY_VALUES,
        )
    ]


def gated_grid_profiles() -> list[Profile]:
    return [
        Profile(
            f"a{_code(a)}_ga{_code(ga)}_mc{_code(mc)}_sr{_code(sr)}_sd{_code(sd)}",
            assay_lambda=a,
            gated_assay_lambda=ga,
            molecular_lambda=mc,
            semantic_relevance_lambda=sr,
            semantic_diversity_lambda=sd,
        )
        for a, ga, mc, sr, sd in itertools.product(
            (0.0, 0.1, 0.25),
            GATED_ASSAY_VALUES,
            GATED_MOLECULAR_VALUES,
            GATED_SEMANTIC_VALUES,
            GATED_SEMANTIC_VALUES,
        )
    ]


def fixed_gated_grid_profiles() -> list[Profile]:
    return [
        Profile(
            f"a{_code(a)}_ga100_mc{_code(mc)}_sr{_code(sr)}_sd{_code(sd)}",
            assay_lambda=a,
            gated_assay_lambda=FIXED_GATED_ASSAY,
            molecular_lambda=mc,
            semantic_relevance_lambda=sr,
            semantic_diversity_lambda=sd,
        )
        for a, mc, sr, sd in itertools.product(FIXED_GATED_VALUES, repeat=4)
    ]


def fixed_gated_anchor_names() -> list[str]:
    return [
        "a000_ga100_mc025_sr000_sd025",  # prior best profile, with the fixed gate
        "a000_ga100_mc000_sr000_sd000",
        "a050_ga100_mc000_sr000_sd000",
        "a000_ga100_mc050_sr000_sd000",
        "a000_ga100_mc000_sr050_sd000",
        "a000_ga100_mc000_sr000_sd050",
    ]


def normalized_gated_grid_profiles(
    semantic_values: Sequence[float] = NORMALIZED_SEMANTIC_VALUES,
) -> list[Profile]:
    crossed = [
        Profile(
            f"ga{_code(ga)}_mc{_code(mc)}_sr{_code(sr)}_sd{_code(sd)}",
            gated_assay_lambda=ga,
            molecular_lambda=mc,
            semantic_relevance_lambda=sr,
            semantic_diversity_lambda=sd,
        )
        for ga, mc, sr, sd in itertools.product(
            NORMALIZED_GATED_ASSAY_VALUES,
            NORMALIZED_MOLECULAR_VALUES,
            semantic_values,
            semantic_values,
        )
    ]
    controls = [
        Profile(f"ga{_code(ga)}_mc000_sr000_sd000", gated_assay_lambda=ga)
        for ga in NORMALIZED_GATED_ASSAY_VALUES
    ]
    return [*crossed, *controls]


def anchor_names() -> list[str]:
    values = [
        (0, 0, 0, 0),
        *((value, 0, 0, 0) for value in ASSAY_VALUES[1:]),
        (0, max(MOLECULAR_VALUES), 0, 0),
        (0, 0, max(SEMANTIC_RELEVANCE_VALUES), 0),
        (0, 0, 0, max(SEMANTIC_DIVERSITY_VALUES)),
    ]
    return [profile_name(*value) for value in values]


_WORK_UNIVERSES: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}
_WORK_PROFILES: Sequence[Profile] = ()
_WORK_K = 10
_WORK_NORMALIZED_GATED = False


def _initialize_worker(
    universes: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    profiles: Sequence[Profile],
    k: int,
    normalized_gated: bool,
) -> None:
    global _WORK_UNIVERSES, _WORK_PROFILES, _WORK_K, _WORK_NORMALIZED_GATED
    _WORK_UNIVERSES, _WORK_PROFILES, _WORK_K = universes, profiles, k
    _WORK_NORMALIZED_GATED = normalized_gated


def _select_query_level(key: tuple[str, str]) -> tuple[str, str, list[dict[str, Any]]]:
    query_id, level = key
    candidates = _WORK_UNIVERSES[level][query_id]
    output = []
    for profile in _WORK_PROFILES:
        selected, summary = select_records(
            candidates,
            k=_WORK_K,
            assay_lambda=profile.assay_lambda,
            gated_assay_lambda=profile.gated_assay_lambda,
            molecular_lambda=profile.molecular_lambda,
            semantic_relevance_lambda=profile.semantic_relevance_lambda,
            semantic_diversity_lambda=profile.semantic_diversity_lambda,
            normalized_gated_objective=_WORK_NORMALIZED_GATED,
        )
        output.append({
            "profile": profile.name,
            "selected": [str(row["item_id"]) for row in selected],
            "summary": summary,
        })
    return query_id, level, output


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return all(left[key] >= right[key] for key in COMPONENTS) and any(
        left[key] > right[key] for key in COMPONENTS
    )


def pareto_layers(names: Sequence[str], components: Mapping[str, Mapping[str, float]]) -> list[list[str]]:
    remaining = set(names)
    layers = []
    while remaining:
        layer = sorted(
            name for name in remaining
            if not any(
                _dominates(components[other], components[name])
                for other in remaining if other != name
            )
        )
        if not layer:
            raise RuntimeError("Pareto screening made no progress")
        layers.append(layer)
        remaining.difference_update(layer)
    return layers


def _normalized_vectors(
    names: Sequence[str], components: Mapping[str, Mapping[str, float]]
) -> dict[str, tuple[float, ...]]:
    bounds = {
        key: (
            min(components[name][key] for name in names),
            max(components[name][key] for name in names),
        )
        for key in COMPONENTS
    }
    return {
        name: tuple(
            0.0 if bounds[key][0] == bounds[key][1]
            else (components[name][key] - bounds[key][0]) / (bounds[key][1] - bounds[key][0])
            for key in COMPONENTS
        )
        for name in names
    }


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def _farthest_fill(
    candidates: Iterable[str],
    selected: list[str],
    vectors: Mapping[str, Sequence[float]],
    count: int,
) -> list[str]:
    pool = set(candidates).difference(selected)
    while pool and len(selected) < count:
        choice = min(
            pool,
            key=lambda name: (
                -min(_distance(vectors[name], vectors[prior]) for prior in selected),
                name,
            ),
        )
        selected.append(choice)
        pool.remove(choice)
    return selected


def screen_profiles(
    profiles: Sequence[Profile],
    signatures: Mapping[str, str],
    components: Mapping[str, Mapping[str, float]],
    *,
    count: int = 32,
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    by_signature: dict[str, list[Profile]] = {}
    for profile in profiles:
        by_signature.setdefault(signatures[profile.name], []).append(profile)
    aliases: dict[str, str] = {}
    representatives = []
    for members in by_signature.values():
        representative = min(
            members,
            key=lambda profile: (sum(profile.lambdas().values()), profile.name),
        )
        representatives.append(representative.name)
        aliases.update({member.name: representative.name for member in members})
    representatives.sort()
    if len(representatives) < count:
        raise ValueError(f"Only {len(representatives)} distinct panels; cannot select {count}")

    required = []
    for anchor in anchor_names():
        representative = aliases[anchor]
        if representative not in required:
            required.append(representative)
    if len(required) > count:
        raise ValueError("Distinct anchor panels exceed screen size")

    vectors = _normalized_vectors(representatives, components)
    ranks: dict[str, int] = {}
    selected = list(required)
    for rank, layer in enumerate(pareto_layers(representatives, components), start=1):
        ranks.update({name: rank for name in layer})
        if len(selected) >= count:
            continue
        _farthest_fill(layer, selected, vectors, count)
    _farthest_fill(representatives, selected, vectors, count)
    return selected, aliases, ranks


def _selected_uid_panels(
    path: Path, profile_names: Sequence[str], k: int
) -> dict[str, dict[tuple[str, str, str], frozenset[str]]]:
    rows: dict[str, dict[tuple[str, str, str], list[str]]] = {
        name: {} for name in profile_names
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row["profile"]
            if name not in rows:
                raise ValueError(f"Unexpected profile in UID selections: {name}")
            panel = (row["task"], row["benchmark_row_id"], row["level"])
            rows[name].setdefault(panel, []).append(row["source_row_uid"])

    expected_panels = set(rows[profile_names[0]])
    output: dict[str, dict[tuple[str, str, str], frozenset[str]]] = {}
    for name in profile_names:
        if set(rows[name]) != expected_panels:
            raise ValueError(f"UID panels differ for profile {name}")
        output[name] = {}
        for panel, uids in rows[name].items():
            if len(uids) != k or len(set(uids)) != k:
                raise ValueError(f"Profile {name} panel {panel} does not contain {k} unique UIDs")
            output[name][panel] = frozenset(uids)
    return output


def screen_by_uid_overlap(
    profiles: Sequence[Profile],
    panels: Mapping[str, Mapping[tuple[str, str, str], frozenset[str]]],
    components: Mapping[str, Mapping[str, float]],
    anchors: Sequence[str],
    *,
    k: int,
    threshold: float = FIXED_GATED_OVERLAP,
    profiles_per_assay_stratum: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select a maximum independent set, then prefer Morgan and assay quality."""
    names = [profile.name for profile in profiles]
    if not 0 <= threshold <= 1 or not names or len(names) != len(set(names)):
        raise ValueError("Overlap screening requires unique profiles and a unit threshold")
    if profiles_per_assay_stratum is not None and profiles_per_assay_stratum < 1:
        raise ValueError("Profiles per assay stratum must be positive")
    if not set(anchors).issubset(names):
        raise ValueError("Every overlap-screen anchor must be in the grid")
    panel_keys = set(panels[names[0]])
    if not panel_keys or any(set(panels[name]) != panel_keys for name in names):
        raise ValueError("Every profile must contain the same non-empty UID panels")

    anchor_set = set(anchors)
    pair_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []
    for left_index, left in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right = names[right_index]
            overlap = sum(
                len(panels[left][panel] & panels[right][panel]) / k
                for panel in panel_keys
            ) / len(panel_keys)
            conflict = overlap > threshold
            exception = conflict and left in anchor_set and right in anchor_set
            row = {
                "left_profile": left,
                "right_profile": right,
                "average_exact_uid_overlap": overlap,
                "conflict": conflict,
                "anchor_exception": exception,
            }
            pair_rows.append(row)
            if conflict:
                edge_rows.append(row)
                if not exception:
                    edges.append((left_index, right_index))

    size = len(names)
    base_constraints: list[tuple[np.ndarray, float, float]] = []
    for left, right in edges:
        row = np.zeros(size)
        row[left] = row[right] = 1
        base_constraints.append((row, -np.inf, 1))
    for anchor in anchors:
        row = np.zeros(size)
        row[names.index(anchor)] = 1
        base_constraints.append((row, 1, 1))

    def solve(
        objective: np.ndarray,
        extra: Sequence[tuple[np.ndarray, float, float]] = (),
    ) -> Any:
        constraints = [*base_constraints, *extra]
        matrix = np.stack([row for row, _, _ in constraints])
        result = milp(
            c=objective,
            integrality=np.ones(size),
            bounds=Bounds(np.zeros(size), np.ones(size)),
            constraints=LinearConstraint(
                matrix,
                np.array([lower for _, lower, _ in constraints]),
                np.array([upper for _, _, upper in constraints]),
            ),
        )
        if not result.success:
            raise RuntimeError(f"Overlap graph optimization failed: {result.message}")
        return result

    assay_values = sorted({profile.assay_lambda for profile in profiles})
    stratum_rows = {
        value: np.array([profile.assay_lambda == value for profile in profiles], dtype=float)
        for value in assay_values
    }
    balance_constraints = [
        (stratum_rows[left] - stratum_rows[right], 0, 0)
        for left, right in itertools.combinations(assay_values, 2)
    ]
    unconstrained_maximum = solve(-np.ones(size))
    unconstrained_maximum_size = round(float(sum(unconstrained_maximum.x)))
    target_constraints = [
        (row, profiles_per_assay_stratum, profiles_per_assay_stratum)
        for row in stratum_rows.values()
    ] if profiles_per_assay_stratum is not None else balance_constraints
    balanced_maximum = solve(
        np.zeros(size) if profiles_per_assay_stratum is not None else -np.ones(size),
        target_constraints,
    )
    maximum_size = round(float(sum(balanced_maximum.x)))
    selected_counts = tuple(
        round(float(stratum_rows[value] @ balanced_maximum.x))
        for value in assay_values
    )
    fixed_constraints = [
        *target_constraints,
        (np.ones(size), maximum_size, maximum_size),
    ]

    def normalized_component(component: str) -> np.ndarray:
        values = np.zeros(size)
        for assay in assay_values:
            indices = [
                index for index, profile in enumerate(profiles)
                if profile.assay_lambda == assay
            ]
            raw = [float(components[names[index]][component]) for index in indices]
            lower, upper = min(raw), max(raw)
            if upper > lower:
                for index, value in zip(indices, raw):
                    values[index] = (value - lower) / (upper - lower)
        return values

    morgan = normalized_component("morgan")
    morgan_result = solve(-morgan, fixed_constraints)
    morgan_optimum = float(morgan @ morgan_result.x)
    quality_constraints = [*fixed_constraints, (morgan, morgan_optimum - 1e-9, np.inf)]
    assay = normalized_component("assay")
    assay_result = solve(-assay, quality_constraints)
    assay_optimum = float(assay @ assay_result.x)
    final_constraints = [
        *quality_constraints, (assay, assay_optimum - 1e-9, np.inf)
    ]
    for index in sorted(range(size), key=lambda value: names[value]):
        row = np.zeros(size)
        row[index] = 1
        try:
            solve(np.zeros(size), [*final_constraints, (row, 1, 1)])
        except RuntimeError:
            final_constraints.append((row, 0, 0))
        else:
            final_constraints.append((row, 1, 1))
    final = solve(np.zeros(size), final_constraints)
    selected = [name for name, value in zip(names, final.x) if value > 0.5]

    frontier = []
    for limit in range(unconstrained_maximum_size + 1):
        balance_constraints = []
        for left, right in itertools.combinations(assay_values, 2):
            difference = stratum_rows[left] - stratum_rows[right]
            balance_constraints.append((difference, -limit, limit))
        result = solve(-np.ones(size), balance_constraints)
        frontier.append({
            "maximum_stratum_spread": limit,
            "maximum_profiles": round(float(sum(result.x))),
            **{
                f"assay_{_code(value)}_profiles": round(float(stratum_rows[value] @ result.x))
                for value in assay_values
            },
        })
        if frontier[-1]["maximum_profiles"] == unconstrained_maximum_size:
            break

    metadata = {
        "threshold": threshold,
        "panel_count": len(panel_keys),
        "unconstrained_maximum_size": unconstrained_maximum_size,
        "maximum_size": maximum_size,
        "profiles_per_assay_stratum": profiles_per_assay_stratum,
        "selected_stratum_counts": {
            str(value): count for value, count in zip(assay_values, selected_counts)
        },
        "anchor_exceptions": [row for row in edge_rows if row["anchor_exception"]],
        "balance_frontier": frontier,
    }
    return selected, pair_rows, edge_rows, metadata


def _task_inputs(
    task: str, *, query_set: str = "valid_small"
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    evaluation_subset = "test" if query_set == "test" else "valid"
    query_path = (
        Path("data/gold_labels")
        / GOLD_TASKS[task]
        / "v1/scaffold"
        / f"{query_set}.jsonl"
    ).resolve()
    release_index = _default_release_index(task).resolve()
    query_rows = read_jsonl(query_path)
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in query_rows}
    if not queries or len(queries) != len(query_rows):
        raise ValueError(f"{task} {query_set} requires unique benchmark rows")
    levels = TASK_LEVELS[task]
    ranked, _, cache_audit = load_ranked_universe(
        release_index,
        task=task,
        subset=evaluation_subset,
        levels=levels,
        queries=queries,
    )
    semantics, semantic_audit = _semantic_rows(task)
    universes = {
        level: {
            query_id: _attach_semantics(rows, level=level, semantics=semantics)
            for query_id, rows in by_query.items()
        }
        for level, by_query in ranked.items()
    }
    audit = {
        "query_path": str(query_path),
        "query_sha256": sha256_file(query_path),
        "evaluation_subset": evaluation_subset,
        "release_index": str(release_index),
        "release_index_sha256": sha256_file(release_index),
        "cache": {key: value for key, value in cache_audit.items() if key != "query_identities"},
        "semantics": semantic_audit,
    }
    return list(queries), universes, audit


def _write_selected_manifests(
    root: Path,
    selected_names: Sequence[str],
    profiles: Mapping[str, Profile],
    selected_path: Path,
    diagnostics_path: Path,
    task_audits: Mapping[str, Mapping[str, Any]],
    query_ids: Mapping[str, Sequence[str]],
    k: int,
    objective_version: str,
    evaluation_subset: str,
) -> dict[str, dict[str, str]]:
    selected_set = set(selected_names)
    records = {
        (profile, task): [] for profile in selected_names for task in TASK_LEVELS
    }
    diagnostics = {
        (profile, task): [] for profile in selected_names for task in TASK_LEVELS
    }
    with selected_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["profile"] in selected_set:
                records[(row["profile"], row["task"])].append(row)
    with diagnostics_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["profile"] in selected_set:
                diagnostics[(row["profile"], row["task"])].append(row)

    paths: dict[str, dict[str, str]] = {name: {} for name in selected_names}
    record_fields = ("benchmark_row_id", "level", "selection_rank", "source_row_uid")
    diagnostic_fields = (
        "benchmark_row_id", "level", "candidate_records", "selected_records",
        "distinct_selected_parents", "distinct_selected_semantic_buckets",
        "assay_available", "assay_lambda_effective", "gated_assay_lambda_effective",
        "morgan", "assay", "assay_gated", "molecular_coverage",
        "normalized_relevance", "molecular_coverage_ceiling",
        "molecular_coverage_normalized",
        "semantic_relevance", "semantic_coverage",
        "molecular_diversity", "semantic_diversity", "objective_score",
    )
    for name in selected_names:
        for task, levels in TASK_LEVELS.items():
            output = root / "selected" / name / task
            output.mkdir(parents=True)
            selected_path = output / "selected_uids.tsv"
            with selected_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=record_fields, delimiter="\t")
                writer.writeheader()
                writer.writerows({key: row[key] for key in record_fields} for row in records[(name, task)])
            diagnostics_path = output / "selection_diagnostics.tsv"
            with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=diagnostic_fields, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(diagnostics[(name, task)])
            manifest = {
                "schema_version": UID_MANIFEST_SCHEMA,
                "status": "complete",
                "task_id": task,
                "subset": evaluation_subset,
                "profile": name,
                "benchmark_row_ids": list(query_ids[task]),
                "k_per_level": {level: k for level in levels},
                "release_index": task_audits[task]["release_index"],
                "release_index_sha256": task_audits[task]["release_index_sha256"],
                "objective": {
                    "version": objective_version,
                    "formula": (
                        "(M + alpha*MGA)/(1 + alpha) + lambda_mc*Cmol_normalized + lambda_sr*Srel + lambda_sd*Csem"
                        if objective_version == NORMALIZED_GATED_OBJECTIVE_VERSION
                        else "M + lambda_a*A + lambda_ga*MGA + lambda_mc*Cmol + lambda_sr*Srel + lambda_sd*Csem"
                        if objective_version == GATED_OBJECTIVE_VERSION
                        else "M + lambda_a*A + lambda_mc*Cmol + lambda_sr*Srel + lambda_sd*Csem"
                    ),
                    "lambdas": profiles[name].lambdas(),
                    "optimizer": "apricot.CustomSelection:lazy",
                },
                "inputs": task_audits[task],
                "records": {
                    "path": selected_path.name,
                    "sha256": sha256_file(selected_path),
                    "row_count": len(records[(name, task)]),
                },
                "outputs": {diagnostics_path.name: sha256_file(diagnostics_path)},
                "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
            }
            manifest_path = output / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            paths[name][task] = str(manifest_path.relative_to(root))
    return paths


def run_grid(
    output_root: Path,
    *,
    workers: int,
    k: int = 10,
    profile_names: Sequence[str] | None = None,
    query_set: str = "valid_small",
    grid: str = "legacy",
    overlap_threshold: float = FIXED_GATED_OVERLAP,
    profiles_per_assay_stratum: int | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=False)
    fixed_gated = grid == "gated-fixed-81"
    normalized_gated = grid in (
        "normalized-gated-39", "normalized-gated-semantic-high-39"
    )
    gated = grid == "gated-324" or fixed_gated or normalized_gated
    grid_components = (
        NORMALIZED_COMPONENTS if normalized_gated
        else GATED_COMPONENTS if gated else COMPONENTS
    )
    objective_version = (
        NORMALIZED_GATED_OBJECTIVE_VERSION if normalized_gated
        else GATED_OBJECTIVE_VERSION if gated else OBJECTIVE_VERSION
    )
    available_profiles = (
        normalized_gated_grid_profiles(
            NORMALIZED_HIGH_SEMANTIC_VALUES
            if grid == "normalized-gated-semantic-high-39"
            else NORMALIZED_SEMANTIC_VALUES
        )
        if normalized_gated
        else fixed_gated_grid_profiles()
        if fixed_gated
        else gated_grid_profiles() if gated else grid_profiles()
    )
    all_profiles = {
        profile.name: profile
        for profile in available_profiles
    }
    profiles = (
        [all_profiles[name] for name in profile_names]
        if profile_names is not None
        else list(all_profiles.values())
    )
    profile_map = {profile.name: profile for profile in profiles}
    property_fields = (
        *grid_components,
        "molecular_diversity",
        "semantic_diversity",
        "distinct_selected_parents",
    )
    accumulators = {
        profile.name: {level: {key: 0.0 for key in property_fields} | {"count": 0.0}
                       for task in TASK_LEVELS for level in (f"{task}:{value}" for value in TASK_LEVELS[task])}
        for profile in profiles
    }
    signature_hashes = {profile.name: hashlib.sha256() for profile in profiles}
    task_audits, task_query_ids = {}, {}

    selected_path = output_root / "selected_uids.tsv"
    diagnostics_path = output_root / "selection_diagnostics.tsv"
    selected_handle = selected_path.open("w", encoding="utf-8", newline="")
    diagnostics_handle = diagnostics_path.open("w", encoding="utf-8", newline="")
    selected_fields = ("profile", "task", "benchmark_row_id", "level", "selection_rank", "source_row_uid")
    diagnostic_fields = (
        "profile", "task", "benchmark_row_id", "level", "candidate_records",
        "selected_records", "distinct_selected_parents",
        "distinct_selected_semantic_buckets", "assay_available",
        "assay_lambda_effective", "gated_assay_lambda_effective",
        "morgan", "assay", "assay_gated", "molecular_coverage",
        "normalized_relevance", "molecular_coverage_ceiling",
        "molecular_coverage_normalized",
        "semantic_relevance", "semantic_coverage", "molecular_diversity",
        "semantic_diversity", "objective_score",
    )
    selected_writer = csv.DictWriter(selected_handle, fieldnames=selected_fields, delimiter="\t")
    diagnostics_writer = csv.DictWriter(
        diagnostics_handle, fieldnames=diagnostic_fields, delimiter="\t", extrasaction="ignore"
    )
    selected_writer.writeheader()
    diagnostics_writer.writeheader()

    try:
        for task, levels in TASK_LEVELS.items():
            query_ids, universes, audit = _task_inputs(task, query_set=query_set)
            task_audits[task], task_query_ids[task] = audit, query_ids
            keys = [(query_id, level) for query_id in query_ids for level in levels]
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(universes, profiles, k, normalized_gated),
            ) as executor:
                for query_id, level, results in executor.map(_select_query_level, keys, chunksize=1):
                    for result in results:
                        name, summary = result["profile"], result["summary"]
                        for rank, uid in enumerate(result["selected"], start=1):
                            selected_writer.writerow({
                                "profile": name, "task": task,
                                "benchmark_row_id": query_id, "level": level,
                                "selection_rank": rank, "source_row_uid": uid,
                            })
                            signature_hashes[name].update(f"{task}\0{query_id}\0{level}\0{uid}\n".encode())
                        diagnostics_writer.writerow({
                            "profile": name, "task": task,
                            "benchmark_row_id": query_id, "level": level, **summary,
                        })
                        aggregate = accumulators[name][f"{task}:{level}"]
                        aggregate["count"] += 1
                        for component in property_fields:
                            aggregate[component] += float(summary[component])
    finally:
        selected_handle.close()
        diagnostics_handle.close()

    signatures = {name: digest.hexdigest() for name, digest in signature_hashes.items()}
    properties = {}
    for name, strata in accumulators.items():
        properties[name] = {
            component: sum(values[component] / values["count"] for values in strata.values()) / len(strata)
            for component in property_fields
        }
    overlap_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    screen_details: dict[str, Any] = {}
    if fixed_gated and profile_names is None:
        anchors = fixed_gated_anchor_names()
        selected_names, overlap_rows, edge_rows, screen_details = screen_by_uid_overlap(
            profiles,
            _selected_uid_panels(selected_path, list(profile_map), k),
            properties,
            anchors,
            k=k,
            threshold=overlap_threshold,
            profiles_per_assay_stratum=profiles_per_assay_stratum,
        )
        expected_per_stratum = profiles_per_assay_stratum or 6
        expected_counts = {
            "0.0": expected_per_stratum,
            "0.25": expected_per_stratum,
            "0.5": expected_per_stratum,
        }
        if (
            screen_details["panel_count"] != 900
            or screen_details["maximum_size"] != 3 * expected_per_stratum
            or screen_details["selected_stratum_counts"] != expected_counts
        ):
            raise RuntimeError(f"Fixed-gate overlap screen drifted: {screen_details}")
        aliases = {name: name for name in profile_map}
        ranks = {}
        selection_policy = "equal_assay_strata_then_maximum_independent_set_then_morgan_then_assay"
    elif normalized_gated and profile_names is None:
        selected_names = [profile.name for profile in profiles]
        aliases = {name: name for name in selected_names}
        ranks = {name: 1 for name in selected_names}
        selection_policy = "complete_factorial_plus_assay_only_controls"
        anchors = [
            profile.name for profile in profiles
            if profile.molecular_lambda == 0
            and profile.semantic_relevance_lambda == 0
            and profile.semantic_diversity_lambda == 0
        ]
    elif gated and profile_names is None:
        selected_names, aliases, ranks = [], {name: name for name in profile_map}, {}
        selection_policy = "properties_only_no_pruning"
        anchors = []
    elif profile_names is None:
        selected_names, aliases, ranks = screen_profiles(profiles, signatures, properties)
        selection_policy = "anchors_then_pareto_layers_with_farthest_point_fill"
        anchors = anchor_names()
    else:
        selected_names = [profile.name for profile in profiles]
        aliases = {name: name for name in selected_names}
        ranks = {name: 1 for name in selected_names}
        selection_policy = "explicit_profiles"
        anchors = []

    profiles_path = output_root / "profiles.tsv"
    with profiles_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "profile", "assay_lambda", "gated_assay_lambda", "molecular_coverage_lambda",
            "semantic_relevance_lambda", "semantic_diversity_lambda",
            "panel_signature", "representative", "pareto_layer", "selected",
            *property_fields,
        )
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for profile in profiles:
            writer.writerow({
                "profile": profile.name,
                "assay_lambda": profile.assay_lambda,
                "gated_assay_lambda": profile.gated_assay_lambda,
                "molecular_coverage_lambda": profile.molecular_lambda,
                "semantic_relevance_lambda": profile.semantic_relevance_lambda,
                "semantic_diversity_lambda": profile.semantic_diversity_lambda,
                "panel_signature": signatures[profile.name],
                "representative": aliases[profile.name],
                "pareto_layer": ranks.get(aliases[profile.name], ""),
                "selected": profile.name in selected_names,
                **properties[profile.name],
            })

    by_level_path = output_root / "profile_properties.by_task_level.tsv"
    with by_level_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("profile", "task", "level", "queries", *property_fields)
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for profile in profiles:
            for task, levels in TASK_LEVELS.items():
                for level in levels:
                    values = accumulators[profile.name][f"{task}:{level}"]
                    writer.writerow({
                        "profile": profile.name,
                        "task": task,
                        "level": level,
                        "queries": int(values["count"]),
                        **{
                            field: values[field] / values["count"]
                            for field in property_fields
                        },
                    })

    extra_outputs: list[Path] = []
    if fixed_gated and profile_names is None:
        overlap_path = output_root / "profile_pair_overlap.tsv"
        edge_path = output_root / "overlap_conflict_edges.tsv"
        frontier_path = output_root / "balance_frontier.tsv"
        summary_path = output_root / "screening_summary.tsv"
        report_path = output_root / "report.md"
        overlap_fields = (
            "left_profile", "right_profile", "average_exact_uid_overlap",
            "conflict", "anchor_exception",
        )
        for path, rows in ((overlap_path, overlap_rows), (edge_path, edge_rows)):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=overlap_fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
        frontier_fields = tuple(screen_details["balance_frontier"][0])
        with frontier_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=frontier_fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(screen_details["balance_frontier"])
        summary_rows = [
            ("profiles", len(profiles)),
            ("query_level_panels", screen_details["panel_count"]),
            ("overlap_threshold", screen_details["threshold"]),
            ("conflict_edges", len(edge_rows)),
            ("anchor_exceptions", len(screen_details["anchor_exceptions"])),
            ("unconstrained_maximum_profiles", screen_details["unconstrained_maximum_size"]),
            ("selected_profiles", len(selected_names)),
            *(
                (f"selected_assay_{_code(float(value))}", count)
                for value, count in screen_details["selected_stratum_counts"].items()
            ),
        ]
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("metric", "value"))
            writer.writerows(summary_rows)
        exceptions = screen_details["anchor_exceptions"]
        exception_text = (
            "Every selected pair satisfies the overlap threshold."
            if not exceptions else
            "Required anchor exceptions: " + "; ".join(
                f"`{row['left_profile']}` / `{row['right_profile']}` "
                f"({row['average_exact_uid_overlap']:.6f})"
                for row in exceptions
            ) + "."
        )
        count_text = " / ".join(
            str(count) for count in screen_details["selected_stratum_counts"].values()
        )
        report_path.write_text(
            "\n".join([
                "# Fixed-gate record-selection overlap screen",
                "",
                "This label-blind analysis screened 81 profiles on 900 valid-small "
                "query-level panels. The Morgan-gated assay coefficient is fixed at 1; "
                "all other coefficients use `{0, 0.25, 0.5}`.",
                "",
                f"The unconstrained maximum contains {screen_details['unconstrained_maximum_size']} "
                f"profiles. Equal additive-assay strata select {len(selected_names)} profiles: "
                f"{count_text} for 0 / 0.25 / 0.5. Selection maximizes Morgan quality "
                "before assay-transfer quality within those strata.",
                "",
                f"The overlap threshold is {screen_details['threshold']:.2f}. "
                + exception_text,
                "",
                "All numeric results are exported in the adjacent TSV files.",
                "",
            ]),
            encoding="utf-8",
        )
        extra_outputs = [overlap_path, edge_path, frontier_path, summary_path, report_path]

    task_manifests = _write_selected_manifests(
        output_root, selected_names, profile_map, selected_path, diagnostics_path,
        task_audits, task_query_ids, k, objective_version,
        "test" if query_set == "test" else "valid",
    )
    screen_path = output_root / "screen_manifest.json"
    write_json_atomic(screen_path, {
        "schema_version": SCREEN_SCHEMA,
        "status": "complete",
        "objective_version": objective_version,
        "grid_size": len(profiles),
        "screen_size": len(selected_names),
        "selection_policy": selection_policy,
        "anchors": anchors,
        "screen_details": screen_details,
        "selected_profiles": [
            {
                "name": name,
                "lambdas": profile_map[name].lambdas(),
                "panel_signature": signatures[name],
                **({"pareto_layer": ranks[name]} if name in ranks else {}),
                "task_manifests": {
                    task: {
                        "path": task_manifests[name][task],
                        "sha256": sha256_file(output_root / task_manifests[name][task]),
                    }
                    for task in TASK_LEVELS
                },
            }
            for name in selected_names
        ],
        "aliases": aliases,
        "outputs": {
            path.name: sha256_file(path)
            for path in (
                profiles_path, by_level_path, selected_path, diagnostics_path,
                *extra_outputs,
            )
        },
    })
    write_json_atomic(output_root / "manifest.json", {
        "schema_version": GRID_SCHEMA,
        "status": "complete",
        "profile_count": len(profiles),
        "grid": grid,
        "objective_version": objective_version,
        "query_count": sum(len(value) for value in task_query_ids.values()),
        "query_level_count": sum(len(task_query_ids[task]) * len(TASK_LEVELS[task]) for task in TASK_LEVELS),
        "k": k,
        "workers": workers,
        "query_set": query_set,
        "overlap_threshold": overlap_threshold if fixed_gated else None,
        "profiles_per_assay_stratum": (
            profiles_per_assay_stratum if fixed_gated else None
        ),
        "screen_manifest": {"path": screen_path.name, "sha256": sha256_file(screen_path)},
        "task_inputs": task_audits,
    })
    return screen_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--query-set", choices=("valid_small", "valid", "test"), default="valid_small"
    )
    parser.add_argument(
        "--grid",
        choices=(
            "legacy", "gated-324", "gated-fixed-81", "normalized-gated-39",
            "normalized-gated-semantic-high-39",
        ),
        default="legacy",
    )
    parser.add_argument("--overlap-threshold", type=float, default=FIXED_GATED_OVERLAP)
    parser.add_argument("--profiles-per-assay-stratum", type=int)
    parser.add_argument("--profiles", nargs="+")
    args = parser.parse_args(argv)
    if (args.workers < 1 or args.k < 1
            or not 0 <= args.overlap_threshold <= 1
            or (args.profiles_per_assay_stratum is not None
                and args.profiles_per_assay_stratum < 1)):
        parser.error("workers, K, overlap threshold, and stratum size must be valid")
    if args.grid != "gated-fixed-81" and (
        args.overlap_threshold != FIXED_GATED_OVERLAP
        or args.profiles_per_assay_stratum is not None
    ):
        parser.error("overlap screening options require --grid gated-fixed-81")
    print(run_grid(
        args.output_root.resolve(), workers=args.workers, k=args.k,
        profile_names=args.profiles, query_set=args.query_set, grid=args.grid,
        overlap_threshold=args.overlap_threshold,
        profiles_per_assay_stratum=args.profiles_per_assay_stratum,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
