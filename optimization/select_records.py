"""Optimize ordered evidence-record panels for the full-flat V5 harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from apricot import CustomSelection
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_ranked_universe
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from semantic_buckets.artifacts import resolve_semantic_bucket_artifacts


OBJECTIVE_VERSION = "morgan_assay_feature_semantic.v1"
GATED_OBJECTIVE_VERSION = "morgan_assay_gated_feature_semantic.v1"
NORMALIZED_GATED_OBJECTIVE_VERSION = "morgan_normalized_gated_feature_semantic.v1"
UID_MANIFEST_SCHEMA = "flat_preselected_uids.v1"
SEMANTIC_RELEASE = "v10_main_universe_v1"
TASK_LEVELS = {
    "bbb_martins": ("L2", "L3", "L4", "L5"),
    "bioavailability_ma": ("L2", "L3", "L4", "L5", "L6"),
}
GOLD_TASKS = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
}


@dataclass(frozen=True)
class Profile:
    name: str
    assay_lambda: float = 0.0
    molecular_lambda: float = 0.0
    semantic_relevance_lambda: float = 0.0
    semantic_diversity_lambda: float = 0.0
    gated_assay_lambda: float = 0.0

    def lambdas(self) -> dict[str, float]:
        return {
            "assay": self.assay_lambda,
            "assay_gated": self.gated_assay_lambda,
            "molecular_coverage": self.molecular_lambda,
            "semantic_relevance": self.semantic_relevance_lambda,
            "semantic_coverage": self.semantic_diversity_lambda,
        }


def _profiles() -> dict[str, Profile]:
    profiles = {"baseline": Profile("baseline")}
    fields = {
        "assay": "assay_lambda",
        "molecular": "molecular_lambda",
        "semantic_rel": "semantic_relevance_lambda",
        "semantic_div": "semantic_diversity_lambda",
    }
    for prefix, field in fields.items():
        for suffix, value in (("025", 0.25), ("05", 0.5), ("10", 1.0)):
            profiles[f"{prefix}_{suffix}"] = Profile(
                f"{prefix}_{suffix}", **{field: value}
            )
    for suffix, value in (("025", 0.25), ("05", 0.5), ("10", 1.0)):
        profiles[f"balanced_{suffix}"] = Profile(
            f"balanced_{suffix}", value, value, value, value
        )
    return profiles


PROFILES = _profiles()
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _finite_unit(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return number


@lru_cache(maxsize=None)
def _fingerprint(smiles: str) -> DataStructs.ExplicitBitVect:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid cached parent SMILES: {smiles!r}")
    return _MORGAN.GetFingerprint(molecule)


def _feature_coverage(selected: Iterable[Iterable[int]], universe: Iterable[int]) -> float:
    universe_bits = frozenset(universe)
    if not universe_bits:
        return 0.0
    covered: set[int] = set()
    for bits in selected:
        covered.update(bits)
    return len(covered) / len(universe_bits)


def _greedy_coverage_ceiling(
    fingerprint_bits: Sequence[frozenset[int]], universe_bits: frozenset[int], k: int
) -> float:
    """Return the deterministic K-item greedy ceiling used to rescale coverage."""
    remaining = list(dict.fromkeys(fingerprint_bits))
    covered: set[int] = set()
    for _ in range(min(k, len(remaining))):
        best = max(range(len(remaining)), key=lambda index: len(remaining[index] - covered))
        covered.update(remaining.pop(best))
    return len(covered) / len(universe_bits) if universe_bits else 0.0


def _objective_components(
    indices: Sequence[int],
    *,
    candidates: Sequence[Mapping[str, Any]],
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
    fingerprint_bits: Sequence[frozenset[int]],
    universe_feature_bits: frozenset[int],
    k: int,
    assay_available: bool,
    include_pairwise_diagnostic: bool = True,
) -> dict[str, float]:
    if not indices:
        return {
            "morgan": 0.0,
            "assay": 0.0,
            "assay_gated": 0.0,
            "molecular_coverage": 0.0,
            "molecular_diversity": 0.0,
            "semantic_relevance": 0.0,
            "semantic_coverage": 0.0,
            "semantic_diversity": 0.0,
        }
    rows = [candidates[index] for index in indices]
    pair_sum = (
        sum(
            1.0 - DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right])
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        )
        if include_pairwise_diagnostic else 0.0
    )
    pair_denominator = k * (k - 1) / 2
    unique_buckets = len({str(row["semantic_bucket_id"]) for row in rows})
    return {
        "morgan": sum(float(row["morgan_similarity"]) for row in rows) / k,
        "assay": (
            sum(float(row["assay_transfer_score"]) for row in rows) / k
            if assay_available else 0.0
        ),
        "assay_gated": (
            sum(
                float(row["morgan_similarity"]) * float(row["assay_transfer_score"])
                for row in rows
            ) / k
            if assay_available else 0.0
        ),
        "molecular_coverage": _feature_coverage(
            (fingerprint_bits[index] for index in indices),
            universe_feature_bits,
        ),
        "molecular_diversity": pair_sum / pair_denominator if pair_denominator else 0.0,
        "semantic_relevance": sum(float(row["semantic_weight"]) for row in rows) / k,
        "semantic_coverage": unique_buckets / k,
        "semantic_diversity": (
            (unique_buckets - 1) / (k - 1) if k > 1 else 1.0
        ),
    }


def select_records(
    candidates: Sequence[Mapping[str, Any]],
    *,
    k: int = 10,
    assay_lambda: float = 0.0,
    gated_assay_lambda: float = 0.0,
    molecular_lambda: float = 0.0,
    semantic_relevance_lambda: float = 0.0,
    semantic_diversity_lambda: float = 0.0,
    normalized_gated_objective: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily select exactly K UIDs without constructing an N-by-N matrix."""
    lambdas = {
        "assay": float(assay_lambda),
        "assay_gated": float(gated_assay_lambda),
        "molecular_coverage": float(molecular_lambda),
        "semantic_relevance": float(semantic_relevance_lambda),
        "semantic_coverage": float(semantic_diversity_lambda),
    }
    if k < 1 or k > len(candidates):
        raise ValueError("K must be positive and no greater than the candidate count")
    if any(not math.isfinite(value) or value < 0 for value in lambdas.values()):
        raise ValueError("Objective lambdas must be finite and non-negative")
    if normalized_gated_objective and assay_lambda:
        raise ValueError("The normalized gated objective does not use additive assay")

    rows = [dict(row) for row in candidates]
    item_ids = [str(row["item_id"]) for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Candidate source_row_uid values must be unique")
    parent_similarity: dict[str, float] = {}
    parent_fingerprints: dict[str, DataStructs.ExplicitBitVect] = {}
    fingerprints = []
    assay_present = []
    for row in rows:
        similarity = _finite_unit(row["morgan_similarity"], "Morgan similarity")
        weight = _finite_unit(row["semantic_weight"], "semantic weight")
        row["morgan_similarity"], row["semantic_weight"] = similarity, weight
        parent_id = str(row["parent_id"])
        if parent_id in parent_similarity and not math.isclose(
            parent_similarity[parent_id], similarity, abs_tol=1e-12
        ):
            raise ValueError(f"Morgan similarity varies within parent {parent_id}")
        parent_similarity[parent_id] = similarity
        if parent_id not in parent_fingerprints:
            parent_fingerprints[parent_id] = _fingerprint(str(row["parent_smiles"]))
        fingerprints.append(parent_fingerprints[parent_id])
        present = row.get("assay_transfer_score") is not None
        assay_present.append(present)
        if present:
            row["assay_transfer_score"] = _finite_unit(
                row["assay_transfer_score"], "assay-transfer score"
            )
        if not str(row.get("semantic_bucket_id") or ""):
            raise ValueError("Every candidate requires a semantic bucket")
    if any(assay_present) and not all(assay_present):
        raise ValueError("Assay-transfer scores must be complete or absent for a level")
    assay_available = all(assay_present)
    effective_lambdas = {
        **lambdas,
        "assay": lambdas["assay"] if assay_available else 0.0,
        "assay_gated": lambdas["assay_gated"] if assay_available else 0.0,
    }
    fingerprint_bits = [frozenset(fingerprint.GetOnBits()) for fingerprint in fingerprints]
    universe_feature_bits = frozenset(
        bit for bits in fingerprint_bits for bit in bits
    )
    coverage_ceiling = (
        _greedy_coverage_ceiling(fingerprint_bits, universe_feature_bits, k)
        if normalized_gated_objective else 1.0
    )
    morgan_scores = [float(row["morgan_similarity"]) for row in rows]
    assay_scores = [
        float(row["assay_transfer_score"]) if assay_available else 0.0 for row in rows
    ]
    semantic_weights = [float(row["semantic_weight"]) for row in rows]
    semantic_buckets = [str(row["semantic_bucket_id"]) for row in rows]

    def objective(subset: np.ndarray) -> float:
        indices = [int(value) for value in np.asarray(subset)[:, 0]]
        morgan = sum(morgan_scores[index] for index in indices) / k
        assay_gated = sum(
            morgan_scores[index] * assay_scores[index] for index in indices
        ) / k
        if normalized_gated_objective:
            alpha = effective_lambdas["assay_gated"]
            score = (morgan + alpha * assay_gated) / (1 + alpha)
        else:
            score = morgan
            score += effective_lambdas["assay"] * (
                sum(assay_scores[index] for index in indices) / k
            )
            score += effective_lambdas["assay_gated"] * assay_gated
        coverage = _feature_coverage(
            (fingerprint_bits[index] for index in indices), universe_feature_bits
        )
        if normalized_gated_objective and coverage_ceiling:
            coverage = min(1.0, coverage / coverage_ceiling)
        score += effective_lambdas["molecular_coverage"] * coverage
        score += effective_lambdas["semantic_relevance"] * (
            sum(semantic_weights[index] for index in indices) / k
        )
        score += effective_lambdas["semantic_coverage"] * (
            len({semantic_buckets[index] for index in indices}) / k
        )
        return score

    selector = CustomSelection(k, objective, optimizer="lazy", n_jobs=1)
    selector.fit(np.arange(len(rows), dtype=np.float64).reshape(-1, 1))
    ranking = [int(index) for index in selector.ranking]
    selected = []
    for selection_rank, (index, gain) in enumerate(zip(ranking, selector.gains), start=1):
        selected.append({
            **rows[index],
            "selection_rank": selection_rank,
            "marginal_gain": float(gain),
        })
    components = _objective_components(
        ranking,
        candidates=rows,
        fingerprints=fingerprints,
        fingerprint_bits=fingerprint_bits,
        universe_feature_bits=universe_feature_bits,
        k=k,
        assay_available=assay_available,
    )
    normalized_coverage = (
        min(1.0, components["molecular_coverage"] / coverage_ceiling)
        if normalized_gated_objective and coverage_ceiling
        else components["molecular_coverage"]
    )
    alpha = effective_lambdas["assay_gated"]
    normalized_relevance = (
        (components["morgan"] + alpha * components["assay_gated"]) / (1 + alpha)
        if normalized_gated_objective else components["morgan"]
    )
    objective_score = (
        normalized_relevance
        + effective_lambdas["molecular_coverage"] * normalized_coverage
        + effective_lambdas["semantic_relevance"] * components["semantic_relevance"]
        + effective_lambdas["semantic_coverage"] * components["semantic_coverage"]
        if normalized_gated_objective else
        components["morgan"] + sum(
            effective_lambdas[name] * components[name] for name in effective_lambdas
        )
    )
    return selected, {
        **components,
        "normalized_relevance": normalized_relevance,
        "molecular_coverage_ceiling": coverage_ceiling,
        "molecular_coverage_normalized": normalized_coverage,
        "objective_score": objective_score,
        "assay_available": assay_available,
        "assay_lambda_effective": effective_lambdas["assay"],
        "gated_assay_lambda_effective": effective_lambdas["assay_gated"],
        "candidate_records": len(rows),
        "selected_records": k,
        "distinct_selected_parents": len({str(row["parent_id"]) for row in selected}),
        "distinct_selected_semantic_buckets": len({
            str(row["semantic_bucket_id"]) for row in selected
        }),
    }


def _semantic_rows(task: str) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    artifacts = resolve_semantic_bucket_artifacts(task, SEMANTIC_RELEASE)
    weighting_manifest_path = artifacts.record_relevance_rankings.parent / "manifest.json"
    weighting_manifest = json.loads(weighting_manifest_path.read_text(encoding="utf-8"))
    filename = artifacts.record_relevance_rankings.name
    if (
        weighting_manifest.get("status") != "complete_reviewed"
        or weighting_manifest.get("task") != task
        or weighting_manifest.get("files", {}).get(filename)
        != sha256_file(artifacts.record_relevance_rankings)
    ):
        raise ValueError(f"Semantic weighting manifest is incompatible: {weighting_manifest_path}")
    table = pq.read_table(
        artifacts.record_relevance_rankings,
        columns=["source_row_uid", "level", "semantic_bucket_id", "weight"],
    )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in table.to_pylist():
        key = (str(row["source_row_uid"]), str(row["level"]))
        if key in rows:
            raise ValueError(f"Semantic rankings repeat a record-level assignment: {key}")
        rows[key] = {
            "semantic_bucket_id": str(row["semantic_bucket_id"]),
            "semantic_weight": _finite_unit(row["weight"], "semantic weight"),
        }
    return rows, {
        "release": SEMANTIC_RELEASE,
        "release_manifest": str(artifacts.manifest),
        "release_manifest_sha256": sha256_file(artifacts.manifest),
        "weighting_manifest": str(weighting_manifest_path),
        "record_relevance_rankings": str(artifacts.record_relevance_rankings),
        "record_relevance_rankings_sha256": sha256_file(artifacts.record_relevance_rankings),
    }


def _attach_semantics(
    candidates: Sequence[Mapping[str, Any]],
    *,
    level: str,
    semantics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    missing = []
    for candidate in candidates:
        uid = str(candidate["item_id"])
        semantic = semantics.get((uid, level))
        if semantic is None:
            missing.append(uid)
        else:
            output.append({**candidate, **semantic})
    if missing:
        raise ValueError(f"Semantic rankings omit {level} candidate UIDs: {missing[:5]}")
    return output


def _write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _default_query_path(task: str) -> Path:
    return Path("data/gold_labels") / GOLD_TASKS[task] / "v1/scaffold/valid_small.jsonl"


def _default_release_index(task: str) -> Path:
    return Path("data/caches/assay_reranking/active/ranked_level_retrieval_v4") / task / "RELEASE_INDEX.json"


def _profile_output(
    *,
    profile: Profile,
    task: str,
    output_root: Path,
    queries: Mapping[str, str],
    levels: Sequence[str],
    k: int,
    universes: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    release_index: Path,
    cache_audit: Mapping[str, Any],
    semantic_audit: Mapping[str, Any],
    query_path: Path,
) -> Path:
    output = output_root / profile.name / task
    output.mkdir(parents=True, exist_ok=False)
    selected_rows, diagnostic_rows = [], []
    for query_id in queries:
        for level in levels:
            selected, summary = select_records(
                universes[level][query_id], k=k,
                assay_lambda=profile.assay_lambda,
                gated_assay_lambda=profile.gated_assay_lambda,
                molecular_lambda=profile.molecular_lambda,
                semantic_relevance_lambda=profile.semantic_relevance_lambda,
                semantic_diversity_lambda=profile.semantic_diversity_lambda,
            )
            selected_rows.extend({
                "benchmark_row_id": query_id,
                "level": level,
                "selection_rank": row["selection_rank"],
                "source_row_uid": row["item_id"],
            } for row in selected)
            diagnostic_rows.append({
                "benchmark_row_id": query_id,
                "level": level,
                **summary,
            })

    selected_path = output / "selected_uids.tsv"
    selected_count = _write_tsv(
        selected_path, selected_rows,
        ("benchmark_row_id", "level", "selection_rank", "source_row_uid"),
    )
    diagnostic_fields = (
        "benchmark_row_id", "level", "candidate_records", "selected_records",
        "distinct_selected_parents", "distinct_selected_semantic_buckets",
        "assay_available", "assay_lambda_effective", "gated_assay_lambda_effective",
        "morgan", "assay", "assay_gated",
        "molecular_coverage", "molecular_diversity", "semantic_relevance",
        "semantic_coverage", "semantic_diversity",
        "objective_score",
    )
    diagnostics_path = output / "selection_diagnostics.tsv"
    _write_tsv(diagnostics_path, diagnostic_rows, diagnostic_fields)
    summary_rows = []
    for level in levels:
        local = [row for row in diagnostic_rows if row["level"] == level]
        summary_rows.append({
            "level": level,
            "queries": len(local),
            **{
                f"mean_{field}": sum(float(row[field]) for row in local) / len(local)
                for field in (
                    "morgan", "assay", "assay_gated", "molecular_coverage", "molecular_diversity",
                    "semantic_relevance", "semantic_coverage",
                    "semantic_diversity", "objective_score",
                )
            },
        })
    summary_path = output / "summary.tsv"
    _write_tsv(summary_path, summary_rows, tuple(summary_rows[0]))
    report_path = output / "report.md"
    report_path.write_text(
        "\n".join([
            f"# Record selection: {profile.name}", "",
            f"- Task: `{task}`",
            f"- Queries: {len(queries)}",
            f"- Levels: {', '.join(levels)}",
            f"- Records per query and level: {k}",
            f"- Lambdas: `{json.dumps(profile.lambdas(), sort_keys=True)}`",
            "- Molecular feature coverage is optimized; mean pairwise Morgan distance is diagnostic only.",
            "- Semantic bucket coverage is optimized; fixed-K normalized semantic diversity is also reported.",
            "- Assay contribution is zero only on levels whose frozen cache has no assay-transfer scores.",
            "",
        ]),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": UID_MANIFEST_SCHEMA,
        "status": "complete",
        "task_id": task,
        "subset": "valid",
        "profile": profile.name,
        "benchmark_row_ids": list(queries),
        "k_per_level": {level: k for level in levels},
        "release_index": str(release_index),
        "release_index_sha256": sha256_file(release_index),
        "objective": {
            "version": OBJECTIVE_VERSION,
            "formula": "M + lambda_a*A + lambda_mc*Cmol + lambda_sr*Srel + lambda_sd*Csem",
            "lambdas": profile.lambdas(),
            "optimizer": "apricot.CustomSelection:lazy",
        },
        "inputs": {
            "queries": str(query_path),
            "queries_sha256": sha256_file(query_path),
            "cache": {key: value for key, value in cache_audit.items() if key != "query_identities"},
            "semantics": dict(semantic_audit),
        },
        "records": {
            "path": selected_path.name,
            "sha256": sha256_file(selected_path),
            "row_count": selected_count,
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (diagnostics_path, summary_path, report_path)
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASK_LEVELS))
    parser.add_argument("--profiles", nargs="+", choices=("all", *PROFILES), default=["baseline"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--query-jsonl", type=Path)
    parser.add_argument("--release-index", type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/analysis/record_selection") / OBJECTIVE_VERSION,
    )
    args = parser.parse_args(argv)
    if args.k < 1 or args.limit < 0:
        parser.error("--k must be positive and --limit must be non-negative")
    if "all" in args.profiles and len(args.profiles) != 1:
        parser.error("--profiles all cannot be combined with named profiles")

    query_path = (args.query_jsonl or _default_query_path(args.task)).resolve()
    release_index = (args.release_index or _default_release_index(args.task)).resolve()
    query_rows = read_jsonl(query_path)
    if args.limit:
        query_rows = query_rows[: args.limit]
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in query_rows}
    if not queries or len(queries) != len(query_rows):
        raise ValueError("Queries require unique benchmark_row_id and drug fields")
    levels = TASK_LEVELS[args.task]
    ranked, _, cache_audit = load_ranked_universe(
        release_index, task=args.task, subset="valid", levels=levels, queries=queries
    )
    semantics, semantic_audit = _semantic_rows(args.task)
    universes = {
        level: {
            query_id: _attach_semantics(rows, level=level, semantics=semantics)
            for query_id, rows in by_query.items()
        }
        for level, by_query in ranked.items()
    }
    selected_profiles = list(PROFILES) if args.profiles == ["all"] else args.profiles
    for name in selected_profiles:
        output = _profile_output(
            profile=PROFILES[name], task=args.task, output_root=args.output_root.resolve(),
            queries=queries, levels=levels, k=args.k, universes=universes,
            release_index=release_index, cache_audit=cache_audit,
            semantic_audit=semantic_audit, query_path=query_path,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
