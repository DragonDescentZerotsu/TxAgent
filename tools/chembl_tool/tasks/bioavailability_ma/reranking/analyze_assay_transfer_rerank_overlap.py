"""Compare v6.5 assay-transfer and Morgan rankings on one frozen candidate pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from rdkit import DataStructs

from tools.chembl_tool.common.task_workflows.evidence_library import (
    fingerprint_metadata,
    standardize_smiles_and_fp,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    SCORING_CONTRACT_VERSION,
    V6_5_TEMPLATE_PROFILE,
    AssayTransferCatalog,
    AssayTransferPromptRenderer,
    flat_score_key,
)

RERANK_ROOT = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/assay_transfer_rerank"
)
DEFAULT_INPUT_DIR = RERANK_ROOT / "prepared_hf_validation_r100_c50_min0"
DEFAULT_CACHE_DIR = RERANK_ROOT / "cache_soft_v6_5_3572110c"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_INPUT_DIR / "manifest.jsonl"
DEFAULT_CATALOG = DEFAULT_INPUT_DIR / "catalog.jsonl"
DEFAULT_CACHE = DEFAULT_CACHE_DIR / "scores.sqlite3"
DEFAULT_CACHE_VERSION = DEFAULT_CACHE_DIR / "VERSION.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "analysis" / "v6_5_vs_morgan_k5_sim0p3_v2"
DEFAULT_RETRIEVAL_ARTIFACTS = Path(
    "outputs/paper/molecular_evidence_agent/"
    "validation_prepared_hf_assay_transfer_soft_v6_5_score_hidden_morgan_sim0p3_k5_parent_disjoint"
)
ANALYSIS_VERSION = "assay_transfer_v6_5_morgan_ranking_diversity.v2"
PAIRWISE_HIGH_SIMILARITY_THRESHOLD = 0.80


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact_dir = None if args.skip_artifact_validation else Path(args.retrieval_artifacts_dir)
    result = analyze_overlap(
        manifest_path=Path(args.candidate_manifest),
        catalog_path=Path(args.rerank_catalog),
        cache_path=Path(args.rerank_cache),
        cache_version_path=Path(args.cache_version) if args.cache_version else None,
        assay_transfer_template_profile=args.assay_transfer_template_profile,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        retrieval_artifacts_dir=artifact_dir,
        require_complete_cache=True,
        require_artifact_match=artifact_dir is not None,
    )
    paths = write_analysis_outputs(result, Path(args.output_dir))
    print(json.dumps({**result["summary"], "outputs": paths}, indent=2))
    return 0


def analyze_overlap(
    *,
    manifest_path: Path,
    catalog_path: Path,
    cache_path: Path,
    cache_version_path: Path | None = None,
    assay_transfer_template_profile: str = V6_5_TEMPLATE_PROFILE,
    top_k: int = 5,
    min_similarity: float = 0.30,
    retrieval_artifacts_dir: Path | None = None,
    require_complete_cache: bool = True,
    require_artifact_match: bool = False,
) -> dict[str, Any]:
    """Analyze deterministic molecule rankings without running model inference."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1]")

    manifest_metadata, manifest_rows = _read_manifest(manifest_path)
    catalog_metadata = _read_catalog_metadata(catalog_path)
    if "query_smiles" in _cache_columns(cache_path):
        raise ValueError("This analysis requires a retrieval-agnostic flat score cache")

    catalog = AssayTransferCatalog(catalog_path)
    renderer = AssayTransferPromptRenderer(profile=assay_transfer_template_profile)
    flat_scores, cache_provenance = _read_flat_scores(cache_path)
    if cache_provenance["template_hash"] != renderer.template_hash:
        raise ValueError(
            "Cache/template mismatch: "
            f"cache={cache_provenance['template_hash']}, current={renderer.template_hash}"
        )
    if cache_provenance["scoring_contract_version"] != SCORING_CONTRACT_VERSION:
        raise ValueError(
            "Cache/scoring-contract mismatch: "
            f"cache={cache_provenance['scoring_contract_version']}, "
            f"current={SCORING_CONTRACT_VERSION}"
        )

    manifest_sha256 = _sha256_file(manifest_path)
    cache_version = _read_optional_json(cache_version_path)
    _validate_provenance(
        manifest_metadata=manifest_metadata,
        manifest_sha256=manifest_sha256,
        catalog_metadata=catalog_metadata,
        cache_provenance=cache_provenance,
        cache_version=cache_version,
        renderer=renderer,
    )

    fingerprint_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for manifest_row in manifest_rows:
        query_smiles = str(manifest_row["query_smiles"])
        group_id = str(manifest_row["group_id"])
        raw_candidates = list(manifest_row.get("candidates") or [])
        counts["manifest_candidate_rows"] += len(raw_candidates)
        eligible_rows = [
            candidate
            for candidate in raw_candidates
            if float(candidate.get("similarity") or 0.0) >= min_similarity
        ]
        collapsed = _collapse_candidates(eligible_rows)
        counts["candidate_rows_at_similarity_floor"] += len(eligible_rows)
        counts["unique_molecules_at_similarity_floor"] += len(collapsed)
        counts["deduplicated_candidate_rows"] += len(eligible_rows) - len(collapsed)

        scored: list[dict[str, Any]] = []
        group_missing = 0
        group_record_references = 0
        for candidate in collapsed:
            scored_candidate, missing = _score_candidate_flat(
                candidate,
                query_smiles,
                group_id,
                catalog,
                renderer,
                cache_provenance,
                flat_scores,
            )
            group_record_references += int(scored_candidate["record_count"])
            group_missing += missing
            if int(scored_candidate["scored_record_count"]) > 0:
                scored.append(scored_candidate)

        counts["candidate_record_references"] += group_record_references
        counts["missing_candidate_record_scores"] += group_missing
        counts["scoreable_unique_molecules"] += len(scored)
        counts["unscoreable_unique_molecules"] += len(collapsed) - len(scored)

        morgan_ranking = _rank_molecules(scored, method="morgan")
        assay_ranking = _rank_molecules(scored, method="assay_transfer")
        morgan_top = morgan_ranking[:top_k]
        assay_top = assay_ranking[:top_k]
        overlap = _overlap_metrics(morgan_top, assay_top, top_k=top_k, pool_size=len(scored))
        pool_status = "complete" if len(scored) >= top_k else ("short" if scored else "empty")

        row = {
            "query_index": int(manifest_row["query_index"]),
            "query_smiles": query_smiles,
            "group_id": group_id,
            "pool_status": pool_status,
            "manifest_candidate_count": len(raw_candidates),
            "candidate_count_at_similarity_floor": len(eligible_rows),
            "unique_candidate_count_at_similarity_floor": len(collapsed),
            "candidate_record_reference_count": group_record_references,
            "missing_candidate_record_score_count": group_missing,
            "candidate_count": len(scored),
            "scoreable_molecule_count": len(scored),
            "unscoreable_molecule_count": len(collapsed) - len(scored),
            "top_k": top_k,
            "morgan_top": [
                _selection_row(candidate, rank, method="morgan")
                for rank, candidate in enumerate(morgan_top, 1)
            ],
            "assay_transfer_top": [
                _selection_row(candidate, rank, method="assay_transfer")
                for rank, candidate in enumerate(assay_top, 1)
            ],
            **overlap,
            "rank_correlation_candidate_count": len(scored),
            "spearman_rank_correlation": _spearman_between_rankings(
                [str(candidate["molecule_id"]) for candidate in morgan_ranking],
                [str(candidate["molecule_id"]) for candidate in assay_ranking],
            ),
            "structural_statistics": {
                "morgan_top": _structural_statistics(
                    morgan_top, query_smiles, fingerprint_cache=fingerprint_cache
                ),
                "assay_transfer_top": _structural_statistics(
                    assay_top, query_smiles, fingerprint_cache=fingerprint_cache
                ),
                "full_scoreable_pool": _structural_statistics(
                    scored, query_smiles, fingerprint_cache=fingerprint_cache
                ),
            },
        }
        rows.append(row)

    missing_scores = int(counts["missing_candidate_record_scores"])
    if require_complete_cache and missing_scores:
        raise ValueError(f"Cache is missing {missing_scores} candidate record scores")

    rows.sort(key=lambda row: (int(row["query_index"]), str(row["group_id"])))
    query_averages = _query_average_rows(rows)
    artifact_validation = _validate_retrieval_artifacts(
        rows,
        retrieval_artifacts_dir,
        expected_provenance={
            **cache_provenance,
            "template_profile": renderer.profile,
        },
    )
    if require_artifact_match and (
        not artifact_validation["performed"] or artifact_validation["mismatch_count"]
    ):
        raise ValueError(
            "Reconstructed selections do not match completed retrieval artifacts: "
            f"{artifact_validation}"
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "comparison": "assay-transfer v6.5 versus Morgan",
        "top_k": top_k,
        "similarity_floor": min_similarity,
        "pool_coverage": _pool_coverage(rows, top_k),
        "complete_pools": _aggregate_rows(
            [row for row in rows if row["pool_status"] == "complete"]
        ),
        "variable_depth_nonempty_pools": _aggregate_rows(
            [row for row in rows if row["pool_status"] != "empty"]
        ),
        "by_group": {
            group_id: {
                "pool_coverage": _pool_coverage(group_rows, top_k),
                "complete_pools": _aggregate_rows(
                    [row for row in group_rows if row["pool_status"] == "complete"]
                ),
                "variable_depth_nonempty_pools": _aggregate_rows(
                    [row for row in group_rows if row["pool_status"] != "empty"]
                ),
            }
            for group_id, group_rows in sorted(by_group.items())
        },
        "query_average": {
            "complete_pools": _query_balanced_aggregate(
                [row for row in rows if row["pool_status"] == "complete"]
            ),
            "variable_depth_nonempty_pools": _query_balanced_aggregate(
                [row for row in rows if row["pool_status"] != "empty"]
            ),
        },
        "artifact_validation": artifact_validation,
        "provenance": {
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": manifest_sha256,
            "candidate_manifest_metadata": manifest_metadata,
            "catalog": str(catalog_path),
            "catalog_version": str(catalog_metadata.get("catalog_version") or ""),
            "cache": str(cache_path),
            "cache_version_file": str(cache_version_path) if cache_version_path else "",
            "cache_version": cache_version,
            "cache_schema": "flat",
            "candidate_min_similarity": min_similarity,
            "fingerprint": fingerprint_metadata(),
            "pairwise_high_similarity_threshold": PAIRWISE_HIGH_SIMILARITY_THRESHOLD,
            "template_profile": renderer.profile,
            "query_context_policy": renderer.query_context_policy,
            **cache_provenance,
            **{f"n_{key}": int(value) for key, value in sorted(counts.items())},
        },
    }
    return {"rows": rows, "query_averages": query_averages, "summary": summary}


def write_analysis_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    paths = {
        "query_family_jsonl": output_dir / "query_family_comparison.jsonl",
        "query_family_tsv": output_dir / "query_family_comparison.tsv",
        "legacy_overlap_jsonl": output_dir / "query_mechanism_overlap.jsonl",
        "query_averages_jsonl": output_dir / "query_rank_correlations.jsonl",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    paths["query_family_jsonl"].write_text(payload, encoding="utf-8")
    # Preserve the established overlap output while adding the clearer family filename.
    paths["legacy_overlap_jsonl"].write_text(payload, encoding="utf-8")
    with paths["query_averages_jsonl"].open("w", encoding="utf-8") as handle:
        for row in result["query_averages"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_tsv(paths["query_family_tsv"], rows)
    paths["summary"].write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(_render_report(result["summary"]), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _collapse_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one deterministic candidate row per molecule with merged record IDs."""
    by_molecule: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        molecule_id = str(candidate["molecule_id"])
        smiles = str(candidate["canonical_smiles"])
        similarity = float(candidate.get("similarity") or 0.0)
        structural_rank = int(candidate.get("structural_rank") or 0)
        record_ids = {str(value) for value in (candidate.get("record_ids") or [])}
        existing = by_molecule.get(molecule_id)
        if existing is None:
            by_molecule[molecule_id] = {
                **candidate,
                "molecule_id": molecule_id,
                "canonical_smiles": smiles,
                "similarity": similarity,
                "structural_rank": structural_rank,
                "record_ids": sorted(record_ids),
            }
            continue
        if str(existing["canonical_smiles"]) != smiles:
            raise ValueError(f"Conflicting SMILES for duplicate molecule {molecule_id}")
        existing["similarity"] = max(float(existing["similarity"]), similarity)
        positive_ranks = [rank for rank in (int(existing["structural_rank"]), structural_rank) if rank > 0]
        existing["structural_rank"] = min(positive_ranks) if positive_ranks else 0
        existing["record_ids"] = sorted(set(existing["record_ids"]) | record_ids)
    return sorted(
        by_molecule.values(),
        key=lambda row: (-float(row["similarity"]), str(row["molecule_id"])),
    )


def _score_candidate_flat(
    candidate: dict[str, Any],
    query_smiles: str,
    group_id: str,
    catalog: AssayTransferCatalog,
    renderer: AssayTransferPromptRenderer,
    provenance: dict[str, str],
    flat_scores: dict[str, float],
) -> tuple[dict[str, Any], int]:
    del group_id  # The exact prompt record already fixes the assay family and context.
    available: list[tuple[float, str]] = []
    missing = 0
    record_ids = sorted(set(str(value) for value in (candidate.get("record_ids") or [])))
    for record in catalog.records_by_id(record_ids):
        prompt = renderer.render(record, query_smiles)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = flat_score_key(
            prompt_hash=prompt_hash,
            model=provenance["model"],
            model_revision=provenance["model_revision"],
            scoring_contract_version=provenance["scoring_contract_version"],
            template_hash=provenance["template_hash"],
        )
        probability = flat_scores.get(cache_key)
        if probability is None:
            missing += 1
        else:
            available.append((probability, str(record["record_id"])))
    available.sort(key=lambda item: (-item[0], item[1]))
    probability, winning_record_id = available[0] if available else (-1.0, "")
    return (
        {
            **candidate,
            "record_count": len(record_ids),
            "transfer_probability": probability,
            "winning_record_id": winning_record_id,
            "scored_record_count": len(available),
        },
        missing,
    )


def _rank_molecules(scored: Iterable[dict[str, Any]], *, method: str) -> list[dict[str, Any]]:
    rows = list(scored)
    molecule_ids = [str(row["molecule_id"]) for row in rows]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise ValueError("Rankings must receive one row per molecule")
    if method == "morgan":
        return sorted(
            rows,
            key=lambda row: (-float(row["similarity"]), str(row["molecule_id"])),
        )
    if method == "assay_transfer":
        return sorted(
            rows,
            key=lambda row: (
                -float(row["transfer_probability"]),
                -float(row["similarity"]),
                str(row["molecule_id"]),
            ),
        )
    raise ValueError(f"Unknown ranking method: {method}")


def _spearman_between_rankings(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or set(left) != set(right):
        raise ValueError("Spearman rankings must contain the same unique molecules")
    if len(left) != len(set(left)):
        raise ValueError("Spearman rankings cannot contain duplicate molecules")
    n = len(left)
    if n < 2:
        return None
    right_position = {molecule_id: rank for rank, molecule_id in enumerate(right, 1)}
    squared_distance = sum(
        (left_rank - right_position[molecule_id]) ** 2
        for left_rank, molecule_id in enumerate(left, 1)
    )
    value = 1.0 - (6.0 * squared_distance) / (n * (n * n - 1))
    return max(-1.0, min(1.0, value))


def _overlap_metrics(
    morgan_top: list[dict[str, Any]],
    assay_top: list[dict[str, Any]],
    *,
    top_k: int,
    pool_size: int,
) -> dict[str, Any]:
    morgan_ids = [str(row["molecule_id"]) for row in morgan_top]
    assay_ids = [str(row["molecule_id"]) for row in assay_top]
    intersection = set(morgan_ids) & set(assay_ids)
    overlap_count = len(intersection)
    denominator = min(top_k, pool_size)
    shared_morgan = [molecule_id for molecule_id in morgan_ids if molecule_id in intersection]
    shared_assay = [molecule_id for molecule_id in assay_ids if molecule_id in intersection]
    union = set(morgan_ids) | set(assay_ids)
    return {
        "overlap_count": overlap_count,
        "overlap_denominator": denominator,
        "overlap_fraction": overlap_count / denominator if denominator else None,
        "overlap_percentage": 100.0 * overlap_count / denominator if denominator else None,
        "jaccard": overlap_count / len(union) if union else None,
        "exact_set_match": set(morgan_ids) == set(assay_ids) if denominator else None,
        "exact_order_match": morgan_ids == assay_ids if denominator else None,
        "shared_order_spearman": _spearman_between_rankings(shared_morgan, shared_assay),
        "shared_order_molecule_count": overlap_count,
    }


def _structural_statistics(
    molecules: Iterable[dict[str, Any]],
    query_smiles: str,
    *,
    fingerprint_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache = fingerprint_cache if fingerprint_cache is not None else {}
    rows = list(molecules)
    molecule_ids = [str(row["molecule_id"]) for row in rows]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise ValueError("Structural statistics require unique molecules")

    query_fp = _fingerprint(query_smiles, cache)
    fingerprints = [_fingerprint(str(row["canonical_smiles"]), cache) for row in rows]
    pairwise = [
        float(DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]))
        for left in range(len(fingerprints))
        for right in range(left + 1, len(fingerprints))
    ]
    query_similarities = [
        float(DataStructs.TanimotoSimilarity(query_fp, fingerprint))
        for fingerprint in fingerprints
    ]
    for value in [*pairwise, *query_similarities]:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Tanimoto similarity outside [0, 1]: {value}")
    mean_pairwise = statistics.fmean(pairwise) if pairwise else None
    return {
        "molecule_count": len(rows),
        "pair_count": len(pairwise),
        "mean_pairwise_tanimoto": mean_pairwise,
        "median_pairwise_tanimoto": statistics.median(pairwise) if pairwise else None,
        "min_pairwise_tanimoto": min(pairwise) if pairwise else None,
        "max_pairwise_tanimoto": max(pairwise) if pairwise else None,
        "structural_diversity": 1.0 - mean_pairwise if mean_pairwise is not None else None,
        "fraction_pairs_tanimoto_ge_0_80": (
            sum(value >= PAIRWISE_HIGH_SIMILARITY_THRESHOLD for value in pairwise) / len(pairwise)
            if pairwise
            else None
        ),
        "mean_similarity_to_query": (
            statistics.fmean(query_similarities) if query_similarities else None
        ),
    }


def _fingerprint(smiles: str, cache: dict[str, Any]) -> Any:
    if smiles not in cache:
        _, _, fingerprint = standardize_smiles_and_fp(smiles)
        if fingerprint is None:
            raise ValueError(f"Could not fingerprint SMILES: {smiles!r}")
        cache[smiles] = fingerprint
    return cache[smiles]


def _selection_row(row: dict[str, Any], rank: int, *, method: str) -> dict[str, Any]:
    output = {
        "rank": rank,
        "molecule_id": row["molecule_id"],
        "canonical_smiles": row["canonical_smiles"],
        "similarity": row["similarity"],
        "structural_rank": row["structural_rank"],
    }
    if method == "assay_transfer":
        output.update(
            {
                "transfer_probability": row["transfer_probability"],
                "winning_record_id": row["winning_record_id"],
                "scored_record_count": row["scored_record_count"],
            }
        )
    return output


def _pool_coverage(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    histogram: dict[str, int] = defaultdict(int)
    for row in rows:
        histogram[str(int(row["candidate_count"]))] += 1
    return {
        "n_query_family_pools": len(rows),
        "n_queries": len({int(row["query_index"]) for row in rows}),
        "n_complete_pools": sum(row["pool_status"] == "complete" for row in rows),
        "n_short_pools": sum(row["pool_status"] == "short" for row in rows),
        "n_empty_pools": sum(row["pool_status"] == "empty" for row in rows),
        "n_nonempty_pools": sum(row["pool_status"] != "empty" for row in rows),
        "complete_pool_definition": f"scoreable_molecule_count >= {top_k}",
        "scoreable_pool_size_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


_STRUCTURAL_FIELDS = (
    "mean_pairwise_tanimoto",
    "median_pairwise_tanimoto",
    "min_pairwise_tanimoto",
    "max_pairwise_tanimoto",
    "structural_diversity",
    "fraction_pairs_tanimoto_ge_0_80",
    "mean_similarity_to_query",
)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_pools": 0, "n_queries": 0}
    output = {
        "n_pools": len(rows),
        "n_queries": len({int(row["query_index"]) for row in rows}),
        "mean_scoreable_molecule_count": _mean(rows, lambda row: row["candidate_count"]),
        "mean_overlap_count": _mean(rows, lambda row: row["overlap_count"]),
        "mean_overlap_fraction": _mean(rows, lambda row: row["overlap_fraction"]),
        "mean_overlap_percentage": _mean(rows, lambda row: row["overlap_percentage"]),
        "exact_set_match_rate": _mean(rows, lambda row: row["exact_set_match"]),
        "exact_order_match_rate": _mean(rows, lambda row: row["exact_order_match"]),
        "mean_full_pool_spearman": _mean(rows, lambda row: row["spearman_rank_correlation"]),
        "n_with_full_pool_spearman": sum(
            row["spearman_rank_correlation"] is not None for row in rows
        ),
        "mean_shared_order_spearman": _mean(rows, lambda row: row["shared_order_spearman"]),
        "n_with_shared_order_spearman": sum(
            row["shared_order_spearman"] is not None for row in rows
        ),
        "structural_statistics": {},
    }
    for selection in ("morgan_top", "assay_transfer_top", "full_scoreable_pool"):
        selection_output = {
            "n_sets": len(rows),
            "n_sets_with_pairs": sum(
                row["structural_statistics"][selection]["pair_count"] > 0 for row in rows
            ),
        }
        for field in _STRUCTURAL_FIELDS:
            selection_output[f"mean_{field}"] = _mean(
                rows, lambda row, s=selection, f=field: row["structural_statistics"][s][f]
            )
        output["structural_statistics"][selection] = selection_output
    return output


def _query_balanced_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average within query first so each represented query has equal weight."""
    if not rows:
        return {"n_queries": 0, "n_pools": 0}
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[int(row["query_index"])].append(row)

    def query_mean(getter: Callable[[dict[str, Any]], Any]) -> float | None:
        values = []
        for query_rows in by_query.values():
            value = _mean(query_rows, getter)
            if value is not None:
                values.append(value)
        return statistics.fmean(values) if values else None

    output = {
        "n_queries": len(by_query),
        "n_pools": len(rows),
        "mean_scoreable_molecule_count": query_mean(lambda row: row["candidate_count"]),
        "mean_overlap_count": query_mean(lambda row: row["overlap_count"]),
        "mean_overlap_fraction": query_mean(lambda row: row["overlap_fraction"]),
        "mean_overlap_percentage": query_mean(lambda row: row["overlap_percentage"]),
        "exact_set_match_rate": query_mean(lambda row: row["exact_set_match"]),
        "exact_order_match_rate": query_mean(lambda row: row["exact_order_match"]),
        "mean_full_pool_spearman": query_mean(lambda row: row["spearman_rank_correlation"]),
        "mean_shared_order_spearman": query_mean(lambda row: row["shared_order_spearman"]),
        "structural_statistics": {},
    }
    for selection in ("morgan_top", "assay_transfer_top", "full_scoreable_pool"):
        output["structural_statistics"][selection] = {
            f"mean_{field}": query_mean(
                lambda row, s=selection, f=field: row["structural_statistics"][s][f]
            )
            for field in _STRUCTURAL_FIELDS
        }
    return output


def _query_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[int(row["query_index"])].append(row)
    return [
        {
            "query_index": query_index,
            "query_smiles": query_rows[0]["query_smiles"],
            "pool_coverage": _pool_coverage(query_rows, top_k=int(query_rows[0]["top_k"])),
            "complete_pools": _aggregate_rows(
                [row for row in query_rows if row["pool_status"] == "complete"]
            ),
            "variable_depth_nonempty_pools": _aggregate_rows(
                [row for row in query_rows if row["pool_status"] != "empty"]
            ),
        }
        for query_index, query_rows in sorted(by_query.items())
    ]


def _mean(rows: Iterable[dict[str, Any]], getter: Callable[[dict[str, Any]], Any]) -> float | None:
    values = [getter(row) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def _validate_retrieval_artifacts(
    rows: list[dict[str, Any]],
    root: Path | None,
    *,
    expected_provenance: dict[str, str],
) -> dict[str, Any]:
    if root is None:
        return {"performed": False, "mismatch_count": 0}
    files = sorted(root.glob("**/retrieval.json"))
    expected = {
        (str(row["query_smiles"]), str(row["group_id"])): [
            str(item["molecule_id"]) for item in row["assay_transfer_top"]
        ]
        for row in rows
    }
    observed: dict[tuple[str, str], list[str]] = {}
    provenance_mismatches: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        query_smiles = str(payload.get("query", {}).get("input_smiles") or "")
        provenance = payload.get("experiment", {}).get("retrieval_reranker") or {}
        for field in ("model", "model_revision", "scoring_contract_version", "template_hash", "template_profile"):
            if str(provenance.get(field) or "") != str(expected_provenance.get(field) or ""):
                provenance_mismatches.append(f"{path}:{field}")
        for group in payload.get("groups") or []:
            key = (query_smiles, str(group["group_id"]))
            if key in observed:
                raise ValueError(f"Duplicate retrieval artifact group: {key}")
            observed[key] = [
                str(neighbor["molecule_chembl_id"])
                for neighbor in (group.get("neighbors") or [])
            ]
    mismatches = [
        {
            "query_smiles": key[0],
            "group_id": key[1],
            "expected": expected[key],
            "observed": observed.get(key),
        }
        for key in sorted(expected)
        if observed.get(key) != expected[key]
    ]
    unexpected = sorted(set(observed) - set(expected))
    mismatch_count = len(mismatches) + len(unexpected) + len(provenance_mismatches)
    return {
        "performed": True,
        "root": str(root),
        "retrieval_file_count": len(files),
        "expected_query_family_count": len(expected),
        "observed_query_family_count": len(observed),
        "selected_neighbor_count": sum(len(value) for value in observed.values()),
        "selection_mismatch_count": len(mismatches),
        "unexpected_query_family_count": len(unexpected),
        "provenance_mismatch_count": len(provenance_mismatches),
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatches[:10],
        "provenance_mismatch_examples": provenance_mismatches[:10],
    }


def _validate_provenance(
    *,
    manifest_metadata: dict[str, Any],
    manifest_sha256: str,
    catalog_metadata: dict[str, Any],
    cache_provenance: dict[str, str],
    cache_version: dict[str, Any],
    renderer: AssayTransferPromptRenderer,
) -> None:
    if not cache_version:
        return
    checks = {
        "model": cache_provenance["model"],
        "model_revision": cache_provenance["model_revision"],
        "scoring_contract_version": cache_provenance["scoring_contract_version"],
        "template_hash": renderer.template_hash,
        "template_profile": renderer.profile,
        "catalog_version": str(catalog_metadata.get("catalog_version") or ""),
        "condition_id": str(manifest_metadata.get("condition_id") or ""),
        "candidate_manifest_sha256": manifest_sha256,
    }
    mismatches = {
        field: {"version": cache_version.get(field), "observed": observed}
        for field, observed in checks.items()
        if str(cache_version.get(field) or "") != str(observed)
    }
    if mismatches:
        raise ValueError(f"Cache VERSION provenance mismatch: {mismatches}")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query_index", "query_smiles", "group_id", "pool_status",
        "manifest_candidate_count", "candidate_count_at_similarity_floor",
        "unique_candidate_count_at_similarity_floor", "candidate_record_reference_count",
        "missing_candidate_record_score_count", "scoreable_molecule_count",
        "unscoreable_molecule_count", "overlap_count", "overlap_denominator",
        "overlap_percentage", "exact_set_match", "exact_order_match",
        "spearman_rank_correlation", "shared_order_spearman",
        "morgan_top_molecule_ids", "assay_transfer_top_molecule_ids",
    ]
    for selection in ("morgan_top", "assay_transfer_top", "full_scoreable_pool"):
        fields.extend(f"{selection}_{field}" for field in ("molecule_count", "pair_count", *_STRUCTURAL_FIELDS))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field) for field in fields}
            flat["morgan_top_molecule_ids"] = ",".join(
                str(item["molecule_id"]) for item in row["morgan_top"]
            )
            flat["assay_transfer_top_molecule_ids"] = ",".join(
                str(item["molecule_id"]) for item in row["assay_transfer_top"]
            )
            for selection, stats in row["structural_statistics"].items():
                for key, value in stats.items():
                    flat[f"{selection}_{key}"] = value
            writer.writerow(flat)


def _render_report(summary: dict[str, Any]) -> str:
    coverage = summary["pool_coverage"]
    complete = summary["complete_pools"]
    variable = summary["variable_depth_nonempty_pools"]
    provenance = summary["provenance"]
    validation = summary["artifact_validation"]

    def fmt(value: Any, digits: int = 3) -> str:
        return "not available" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# V6.5 Assay-Transfer versus Morgan Ranking and Structural Diversity",
        "",
        "## Scope",
        "",
        (
            f"This analysis compares deterministic molecule rankings for {coverage['n_queries']} validation "
            f"queries and {coverage['n_query_family_pools']} query-family pools at a Morgan similarity floor "
            f"of {summary['similarity_floor']:.2f}. Both methods use exactly the same scoreable unique-molecule pool."
        ),
        "",
        "- V6.5 ranks molecules by maximum cached record transfer probability, then Morgan similarity, then molecule ID.",
        "- Morgan ranks each molecule once by similarity, then molecule ID; it does not select or inherit an assay record.",
        "- Structural diversity uses RDKit Morgan fingerprints with radius 2 and 2,048 bits, without features or chirality and with bond types enabled.",
        "",
        "## Pool coverage",
        "",
        "| Pool category | Count |",
        "|---|---:|",
        f"| Complete (at least {summary['top_k']}) | {coverage['n_complete_pools']} |",
        f"| Short (1-{summary['top_k'] - 1}) | {coverage['n_short_pools']} |",
        f"| Empty | {coverage['n_empty_pools']} |",
        f"| All nonempty | {coverage['n_nonempty_pools']} |",
        "",
        "## Ranking agreement",
        "",
        "| Analysis set | Pools | Mean overlap | Mean overlap | Exact-set match | Full-pool Spearman | Shared-top Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Complete top five | {complete['n_pools']} | {fmt(complete['mean_overlap_count'])} / 5 | "
            f"{fmt(complete['mean_overlap_percentage'], 1)}% | {fmt(100 * complete['exact_set_match_rate'], 1)}% | "
            f"{fmt(complete['mean_full_pool_spearman'])} | {fmt(complete['mean_shared_order_spearman'])} |"
        ),
        (
            f"| Variable depth, all nonempty | {variable['n_pools']} | {fmt(variable['mean_overlap_count'])} / available | "
            f"{fmt(variable['mean_overlap_percentage'], 1)}% | {fmt(100 * variable['exact_set_match_rate'], 1)}% | "
            f"{fmt(variable['mean_full_pool_spearman'])} | {fmt(variable['mean_shared_order_spearman'])} |"
        ),
        "",
        "The primary overlap percentage uses intersection size divided by five on complete pools. The variable-depth result uses intersection size divided by `min(5, pool size)`.",
        "",
        "## Structural diversity",
        "",
        "Complete-pool top-five results are shown below. Higher structural diversity means lower mean within-set Tanimoto similarity.",
        "",
        "| Selection | Mean pairwise Tanimoto | Structural diversity | Pair fraction >=0.80 | Mean similarity to query |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Morgan top five", "morgan_top"),
        ("V6.5 top five", "assay_transfer_top"),
        ("Full scoreable pool", "full_scoreable_pool"),
    ):
        stats = complete["structural_statistics"][key]
        lines.append(
            f"| {label} | {fmt(stats['mean_mean_pairwise_tanimoto'])} | "
            f"{fmt(stats['mean_structural_diversity'])} | "
            f"{fmt(stats['mean_fraction_pairs_tanimoto_ge_0_80'])} | "
            f"{fmt(stats['mean_mean_similarity_to_query'])} |"
        )
    lines.extend(
        [
            "",
            "## Results by mechanism family",
            "",
            "| Mechanism family | Complete | Short | Empty | Complete overlap | Complete V6.5 diversity | Complete Morgan diversity |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group_id, group in summary["by_group"].items():
        group_coverage = group["pool_coverage"]
        group_complete = group["complete_pools"]
        v65_diversity = (group_complete.get("structural_statistics") or {}).get("assay_transfer_top", {}).get("mean_structural_diversity")
        morgan_diversity = (group_complete.get("structural_statistics") or {}).get("morgan_top", {}).get("mean_structural_diversity")
        lines.append(
            f"| `{group_id}` | {group_coverage['n_complete_pools']} | {group_coverage['n_short_pools']} | "
            f"{group_coverage['n_empty_pools']} | {fmt(group_complete.get('mean_overlap_percentage'), 1)}% | "
            f"{fmt(v65_diversity)} | {fmt(morgan_diversity)} |"
        )
    lines.extend(
        [
            "",
            "## Provenance and reconstruction checks",
            "",
            f"- Model: `{provenance['model']}` at immutable revision `{provenance['model_revision']}`.",
            f"- Template profile: `{provenance['template_profile']}`; hash `{provenance['template_hash']}`.",
            f"- Candidate manifest SHA-256: `{provenance['candidate_manifest_sha256']}`.",
            f"- Missing cached record scores: {provenance['n_missing_candidate_record_scores']}.",
            f"- Reconstructed retrieval artifacts: {validation.get('observed_query_family_count', 0)} query-family pools and {validation.get('selected_neighbor_count', 0)} selected neighbors.",
            f"- Artifact mismatches: {validation.get('mismatch_count', 0)}.",
            "",
        ]
    )
    return "\n".join(lines)


def _cache_columns(path: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        return {row[1] for row in connection.execute("PRAGMA table_info(prompt_scores)")}
    finally:
        connection.close()


def _read_flat_scores(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT cache_key, transfer_probability, model, model_revision, "
            "scoring_contract_version, template_hash FROM prompt_scores"
        ).fetchall()
    finally:
        connection.close()
    lookup = {str(row["cache_key"]): float(row["transfer_probability"]) for row in rows}
    provenance_fields = ("model", "model_revision", "scoring_contract_version", "template_hash")
    distinct = {tuple(str(row[field]) for field in provenance_fields) for row in rows}
    if len(distinct) != 1:
        raise ValueError(f"Expected one cache provenance tuple, found {len(distinct)}")
    return lookup, dict(zip(provenance_fields, next(iter(distinct))))


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    if not rows or rows[0].get("record_type") != "manifest_metadata":
        raise ValueError(f"Invalid candidate manifest: {path}")
    candidates = [row for row in rows[1:] if row.get("record_type") == "candidate_group"]
    if len(candidates) != len(rows) - 1:
        raise ValueError(f"Invalid candidate-group row in manifest: {path}")
    return rows[0], candidates


def _read_catalog_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Cache version metadata does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", default=str(DEFAULT_CANDIDATE_MANIFEST))
    parser.add_argument("--rerank-catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--rerank-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--cache-version", default=str(DEFAULT_CACHE_VERSION))
    parser.add_argument(
        "--assay-transfer-template-profile",
        default=V6_5_TEMPLATE_PROFILE,
        help="Prompt profile used to reconstruct flat-cache keys.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.30,
        help="Keep manifest candidates with Morgan Tanimoto similarity at or above this floor.",
    )
    parser.add_argument("--retrieval-artifacts-dir", default=str(DEFAULT_RETRIEVAL_ARTIFACTS))
    parser.add_argument(
        "--skip-artifact-validation",
        action="store_true",
        help="Do not compare reconstructed v6.5 selections with completed k=5 retrieval artifacts.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
