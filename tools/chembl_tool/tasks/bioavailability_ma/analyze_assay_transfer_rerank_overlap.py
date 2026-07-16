"""Compare Morgan top-k with cached assay-transfer reranking on frozen candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.tasks.bioavailability_ma.assay_transfer_rerank import (
    DEFAULT_CACHE,
    DEFAULT_CATALOG,
    SCORING_CONTRACT_VERSION,
    AssayTransferCatalog,
    AssayTransferPromptRenderer,
    flat_score_key,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_assay_transfer_rerank_catalog import (
    DEFAULT_CANDIDATE_MANIFEST,
)

DEFAULT_OUTPUT_DIR = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "assay_transfer_rerank/overlap_analysis"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = analyze_overlap(
        manifest_path=Path(args.candidate_manifest),
        catalog_path=Path(args.rerank_catalog),
        cache_path=Path(args.rerank_cache),
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "query_mechanism_overlap.jsonl"
    query_correlations_path = output_dir / "query_rank_correlations.jsonl"
    summary_path = output_dir / "summary.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with query_correlations_path.open("w", encoding="utf-8") as handle:
        for row in result["query_correlations"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **result["summary"],
                "rows_path": str(rows_path),
                "query_correlations_path": str(query_correlations_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


def analyze_overlap(
    *,
    manifest_path: Path,
    catalog_path: Path,
    cache_path: Path,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    catalog_metadata = _read_catalog_metadata(catalog_path)
    flat_schema = "query_smiles" not in _cache_columns(cache_path)
    n_missing_scores = 0
    if flat_schema:
        # Retrieval-agnostic (flat) cache: join candidate records to scores by
        # rendering each prompt to its cache_key, exactly as the reranker does.
        catalog = AssayTransferCatalog(catalog_path)
        renderer = AssayTransferPromptRenderer()
        flat_scores, cache_provenance = _read_flat_scores(cache_path)
        if cache_provenance["template_hash"] != renderer.template_hash:
            raise ValueError(
                "Cache/template mismatch: "
                f"cache={cache_provenance['template_hash']}, current={renderer.template_hash}"
            )

        def score_candidate(candidate: dict[str, Any], qs: str, gid: str) -> dict[str, Any]:
            nonlocal n_missing_scores
            scored_row, missing = _score_candidate_flat(
                candidate, qs, gid, catalog, renderer, cache_provenance, flat_scores
            )
            n_missing_scores += missing
            return scored_row

    else:
        score_lookup, cache_provenance = _read_scores(cache_path)
        expected_catalog = str(catalog_metadata["catalog_version"])
        if cache_provenance.get("catalog_version") != expected_catalog:
            raise ValueError(
                "Cache/catalog provenance mismatch: "
                f"cache={cache_provenance.get('catalog_version')}, catalog={expected_catalog}"
            )

        def score_candidate(candidate: dict[str, Any], qs: str, gid: str) -> dict[str, Any]:
            return _score_candidate(candidate, qs, gid, score_lookup)

    rows = []
    for manifest_row in _read_jsonl(manifest_path):
        # Flat manifests carry a leading manifest_metadata header row; legacy
        # manifests are bare candidate rows. Score only candidate groups.
        if "candidates" not in manifest_row:
            continue
        query_smiles = str(manifest_row["query_smiles"])
        group_id = str(manifest_row["group_id"])
        candidates = [
            candidate
            for candidate in manifest_row["candidates"]
            if float(candidate.get("similarity") or 0.0) >= min_similarity
        ]
        if not candidates:
            continue
        scored = [
            score_candidate(candidate, query_smiles, group_id)
            for candidate in candidates
        ]
        morgan_top = scored[:top_k]
        reranked_top = sorted(
            scored,
            key=lambda row: (
                -float(row["transfer_probability"]),
                -float(row["similarity"]),
                str(row["molecule_id"]),
            ),
        )[:top_k]
        morgan_ids = [str(row["molecule_id"]) for row in morgan_top]
        reranked_ids = [str(row["molecule_id"]) for row in reranked_top]
        morgan_set, reranked_set = set(morgan_ids), set(reranked_ids)
        overlap = len(morgan_set & reranked_set)
        union = len(morgan_set | reranked_set)
        position_matches = sum(left == right for left, right in zip(morgan_ids, reranked_ids))
        rank_correlation = _rank_correlation(scored)
        rows.append(
            {
                "query_index": int(manifest_row["query_index"]),
                "query_smiles": query_smiles,
                "group_id": group_id,
                "candidate_count": len(candidates),
                "top_k": top_k,
                "morgan_top": [_selection_row(row, rank) for rank, row in enumerate(morgan_top, 1)],
                "assay_transfer_top": [
                    _selection_row(row, rank) for rank, row in enumerate(reranked_top, 1)
                ],
                "overlap_count": overlap,
                "overlap_fraction": overlap / top_k,
                "jaccard": overlap / union if union else 1.0,
                "exact_set_match": morgan_set == reranked_set,
                "exact_order_match": morgan_ids == reranked_ids,
                "position_match_count": position_matches,
                "position_match_fraction": position_matches / top_k,
                **rank_correlation,
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    query_correlations = _query_correlation_rows(rows)
    summary = {
        "analysis_version": "assay_transfer_morgan_overlap.v1",
        "top_k": top_k,
        "n_queries": len({int(row["query_index"]) for row in rows}),
        "n_query_mechanism_rows": len(rows),
        "overall": _aggregate(rows, top_k),
        "by_group": {group: _aggregate(group_rows, top_k) for group, group_rows in sorted(by_group.items())},
        "rank_correlation": {
            "candidate_policy": "compatible_numeric_records_only",
            "query_mechanism": _aggregate_correlations(rows),
            "query_average": _aggregate_correlations(query_correlations),
            "by_group": {
                group: _aggregate_correlations(group_rows)
                for group, group_rows in sorted(by_group.items())
            },
        },
        "provenance": {
            "candidate_manifest": str(manifest_path),
            "catalog": str(catalog_path),
            "cache": str(cache_path),
            "cache_schema": "flat" if flat_schema else "condition_scoped",
            "candidate_min_similarity": min_similarity,
            "n_missing_candidate_record_scores": n_missing_scores,
            **cache_provenance,
        },
    }
    return {"rows": rows, "query_correlations": query_correlations, "summary": summary}


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
    lookup = {row["cache_key"]: float(row["transfer_probability"]) for row in rows}
    provenance_fields = ("model", "model_revision", "scoring_contract_version", "template_hash")
    distinct = {tuple(row[field] for field in provenance_fields) for row in rows}
    if len(distinct) != 1:
        raise ValueError(f"Expected one cache provenance tuple, found {len(distinct)}")
    provenance = dict(zip(provenance_fields, next(iter(distinct))))
    return lookup, provenance


def _score_candidate_flat(
    candidate: dict[str, Any],
    query_smiles: str,
    group_id: str,
    catalog: AssayTransferCatalog,
    renderer: AssayTransferPromptRenderer,
    provenance: dict[str, str],
    flat_scores: dict[str, float],
) -> tuple[dict[str, Any], int]:
    available: list[tuple[float, str]] = []
    missing = 0
    record_ids = [str(record_id) for record_id in (candidate.get("record_ids") or [])]
    for record in catalog.records_by_id(record_ids):
        prompt = renderer.render(record, query_smiles)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = flat_score_key(
            prompt_hash=prompt_hash,
            model=provenance["model"],
            model_revision=provenance["model_revision"],
            scoring_contract_version=SCORING_CONTRACT_VERSION,
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
            "transfer_probability": probability,
            "winning_record_id": winning_record_id,
            "scored_record_count": len(available),
        },
        missing,
    )


def _rank_correlation(scored: list[dict[str, Any]]) -> dict[str, Any]:
    compatible = [row for row in scored if int(row["scored_record_count"]) > 0]
    n = len(compatible)
    if n < 2:
        return {
            "rank_correlation_candidate_count": n,
            "spearman_rank_correlation": None,
            "kendall_rank_correlation": None,
        }
    morgan_ids = [str(row["molecule_id"]) for row in compatible]
    assay_ids = [
        str(row["molecule_id"])
        for row in sorted(
            compatible,
            key=lambda row: (
                -float(row["transfer_probability"]),
                -float(row["similarity"]),
                str(row["molecule_id"]),
            ),
        )
    ]
    assay_position = {molecule_id: position for position, molecule_id in enumerate(assay_ids, 1)}
    squared_distance = sum(
        (morgan_position - assay_position[molecule_id]) ** 2
        for morgan_position, molecule_id in enumerate(morgan_ids, 1)
    )
    spearman = 1.0 - (6.0 * squared_distance) / (n * (n * n - 1))
    concordant = 0
    discordant = 0
    for left in range(n):
        for right in range(left + 1, n):
            if assay_position[morgan_ids[left]] < assay_position[morgan_ids[right]]:
                concordant += 1
            else:
                discordant += 1
    kendall = (concordant - discordant) / (concordant + discordant)
    return {
        "rank_correlation_candidate_count": n,
        "spearman_rank_correlation": spearman,
        "kendall_rank_correlation": kendall,
    }


def _query_correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[int(row["query_index"])].append(row)
    output = []
    for query_index, query_rows in sorted(by_query.items()):
        valid = [row for row in query_rows if row["spearman_rank_correlation"] is not None]
        output.append(
            {
                "query_index": query_index,
                "query_smiles": query_rows[0]["query_smiles"],
                "n_mechanisms": len(query_rows),
                "n_mechanisms_with_correlation": len(valid),
                "mean_spearman_rank_correlation": (
                    sum(float(row["spearman_rank_correlation"]) for row in valid) / len(valid)
                    if valid
                    else None
                ),
                "mean_kendall_rank_correlation": (
                    sum(float(row["kendall_rank_correlation"]) for row in valid) / len(valid)
                    if valid
                    else None
                ),
                "mechanisms": [
                    {
                        "group_id": row["group_id"],
                        "candidate_count": row["rank_correlation_candidate_count"],
                        "spearman_rank_correlation": row["spearman_rank_correlation"],
                        "kendall_rank_correlation": row["kendall_rank_correlation"],
                    }
                    for row in sorted(query_rows, key=lambda item: str(item["group_id"]))
                ],
            }
        )
    return output


def _aggregate_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows and "mean_spearman_rank_correlation" in rows[0]:
        spearman_key = "mean_spearman_rank_correlation"
        kendall_key = "mean_kendall_rank_correlation"
    else:
        spearman_key = "spearman_rank_correlation"
        kendall_key = "kendall_rank_correlation"
    valid = [row for row in rows if row.get(spearman_key) is not None]
    return {
        "n": len(rows),
        "n_with_correlation": len(valid),
        "mean_spearman_rank_correlation": (
            sum(float(row[spearman_key]) for row in valid) / len(valid) if valid else None
        ),
        "mean_kendall_rank_correlation": (
            sum(float(row[kendall_key]) for row in valid) / len(valid) if valid else None
        ),
    }


def _score_candidate(
    candidate: dict[str, Any],
    query_smiles: str,
    group_id: str,
    scores: dict[tuple[str, str, str, str], tuple[float, str]],
) -> dict[str, Any]:
    molecule_id = str(candidate["molecule_id"])
    available = []
    for record_id in candidate.get("record_ids") or []:
        value = scores.get((query_smiles, group_id, molecule_id, str(record_id)))
        if value is not None:
            available.append((value[0], str(record_id)))
    available.sort(key=lambda item: (-item[0], item[1]))
    probability, winning_record_id = available[0] if available else (-1.0, "")
    return {
        **candidate,
        "transfer_probability": probability,
        "winning_record_id": winning_record_id,
        "scored_record_count": len(available),
    }


def _selection_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "molecule_id": row["molecule_id"],
        "canonical_smiles": row["canonical_smiles"],
        "similarity": row["similarity"],
        "structural_rank": row["structural_rank"],
        "transfer_probability": row["transfer_probability"],
        "winning_record_id": row["winning_record_id"],
        "scored_record_count": row["scored_record_count"],
    }


def _aggregate(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"n": 0}
    histogram = {str(value): 0 for value in range(top_k + 1)}
    for row in rows:
        histogram[str(int(row["overlap_count"]))] += 1
    full_pool_rows = [row for row in rows if int(row["candidate_count"]) >= top_k]
    result = {
        "n": count,
        "n_with_full_top_k_pool": len(full_pool_rows),
        "mean_overlap_count": sum(float(row["overlap_count"]) for row in rows) / count,
        "mean_overlap_fraction": sum(float(row["overlap_fraction"]) for row in rows) / count,
        "mean_jaccard": sum(float(row["jaccard"]) for row in rows) / count,
        "exact_set_match_rate": sum(bool(row["exact_set_match"]) for row in rows) / count,
        "exact_order_match_rate": sum(bool(row["exact_order_match"]) for row in rows) / count,
        "mean_position_match_fraction": sum(float(row["position_match_fraction"]) for row in rows) / count,
        "overlap_count_histogram": histogram,
    }
    if len(full_pool_rows) != count:
        result["full_top_k_pool_only"] = _aggregate(full_pool_rows, top_k)
    return result


def _read_scores(
    path: Path,
) -> tuple[dict[tuple[str, str, str, str], tuple[float, str]], dict[str, str]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT query_smiles, group_id, molecule_id, record_id, transfer_probability, "
            "model, model_revision, scoring_contract_version, template_hash, catalog_version "
            "FROM prompt_scores"
        ).fetchall()
    finally:
        connection.close()
    lookup = {
        (row["query_smiles"], row["group_id"], row["molecule_id"], row["record_id"]): (
            float(row["transfer_probability"]),
            row["record_id"],
        )
        for row in rows
    }
    provenance_fields = (
        "model",
        "model_revision",
        "scoring_contract_version",
        "template_hash",
        "catalog_version",
    )
    distinct = {tuple(row[field] for field in provenance_fields) for row in rows}
    if len(distinct) != 1:
        raise ValueError(f"Expected one cache provenance tuple, found {len(distinct)}")
    provenance = dict(zip(provenance_fields, next(iter(distinct))))
    return lookup, provenance


def _read_catalog_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--rerank-catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--rerank-cache", default=DEFAULT_CACHE)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Keep only manifest candidates with Tanimoto similarity >= this floor (e.g. 0.3).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
