"""Select V10 evidence records with Morgan relevance and parent diminishing returns."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from functools import lru_cache
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
from apricot import CustomSelection

from predict.retrieval.assay_reranking.ranked_uid_retrieval import (
    hydrate_uids,
    load_ranked_universe,
)
from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_VERSION = "morgan_assay_feature_diversity.v1"
FINGERPRINT_BITS = 2048
FINGERPRINT_CHUNK_BITS = 32
TASKS = {
    "bbb_martins": {
        "gold_task": "BBB_Martins",
        "levels": ("L2", "L3", "L4", "L5"),
    },
    "bioavailability_ma": {
        "gold_task": "Bioavailability_Ma",
        "levels": ("L2", "L3", "L4", "L5", "L6"),
    },
}


def _covered_bit_count(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    covered = np.bitwise_or.reduce(matrix[:, 3:].astype(np.uint32), axis=0)
    return sum(int(chunk).bit_count() for chunk in covered)


def _objective(
    matrix: np.ndarray, *, k: int, parent_lambda: float, assay_lambda: float,
    diversity_lambda: float, fingerprint_universe_bits: int,
) -> float:
    if matrix.size == 0:
        return 0.0
    parent_counts = np.bincount(matrix[:, 2].astype(np.int64))
    return float(
        matrix[:, 0].sum() / k
        + assay_lambda * matrix[:, 1].sum() / k
        + parent_lambda * np.sqrt(parent_counts).sum() / k
        + diversity_lambda * _covered_bit_count(matrix) / fingerprint_universe_bits
    )


@lru_cache(maxsize=None)
def _fingerprint_chunks(smiles: str) -> tuple[int, ...]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot fingerprint parent SMILES: {smiles}")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FINGERPRINT_BITS,
    ).GetFingerprint(molecule)
    chunks = [0] * (FINGERPRINT_BITS // FINGERPRINT_CHUNK_BITS)
    for bit in fingerprint.GetOnBits():
        chunks[bit // FINGERPRINT_CHUNK_BITS] |= 1 << (bit % FINGERPRINT_CHUNK_BITS)
    return tuple(chunks)


def select_records(
    candidates: Sequence[Mapping[str, Any]], *, k: int, parent_lambda: float,
    assay_lambda: float = 0.0, diversity_lambda: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily select exactly K records under the versioned objective."""
    if k < 1 or len(candidates) < k:
        raise ValueError(f"K must be positive and no larger than {len(candidates)}")
    lambdas = (parent_lambda, assay_lambda, diversity_lambda)
    if any(not math.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("Objective lambdas must be finite and nonnegative")

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (int(row["morgan_rank"]), str(row["item_id"])),
    )
    if len({str(row["item_id"]) for row in ordered}) != len(ordered):
        raise ValueError("Candidate item IDs must be unique")

    parent_similarities: dict[str, float] = {}
    parent_smiles: dict[str, str] = {}
    parent_codes: dict[str, int] = {}
    assay_available: list[bool] = []
    features = []
    for row in ordered:
        parent_id = str(row["parent_id"])
        smiles = str(row["parent_smiles"])
        similarity = float(row["morgan_similarity"])
        if not math.isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError(f"Invalid Morgan similarity for {row['item_id']}")
        prior = parent_similarities.setdefault(parent_id, similarity)
        if not math.isclose(prior, similarity, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Morgan similarity varies within parent {parent_id}")
        prior_smiles = parent_smiles.setdefault(parent_id, smiles)
        if prior_smiles != smiles:
            raise ValueError(f"Parent SMILES varies within parent {parent_id}")
        raw_assay = row.get("assay_transfer_score")
        assay_available.append(raw_assay is not None)
        assay_score = 0.0 if raw_assay is None else float(raw_assay)
        if not math.isfinite(assay_score) or not 0 <= assay_score <= 1:
            raise ValueError(f"Invalid assay-transfer score for {row['item_id']}")
        parent_codes.setdefault(parent_id, len(parent_codes))
        features.append((
            similarity, assay_score, parent_codes[parent_id], *_fingerprint_chunks(smiles),
        ))

    if any(assay_available) and not all(assay_available):
        raise ValueError("Assay-transfer scores must be complete or absent for a level")
    has_assay_scores = all(assay_available)
    if assay_lambda > 0 and not has_assay_scores:
        raise ValueError("Positive assay_lambda requires complete assay-transfer scores")

    matrix = np.asarray(features, dtype=np.float64)
    fingerprint_universe_bits = _covered_bit_count(matrix)
    if fingerprint_universe_bits == 0:
        raise ValueError("Candidate universe has no Morgan fingerprint bits")
    selector = CustomSelection(
        n_samples=k,
        function=_objective,
        function_kwds={
            "k": k,
            "parent_lambda": parent_lambda,
            "assay_lambda": assay_lambda,
            "diversity_lambda": diversity_lambda,
            "fingerprint_universe_bits": fingerprint_universe_bits,
        },
        optimizer="naive",
        n_jobs=1,
    ).fit(matrix)
    selected = [ordered[int(index)] for index in selector.ranking]
    if not any(lambdas) and [row["item_id"] for row in selected] != [
        row["item_id"] for row in ordered[:k]
    ]:
        raise RuntimeError("Pure Morgan selection differs from the frozen Morgan order")

    final_counts = Counter(str(row["parent_id"]) for row in selected)
    running_counts: Counter[str] = Counter()
    result = []
    for rank, (row, gain) in enumerate(zip(selected, selector.gains), start=1):
        parent_id = str(row["parent_id"])
        running_counts[parent_id] += 1
        result.append({
            **row,
            "selection_rank": rank,
            "marginal_gain": float(gain),
            "parent_count_after_selection": running_counts[parent_id],
            "records_from_parent": final_counts[parent_id],
        })

    similarity_sum = sum(float(row["morgan_similarity"]) for row in selected)
    assay_sum = (
        sum(float(row["assay_transfer_score"]) for row in selected)
        if has_assay_scores else None
    )
    parent_reward = sum(math.sqrt(count) for count in final_counts.values())
    selected_matrix = matrix[np.asarray(selector.ranking, dtype=np.int64)]
    fingerprint_bits_covered = _covered_bit_count(selected_matrix)
    fingerprint_coverage = fingerprint_bits_covered / fingerprint_universe_bits
    objective = (
        similarity_sum / k
        + assay_lambda * (assay_sum or 0.0) / k
        + parent_lambda * parent_reward / k
        + diversity_lambda * fingerprint_coverage
    )
    if not math.isclose(objective, float(np.sum(selector.gains)), abs_tol=1e-10):
        raise RuntimeError("Apricot gains do not reproduce the objective")
    return result, {
        "candidate_records": len(ordered),
        "candidate_parents": len(parent_codes),
        "selected_records": k,
        "distinct_selected_parents": len(final_counts),
        "max_records_one_parent": max(final_counts.values()),
        "similarity_sum": similarity_sum,
        "similarity_mean": similarity_sum / k,
        "assay_available": has_assay_scores,
        "assay_score_sum": assay_sum,
        "assay_score_mean": assay_sum / k if assay_sum is not None else None,
        "parent_reward_sum": parent_reward,
        "fingerprint_bits_covered": fingerprint_bits_covered,
        "fingerprint_universe_bits": fingerprint_universe_bits,
        "fingerprint_coverage": fingerprint_coverage,
        "objective_score": objective,
    }


def _read_query(path: Path, *, query_index: int, benchmark_row_id: str | None) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if benchmark_row_id is not None:
        matches = [row for row in rows if str(row.get("benchmark_row_id")) == benchmark_row_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one gold row for {benchmark_row_id}, found {len(matches)}")
        row = matches[0]
    else:
        if not 0 <= query_index < len(rows):
            raise ValueError(f"query-index must be between 0 and {len(rows) - 1}")
        row = rows[query_index]
    return {
        "benchmark_row_id": str(row["benchmark_row_id"]),
        "drug": str(row["drug"]),
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run(args: argparse.Namespace) -> Path:
    config = TASKS[args.task]
    gold_path = (
        args.gold_query_file.resolve()
        if args.gold_query_file
        else REPO_ROOT / "data" / "gold_labels" / config["gold_task"] / "v1" /
        "scaffold" / "valid_small.jsonl"
    )
    release_index = (
        args.release_index.resolve()
        if args.release_index
        else REPO_ROOT / "data" / "caches" / "assay_reranking" / "active" /
        "ranked_level_retrieval_v3" / args.task / "RELEASE_INDEX.json"
    )
    query = _read_query(
        gold_path, query_index=args.query_index, benchmark_row_id=args.benchmark_row_id,
    )
    query_id = query["benchmark_row_id"]
    levels = tuple(config["levels"])
    ranked, evidence_manifest, cache_audit = load_ranked_universe(
        release_index,
        task=args.task,
        subset="valid",
        levels=levels,
        queries={query_id: query["drug"]},
    )

    selected_by_run: list[
        tuple[str, float, float, float, list[dict[str, Any]], dict[str, Any]]
    ] = []
    selected_uids: set[str] = set()
    assay_unavailable_levels = []
    for level in levels:
        candidates = ranked[level][query_id]
        has_assay_scores = all(row.get("assay_transfer_score") is not None for row in candidates)
        assay_lambdas = args.assay_lambdas if has_assay_scores else [0.0]
        if not has_assay_scores:
            assay_unavailable_levels.append(level)
        for parent_lambda in args.parent_lambdas:
            for assay_lambda in assay_lambdas:
                for diversity_lambda in args.diversity_lambdas:
                    selected, summary = select_records(
                        candidates,
                        k=args.k,
                        parent_lambda=parent_lambda,
                        assay_lambda=assay_lambda,
                        diversity_lambda=diversity_lambda,
                    )
                    selected_by_run.append((
                        level, parent_lambda, assay_lambda, diversity_lambda,
                        selected, summary,
                    ))
                    selected_uids.update(str(row["item_id"]) for row in selected)

    hydrated = hydrate_uids(evidence_manifest, selected_uids)
    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (
        level, parent_lambda, assay_lambda, diversity_lambda, selected, summary,
    ) in selected_by_run:
        for row in selected:
            evidence = hydrated[str(row["item_id"])]
            payload = json.loads(str(evidence["payload"]))
            source_fields = payload.get("source_fields") or {}
            selection_rows.append({
                "task_id": args.task,
                "benchmark_row_id": query_id,
                "level": level,
                "parent_lambda": parent_lambda,
                "assay_lambda": assay_lambda,
                "diversity_lambda": diversity_lambda,
                "selection_rank": row["selection_rank"],
                "source_row_uid": row["item_id"],
                "external_record_id": evidence["external_record_id"],
                "parent_id": row["parent_id"],
                "parent_smiles": row["parent_smiles"],
                "parent_morgan_rank": row["parent_morgan_rank"],
                "within_parent_rank": row["within_parent_rank"],
                "morgan_rank": row["morgan_rank"],
                "morgan_similarity": row["morgan_similarity"],
                "assay_transfer_score": row["assay_transfer_score"],
                "marginal_gain": row["marginal_gain"],
                "parent_count_after_selection": row["parent_count_after_selection"],
                "records_from_parent": row["records_from_parent"],
                "family_key": payload.get("family_key", ""),
                "endpoint_name": source_fields.get("endpoint_name", ""),
                "measurement_text": payload.get("display_measurement_text", ""),
                "unit_text": payload.get("display_unit_text", ""),
                "support_text": source_fields.get("support_text", ""),
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            })
        summary_rows.append({
            "task_id": args.task,
            "benchmark_row_id": query_id,
            "level": level,
            "parent_lambda": parent_lambda,
            "assay_lambda": assay_lambda,
            "diversity_lambda": diversity_lambda,
            **summary,
        })

    output_root = args.output_root.resolve() / args.task / query_id
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selections.tsv"
    summary_path = output_root / "summary.tsv"
    report_path = output_root / "report.md"
    manifest_path = output_root / "manifest.json"
    _write_tsv(selection_path, selection_rows, [
        "task_id", "benchmark_row_id", "level", "parent_lambda", "assay_lambda",
        "diversity_lambda", "selection_rank",
        "source_row_uid", "external_record_id", "parent_id", "parent_smiles",
        "parent_morgan_rank", "within_parent_rank", "morgan_rank", "morgan_similarity",
        "assay_transfer_score", "marginal_gain", "parent_count_after_selection",
        "records_from_parent",
        "family_key", "endpoint_name", "measurement_text", "unit_text", "support_text",
        "payload_json",
    ])
    _write_tsv(summary_path, summary_rows, [
        "task_id", "benchmark_row_id", "level", "parent_lambda", "assay_lambda",
        "diversity_lambda", "candidate_records", "candidate_parents", "selected_records",
        "distinct_selected_parents",
        "max_records_one_parent", "similarity_sum", "similarity_mean",
        "assay_available", "assay_score_sum", "assay_score_mean", "parent_reward_sum",
        "fingerprint_bits_covered", "fingerprint_universe_bits", "fingerprint_coverage",
        "objective_score",
    ])

    report_lines = [
        f"# Morgan, assay-transfer, and feature-diversity selection: {args.task}",
        "",
        f"Query: `{query_id}`",
        "",
        f"SMILES: `{query['drug']}`",
        "",
        f"Objective: `{OBJECTIVE_VERSION}` with K={args.k}. Gold labels were not used by selection.",
        "",
        "Diversity is Morgan fingerprint-bit coverage over the candidate universe; no pairwise similarity matrix is constructed.",
        "",
        f"Assay-unavailable levels: `{', '.join(assay_unavailable_levels)}`.",
        "",
        "| level | parent lambda | assay lambda | diversity lambda | parents | max/parent | similarity mean | assay mean | FP coverage | objective |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        assay_mean = (
            f"{row['assay_score_mean']:.6f}"
            if row["assay_score_mean"] is not None else "—"
        )
        report_lines.append(
            f"| {row['level']} | {row['parent_lambda']:.3g} | {row['assay_lambda']:.3g} | "
            f"{row['diversity_lambda']:.3g} | {row['distinct_selected_parents']} | "
            f"{row['max_records_one_parent']} | {row['similarity_mean']:.6f} | "
            f"{assay_mean} | {row['fingerprint_coverage']:.6f} | "
            f"{row['objective_score']:.6f} |"
        )
    report_lines.extend([
        "",
        "All selected physical records and the complete numeric summary are in "
        "`selections.tsv` and `summary.tsv`.",
        "",
    ])
    with atomic_output_path(report_path) as temporary:
        temporary.write_text("\n".join(report_lines), encoding="utf-8")

    write_json_atomic(manifest_path, {
        "schema_version": "record_selection_study.v1",
        "status": "complete",
        "objective": {
            "version": OBJECTIVE_VERSION,
            "formula": (
                "mean_morgan_similarity + assay_lambda * mean_assay_transfer_score + "
                "parent_lambda * sum_sqrt_parent_counts / K + diversity_lambda * "
                "selected_fingerprint_union_bits / candidate_fingerprint_union_bits"
            ),
            "k_per_level": args.k,
            "parent_lambdas": args.parent_lambdas,
            "assay_lambdas": args.assay_lambdas,
            "diversity_lambdas": args.diversity_lambdas,
            "assay_unavailable_levels": assay_unavailable_levels,
            "fingerprint": {
                "radius": 2,
                "bits": FINGERPRINT_BITS,
                "diversity": "selected_union_bits / candidate_universe_union_bits",
                "pairwise_similarity": False,
            },
            "optimizer": "apricot.CustomSelection:naive",
        },
        "task_id": args.task,
        "subset": "valid_small",
        "cache_subset": "valid",
        "query": query,
        "levels": list(levels),
        "inputs": {
            "gold_query_file": str(gold_path),
            "gold_query_file_sha256": sha256_file(gold_path),
            **cache_audit,
        },
        "environment": {
            "python": platform.python_version(),
            "apricot-select": _package_version("apricot-select"),
            "numpy": _package_version("numpy"),
            "pyarrow": _package_version("pyarrow"),
            "rdkit": _package_version("rdkit"),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (selection_path, summary_path, report_path)
        },
    })
    return output_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task", required=True, choices=sorted(TASKS))
    query = result.add_mutually_exclusive_group()
    query.add_argument("--query-index", type=int, default=0)
    query.add_argument("--benchmark-row-id")
    result.add_argument("--k", type=int, default=10)
    result.add_argument("--parent-lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.15])
    result.add_argument("--assay-lambdas", type=float, nargs="+", default=[0.0, 0.25])
    result.add_argument(
        "--diversity-lambdas", type=float, nargs="+", default=[0.0, 0.25, 1.0],
    )
    result.add_argument("--gold-query-file", type=Path)
    result.add_argument("--release-index", type=Path)
    result.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "record_selection" /
        "morgan_assay_feature_diversity_v1",
    )
    return result


def main() -> None:
    output = run(parser().parse_args())
    print(output)


if __name__ == "__main__":
    main()
