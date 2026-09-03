"""Compare historical broad-skin and sensitization-only direct retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)
from tools.chembl_tool.tasks.skin_reaction.experiment_config import STARLING


DEFAULT_OLD_INDEX = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/"
    "evidence/skin_reaction_starling_full/starling_skin_reaction_neighbor_index.pkl"
)
DEFAULT_NEW_INDEX = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_"
    "skin_direct_scope_v2/evidence/skin_reaction_starling_full/"
    "starling_skin_reaction_neighbor_index.pkl"
)
DEFAULT_VALID = Path(
    "data/gold_labels/legacy/processed_starling_record_supported_v2/Skin_Reaction/scaffold/valid.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "outputs/paper/skin_direct_scope_retrieval_audit_record_supported_v2_valid"
)
DEFAULT_REFERENCE_PREDICTIONS = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_"
    "valid_gpt_oss_120b_bugfix_v2/runs_identity_blind_parent_disjoint/skin_reaction/"
    "skin_reaction__starling_direct/predictions.jsonl"
)
DEFAULT_CANDIDATE_PREDICTIONS = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_"
    "valid_gpt_oss_120b_skin_direct_scope_v2/runs_identity_blind_parent_disjoint/"
    "skin_reaction/skin_reaction__starling_direct/predictions.jsonl"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    old_path = Path(args.old_index)
    new_path = Path(args.new_index)
    valid_path = Path(args.valid_jsonl)
    old_index = _load_index(old_path)
    new_index = _load_index(new_path)
    samples = _read_jsonl(valid_path)

    rows = [
        compare_query(
            query_index=index,
            sample=sample,
            old_index=old_index,
            new_index=new_index,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
        )
        for index, sample in enumerate(samples)
    ]
    summary = summarize(rows)
    summary["inputs"] = {
        "old_index": str(old_path),
        "old_index_sha256": sha256_file(old_path),
        "new_index": str(new_path),
        "new_index_sha256": sha256_file(new_path),
        "valid_jsonl": str(valid_path),
        "valid_jsonl_sha256": sha256_file(valid_path),
        "top_k": args.top_k,
        "min_similarity": args.min_similarity,
        "neighbor_identity_policy": "parent_disjoint",
    }
    summary["index_sources"] = {
        "old": old_index.get("source", {}),
        "new": new_index.get("source", {}),
    }
    prediction_rows, prediction_summary = compare_predictions(
        Path(args.reference_predictions),
        Path(args.candidate_predictions),
    )
    summary["prediction_comparison"] = prediction_summary
    predictions_by_index = {row["query_index"]: row for row in prediction_rows}
    for row in rows:
        row["prediction_comparison"] = predictions_by_index[row["query_index"]]

    out_dir = Path(args.out_dir)
    write_jsonl_atomic(out_dir / "query_differences.jsonl", rows)
    write_json_atomic(out_dir / "summary.json", summary)
    _write_report(out_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def compare_query(
    *,
    query_index: int,
    sample: Mapping[str, Any],
    old_index: Mapping[str, Any],
    new_index: Mapping[str, Any],
    top_k: int = 3,
    min_similarity: float = 0.3,
) -> dict[str, Any]:
    query = str(sample.get("drug") or "")
    old_group = _direct_group(
        query, old_index, top_k=top_k, min_similarity=min_similarity
    )
    new_group = _direct_group(
        query, new_index, top_k=top_k, min_similarity=min_similarity
    )
    old_cards = [_card_summary(neighbor) for neighbor in old_group.get("neighbors", [])]
    new_cards = [_card_summary(neighbor) for neighbor in new_group.get("neighbors", [])]
    old_smiles = [card["canonical_smiles"] for card in old_cards]
    new_smiles = [card["canonical_smiles"] for card in new_cards]
    old_payload = canonical_json_bytes(old_group.get("neighbors", []))
    new_payload = canonical_json_bytes(new_group.get("neighbors", []))
    return {
        "query_index": query_index,
        "Y": int(sample["Y"]),
        "old_candidate_molecules": old_group.get("n_candidate_molecules", 0),
        "new_candidate_molecules": new_group.get("n_candidate_molecules", 0),
        "neighbor_identity_changed": old_smiles != new_smiles,
        "direct_context_changed": old_payload != new_payload,
        "old_neighbor_smiles": old_smiles,
        "new_neighbor_smiles": new_smiles,
        "old_cards": old_cards,
        "new_cards": new_cards,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    by_label: dict[str, Counter[str]] = {"0": Counter(), "1": Counter()}
    old_neighbors: set[str] = set()
    new_neighbors: set[str] = set()
    for row in rows:
        label_counts = by_label[str(row["Y"])]
        for key in ("neighbor_identity_changed", "direct_context_changed"):
            counts[key] += bool(row[key])
            label_counts[key] += bool(row[key])
        counts["n_queries"] += 1
        label_counts["n_queries"] += 1
        old_neighbors.update(row["old_neighbor_smiles"])
        new_neighbors.update(row["new_neighbor_smiles"])
    return {
        **dict(counts),
        "n_unique_old_neighbors": len(old_neighbors),
        "n_unique_new_neighbors": len(new_neighbors),
        "n_old_neighbors_absent_from_new_retrieval": len(old_neighbors - new_neighbors),
        "n_new_backfill_neighbors": len(new_neighbors - old_neighbors),
        "by_label": {label: dict(values) for label, values in by_label.items()},
    }


def compare_predictions(
    reference_path: Path,
    candidate_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = _prediction_map(_read_jsonl(reference_path), reference_path)
    candidate = _prediction_map(_read_jsonl(candidate_path), candidate_path)
    if set(reference) != set(candidate):
        raise ValueError("Reference and candidate query-index sets do not match")
    rows: list[dict[str, Any]] = []
    direction_counts: Counter[str] = Counter()
    direction_by_gold: Counter[str] = Counter()
    negative_transfer: Counter[str] = Counter()
    for query_index in sorted(reference):
        left = reference[query_index]
        right = candidate[query_index]
        if int(left["label"]) != int(right["label"]):
            raise ValueError(f"Gold-label mismatch at query index {query_index}")
        direction, transferability = _direct_group_assessment(Path(right["run_dir"]))
        label = int(right["label"])
        direction_counts[direction] += 1
        direction_by_gold[f"{direction}|Y={label}"] += 1
        if direction == "argues_against_sensitizer":
            negative_transfer[f"{transferability}|Y={label}"] += 1
        rows.append(
            {
                "query_index": query_index,
                "Y": label,
                "reference_prediction": int(left["pred_label"]),
                "candidate_prediction": int(right["pred_label"]),
                "prediction_flipped": int(left["pred_label"]) != int(right["pred_label"]),
                "reference_correct": bool(left["correct"]),
                "candidate_correct": bool(right["correct"]),
                "candidate_direct_direction": direction,
                "candidate_direct_transferability": transferability,
            }
        )
    labels = [row["Y"] for row in rows]
    reference_predictions = [row["reference_prediction"] for row in rows]
    candidate_predictions = [row["candidate_prediction"] for row in rows]
    return rows, {
        "reference_predictions": str(reference_path),
        "reference_predictions_sha256": sha256_file(reference_path),
        "candidate_predictions": str(candidate_path),
        "candidate_predictions_sha256": sha256_file(candidate_path),
        "paired": paired_binary_summary(
            labels,
            reference_predictions,
            candidate_predictions,
        ),
        "candidate_direct_directions": dict(direction_counts),
        "candidate_direct_direction_by_gold": dict(direction_by_gold),
        "candidate_negative_transferability_by_gold": dict(negative_transfer),
    }


def _direct_group(
    query: str,
    index: Mapping[str, Any],
    *,
    top_k: int,
    min_similarity: float,
) -> dict[str, Any]:
    result = retrieve_experiment_view(
        query,
        index,
        mode="direct",
        config=STARLING,
        top_k_per_group=top_k,
        min_similarity=min_similarity,
        neighbor_identity_policy="parent_disjoint",
    )
    if result.get("status") != "ok" or len(result.get("groups", [])) != 1:
        raise ValueError(f"Unexpected direct retrieval result: {result.get('status')}")
    return result["groups"][0]


def _card_summary(neighbor: Mapping[str, Any]) -> dict[str, Any]:
    rows = neighbor.get("evidence_rows") or []
    if len(rows) != 1:
        raise ValueError("Skin direct neighbor must contain exactly one evidence row")
    evidence = rows[0]
    counts = evidence.get("source_endpoint_counts") or {}
    return {
        "canonical_smiles": neighbor.get("canonical_smiles"),
        "similarity": neighbor.get("similarity"),
        "source_record_count": evidence.get("source_record_count", 0),
        "endpoint_counts": counts,
        "agreement_state": card_agreement_state(counts),
    }


def card_agreement_state(counts: Mapping[str, Any]) -> str:
    positive = int(counts.get("positive") or 0)
    negative = int(counts.get("negative") or 0)
    binary = positive + negative
    if binary == 0:
        return "insufficient"
    if max(positive, negative) / binary < 0.7:
        return "mixed"
    return "consistent_positive" if positive > negative else "consistent_negative"


def _load_index(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict index: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prediction_map(
    rows: list[dict[str, Any]],
    path: Path,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        query_index = int(row["query_index"])
        if query_index in result:
            raise ValueError(f"Duplicate query index {query_index} in {path}")
        result[query_index] = row
    return result


def _direct_group_assessment(run_dir: Path) -> tuple[str, str]:
    rows = _read_jsonl(run_dir / "group_reasoning_outputs.jsonl")
    if not rows:
        return "absent", "not_applicable"
    if len(rows) != 1:
        raise ValueError(f"Expected one direct group output: {run_dir}")
    content = ((rows[0].get("llm") or {}).get("content") or {})
    return (
        str(content.get("sensitization_evidence_direction") or "missing"),
        str(content.get("transferability") or "missing"),
    )


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Skin direct-scope retrieval audit",
        "",
        "This deterministic audit compares the historical broad-skin direct index with the "
        "sensitization/contact-allergy-only index. It makes no model calls.",
        "",
        f"- Queries: {summary['n_queries']}",
        f"- Direct contexts changed: {summary['direct_context_changed']}",
        f"- Top-k neighbor identities changed: {summary['neighbor_identity_changed']}",
        f"- Historical unique retrieved neighbors: {summary['n_unique_old_neighbors']}",
        f"- Scoped unique retrieved neighbors: {summary['n_unique_new_neighbors']}",
        f"- Historical neighbors removed from retrieved union: "
        f"{summary['n_old_neighbors_absent_from_new_retrieval']}",
        f"- New backfill neighbors: {summary['n_new_backfill_neighbors']}",
        f"- Candidate macro-F1: "
        f"{summary['prediction_comparison']['paired']['right_macro_f1']:.4f}",
        f"- Paired macro-F1 delta: "
        f"{summary['prediction_comparison']['paired']['delta_macro_f1']:+.4f}",
        "",
        "`query_differences.jsonl` preserves both ranked neighbor lists and card-level record "
        "agreement states for every query.",
    ]
    with atomic_output_path(path) as temporary:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-index", default=str(DEFAULT_OLD_INDEX))
    parser.add_argument("--new-index", default=str(DEFAULT_NEW_INDEX))
    parser.add_argument("--valid-jsonl", default=str(DEFAULT_VALID))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--reference-predictions", default=str(DEFAULT_REFERENCE_PREDICTIONS)
    )
    parser.add_argument(
        "--candidate-predictions", default=str(DEFAULT_CANDIDATE_PREDICTIONS)
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
