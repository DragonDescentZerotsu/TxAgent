"""Audit Skin causal-panel retrieval parity and freeze a label-blind seed subset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.json_utils import read_jsonl, write_jsonl_atomic
from tools.chembl_tool.common.retrieval_features import load_retrieval_index
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.build_skin_seed import (
    DEFAULT_BASE_DESCRIPTOR,
    DEFAULT_OUTPUT_ROOT,
)
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.skin_contract import (
    causal_panel_retrieval_config,
    panel_direction_from_evidence_row,
    stable_json_sha256,
)
from tools.chembl_tool.tasks.skin_reaction.experiment_config import STARLING


DEFAULT_INPUT = Path(
    "data/gold_labels/legacy/processed_starling_record_supported_v2/Skin_Reaction/scaffold/valid.jsonl"
)
DEFAULT_SEED_DESCRIPTOR = DEFAULT_OUTPUT_ROOT / (
    "retrieval_features/scaffold/descriptors/"
    "skin_reaction__starling_causal_panel_seed_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = audit_skin_seed(
        input_jsonl=args.input_jsonl,
        canonical_descriptor=args.canonical_descriptor,
        seed_descriptor=args.seed_descriptor,
        output_root=args.output_root,
        seed_size=args.seed_size,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["zero_cost_gate"]["passed"] else 2


def audit_skin_seed(
    *,
    input_jsonl: Path,
    canonical_descriptor: Path,
    seed_descriptor: Path,
    output_root: Path,
    seed_size: int,
    top_k: int,
    min_similarity: float,
) -> dict[str, Any]:
    query_rows = read_jsonl(input_jsonl)
    canonical_index = load_retrieval_index(canonical_descriptor)
    seed_index = load_retrieval_index(seed_descriptor)
    canonical_config = STARLING
    seed_config = causal_panel_retrieval_config()

    audit_rows: list[dict[str, Any]] = []
    parity_failures: list[int] = []
    for query_index, query in enumerate(query_rows):
        kwargs = {
            "mode": "full_mechanism",
            "top_k_per_group": top_k,
            "min_similarity": min_similarity,
            "neighbor_identity_policy": "parent_disjoint",
        }
        canonical = retrieve_experiment_view(
            str(query["drug"]), canonical_index, config=canonical_config, **kwargs
        )
        seed = retrieve_experiment_view(
            str(query["drug"]), seed_index, config=seed_config, **kwargs
        )
        canonical_direct = _group(canonical, "Mechanism.tier_1")
        seed_direct = _group(seed, "Mechanism.tier_1")
        canonical_hash = stable_json_sha256(canonical_direct)
        seed_hash = stable_json_sha256(seed_direct)
        if canonical_hash != seed_hash:
            parity_failures.append(query_index)
        causal_group = _group(seed, "Mechanism.causal_panel")
        neighbors = []
        for neighbor in causal_group.get("neighbors") or []:
            evidence_rows = neighbor.get("evidence_rows") or []
            evidence = evidence_rows[0] if evidence_rows else {}
            panel = evidence.get("causal_panel") or {}
            neighbors.append(
                {
                    "rank": int(neighbor.get("rank") or 0),
                    "molecule_chembl_id": str(neighbor.get("molecule_chembl_id") or ""),
                    "similarity": float(neighbor.get("similarity") or 0.0),
                    "observed_direction": panel_direction_from_evidence_row(evidence),
                    "events": list(panel.get("events") or []),
                    "source_record_ids_sha256": str(
                        panel.get("source_record_ids_sha256") or ""
                    ),
                }
            )
        top = neighbors[0] if neighbors else {}
        audit_rows.append(
            {
                "query_index": query_index,
                "direct_group_hash": seed_hash,
                "direct_group_matches_frozen_canonical": canonical_hash == seed_hash,
                "causal_neighbors": neighbors,
                "top_causal_similarity": top.get("similarity"),
                "top_causal_direction": top.get("observed_direction", ""),
            }
        )

    selected = select_balanced_seed(audit_rows, seed_size=seed_size)
    selected_set = set(selected)
    for row in audit_rows:
        row["selected_for_llm_seed"] = row["query_index"] in selected_set
    selected_direction_counts = Counter(
        str(row["top_causal_direction"])
        for row in audit_rows
        if row["query_index"] in selected_set
    )
    all_direction_counts = Counter(str(row["top_causal_direction"]) for row in audit_rows)
    source_build = json.loads((output_root / "source_build_audit.json").read_text(encoding="utf-8"))
    negative_cards = int(source_build["compiler"]["direction_counts"].get("negative", 0))
    negative_query_coverage = int(all_direction_counts.get("negative", 0))
    gate_checks = {
        "direct_retrieval_parity_245_of_245": len(parity_failures) == 0,
        "materialized_cards_at_least_80": int(
            source_build["materialization"]["n_materialized_cards"]
        )
        >= 80,
        "negative_reference_cards_at_least_10": negative_cards >= 10,
        "negative_top_neighbor_queries_at_least_24": negative_query_coverage >= 24,
        "selected_queries_equal_seed_size": len(selected) == seed_size,
    }
    result = {
        "type": "skin_causal_panel_seed_retrieval_audit.v1",
        "selection_policy": {
            "label_blind": True,
            "policy": "balanced_top_causal_direction_then_descending_similarity.v1",
            "seed_size": seed_size,
            "top_k_per_group": top_k,
            "min_similarity": min_similarity,
        },
        "n_queries": len(query_rows),
        "n_direct_parity_failures": len(parity_failures),
        "direct_parity_failure_indices": parity_failures,
        "top_causal_direction_counts": dict(sorted(all_direction_counts.items())),
        "selected_direction_counts": dict(sorted(selected_direction_counts.items())),
        "selected_indices": selected,
        "zero_cost_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
        },
        "source_build_audit": str(output_root / "source_build_audit.json"),
        "paths": {
            "retrieval_rows": str(output_root / "retrieval_audit.jsonl"),
            "selected_indices": str(output_root / "selected_indices.json"),
        },
    }
    write_jsonl_atomic(output_root / "retrieval_audit.jsonl", audit_rows)
    (output_root / "selected_indices.json").write_text(
        json.dumps(
            {
                "type": "skin_causal_panel_seed_indices.v1",
                "selection_policy": result["selection_policy"],
                "selected_indices": selected,
                "selected_direction_counts": result["selected_direction_counts"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "retrieval_audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def select_balanced_seed(rows: list[dict[str, Any]], *, seed_size: int) -> list[int]:
    """Balance source direction without using query labels or baseline correctness."""
    if seed_size <= 0 or seed_size > len(rows):
        raise ValueError("seed_size must be between 1 and the number of query rows")
    target_each = seed_size // 2
    ranked: dict[str, list[dict[str, Any]]] = {}
    for direction in ("negative", "positive"):
        ranked[direction] = sorted(
            [row for row in rows if row.get("top_causal_direction") == direction],
            key=lambda row: (
                -float(row.get("top_causal_similarity") or -1.0),
                int(row["query_index"]),
            ),
        )
    selected_rows = ranked["negative"][:target_each] + ranked["positive"][:target_each]
    selected_indices = {int(row["query_index"]) for row in selected_rows}
    if len(selected_indices) < seed_size:
        remainder = sorted(
            [row for row in rows if int(row["query_index"]) not in selected_indices],
            key=lambda row: (
                -float(row.get("top_causal_similarity") or -1.0),
                int(row["query_index"]),
            ),
        )
        for row in remainder[: seed_size - len(selected_indices)]:
            selected_indices.add(int(row["query_index"]))
    return sorted(selected_indices)


def _group(retrieval: dict[str, Any], group_id: str) -> dict[str, Any]:
    matches = [group for group in retrieval.get("groups") or [] if group.get("group_id") == group_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one retrieval group {group_id}, found {len(matches)}")
    return matches[0]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--canonical-descriptor", type=Path, default=DEFAULT_BASE_DESCRIPTOR)
    parser.add_argument("--seed-descriptor", type=Path, default=DEFAULT_SEED_DESCRIPTOR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
