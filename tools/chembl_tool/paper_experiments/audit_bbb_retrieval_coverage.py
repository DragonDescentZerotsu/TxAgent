"""Audit BBB direct/mechanism retrieval coverage without model calls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.retrieval_policy import NeighborIdentityPolicy
from tools.chembl_tool.tasks.bbb_martins.experiment_config import STARLING
from tools.chembl_tool.tasks.bbb_martins.retrieve_neighbors import load_index


DEFAULT_LINEAGE = "experimental_meaningful_cns_access_v2"
DEFAULT_DATA_ROOT = Path("data/gold_labels/legacy/processed_starling_experimental_meaningful_cns_access_v2")
DEFAULT_INDEX_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_"
    "experimental_meaningful_cns_access_v2/evidence"
)
GROUP_NAMES = {
    "Mechanism.tier_1": "direct_brain_exposure",
    "Mechanism.tier_2": "passive_permeability",
    "Mechanism.tier_3": "efflux_transport",
    "Mechanism.tier_4": "influx_transport",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input_jsonl) if args.input_jsonl else (
        DEFAULT_DATA_ROOT / "BBB_Martins/scaffold" / f"{args.evaluation_subset}.jsonl"
    )
    direct_path = Path(args.direct_index)
    full_path = Path(args.full_index)
    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_INDEX_ROOT.parent
        / "diagnostics"
        / f"retrieval_family_coverage_{args.evaluation_subset}"
    )
    rows = _read_jsonl(input_path)
    direct_index = load_index(direct_path)
    full_index = load_index(full_path)
    heldout_path = input_path.with_name("heldout_molecule_labels.jsonl")
    _validate_index_provenance(
        direct_index,
        full_index,
        heldout_path=heldout_path,
        dataset_lineage=args.dataset_lineage,
    )

    query_rows: list[dict[str, Any]] = []
    for query_index, row in enumerate(rows):
        query_rows.append(
            audit_query(
                query_index,
                row,
                direct_index=direct_index,
                full_index=full_index,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
            )
        )

    summary = summarize_query_rows(query_rows, top_k=args.top_k)
    summary.update(
        {
            "schema_version": "bbb_retrieval_family_coverage.v1",
            "dataset_lineage": args.dataset_lineage,
            "evaluation_subset": args.evaluation_subset,
            "retrieval_contract": {
                "top_k_per_group": args.top_k,
                "min_similarity": args.min_similarity,
                "neighbor_identity_policy": NeighborIdentityPolicy.PARENT_DISJOINT.value,
            },
            "inputs": {
                "input_jsonl": str(input_path),
                "input_jsonl_sha256": sha256_file(input_path),
                "direct_index": str(direct_path),
                "direct_index_sha256": sha256_file(direct_path),
                "full_index": str(full_path),
                "full_index_sha256": sha256_file(full_path),
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "summary.json", summary)
    write_jsonl_atomic(output_dir / "per_query.jsonl", query_rows)
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summary["overall"]}, indent=2))
    return 0


def audit_query(
    query_index: int,
    row: dict[str, Any],
    *,
    direct_index: dict[str, Any],
    full_index: dict[str, Any],
    top_k: int,
    min_similarity: float,
) -> dict[str, Any]:
    query_smiles = str(row["drug"])
    common = {
        "top_k_per_group": top_k,
        "min_similarity": min_similarity,
        "neighbor_identity_policy": NeighborIdentityPolicy.PARENT_DISJOINT.value,
    }
    direct = retrieve_experiment_view(
        query_smiles,
        direct_index,
        mode="direct",
        config=STARLING,
        **common,
    )
    full = retrieve_experiment_view(
        query_smiles,
        full_index,
        mode="full_mechanism",
        config=STARLING,
        **common,
    )
    if direct.get("status") != "ok" or full.get("status") != "ok":
        raise ValueError(f"retrieval failed for query index {query_index}")

    groups = {
        str(group["group_id"]): _compact_group(group)
        for group in full.get("groups") or []
    }
    missing = sorted(set(GROUP_NAMES) - set(groups))
    if missing:
        raise ValueError(f"missing BBB mechanism groups: {missing}")
    direct_groups = direct.get("groups") or []
    direct_neighbors = direct_groups[0].get("neighbors") if direct_groups else []
    full_direct_neighbors = next(
        group.get("neighbors") or []
        for group in full["groups"]
        if group["group_id"] == "Mechanism.tier_1"
    )
    parity = _neighbor_signature(direct_neighbors or []) == _neighbor_signature(
        full_direct_neighbors
    )
    return {
        "query_index": query_index,
        "label": int(row["Y"]),
        "groups": groups,
        "n_groups_with_neighbors": sum(
            bool(group["n_neighbors"]) for group in groups.values()
        ),
        "all_groups_covered": all(
            bool(group["n_neighbors"]) for group in groups.values()
        ),
        "direct_index_full_index_parity": parity,
    }


def _validate_index_provenance(
    direct_index: dict[str, Any],
    full_index: dict[str, Any],
    *,
    heldout_path: Path,
    dataset_lineage: str,
) -> None:
    expected_hash = sha256_file(heldout_path)
    for name, index in (("direct", direct_index), ("full", full_index)):
        source = index.get("source") or {}
        if source.get("type") != "starling_heldout_parent_filtered_index":
            raise ValueError(f"{name} index is not heldout-parent filtered")
        if source.get("zero_parent_overlap") is not True:
            raise ValueError(f"{name} index does not certify zero parent overlap")
        if source.get("heldout_labels_sha256") != expected_hash:
            raise ValueError(f"{name} index heldout label hash mismatch")
        if dataset_lineage not in str(source.get("heldout_labels_jsonl") or ""):
            raise ValueError(f"{name} index dataset lineage mismatch")


def summarize_query_rows(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("coverage audit requires at least one query")
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    similarities: dict[str, list[float]] = defaultdict(list)
    label_counts: Counter[int] = Counter()
    pattern_counts: Counter[str] = Counter()
    for row in rows:
        label = int(row["label"])
        label_counts[label] += 1
        covered: list[str] = []
        for group_id, group in row["groups"].items():
            n_neighbors = int(group["n_neighbors"])
            group_counts[group_id]["queries"] += 1
            group_counts[group_id][f"label_{label}_queries"] += 1
            if n_neighbors:
                covered.append(GROUP_NAMES[group_id])
                group_counts[group_id]["covered"] += 1
                group_counts[group_id][f"label_{label}_covered"] += 1
                similarities[group_id].append(float(group["top_similarity"]))
            if n_neighbors == top_k:
                group_counts[group_id]["full_k"] += 1
        pattern_counts["+".join(covered) or "none"] += 1

    families: dict[str, Any] = {}
    for group_id in GROUP_NAMES:
        counts = group_counts[group_id]
        sims = similarities[group_id]
        families[GROUP_NAMES[group_id]] = {
            "group_id": group_id,
            "n_queries": counts["queries"],
            "n_queries_with_neighbors": counts["covered"],
            "coverage": _ratio(counts["covered"], counts["queries"]),
            "n_queries_with_full_k": counts["full_k"],
            "full_k_coverage": _ratio(counts["full_k"], counts["queries"]),
            "coverage_by_label": {
                str(label): {
                    "n": counts[f"label_{label}_queries"],
                    "covered": counts[f"label_{label}_covered"],
                    "coverage": _ratio(
                        counts[f"label_{label}_covered"],
                        counts[f"label_{label}_queries"],
                    ),
                }
                for label in sorted(label_counts)
            },
            "top_similarity_mean": round(mean(sims), 6) if sims else None,
            "top_similarity_median": round(median(sims), 6) if sims else None,
        }
    return {
        "overall": {
            "n_queries": len(rows),
            "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
            "n_all_groups_covered": sum(bool(row["all_groups_covered"]) for row in rows),
            "all_groups_coverage": _ratio(
                sum(bool(row["all_groups_covered"]) for row in rows), len(rows)
            ),
            "n_direct_parity_failures": sum(
                not bool(row["direct_index_full_index_parity"]) for row in rows
            ),
        },
        "families": families,
        "coverage_patterns": dict(
            sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _compact_group(group: dict[str, Any]) -> dict[str, Any]:
    neighbors = group.get("neighbors") or []
    return {
        "family": GROUP_NAMES[str(group["group_id"])],
        "n_candidate_molecules": int(group.get("n_candidate_molecules") or 0),
        "n_neighbors": len(neighbors),
        "top_similarity": float(neighbors[0]["similarity"]) if neighbors else None,
        "neighbor_ids": [str(neighbor["molecule_chembl_id"]) for neighbor in neighbors],
        "similarities": [float(neighbor["similarity"]) for neighbor in neighbors],
    }


def _neighbor_signature(neighbors: Iterable[dict[str, Any]]) -> list[tuple[str, float]]:
    """Compare chemical identities, not source-view-specific display identifiers."""
    return [
        (str(neighbor["standard_inchi_key"]), float(neighbor["similarity"]))
        for neighbor in neighbors
    ]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _render_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# BBB retrieval-family coverage audit",
        "",
        f"- queries: {overall['n_queries']}",
        f"- all four families covered: {overall['n_all_groups_covered']} "
        f"({overall['all_groups_coverage']:.1%})",
        f"- direct/full direct-neighbor parity failures: "
        f"{overall['n_direct_parity_failures']}",
        "",
        "| family | any neighbor | full k | top-1 similarity median |",
        "|---|---:|---:|---:|",
    ]
    for family, row in summary["families"].items():
        lines.append(
            f"| {family} | {row['coverage']:.1%} | "
            f"{row['full_k_coverage']:.1%} | {row['top_similarity_median']} |"
        )
    lines.extend(
        [
            "",
            "Coverage means at least one parent-disjoint neighbor at the frozen "
            "similarity threshold. It measures whether a group can receive analog "
            "evidence, not whether that evidence is correct or improves prediction.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-lineage", default=DEFAULT_LINEAGE)
    parser.add_argument("--evaluation-subset", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--input-jsonl",
        default="",
    )
    parser.add_argument(
        "--direct-index",
        default=str(DEFAULT_INDEX_ROOT / "bbb_starling_direct/starling_bbb_neighbor_index.pkl"),
    )
    parser.add_argument(
        "--full-index",
        default=str(DEFAULT_INDEX_ROOT / "bbb_starling_full/starling_bbb_neighbor_index.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        default="",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.30)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
