"""Audit BBB base parity, cumulative prefixes, coverage, and branch stability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.distance_retrieval import retrieve_all_distance_views
from tools.chembl_tool.common.evidence_distance import (
    resolve_base_source_partition,
    stable_config_hash,
    validate_distance_config,
)
from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.tasks.bbb_martins.distance_config import DISTANCE_CONFIG


PREFIXES = ("D", "D+C", "D+C+H1", "D+C+H1+H2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default="data/processed/BBB_Martins/test.jsonl",
    )
    parser.add_argument(
        "--base-index",
        default="outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl",
    )
    parser.add_argument(
        "--distance-index",
        default=(
            "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/"
            "bbb_distance_superset_index.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/retrieval_audit"
        ),
    )
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.30)
    parser.add_argument("--neighbor-identity-policy", default="operational")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    validate_distance_config(DISTANCE_CONFIG)
    base_index = _load_pickle(Path(args.base_index))
    distance_index = _load_pickle(Path(args.distance_index))
    _validate_source(distance_index)
    base_partition = resolve_base_source_partition(
        DISTANCE_CONFIG.base_source_config,
        base_index["group_to_molecule_indices"],
    )
    records = _read_jsonl(Path(args.input_jsonl), limit=args.limit)
    audit = _audit_records(
        records,
        base_index,
        distance_index,
        top_k_per_group=args.top_k_per_group,
        min_similarity=args.min_similarity,
        neighbor_identity_policy=args.neighbor_identity_policy,
    )
    audit["task_name"] = DISTANCE_CONFIG.task_name
    audit["input_jsonl"] = str(args.input_jsonl)
    audit["base_index"] = str(args.base_index)
    audit["distance_index"] = str(args.distance_index)
    audit["source"] = distance_index["source"]
    audit["distance_config_hash"] = stable_config_hash(DISTANCE_CONFIG)
    audit["retrieval_policy"] = {
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
        "neighbor_identity_policy": args.neighbor_identity_policy,
    }
    audit["base_partition"] = {
        "D": list(base_partition.direct_groups),
        "C": list(base_partition.core_increment_groups),
        "D+C": list(base_partition.full_groups),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_quantity_tsv(out_dir / "prefix_quantity.tsv", audit["prefix_summary"])
    _write_family_tsv(out_dir / "family_coverage.tsv", audit["family_summary"])
    _write_report(out_dir / "report_zh.md", audit)
    if (
        audit["n_base_parity_failures"]
        or audit["n_source_nestedness_failures"]
        or audit["n_extension_retention_failures"]
        or audit["n_h1_stability_failures"]
    ):
        return 1
    return 0


def _audit_records(
    records: list[dict[str, Any]],
    base_index: dict[str, Any],
    distance_index: dict[str, Any],
    *,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
) -> dict[str, Any]:
    base_failures: list[dict[str, Any]] = []
    source_nestedness_failures: list[dict[str, Any]] = []
    extension_retention_failures: list[dict[str, Any]] = []
    base_neighbor_replacements: list[dict[str, Any]] = []
    h1_stability_failures: list[dict[str, Any]] = []
    prefix_stats: dict[str, list[dict[str, int]]] = defaultdict(list)
    family_stats: dict[str, list[dict[str, int]]] = defaultdict(list)
    coverage_labels: dict[str, Counter[int]] = defaultdict(Counter)

    for sample_index, record in enumerate(records):
        query_smiles = str(record.get("drug") or "")
        base_views = retrieve_all_distance_views(
            query_smiles,
            base_index,
            config=DISTANCE_CONFIG,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=neighbor_identity_policy,
        )
        distance_views = retrieve_all_distance_views(
            query_smiles,
            distance_index,
            config=DISTANCE_CONFIG,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=neighbor_identity_policy,
        )
        parity_pairs = (
            ("D", "mechanism"),
            ("D+C", "flat"),
            ("D+C", "mechanism"),
        )
        for prefix, view in parity_pairs:
            if _groups_hash(base_views[prefix][view]["groups"]) != _groups_hash(
                distance_views[prefix][view]["groups"]
            ):
                base_failures.append({"sample_index": sample_index, "prefix": prefix, "view": view})

        previous_rows: set[str] = set()
        previous_groups: set[str] = set()
        for prefix in PREFIXES:
            flat = distance_views[prefix]["flat"]
            mechanism = distance_views[prefix]["mechanism"]
            rows = _evidence_row_signatures(mechanism["groups"])
            source_groups = {
                source_group
                for group in mechanism["groups"]
                for source_group in group.get("source_group_ids") or []
            }
            if previous_groups - source_groups:
                source_nestedness_failures.append({"sample_index": sample_index, "prefix": prefix})
            missing_previous_rows = previous_rows - rows
            if missing_previous_rows and prefix == "D+C":
                base_neighbor_replacements.append(
                    {"sample_index": sample_index, "n_replaced_direct_rows": len(missing_previous_rows)}
                )
            elif missing_previous_rows:
                extension_retention_failures.append(
                    {"sample_index": sample_index, "prefix": prefix, "n_missing_rows": len(missing_previous_rows)}
                )
            previous_rows = rows
            previous_groups = source_groups

            flat_neighbors = flat["groups"][0].get("neighbors") or []
            unique_neighbors = {neighbor["molecule_chembl_id"] for neighbor in flat_neighbors}
            prefix_stats[prefix].append(
                {
                    "covered": int(bool(flat_neighbors)),
                    "n_unique_neighbors": len(unique_neighbors),
                    "n_evidence_rows": sum(int(neighbor.get("n_evidence_rows") or 0) for neighbor in flat_neighbors),
                    "n_source_groups": len(source_groups),
                }
            )
            coverage_labels[prefix][int(record.get("Y") or 0)] += int(bool(flat_neighbors))
            if prefix == "D+C+H1+H2":
                for group in mechanism["groups"]:
                    family_stats[group["group_id"]].append(
                        {
                            "covered": int(bool(group.get("neighbors"))),
                            "n_neighbors": len(group.get("neighbors") or []),
                        }
                    )

        h1_groups = {
            group["group_id"]: group
            for group in distance_views["D+C+H1"]["mechanism"]["groups"]
            if group["tier"] == "Distance H1"
        }
        maximal_h1_groups = {
            group["group_id"]: group
            for group in distance_views["D+C+H1+H2"]["mechanism"]["groups"]
            if group["tier"] == "Distance H1"
        }
        if _groups_hash(list(h1_groups.values())) != _groups_hash(list(maximal_h1_groups.values())):
            h1_stability_failures.append({"sample_index": sample_index})

    return {
        "n_samples": len(records),
        "n_base_parity_failures": len(base_failures),
        "n_source_nestedness_failures": len(source_nestedness_failures),
        "n_extension_retention_failures": len(extension_retention_failures),
        "n_base_d_to_dc_neighbor_replacements": len(base_neighbor_replacements),
        "n_h1_stability_failures": len(h1_stability_failures),
        "base_parity_failures": base_failures[:50],
        "source_nestedness_failures": source_nestedness_failures[:50],
        "extension_retention_failures": extension_retention_failures[:50],
        "base_d_to_dc_neighbor_replacements": base_neighbor_replacements[:50],
        "h1_stability_failures": h1_stability_failures[:50],
        "prefix_summary": _summarize_prefixes(prefix_stats, coverage_labels, len(records)),
        "family_summary": _summarize_families(family_stats, len(records)),
    }


def _summarize_prefixes(
    stats: dict[str, list[dict[str, int]]],
    label_coverage: dict[str, Counter[int]],
    n_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for prefix in PREFIXES:
        values = stats[prefix]
        rows.append(
            {
                "prefix": prefix,
                "n_samples": n_samples,
                "n_covered": sum(value["covered"] for value in values),
                "coverage": sum(value["covered"] for value in values) / n_samples if n_samples else 0.0,
                "mean_unique_neighbors": _mean(value["n_unique_neighbors"] for value in values),
                "mean_evidence_rows": _mean(value["n_evidence_rows"] for value in values),
                "mean_source_groups": _mean(value["n_source_groups"] for value in values),
                "covered_y0": label_coverage[prefix][0],
                "covered_y1": label_coverage[prefix][1],
            }
        )
    return rows


def _summarize_families(stats: dict[str, list[dict[str, int]]], n_samples: int) -> list[dict[str, Any]]:
    rows = []
    for group_id, values in sorted(stats.items()):
        rows.append(
            {
                "group_id": group_id,
                "n_samples": n_samples,
                "n_covered": sum(value["covered"] for value in values),
                "coverage": sum(value["covered"] for value in values) / n_samples if n_samples else 0.0,
                "mean_neighbors": _mean(value["n_neighbors"] for value in values),
            }
        )
    return rows


def _groups_hash(groups: list[dict[str, Any]]) -> str:
    payload_groups = []
    for group in groups:
        payload_group = {key: value for key, value in group.items() if key != "neighbors"}
        payload_group["neighbors"] = []
        for neighbor in group.get("neighbors") or []:
            payload_neighbor = {key: value for key, value in neighbor.items() if key != "evidence_rows"}
            payload_neighbor["evidence_rows"] = [evidence_for_llm(row) for row in neighbor.get("evidence_rows") or []]
            payload_group["neighbors"].append(payload_neighbor)
        payload_groups.append(payload_group)
    payload = json.dumps(payload_groups, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evidence_row_signatures(groups: list[dict[str, Any]]) -> set[str]:
    signatures = set()
    for group in groups:
        for neighbor in group.get("neighbors") or []:
            for row in neighbor.get("evidence_rows") or []:
                payload = {
                    "molecule_chembl_id": neighbor.get("molecule_chembl_id"),
                    "group_id": row.get("group_id"),
                    "minimal_evidence": row.get("minimal_evidence"),
                }
                signatures.add(
                    hashlib.sha256(
                        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
                    ).hexdigest()
                )
    return signatures


def _validate_source(index: dict[str, Any]) -> None:
    source = index.get("source") or {}
    if str(source.get("dataset") or "").lower() != "chembl" or source.get("mixed_sources") is not False:
        raise ValueError("Distance index must be an explicit ChEMBL-only source.")
    if source.get("release") != DISTANCE_CONFIG.source_release:
        raise ValueError("Distance index ChEMBL release does not match distance_config.")


def _write_quantity_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_tsv(path, rows)


def _write_family_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_tsv(path, rows)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# BBB distance-expansion retrieval audit",
        "",
        f"- 样本数：{audit['n_samples']:,}",
        f"- D/D+C base parity failures：{audit['n_base_parity_failures']:,}",
        f"- Source-group nestedness failures：{audit['n_source_nestedness_failures']:,}",
        f"- D+C→H1→H2 evidence-retention failures：{audit['n_extension_retention_failures']:,}",
        f"- D→D+C 旧 family 组织造成的 neighbor replacement query："
        f"{audit['n_base_d_to_dc_neighbor_replacements']:,}",
        f"- H1 加入 H2 后 branch stability failures：{audit['n_h1_stability_failures']:,}",
        "- 数据源：ChEMBL 36；D/C/H1/H2 不允许混入 Starling。",
        "",
        "## Prefix quantity 与 coverage",
        "",
        "| Prefix | Coverage | Mean unique neighbors | Mean evidence rows | Mean source groups |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audit["prefix_summary"]:
        lines.append(
            f"| `{row['prefix']}` | {row['coverage']:.3f} | {row['mean_unique_neighbors']:.2f} | "
            f"{row['mean_evidence_rows']:.2f} | {row['mean_source_groups']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Tree-node branch coverage",
            "",
            "| Tree-node branch | Coverage | Mean neighbors |",
            "|---|---:|---:|",
        ]
    )
    for row in audit["family_summary"]:
        lines.append(f"| `{row['group_id']}` | {row['coverage']:.3f} | {row['mean_neighbors']:.2f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
