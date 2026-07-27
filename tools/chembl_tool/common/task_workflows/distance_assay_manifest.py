"""Shared ChEMBL assay discovery and manifest writer for distance extensions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.assay_loader import (
    iter_assay_metadata,
    load_activity_summary,
    load_target_annotations,
    merge_assay_record,
)
from tools.chembl_tool.common.evidence_distance import (
    DistanceExpansionConfig,
    FamilySelfRelevanceAudit,
    config_to_dict,
    stable_config_hash,
    validate_distance_config,
    validate_self_relevance_audit,
)
from tools.chembl_tool.common.export import ensure_dir, write_jsonl
from tools.chembl_tool.common.sqlite import connect_sqlite
from tools.chembl_tool.common.task_workflows.screen_assays import export_activity_evidence


@dataclass(frozen=True)
class DistanceAssayDecision:
    status: str
    family_id: str = ""
    tree_node_id: str = ""
    parent_c_family_id: str = ""
    distance_level: str = ""
    source_group_id: str = ""
    measured_node: str = ""
    scope_match: str = ""
    quality_status: str = ""
    effect_direction: str = ""
    matched_rules: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class DistanceAssayManifestConfig:
    description: str
    default_base_assays_csv: str
    default_out_dir: str
    distance_config: DistanceExpansionConfig
    self_relevance_audit: tuple[FamilySelfRelevanceAudit, ...]
    classify_assay: Callable[[Mapping[str, Any]], DistanceAssayDecision]
    classify_base_states: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None


CANDIDATE_FIELDS = (
    "assay_chembl_id",
    "assay_id",
    "distance_level",
    "distance_family_id",
    "distance_tree_node_id",
    "parent_c_family_id",
    "source_group_id",
    "measured_node",
    "scope_match",
    "quality_status",
    "effect_direction",
    "matched_rules",
    "assay_type",
    "description",
    "target_chembl_id",
    "target_pref_name",
    "target_genes",
    "target_synonyms",
    "organism",
    "confidence_score",
    "relationship_type",
    "assay_cell_type",
    "assay_tissue",
    "n_activities",
    "n_unique_molecules",
    "standard_types",
    "doc_pubmed_id",
    "doc_doi",
    "mapping_reason",
)

EXCLUSION_FIELDS = (
    "assay_chembl_id",
    "candidate_family_id",
    "candidate_tree_node_id",
    "parent_c_family_id",
    "candidate_level",
    "status",
    "scope_match",
    "quality_status",
    "matched_rules",
    "description",
    "target_chembl_id",
    "target_pref_name",
    "n_activities",
    "n_unique_molecules",
    "reason",
)

BASE_STATE_FIELDS = (
    "assay_chembl_id",
    "tier",
    "measured_nodes",
    "description",
    "target_chembl_id",
    "target_pref_name",
)


def main(config: DistanceAssayManifestConfig, argv: list[str] | None = None) -> int:
    args = _parse_args(config, argv)
    validate_distance_config(config.distance_config)
    validate_self_relevance_audit(
        config.distance_config,
        config.self_relevance_audit,
        require_publishable=True,
    )
    out_dir = ensure_dir(args.out_dir)
    base_rows = _read_csv_rows(Path(args.base_assays_csv))
    base_assay_ids = {
        str(row.get("assay_chembl_id") or "").strip()
        for row in base_rows
        if str(row.get("assay_chembl_id") or "").strip()
    }
    base_state_rows = _base_state_census(config, base_rows)
    base_measured_nodes = {
        node
        for row in base_state_rows
        for node in row.get("measured_nodes") or ()
    }
    overlapping_family_nodes = sorted(
        family.measured_node
        for family in config.distance_config.extension_families
        if family.measured_node in base_measured_nodes
    )
    if overlapping_family_nodes:
        raise ValueError(
            "Extension families measure states already present in frozen D/C: "
            f"{overlapping_family_nodes}"
        )
    _write_tsv(out_dir / "base_measured_state_census.tsv", base_state_rows, BASE_STATE_FIELDS)
    family_by_id = {family.family_id: family for family in config.distance_config.extension_families}

    started = time.monotonic()
    _log("Opening frozen ChEMBL source")
    conn = connect_sqlite(args.chembl_sqlite)
    try:
        _log("Loading activity summaries")
        activity_summaries = load_activity_summary(conn)
        _log(f"Loaded activity summaries for {len(activity_summaries):,} assays")
        _log("Loading target annotations")
        target_annotations = load_target_annotations(conn)
        _log(f"Loaded target annotations for {len(target_annotations):,} targets")

        candidates: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        candidate_assay_ids: list[int] = []
        scanned = 0
        for metadata in iter_assay_metadata(conn):
            scanned += 1
            assay_id = int(metadata["assay_id"])
            row = merge_assay_record(
                metadata,
                activity_summaries.get(assay_id),
                target_annotations.get(int(metadata["tid"])) if metadata.get("tid") is not None else None,
            )
            row["organism"] = row.get("target_organism") or row.get("assay_organism")
            decision = config.classify_assay(row)
            if args.progress_every and scanned % args.progress_every == 0:
                _log(
                    f"scanned={scanned:,} included={len(candidates):,} excluded_matches={len(exclusions):,} "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
            if decision.status == "unmatched":
                if args.limit and scanned >= args.limit:
                    break
                continue
            assay_chembl_id = str(row.get("assay_chembl_id") or "")
            if decision.status == "include" and assay_chembl_id in base_assay_ids:
                decision = DistanceAssayDecision(
                    status="base_overlap",
                    family_id=decision.family_id,
                    tree_node_id=decision.tree_node_id,
                    parent_c_family_id=decision.parent_c_family_id,
                    distance_level=decision.distance_level,
                    source_group_id=decision.source_group_id,
                    measured_node=decision.measured_node,
                    scope_match=decision.scope_match,
                    quality_status=decision.quality_status,
                    effect_direction=decision.effect_direction,
                    matched_rules=decision.matched_rules,
                    reason="Assay already belongs to frozen D/C and cannot be relabeled as H1/H2.",
                )

            if decision.status == "include":
                family = family_by_id.get(decision.family_id)
                if family is None:
                    raise ValueError(f"Classifier returned undeclared family `{decision.family_id}`.")
                if (
                    decision.tree_node_id != family.tree_node_id
                    or decision.parent_c_family_id != family.parent_c_family_id
                    or decision.distance_level != family.declared_level
                    or decision.source_group_id != family.source_group_id
                    or decision.measured_node != family.measured_node
                ):
                    raise ValueError(f"Classifier/config mismatch for family `{decision.family_id}`.")
                candidates.append(_candidate_row(row, decision))
                candidate_assay_ids.append(assay_id)
            else:
                exclusions.append(_exclusion_row(row, decision))

            if args.limit and scanned >= args.limit:
                break

        candidates.sort(
            key=lambda row: (
                str(row["distance_level"]),
                str(row["distance_family_id"]),
                -int(row["n_unique_molecules"] or 0),
                str(row["assay_chembl_id"]),
            )
        )
        exclusions.sort(
            key=lambda row: (
                str(row["candidate_level"]),
                str(row["candidate_family_id"]),
                str(row["status"]),
                str(row["assay_chembl_id"]),
            )
        )
        _write_tsv(out_dir / "assay_to_family.tsv", candidates, CANDIDATE_FIELDS)
        write_jsonl(out_dir / "distance_assay_candidates.jsonl", candidates)
        _write_tsv(out_dir / "assay_exclusions.tsv", exclusions, EXCLUSION_FIELDS)
        if args.export_activities:
            export_activity_evidence(conn, candidate_assay_ids, out_dir / "distance_activity_evidence.csv")

        _write_config_artifacts(config.distance_config, out_dir)
        source_manifest = _source_manifest(
            config.distance_config,
            Path(args.chembl_sqlite),
            Path(args.base_assays_csv),
            hash_source_db=args.hash_source_db,
        )
        source_manifest["scan"] = {
            "n_assays_scanned": scanned,
            "n_base_assays": len(base_assay_ids),
            "n_extension_assays": len(candidates),
            "n_matched_exclusions": len(exclusions),
            "exported_activities": bool(args.export_activities),
        }
        source_manifest["base_measured_nodes"] = sorted(base_measured_nodes)
        source_manifest["n_base_state_assignments"] = len(base_state_rows)
        (out_dir / "source_manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_report(
            out_dir / "report_zh.md",
            config.distance_config,
            candidates,
            exclusions,
            scanned,
            base_measured_nodes=base_measured_nodes,
            n_base_state_assignments=len(base_state_rows),
        )
        _log(
            f"Finished distance assay manifest: scanned={scanned:,} included={len(candidates):,} "
            f"excluded_matches={len(exclusions):,} elapsed={time.monotonic() - started:.1f}s"
        )
    finally:
        conn.close()
    return 0


def _candidate_row(row: Mapping[str, Any], decision: DistanceAssayDecision) -> dict[str, Any]:
    return {
        "assay_chembl_id": row.get("assay_chembl_id"),
        "assay_id": row.get("assay_id"),
        "distance_level": decision.distance_level,
        "distance_family_id": decision.family_id,
        "distance_tree_node_id": decision.tree_node_id,
        "parent_c_family_id": decision.parent_c_family_id,
        "source_group_id": decision.source_group_id,
        "measured_node": decision.measured_node,
        "scope_match": decision.scope_match,
        "quality_status": decision.quality_status,
        "effect_direction": decision.effect_direction,
        "matched_rules": decision.matched_rules,
        "assay_type": row.get("assay_type"),
        "description": row.get("description"),
        "target_chembl_id": row.get("target_chembl_id"),
        "target_pref_name": row.get("target_pref_name"),
        "target_genes": row.get("target_genes"),
        "target_synonyms": row.get("target_synonyms"),
        "organism": row.get("organism"),
        "confidence_score": row.get("confidence_score"),
        "relationship_type": row.get("relationship_type"),
        "assay_cell_type": row.get("assay_cell_type"),
        "assay_tissue": row.get("assay_tissue"),
        "n_activities": row.get("n_activities"),
        "n_unique_molecules": row.get("n_unique_molecules"),
        "standard_types": row.get("standard_types"),
        "doc_pubmed_id": row.get("doc_pubmed_id"),
        "doc_doi": row.get("doc_doi"),
        "mapping_reason": decision.reason,
    }


def _exclusion_row(row: Mapping[str, Any], decision: DistanceAssayDecision) -> dict[str, Any]:
    return {
        "assay_chembl_id": row.get("assay_chembl_id"),
        "candidate_family_id": decision.family_id,
        "candidate_tree_node_id": decision.tree_node_id,
        "parent_c_family_id": decision.parent_c_family_id,
        "candidate_level": decision.distance_level,
        "status": decision.status,
        "scope_match": decision.scope_match,
        "quality_status": decision.quality_status,
        "matched_rules": decision.matched_rules,
        "description": row.get("description"),
        "target_chembl_id": row.get("target_chembl_id"),
        "target_pref_name": row.get("target_pref_name"),
        "n_activities": row.get("n_activities"),
        "n_unique_molecules": row.get("n_unique_molecules"),
        "reason": decision.reason,
    }


def _write_config_artifacts(config: DistanceExpansionConfig, out_dir: Path) -> None:
    payload = config_to_dict(config)
    (out_dir / "distance_graph.json").write_text(
        json.dumps(
            {
                "task_name": config.task_name,
                "source_name": config.source_name,
                "source_release": config.source_release,
                "nodes": payload["nodes"],
                "edges": payload["edges"],
                "config_hash": stable_config_hash(config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "distance_family_tree.json").write_text(
        json.dumps(
            {
                "task_name": config.task_name,
                "source_name": config.source_name,
                "source_release": config.source_release,
                "tree_nodes": payload["extension_tree_nodes"],
                "config_hash": stable_config_hash(config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "distance_measurement_family_manifest.json").write_text(
        json.dumps(
            {
                "task_name": config.task_name,
                "source_name": config.source_name,
                "source_release": config.source_release,
                "measurement_families": payload["extension_families"],
                "config_hash": stable_config_hash(config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _source_manifest(
    config: DistanceExpansionConfig,
    chembl_sqlite: Path,
    base_assays_csv: Path,
    *,
    hash_source_db: bool,
) -> dict[str, Any]:
    return {
        "source_name": config.source_name,
        "source_release": config.source_release,
        "applies_to_levels": ["D", "C", "H1", "H2"],
        "mixed_sources_allowed": False,
        "config_hash": stable_config_hash(config),
        "chembl_sqlite": _file_manifest(chembl_sqlite, include_sha256=hash_source_db),
        "base_assays_csv": _file_manifest(base_assays_csv, include_sha256=True),
    }


def _file_manifest(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path) if include_sha256 else None,
        "sha256_status": "computed" if include_sha256 else "deferred_use_--hash-source-db_for_release",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(
    path: Path,
    config: DistanceExpansionConfig,
    candidates: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    scanned: int,
    *,
    base_measured_nodes: set[str],
    n_base_state_assignments: int,
) -> None:
    family_counts = Counter(str(row["distance_family_id"]) for row in candidates)
    tree_counts = Counter(str(row["distance_tree_node_id"]) for row in candidates)
    family_molecules = Counter()
    for row in candidates:
        family_molecules[str(row["distance_family_id"])] += int(row.get("n_unique_molecules") or 0)
    exclusion_counts = Counter(str(row["status"]) for row in exclusions)
    lines = [
        "# BBB ChEMBL distance assay manifest",
        "",
        f"- ChEMBL release：`{config.source_release}`",
        "- 数据源约束：D/C/H1/H2 全部为 ChEMBL，不允许混入 Starling。",
        f"- 扫描 assay：{scanned:,}",
        f"- 纳入 extension assay：{len(candidates):,}",
        f"- 匹配后排除：{len(exclusions):,}",
        f"- D/C measured-state assignments：{n_base_state_assignments:,}",
        f"- D/C measured nodes：{', '.join(sorted(base_measured_nodes))}",
        "",
        "## 层级 retrieval nodes",
        "",
        "| C parent | Level | Tree node | Parent node | Measurement families | Assays |",
        "|---|---|---|---|---|---:|",
    ]
    for tree_node in config.extension_tree_nodes:
        lines.append(
            f"| `{tree_node.parent_c_family_id}` | {tree_node.declared_level} | "
            f"`{tree_node.tree_node_id}` | `{tree_node.parent_id}` | "
            f"{', '.join(f'`{family_id}`' for family_id in tree_node.measurement_family_ids)} | "
            f"{tree_counts[tree_node.tree_node_id]:,} |"
        )
    lines.extend(
        [
        "",
        "## 纳入 family",
        "",
        "| Level | Family | Assays | Assay 内 unique-molecule 计数之和* |",
        "|---|---|---:|---:|",
        ]
    )
    for family in config.extension_families:
        lines.append(
            f"| {family.declared_level} | `{family.family_id}` | {family_counts[family.family_id]:,} | "
            f"{family_molecules[family.family_id]:,} |"
        )
    lines.extend(
        [
            "",
            "\\* 该列是逐 assay 计数之和，尚未对跨 assay 重复 molecule 去重；真正 quantity 以后续 evidence "
            "library 的 unique molecule 数为准。",
            "",
            "## 排除审计",
            "",
        ]
    )
    if exclusion_counts:
        for status, count in sorted(exclusion_counts.items()):
            lines.append(f"- `{status}`：{count:,}")
    else:
        lines.append("- 无匹配后排除记录。")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- H1/H2 是相对当前 D+C envelope 的额外推理桥，不是绝对生化距离。",
            "- retrieval/reasoning budget 按 tree node 计算；同一 node 内所有 measurement families 共享 top-3。",
            "- H2 assay 只说明存在一条有依据的间接机制路径，不等价于直接 BBB penetration evidence。",
            "- 所有 base-overlap assay 保留在 D/C，绝不重新贴成 H1/H2。",
            "- extension family 的 measured node 若已出现在 D/C census 中，manifest builder 直接失败。",
            "- Binding-only、缺少 functional readout、无 compound activity 或 scope 不合格的记录不进入主实验。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_state_census(
    config: DistanceAssayManifestConfig,
    base_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.classify_base_states is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        nodes = tuple(sorted(set(config.classify_base_states(row))))
        if not nodes:
            continue
        rows.append(
            {
                "assay_chembl_id": row.get("assay_chembl_id"),
                "tier": row.get("tier"),
                "measured_nodes": nodes,
                "description": row.get("description"),
                "target_chembl_id": row.get("target_chembl_id"),
                "target_pref_name": row.get("target_pref_name"),
            }
        )
    rows.sort(key=lambda row: (str(row["tier"]), str(row["assay_chembl_id"])))
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "|".join(str(item) for item in value)
                    if isinstance(value, (list, tuple, set))
                    else ""
                    if value is None
                    else value
                    for field in fieldnames
                    for value in (row.get(field),)
                }
            )


def _parse_args(config: DistanceAssayManifestConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--chembl-sqlite", required=True)
    parser.add_argument("--base-assays-csv", default=config.default_base_assays_csv)
    parser.add_argument("--out-dir", default=config.default_out_dir)
    parser.add_argument("--export-activities", action="store_true")
    parser.add_argument("--hash-source-db", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[distance-assay-manifest] {message}", file=sys.stderr, flush=True)
