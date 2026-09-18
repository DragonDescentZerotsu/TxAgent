"""Audit prepared control/semantic L2 panels against their immutable cache."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
from statistics import mean, median
from typing import Any

import yaml

from predict.utils.json import sha256_file


MODES = ("morgan_parent_control", "morgan_parent_semantic")


def _cache_rows(
    manifest_path: Path,
) -> tuple[dict, list[dict[str, Any]], dict[tuple[str, str], list[str]]]:
    manifest = json.loads(manifest_path.read_text())
    database = manifest_path.with_name(manifest["database"])
    query = """SELECT b.benchmark_row_id,a.mode,a.selection_rank,a.parent_rank,
        r.external_record_id,r.parent_id,a.retrieval_eligible
        FROM l2_assignments AS a JOIN l2_records AS r USING(record_key)
        JOIN benchmark_queries AS b USING(query_id)
        ORDER BY b.query_id,a.mode,a.selection_rank"""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute(query)]
        parents = list(db.execute(
            """SELECT b.benchmark_row_id,p.mode,p.parent_id FROM l2_parent_assignments AS p
               JOIN benchmark_queries AS b USING(query_id) ORDER BY b.query_id,p.mode,p.parent_rank"""
        ))
    panels: dict[tuple[str, str], list[str]] = defaultdict(list)
    for benchmark, mode, parent in parents:
        panels[(str(benchmark), str(mode))].append(str(parent))
    return manifest, rows, panels


def _prepared_ids(root: Path, task: str) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    paths = sorted((root / task / "queries").glob("*/levels/level_2/prepared.json"))
    for path in paths:
        prepared = json.loads(path.read_text())
        records = {
            str(card.get("_canonical_record_id") or card_id)
            for analog in prepared["active_evidence"].values()
            for card_id, card in (analog.get("cards") or {}).items()
            if int(card.get("first_seen_level") or 0) == 2
        }
        output[str(prepared["benchmark_row_id"])] = records
    return output


def _query_audit(
    task: str, benchmark: str, rows: list[dict[str, Any]],
    panels: dict[tuple[str, str], list[str]],
) -> dict[str, Any]:
    by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}
    control, semantic = by_mode[MODES[0]], by_mode[MODES[1]]
    control_ids = {row["external_record_id"] for row in control}
    semantic_ids = {row["external_record_id"] for row in semantic}
    intersection, union = control_ids & semantic_ids, control_ids | semantic_ids
    control_only = [row for row in control if row["external_record_id"] not in semantic_ids]
    control_parents = {row["parent_id"] for row in control}
    semantic_parents = {row["parent_id"] for row in semantic}
    return {
        "task": task, "benchmark_row_id": benchmark,
        "control_records": len(control), "semantic_records": len(semantic),
        "record_intersection": len(intersection), "record_union": len(union),
        "record_jaccard": len(intersection) / len(union) if union else 1.0,
        "identical_record_set": control_ids == semantic_ids,
        "control_selected_parents": len(control_parents),
        "semantic_selected_parents": len(semantic_parents),
        "selected_parent_intersection": len(control_parents & semantic_parents),
        "fixed_parent_panel_identical": panels[(benchmark, MODES[0])] == panels[(benchmark, MODES[1])],
        "control_bad_records": sum(row["retrieval_eligible"] == 0 for row in control),
        "control_unmapped_records": sum(row["retrieval_eligible"] is None for row in control),
        "semantic_bad_records": sum(row["retrieval_eligible"] == 0 for row in semantic),
        "semantic_unmapped_records": sum(row["retrieval_eligible"] is None for row in semantic),
        "control_only_bad_records": sum(row["retrieval_eligible"] == 0 for row in control_only),
        "control_only_unmapped_records": sum(row["retrieval_eligible"] is None for row in control_only),
        "control_only_eligible_records": sum(row["retrieval_eligible"] == 1 for row in control_only),
        "semantic_only_records": len(semantic_ids - control_ids),
    }


def _aggregate(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": task, "queries": len(rows),
        "mean_record_intersection": mean(row["record_intersection"] for row in rows),
        "median_record_intersection": median(row["record_intersection"] for row in rows),
        "mean_record_jaccard": mean(row["record_jaccard"] for row in rows),
        "identical_record_sets": sum(row["identical_record_set"] for row in rows),
        "full_control_25": sum(row["control_records"] == 25 for row in rows),
        "full_semantic_25": sum(row["semantic_records"] == 25 for row in rows),
        "fixed_parent_panels_identical": sum(row["fixed_parent_panel_identical"] for row in rows),
        "control_bad_records": sum(row["control_bad_records"] for row in rows),
        "control_unmapped_records": sum(row["control_unmapped_records"] for row in rows),
        "semantic_bad_records": sum(row["semantic_bad_records"] for row in rows),
        "semantic_unmapped_records": sum(row["semantic_unmapped_records"] for row in rows),
        "control_only_eligible_records": sum(row["control_only_eligible_records"] for row in rows),
        "semantic_only_records": sum(row["semantic_only_records"] for row in rows),
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(cache_bundle: Path, control_root: Path, semantic_root: Path, output: Path) -> None:
    bundle = yaml.safe_load(cache_bundle.read_text())
    query_rows, manifests = [], {}
    for task, subsets in bundle["caches"].items():
        path = (cache_bundle.parent / subsets["valid"]).resolve()
        manifests[task], rows, panels = _cache_rows(path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["benchmark_row_id"])].append(row)
        control_prepared = _prepared_ids(control_root, task)
        semantic_prepared = _prepared_ids(semantic_root, task)
        if control_prepared.keys() != semantic_prepared.keys():
            raise ValueError(f"{task}: prepared arms contain different query identities")
        if not control_prepared.keys() <= grouped.keys():
            raise ValueError(f"{task}: prepared query is absent from the cache")
        for benchmark in sorted(control_prepared):
            selected = grouped[benchmark]
            audit = _query_audit(task, benchmark, selected, panels)
            expected = {mode: {row["external_record_id"] for row in selected if row["mode"] == mode}
                        for mode in MODES}
            if control_prepared.get(benchmark) != expected[MODES[0]]:
                raise ValueError(f"{task}/{benchmark}: control prepared selection differs from cache")
            if semantic_prepared.get(benchmark) != expected[MODES[1]]:
                raise ValueError(f"{task}/{benchmark}: semantic prepared selection differs from cache")
            query_rows.append(audit)
    output.mkdir(parents=True, exist_ok=False)
    aggregates = [_aggregate(task, [row for row in query_rows if row["task"] == task])
                  for task in sorted(manifests)]
    _write_tsv(output / "per_query_overlap.tsv", query_rows)
    _write_tsv(output / "aggregate_overlap.tsv", aggregates)
    identity = {"schema_version": "context_l2_selection_overlap.v1", "cache_bundle": {
        "path": str(cache_bundle.resolve()), "sha256": sha256_file(cache_bundle)},
        "control_root": str(control_root.resolve()), "semantic_root": str(semantic_root.resolve()),
        "cache_content_ids": {task: manifest["content_id"] for task, manifest in manifests.items()},
        "query_rows": len(query_rows), "status": "complete"}
    (output / "manifest.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    (output / "REPORT.md").write_text(
        "# L2 selection overlap\n\nThe TSV files are authoritative. They compare prepared control and "
        "semantic record sets and verify both against the immutable cache.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-bundle", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache_bundle, args.control_root, args.semantic_root, args.output_root)


if __name__ == "__main__":
    main()
