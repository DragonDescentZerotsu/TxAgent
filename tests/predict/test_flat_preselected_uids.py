import csv
import json
import sqlite3
from pathlib import Path

import pytest

from predict.harnesses.branches.flat import (
    PRESELECTED_DIRECT_SCHEMA,
    PRESELECTED_UID_SCHEMA,
    _load_preselected_contexts,
    _load_preselected_uids,
    _preselected_query_indices,
)
from predict.utils.json import sha256_file


def test_exact_score_direct_manifest_uses_frozen_scores_and_coverage(tmp_path: Path) -> None:
    database = tmp_path / "scores.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE scores (query_id TEXT, level TEXT, item_id TEXT, parent_id TEXT, score REAL, score_key TEXT)")
        connection.execute("INSERT INTO scores VALUES ('q1','L1','c1','p1',0.8,'exact')")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"status": "complete", "task": "ames",
                                   "database": database.name, "database_sha256": sha256_file(database)}))
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text("version: 22\n")
    later_index = tmp_path / "RELEASE_INDEX.json"
    later_index.write_text("{}\n")
    records = tmp_path / "selected_contexts.tsv"
    fields = ("benchmark_row_id", "selection_rank", "context_id", "parent_id",
              "gold_label", "assay_transfer_score", "score_key")
    with records.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(dict(zip(fields, ("q1", 1, "c1", "p1", 1, 0.8, "exact"))))
    coverage = tmp_path / "coverage.tsv"
    with coverage.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("benchmark_row_id", "scored_candidates",
            "candidate_count", "coverage_band", "direct_scored", "indirect_scored"), delimiter="\t")
        writer.writeheader()
        writer.writerow(dict(benchmark_row_id="q1", scored_candidates=70, candidate_count=100,
                             coverage_band="high", direct_scored=1, indirect_scored=69))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": PRESELECTED_DIRECT_SCHEMA, "status": "complete",
        "benchmark": "tdc", "task_id": "ames", "budget": 1,
        "benchmark_row_ids": ["q1"],
        "records": {"path": records.name, "sha256": sha256_file(records), "row_count": 1},
        "score_policy": {"version": "tdc_exact_direct_40_70",
            "overlay_manifest": str(overlay), "overlay_manifest_sha256": sha256_file(overlay),
            "database_sha256": sha256_file(database),
            "cache_bundle": str(bundle), "cache_bundle_sha256": sha256_file(bundle),
            "later_index": str(later_index), "later_index_sha256": sha256_file(later_index),
            "coverage": {"path": coverage.name, "sha256": sha256_file(coverage), "row_count": 1}},
    }))
    selected, receipt = _load_preselected_contexts(
        manifest, task="ames", queries={"q1": "CC"}, budget=1, benchmark="tdc")
    assert selected == {"q1": ["c1"]}
    assert receipt["score_visibility"] == {"q1": "high"}
    assert receipt["_selected_scores"] == {"q1": {("L1", "p1"): 0.8}}
    coverage.write_text(coverage.read_text().replace("high", "middle"))
    policy = json.loads(manifest.read_text())
    policy["score_policy"]["coverage"]["sha256"] = sha256_file(coverage)
    manifest.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="coverage band differs"):
        _load_preselected_contexts(
            manifest, task="ames", queries={"q1": "CC"}, budget=1, benchmark="tdc")


def _manifest(tmp_path: Path) -> Path:
    cache_index = tmp_path / "RELEASE_INDEX.json"
    cache_index.write_text("{}\n", encoding="utf-8")
    records = tmp_path / "selected_uids.tsv"
    with records.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark_row_id", "level", "selection_rank", "source_row_uid"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows([
            {"benchmark_row_id": "q1", "level": "L2", "selection_rank": 1, "source_row_uid": "u2"},
            {"benchmark_row_id": "q1", "level": "L2", "selection_rank": 2, "source_row_uid": "u1"},
        ])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": PRESELECTED_UID_SCHEMA,
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "benchmark_row_ids": ["q1"],
        "k_per_level": {"L2": 2},
        "release_index_sha256": sha256_file(cache_index),
        "objective": {"version": "test"},
        "records": {
            "path": records.name,
            "sha256": sha256_file(records),
            "row_count": 2,
        },
    }), encoding="utf-8")
    return manifest


def test_manifest_preserves_optimizer_uid_order(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    selected, receipt = _load_preselected_uids(
        manifest,
        task="bbb_martins",
        subset="valid",
        queries={"q1": "CCO"},
        levels={"L2"},
        limits={"L2": 2},
        cache_index=tmp_path / "RELEASE_INDEX.json",
    )

    assert selected == {"q1": {"L2": ["u2", "u1"]}}
    assert receipt["objective"] == {"version": "test"}


def test_manifest_rejects_tampered_uid_table(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "selected_uids.tsv").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        _load_preselected_uids(
            manifest,
            task="bbb_martins",
            subset="valid",
            queries={"q1": "CCO"},
            levels={"L2"},
            limits={"L2": 2},
            cache_index=tmp_path / "RELEASE_INDEX.json",
        )


def test_manifest_query_ids_select_the_matching_canonical_rows(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    indices = _preselected_query_indices(
        manifest,
        [
            {"benchmark_row_id": "q0", "drug": "CC"},
            {"benchmark_row_id": "q1", "drug": "CCO"},
        ],
        task="bbb_martins",
        subset="valid",
    )

    assert indices == [1]


def test_direct_manifest_preserves_context_order(tmp_path: Path) -> None:
    records = tmp_path / "selected_contexts.tsv"
    with records.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "benchmark_row_id", "selection_rank", "context_id",
                "parent_id", "gold_label",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows([
            {"benchmark_row_id": "q1", "selection_rank": 1, "context_id": "c2", "parent_id": "p2", "gold_label": "pass"},
            {"benchmark_row_id": "q1", "selection_rank": 2, "context_id": "c1", "parent_id": "p1", "gold_label": "fail"},
        ])
    manifest = tmp_path / "direct.json"
    manifest.write_text(json.dumps({
        "schema_version": PRESELECTED_DIRECT_SCHEMA,
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid_small",
        "benchmark_row_ids": ["q1"],
        "budget": 2,
        "objective": {"version": "test"},
        "records": {
            "path": records.name,
            "sha256": sha256_file(records),
            "row_count": 2,
        },
    }), encoding="utf-8")

    selected, receipt = _load_preselected_contexts(
        manifest, task="bbb_martins", queries={"q1": "CCO"}, budget=2,
    )

    assert selected == {"q1": ["c2", "c1"]}
    assert receipt["objective"] == {"version": "test"}
    with pytest.raises(ValueError, match="Incompatible preselected direct manifest"):
        _load_preselected_contexts(
            manifest, task="bbb_martins", queries={"q1": "CCO"}, budget=2,
            benchmark="tdc",
        )
