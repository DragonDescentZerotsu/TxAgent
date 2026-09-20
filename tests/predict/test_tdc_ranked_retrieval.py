from __future__ import annotations

import json
import sqlite3

import pytest

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_candidates
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.utils.json import read_jsonl, sha256_file


EXPECTED = {
    "bbb_martins": (197, 530),
    "bioavailability_ma": (64, 128),
    "skin_reaction": (40, 82),
    "ames": (720, 1449),
    "dili": (47, 96),
    "carcinogens": (27, 56),
}


@pytest.mark.parametrize("task", EXPECTED)
def test_tdc_ranked_release_is_complete_morgan_only(task: str) -> None:
    root = cache_profile_root("ranked_level_retrieval_tdc_v1") / task
    index_path = root / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text())
    assert index["status"] == "complete"
    assert index["ranking_modes"] == ["morgan"]
    assert index["assay_transfer_status"] == "not_computed"
    assert index["label_release"] == {"benchmark": "tdc", "version": "v1"}

    train = {
        str(row["benchmark_row_id"]): list(map(str, row["source_record_ids"]))
        for row in read_jsonl(
            tdc_task_root(task) / "train_molecule_condition_labels.jsonl"
        )
    }
    for subset, expected_queries in zip(("valid", "test"), EXPECTED[task]):
        entry = index["splits"][subset]["levels"]["L1"]
        manifest_path = root / entry["manifest"]
        manifest = json.loads(manifest_path.read_text())
        database = manifest_path.with_name(manifest["database"])
        assert sha256_file(manifest_path) == entry["manifest_sha256"]
        assert sha256_file(database) == manifest["database_sha256"]
        assert manifest["query_count"] == expected_queries
        assert manifest["stored_rows"] == 100 * expected_queries
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == expected_queries
            query_id, query_parent = connection.execute(
                "SELECT benchmark_row_id,query_parent_id FROM queries LIMIT 1"
            ).fetchone()
            rows = connection.execute(
                "SELECT parent_id,morgan_rank,assay_transfer_score,assay_rank,"
                "morgan_context_id,assay_context_id FROM rankings "
                "WHERE benchmark_row_id=? ORDER BY morgan_rank", (query_id,),
            ).fetchall()
            assert len(rows) == len({row[0] for row in rows}) == 100
            assert query_parent not in {row[0] for row in rows}
            assert [row[1] for row in rows] == list(range(1, 101))
            assert all(row[2] is row[3] is row[5] is None for row in rows)
            for context_id in {row[4] for row in rows}:
                members = [
                    row[0] for row in connection.execute(
                        "SELECT source_row_uid FROM context_records "
                        "WHERE context_id=? ORDER BY within_context_rank", (context_id,),
                    )
                ]
                assert members == train[context_id]


def test_generic_reader_loads_tdc_morgan_and_rejects_assay_transfer() -> None:
    task = "bbb_martins"
    subset = "valid"
    root = cache_profile_root("ranked_level_retrieval_tdc_v1") / task
    index = root / "RELEASE_INDEX.json"
    manifest = root / "scaffold" / subset / "L1" / "VERSION.json"
    query = read_jsonl(tdc_task_root(task) / f"{subset}.jsonl")[0]
    queries = {str(query["benchmark_row_id"]): str(query["drug"])}
    policy = {
        "stages": {"L1": "morgan"},
        "cache_manifests": {"L1": str(manifest)},
        "cache_index": str(index),
    }

    molecules, later, audit = load_candidates(
        queries,
        task=task,
        subset=subset,
        policy=policy,
        molecule_limit=10,
        l1_limit=10,
    )
    assert later[str(query["benchmark_row_id"])] == {}
    assert len(molecules[str(query["benchmark_row_id"])]) == 10
    assert all(row["assay_rank"] is None for row in molecules[str(query["benchmark_row_id"])])
    assert all(
        row["gold_context_ids"]["assay_transfer"] is None
        for row in molecules[str(query["benchmark_row_id"])]
    )
    assert audit["cache_capacities"] == {"L1": 100}

    policy["stages"]["L1"] = "assay_transfer"
    with pytest.raises(ValueError, match="Requested 10 rows"):
        load_candidates(
            queries,
            task=task,
            subset=subset,
            policy=policy,
            molecule_limit=10,
            l1_limit=10,
        )
