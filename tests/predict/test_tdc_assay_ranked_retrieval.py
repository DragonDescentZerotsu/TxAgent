from __future__ import annotations

import json
import sqlite3

import pytest

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_candidates
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.retrieval.assay_reranking.score_tdc_ranked_retrieval import (
    BASE_PROFILE,
    PROFILE,
    SKIN_PROFILE,
)
from predict.utils.json import read_jsonl, sha256_file


EXPECTED = {
    "bbb_martins": {"valid": 197, "test": 530},
    "bioavailability_ma": {"valid": 64, "test": 128},
    "skin_reaction": {"valid": 40, "test": 82},
}
PROFILES = {task: (SKIN_PROFILE if task == "skin_reaction" else PROFILE) for task in EXPECTED}


@pytest.mark.parametrize("task", EXPECTED)
def test_tdc_assay_release_preserves_morgan_and_adds_dense_assay_ranks(task: str) -> None:
    root = cache_profile_root(PROFILES[task]) / task
    base = cache_profile_root(BASE_PROFILE) / task
    index = json.loads((root / "RELEASE_INDEX.json").read_text())
    assert index["status"] == "complete"
    assert index["ranking_modes"] == ["morgan", "assay-transfer"]
    assert index["assay_transfer_status"] == "complete"
    assert index["label_release"] == {"benchmark": "tdc", "version": "v1"}
    assert (root / "evidence/VERSION.json").read_bytes() == (base / "evidence/VERSION.json").read_bytes()
    assert (root / "evidence/records.parquet").read_bytes() == (base / "evidence/records.parquet").read_bytes()

    immutable = (
        "benchmark_row_id,item_id,parent_id,parent_smiles,morgan_similarity,"
        "parent_morgan_rank,within_parent_rank,morgan_rank,morgan_context_id,"
        "morgan_member_count"
    )
    for subset, expected_queries in EXPECTED[task].items():
        manifest_path = root / "scaffold" / subset / "L1/VERSION.json"
        manifest = json.loads(manifest_path.read_text())
        base_manifest = json.loads((base / "scaffold" / subset / "L1/VERSION.json").read_text())
        assert sha256_file(manifest_path) == index["splits"][subset]["levels"]["L1"]["manifest_sha256"]
        with (
            sqlite3.connect(manifest_path.with_name(manifest["database"])) as current,
            sqlite3.connect(base / "scaffold" / subset / "L1" / base_manifest["database"]) as original,
        ):
            assert current.execute(f"SELECT {immutable} FROM rankings ORDER BY benchmark_row_id,item_id").fetchall() == original.execute(f"SELECT {immutable} FROM rankings ORDER BY benchmark_row_id,item_id").fetchall()
            assert current.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == expected_queries
            assert current.execute("SELECT COUNT(*) FROM rankings WHERE assay_transfer_score BETWEEN 0 AND 1 AND assay_rank BETWEEN 1 AND 100 AND assay_context_id=morgan_context_id AND assay_member_count=morgan_member_count").fetchone()[0] == 100 * expected_queries


@pytest.mark.parametrize("method", ["morgan", "assay_transfer"])
def test_generic_reader_loads_both_tdc_rankings(method: str) -> None:
    task, subset = "bbb_martins", "valid"
    root = cache_profile_root(PROFILE) / task
    query = read_jsonl(tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl")[0]
    query_id = str(query["benchmark_row_id"])
    molecules, later, audit = load_candidates(
        {query_id: str(query["drug"])},
        task=task,
        subset=subset,
        policy={
            "stages": {"L1": method},
            "cache_manifests": {"L1": str(root / "scaffold" / subset / "L1/VERSION.json")},
            "cache_index": str(root / "RELEASE_INDEX.json"),
        },
        molecule_limit=10,
        l1_limit=10,
    )
    assert len(molecules[query_id]) == 10
    assert later[query_id] == {}
    assert audit["cache_capacities"] == {"L1": 100}
