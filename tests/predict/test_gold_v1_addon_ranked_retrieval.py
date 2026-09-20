from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq
import pytest

from predict.retrieval.assay_reranking.ranked_uid_retrieval import load_candidates
from predict.retrieval.assay_reranking.runtime import DATA_ACTIVE_CACHE_ROOT, cache_profile_root
from predict.utils.json import read_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "ranked_level_retrieval_gold_v1_addon_v1"
TASKS = {"ames": "Ames", "dili": "DILI", "carcinogens": "Carcinogens"}


@pytest.mark.parametrize("task,gold_name", TASKS.items())
def test_gold_v1_contract_and_morgan_only_cache(task: str, gold_name: str) -> None:
    gold = ROOT / "data/gold_labels" / gold_name / "v1/scaffold"
    contract = json.loads((gold / "voter_contract_manifest.json").read_text())
    assert contract["status"] == "complete"
    assert contract["counts"]["physical_members"] == pq.ParquetFile(
        gold / "voter_membership.parquet"
    ).metadata.num_rows
    for output in contract["outputs"].values():
        assert sha256_file(ROOT / output["path"]) == output["sha256"]

    cache = cache_profile_root(PROFILE) / task
    index = json.loads((cache / "RELEASE_INDEX.json").read_text())
    assert cache_profile_root(PROFILE).parent == DATA_ACTIVE_CACHE_ROOT
    assert index["ranking_modes"] == ["morgan"]
    assert index["assay_transfer_status"] == "not_computed"
    for subset in ("valid", "test"):
        manifest_path = cache / index["splits"][subset]["levels"]["L1"]["manifest"]
        manifest = json.loads(manifest_path.read_text())
        with sqlite3.connect(manifest_path.with_name(manifest["database"])) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM rankings WHERE assay_rank IS NOT NULL "
                "OR assay_transfer_score IS NOT NULL OR assay_context_id IS NOT NULL"
            ).fetchone() == (0,)


@pytest.mark.parametrize("task,gold_name", TASKS.items())
def test_gold_v1_addon_reader_loads_morgan_and_rejects_assay(task: str, gold_name: str) -> None:
    cache = cache_profile_root(PROFILE) / task
    query = read_jsonl(ROOT / "data/gold_labels" / gold_name / "v1/scaffold/valid.jsonl")[0]
    queries = {str(query["benchmark_row_id"]): str(query["drug"])}
    manifest = cache / "scaffold/valid/L1/VERSION.json"
    policy = {
        "stages": {"L1": "morgan"},
        "cache_manifests": {"L1": str(manifest)},
        "cache_index": str(cache / "RELEASE_INDEX.json"),
    }
    molecules, later, _ = load_candidates(
        queries, task=task, subset="valid", policy=policy,
        molecule_limit=1, l1_limit=100,
    )
    assert len(molecules[next(iter(queries))]) == 1
    assert molecules[next(iter(queries))][0]["l1_records"]
    assert later[next(iter(queries))] == {}

    policy["stages"]["L1"] = "assay_transfer"
    with pytest.raises(ValueError, match="has 0"):
        load_candidates(
            queries, task=task, subset="valid", policy=policy,
            molecule_limit=1, l1_limit=100,
        )
