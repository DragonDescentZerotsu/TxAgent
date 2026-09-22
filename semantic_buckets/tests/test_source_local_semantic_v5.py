from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from semantic_buckets import source_local_semantic_v5 as workflow


@pytest.mark.parametrize("task", ("ames", "dili", "carcinogens"))
def test_semantic_contract_starts_from_endpoint_and_ignores_unit_species(task):
    pair_columns, refinement = workflow._semantic_columns(workflow._contract(task))
    assert set(pair_columns) == set(refinement)
    for source, columns in refinement.items():
        assert columns[0] == "canonical_endpoint_concept"
        assert "canonical_measurement_scale_id" in columns
        assert "canonical_unit_text" not in columns
        assert not any("species" in column or "population" in column for column in columns)
        assert set(columns) <= set(pair_columns[source])


def test_semantic_contract_retains_assay_context_for_every_source():
    for task in ("ames", "dili", "carcinogens"):
        _, refinement = workflow._semantic_columns(workflow._contract(task))
        assert all("canonical_assay_context" in columns for columns in refinement.values())


def test_endpoint_specs_preserve_independent_endpoint_capacity():
    specs = workflow._endpoint_specs(
        ("http://dgx017:50001/v1", "http://dgx020:50002/v1"), 128
    )
    assert [spec["name"] for spec in specs] == ["dgx017:50001", "dgx020:50002"]
    assert [spec["max_inflight"] for spec in specs] == [128, 128]
    assert all(spec["provider"] == "local" for spec in specs)


def test_endpoint_specs_reject_invalid_capacity():
    with pytest.raises(ValueError, match="positive"):
        workflow._endpoint_specs(("http://dgx017:50001/v1",), 0)


def test_workflow_retains_optional_provider_pool(tmp_path: Path):
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    config = workflow.workflow(
        "ames",
        tmp_path / "release",
        tmp_path / "run",
        retrieval_index=tmp_path / "RELEASE_INDEX.json",
        provider_pool=pool,
    )

    assert config.provider_pool == pool.resolve()


def test_run_forwards_seed_request_cache(monkeypatch, tmp_path: Path):
    captured = {}
    config = SimpleNamespace(
        run=tmp_path / "run",
        review=tmp_path / "review",
        endpoints=({"max_inflight": 7},),
        task="ames",
        release_root=Path("release"),
    )
    monkeypatch.setattr(workflow, "configure_core", lambda _: None)
    monkeypatch.setattr(
        workflow.core,
        "run_semantic",
        lambda *args, **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(workflow, "write_json_atomic", lambda *args: None)
    seed = tmp_path / "seed.sqlite3"

    workflow.run(config, "review-hash", seed)

    assert captured["seed_request_cache"] == seed


def test_retrieval_scope_reads_only_later_level_uids(tmp_path: Path):
    level = tmp_path / "scaffold/valid/L2"
    level.mkdir(parents=True)
    database = level / "rankings.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE rankings(item_id TEXT)")
        connection.executemany("INSERT INTO rankings VALUES (?)", [("u1",), ("u2",)])
    manifest = level / "VERSION.json"
    manifest.write_text(json.dumps({
        "database": database.name,
        "database_sha256": workflow.file_sha256(database),
    }))
    entry = {
        "manifest": str(manifest.relative_to(tmp_path)),
        "manifest_sha256": workflow.file_sha256(manifest),
    }
    index = tmp_path / "RELEASE_INDEX.json"
    index.write_text(json.dumps({
        "status": "complete", "task_id": "ames",
        "splits": {
            "valid": {"levels": {"L1": {}, "L2": entry}},
            "test": {"levels": {"L2": entry}},
        },
    }))

    assert workflow._retrieval_uids(
        SimpleNamespace(task="ames", retrieval_index=index)
    ) == {"u1", "u2"}
