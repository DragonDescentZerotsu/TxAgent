from pathlib import Path
import pandas as pd
import pytest
from optimization.semantic_buckets import ledger as central_ledger


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    rows = [
        ("c1", "u1", "source", "pair-a"),
        ("c2", "u2", "source", "pair-a"),
        ("c3", "u3", "source", "pair-b"),
        ("c4", "u4", "source", "pair-c"),
        ("c5", "u5", "source", "pair-a"),
    ]
    stage = tmp_path / "stage.parquet"
    levels = tmp_path / "levels.parquet"
    mapping = tmp_path / "mapping.parquet"
    pd.DataFrame(
        rows,
        columns=[
            "canonical_record_id",
            "source_row_uid",
            "source_id",
            "pair_bucket_key",
        ],
    ).to_parquet(stage, index=False)
    pd.DataFrame(
        [("c1", "u1", 2), ("c2", "u2", 3), ("c3", "u3", 2), ("c5", "u5", 2)],
        columns=["canonical_record_id", "source_row_uid", "level"],
    ).to_parquet(levels, index=False)
    pd.DataFrame(
        [
            ("c1", "u1", "source", "L2", "family-2"),
            ("c2", "u2", "source", "L3", "family-3"),
            ("c3", "u3", "source", "L2", "family-4"),
        ],
        columns=[
            "canonical_record_id",
            "source_row_uid",
            "source_id",
            "level",
            "semantic_bucket_id",
        ],
    ).to_parquet(mapping, index=False)
    manifest = tmp_path / "selected.json"
    manifest.write_text("{}", encoding="utf-8")
    paths = {
        "stage3": stage,
        "levels": levels,
        "record_map": mapping,
        "semantic_manifest": manifest,
    }
    monkeypatch.setattr(central_ledger, "_inputs", lambda task, release: paths)
    return paths


def test_pair_inherits_cross_level_families_and_mean_weight(tmp_path, monkeypatch):
    _source(tmp_path, monkeypatch)
    root = tmp_path / "ledger"
    manifest = central_ledger.build("ames", "v1", root)
    assert manifest["assignment_status_counts"] == {"assigned": 4, "unleveled": 1}
    ledger = root / "central_ledger.parquet"
    rows = pd.read_parquet(ledger).set_index("source_row_uid")
    assert rows.loc["u1", "own_level_semantic_bucket_id"] == "family-2"
    assert rows.loc["u1", "semantic_memberships"].tolist() == [
        {"level": "L2", "semantic_bucket_id": "family-2"},
        {"level": "L3", "semantic_bucket_id": "family-3"},
    ]
    assert rows.loc["u4", "assignment_status"] == "unleveled"
    assert rows.loc["u5", "own_level_semantic_bucket_id"] == "family-2"
    weights = tmp_path / "weights.parquet"
    retrieval = tmp_path / "retrieval.parquet"
    pd.DataFrame(
        [("L2", "family-2", 0.2), ("L3", "family-3", 0.8), ("L2", "family-4", 0.4)],
        columns=["level", "semantic_bucket_id", "weight"],
    ).to_parquet(weights, index=False)
    pd.DataFrame(
        [("u1", "L2"), ("u2", "L3")], columns=["source_row_uid", "level"]
    ).to_parquet(retrieval, index=False)
    assert central_ledger.audit_retrieval_coverage(ledger, weights, retrieval) == {
        "retrieval_uids": 2,
        "missing_uid": 0,
        "unassigned_uid": 0,
        "uids_missing_component_weight": 0,
        "unweighted_touched_families": 0,
    }
    projection = tmp_path / "projection"
    central_ledger.project_retrieval_weights(ledger, weights, retrieval, projection)
    ranked = pd.read_parquet(projection / "record_relevance_rankings.parquet")
    assert dict(zip(ranked.source_row_uid, ranked.weight)) == {"u1": 0.5, "u2": 0.5}
    assert set(ranked.component_count) == {2}
    assert dict(zip(ranked.source_row_uid, ranked.semantic_bucket_id)) == {
        "u1": "family-2",
        "u2": "family-3",
    }


def test_projection_rejects_partial_weights_and_pending_uid(tmp_path, monkeypatch):
    _source(tmp_path, monkeypatch)
    ledger = tmp_path / "ledger"
    central_ledger.build("ames", "v1", ledger)
    weights = tmp_path / "weights.parquet"
    retrieval = tmp_path / "retrieval.parquet"
    pd.DataFrame(
        [("L2", "family-2", 0.2)], columns=["level", "semantic_bucket_id", "weight"]
    ).to_parquet(weights, index=False)
    pd.DataFrame([("u1", "L2")], columns=["source_row_uid", "level"]).to_parquet(
        retrieval, index=False
    )
    assert (
        central_ledger.audit_retrieval_coverage(
            ledger / "central_ledger.parquet", weights, retrieval
        )["unweighted_touched_families"]
        == 1
    )
    with pytest.raises(ValueError, match="unweighted memberships"):
        central_ledger.project_retrieval_weights(
            ledger / "central_ledger.parquet", weights, retrieval, tmp_path / "missing"
        )
    pd.DataFrame([("u4", "L2")], columns=["source_row_uid", "level"]).to_parquet(
        retrieval, index=False
    )
    with pytest.raises(ValueError, match="coverage incomplete"):
        central_ledger.project_retrieval_weights(
            ledger / "central_ledger.parquet", weights, retrieval, tmp_path / "pending"
        )


def test_provisional_mean_does_not_assign_a_semantic_bucket(tmp_path, monkeypatch):
    paths = _source(tmp_path, monkeypatch)
    mapping = pd.read_parquet(paths["record_map"])
    mapping[mapping.source_row_uid.ne("u3")].to_parquet(
        paths["record_map"], index=False
    )
    ledger = tmp_path / "ledger"
    central_ledger.build("ames", "v1", ledger)
    weights = tmp_path / "weights.parquet"
    retrieval = tmp_path / "retrieval.parquet"
    pd.DataFrame(
        [("L2", "family-2", 0.2), ("L2", "other", 0.8), ("L3", "family-3", 0.9)],
        columns=["level", "semantic_bucket_id", "weight"],
    ).to_parquet(weights, index=False)
    pd.DataFrame([("u3", "L2")], columns=["source_row_uid", "level"]).to_parquet(
        retrieval, index=False
    )
    projection = tmp_path / "projection"
    manifest = central_ledger.project_retrieval_weights(
        ledger / "central_ledger.parquet",
        weights,
        retrieval,
        projection,
        provisional_mean=True,
    )
    row = pd.read_parquet(projection / "record_relevance_rankings.parquet").iloc[0]
    assert manifest["provisional_uids"] == 1
    assert row.weight == 0.5
    assert pd.isna(row.semantic_bucket_id)
    assert row.weight_source == "provisional_task_level_mean"


def test_same_level_pair_cannot_have_conflicting_families(tmp_path, monkeypatch):
    paths = _source(tmp_path, monkeypatch)
    levels = pd.read_parquet(paths["levels"])
    levels.loc[levels.source_row_uid.eq("u2"), "level"] = 2
    levels.to_parquet(paths["levels"], index=False)
    mapping = pd.read_parquet(paths["record_map"])
    mapping.loc[mapping.source_row_uid.eq("u2"), "level"] = "L2"
    mapping.to_parquet(paths["record_map"], index=False)
    with pytest.raises(ValueError, match="conflicting families"):
        central_ledger.build("ames", "v1", tmp_path / "invalid")


def test_pending_atom_worklist_excludes_unleveled_rows(tmp_path, monkeypatch):
    paths = _source(tmp_path, monkeypatch)
    mapping = pd.read_parquet(paths["record_map"])
    mapping[mapping.source_row_uid.ne("u3")].to_parquet(
        paths["record_map"], index=False
    )
    ledger = tmp_path / "ledger"
    central_ledger.build("ames", "v1", ledger)
    worklist = tmp_path / "worklist"
    manifest = central_ledger.prepare_pending_atoms(
        ledger / "central_ledger.parquet", worklist
    )
    assert manifest["atoms"]["rows"] == 1
    row = pd.read_parquet(worklist / "pending_atoms.parquet").iloc[0]
    assert (row.level, row.pair_bucket_key, row.representative_uid) == (
        "L2",
        "pair-b",
        "u3",
    )
