import json
import pandas as pd
import pytest
from optimization.semantic_buckets import inputs as artifacts
from predict.utils.json import sha256_file


def test_record_loader_resolves_manifest_and_filters_eligible_rows(
    tmp_path, monkeypatch
) -> None:
    release_root = tmp_path / "releases"
    root = release_root / "bbb_martins" / "v10"
    selected = {
        "semantic_map": "selected/semantic.parquet",
        "semantic_map_manifest": "selected/semantic.json",
        "readout_bucket_map": "selected/readout.parquet",
        "record_readout_bucket_map": "selected/records.parquet",
        "semantic_bucket_rankings": "selected/bucket_rankings.parquet",
        "record_relevance_rankings": "selected/record_rankings.parquet",
        "retrieval_eligibility": "selected/eligibility.parquet",
        "semantic_bucket_weights": "selected/weights.parquet",
    }
    (root / "selected").mkdir(parents=True)
    current = tmp_path / "data/evidence_libraries/bbb_martins/CURRENT"
    current.parent.mkdir(parents=True)
    current.write_text("v10\n", encoding="utf-8")
    for key, relative in selected.items():
        path = root / relative
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif key == "record_readout_bucket_map":
            pd.DataFrame(
                [
                    {
                        "canonical_record_id": "record-1",
                        "source_row_uid": "row-1",
                        "level": "L2",
                        "semantic_bucket_id": "semantic-1",
                        "readout_bucket_id": "readout-1",
                        "readout_depth": 1,
                    }
                ]
            ).to_parquet(path, index=False)
        elif key == "retrieval_eligibility":
            pd.DataFrame(
                [
                    {
                        "canonical_record_id": "record-1",
                        "source_row_uid": "row-1",
                        "level": "L2",
                        "semantic_bucket_id": "semantic-1",
                        "retrieval_eligible": True,
                    }
                ]
            ).to_parquet(path, index=False)
        else:
            pd.DataFrame({"placeholder": [1]}).to_parquet(path, index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_buckets.release.v1",
                "task": "bbb_martins",
                "evidence_library_version": "v10",
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(artifacts, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(artifacts, "REPOSITORY_ROOT", tmp_path)
    result = artifacts.load_record_bucket_map("bbb", eligible_only=True)
    resolved = artifacts.resolve_semantic_bucket_artifacts("bbb")
    assert result[
        ["canonical_record_id", "semantic_bucket_id", "readout_bucket_id"]
    ].to_dict("records") == [
        {
            "canonical_record_id": "record-1",
            "semantic_bucket_id": "semantic-1",
            "readout_bucket_id": "readout-1",
        }
    ]
    assert resolved.semantic_bucket_weights == root / "selected/weights.parquet"


def test_current_pointer_is_owned_by_evidence_library(tmp_path, monkeypatch) -> None:
    release_root = tmp_path / "releases"
    current = tmp_path / "data/evidence_libraries/bbb_martins/CURRENT"
    current.parent.mkdir(parents=True)
    current.write_text("v10_main_universe_v1\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(artifacts, "REPOSITORY_ROOT", tmp_path)
    assert artifacts.resolve_release("bbb", "CURRENT") == "v10_main_universe_v1"


def test_unweighted_semantic_release_resolves_without_rankings(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "releases/dili/v10_main_universe_v3"
    generation = root / "generation"
    generation.mkdir(parents=True)
    selected = {
        "semantic_map": "generation/semantic.parquet",
        "semantic_map_manifest": "generation/semantic.json",
        "record_semantic_bucket_map": "generation/records.parquet",
    }
    (generation / "semantic.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"atom_id": ["a"], "semantic_bucket_id": ["b"]}).to_parquet(
        generation / "semantic.parquet", index=False
    )
    pd.DataFrame({"source_row_uid": ["u"], "semantic_bucket_id": ["b"]}).to_parquet(
        generation / "records.parquet", index=False
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_buckets.release.v1",
                "task": "dili",
                "evidence_library_version": "v10_main_universe_v3",
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    current = tmp_path / "data/evidence_libraries/dili/CURRENT"
    current.parent.mkdir(parents=True)
    current.write_text("v10_main_universe_v3\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "RELEASE_ROOT", tmp_path / "releases")
    monkeypatch.setattr(artifacts, "REPOSITORY_ROOT", tmp_path)
    release = artifacts.resolve_semantic_bucket_artifacts("dili")
    assert release.semantic_bucket_rankings is None
    assert release.record_relevance_rankings is None
    assert len(artifacts.load_record_bucket_map("dili")) == 1


def test_semantic_only_release_resolves_skin_record_map(tmp_path, monkeypatch) -> None:
    release_root = tmp_path / "releases"
    root = release_root / "skin_reaction" / "v10_main_universe_v6"
    selected = {
        "semantic_map": "generation/semantic.parquet",
        "semantic_map_manifest": "generation/semantic.json",
        "record_semantic_bucket_map": "generation/records.parquet",
        "semantic_bucket_rankings": "weighting/buckets.parquet",
        "record_relevance_rankings": "weighting/records.parquet",
        "semantic_bucket_weights": "weighting/weights.parquet",
    }
    for key, relative in selected.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            pd.DataFrame({"placeholder": [1]}).to_parquet(path, index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_buckets.release.v1",
                "task": "skin_reaction",
                "evidence_library_version": "v10_main_universe_v6",
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    current = tmp_path / "data/evidence_libraries/skin_reaction/CURRENT"
    current.parent.mkdir(parents=True)
    current.write_text("v10_main_universe_v6\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(artifacts, "REPOSITORY_ROOT", tmp_path)
    resolved = artifacts.resolve_semantic_bucket_artifacts("skin", "CURRENT")
    mapping = artifacts.load_record_bucket_map("skin", "CURRENT")
    assert resolved.record_readout_bucket_map is None
    assert (
        resolved.record_semantic_bucket_map
        == root / selected["record_semantic_bucket_map"]
    )
    assert list(mapping.columns) == ["placeholder"]


def test_selected_central_ledger_is_read_separately_from_legacy_record_map(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "releases/ames/v10"
    root.mkdir(parents=True)
    selected = {
        "semantic_map": "semantic.parquet",
        "semantic_map_manifest": "semantic.json",
        "record_semantic_bucket_map": "records.parquet",
        "central_ledger": "central.parquet",
    }
    (root / "semantic.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"atom_id": ["a"]}).to_parquet(root / "semantic.parquet")
    pd.DataFrame({"source_row_uid": ["u"]}).to_parquet(root / "records.parquet")
    pd.DataFrame(
        {"source_row_uid": ["u"], "assignment_status": ["assigned"]}
    ).to_parquet(root / "central.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_buckets.release.v1",
                "task": "ames",
                "evidence_library_version": "v10",
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(artifacts, "RELEASE_ROOT", tmp_path / "releases")
    assert artifacts.load_central_ledger("ames", "v10").assignment_status.tolist() == [
        "assigned"
    ]
    assert artifacts.load_record_bucket_map("ames", "v10").source_row_uid.tolist() == [
        "u"
    ]


def test_central_ledger_requires_reviewed_mean_projection(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "releases/ames/v10"
    weighting = root / "weighting"
    weighting.mkdir(parents=True)
    ledger = root / "central.parquet"
    weights = weighting / "weights.parquet"
    rankings = weighting / "record_relevance_rankings.parquet"
    pd.DataFrame({"source_row_uid": ["u"]}).to_parquet(ledger)
    pd.DataFrame(
        {"level": ["L2"], "semantic_bucket_id": ["family"], "weight": [0.5]}
    ).to_parquet(weights)
    pd.DataFrame(
        {
            "source_row_uid": ["u"],
            "level": ["L2"],
            "semantic_bucket_id": ["family"],
            "weight": [0.5],
        }
    ).to_parquet(rankings)
    (root / "semantic.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"atom_id": ["a"]}).to_parquet(root / "semantic.parquet")
    pd.DataFrame({"source_row_uid": ["u"]}).to_parquet(root / "records.parquet")
    projection = {
        "schema_version": "semantic_retrieval_mean_weights.v1",
        "status": "candidate_unreviewed",
        "inputs": {
            "ledger": {"sha256": sha256_file(ledger)},
            "weights": {"sha256": sha256_file(weights)},
        },
        "projection": {"sha256": sha256_file(rankings)},
    }
    (weighting / "manifest.json").write_text(json.dumps(projection), encoding="utf-8")
    selected = {
        "semantic_map": "semantic.parquet",
        "semantic_map_manifest": "semantic.json",
        "record_semantic_bucket_map": "records.parquet",
        "central_ledger": "central.parquet",
        "semantic_bucket_weights": "weighting/weights.parquet",
        "record_relevance_rankings": "weighting/record_relevance_rankings.parquet",
    }
    files = {
        key: {"sha256": sha256_file(root / path)} for key, path in selected.items()
    }
    files["weight_projection_manifest"] = {
        "sha256": sha256_file(weighting / "manifest.json")
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "semantic_buckets.release.v1",
                "status": "complete_reviewed",
                "task": "ames",
                "evidence_library_version": "v10",
                "selected": selected,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(artifacts, "RELEASE_ROOT", tmp_path / "releases")
    with pytest.raises(ValueError, match="not ledger-bound"):
        artifacts.load_reviewed_record_weights("ames", "v10")
    projection["status"] = "complete_reviewed"
    (weighting / "manifest.json").write_text(json.dumps(projection), encoding="utf-8")
    release = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    release["files"]["weight_projection_manifest"]["sha256"] = sha256_file(
        weighting / "manifest.json"
    )
    (root / "manifest.json").write_text(json.dumps(release), encoding="utf-8")
    rows, _ = artifacts.load_reviewed_record_weights("ames", "v10")
    assert rows["u", "L2"]["semantic_weight"] == 0.5
