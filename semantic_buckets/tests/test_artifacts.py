import json

import pandas as pd

from semantic_buckets import artifacts
from semantic_buckets import publication


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

    assert result[["canonical_record_id", "semantic_bucket_id", "readout_bucket_id"]].to_dict(
        "records"
    ) == [
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


def test_refresh_upstream_preserves_reviewed_release(tmp_path, monkeypatch) -> None:
    release = tmp_path / "release"
    release.mkdir()
    original = {
        "task": "bioavailability_ma",
        "selected": {"semantic_map": "frozen.parquet"},
        "migration": {"version": "reviewed"},
        "upstream": {"pair_bucket_records_sha256": "old"},
    }
    (release / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
    records = (
        tmp_path
        / "data/evidence_libraries/bioavailability_ma/v10/03_pair_buckets/records.parquet"
    )
    records.parent.mkdir(parents=True)
    records.write_bytes(b"new-stage-3")
    monkeypatch.setattr(publication, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(publication, "verify_manifest", lambda root: None)

    refreshed = publication.refresh_upstream("bioavailability_ma", release)

    assert refreshed["selected"] == original["selected"]
    assert refreshed["migration"] == original["migration"]
    assert refreshed["upstream"]["pair_bucket_records_sha256"] == publication.sha256_file(records)
