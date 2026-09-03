from __future__ import annotations

from types import SimpleNamespace

from data.processing.evidence_library.shared.v1.build_runtime import (
    FileDigestCache,
    build_cache_metadata,
    cache_metadata_matches,
)
from data.processing.evidence_library.versions.v7.build_normalized_evidence_library import (
    MANIFEST_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    STAGES,
    _record_build_cache,
    _scientific_assets,
    _stage_output_filenames,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_policy import POLICY


def test_source_snapshot_precedes_one_row_per_source_cleaning() -> None:
    assert STAGES[:2] == ("source", "clean")
    assert SOURCE_INVENTORY_FILENAME == "00_source/source_inventory.json"
    outputs = _stage_output_filenames(POLICY)
    assert "00_source/endpoint_inventory.json" in outputs["source"]
    assert "01_cleaned/endpoint_unit_profile.json" in outputs["clean"]


def test_build_cache_tracks_semantic_arguments_and_artifact_bytes(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "records.parquet"
    source.write_bytes(b"source-v1")
    output.write_bytes(b"records-v1")
    args = SimpleNamespace(
        source=str(source),
        threshold=0.7,
        workers=8,
        cache_mode="auto",
        validation_level="strict",
        from_stage="clean",
        through_stage="organize",
        out_dir=str(tmp_path),
        progress_every=0,
    )
    metadata = build_cache_metadata(
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        input_paths=[source],
        output_paths=[output],
        digests=FileDigestCache(),
    )
    assert cache_metadata_matches(
        metadata,
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        digests=FileDigestCache(),
    )

    args.workers = 2
    assert cache_metadata_matches(
        metadata,
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        digests=FileDigestCache(),
    )
    args.threshold = 0.8
    assert not cache_metadata_matches(
        metadata,
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        digests=FileDigestCache(),
    )
    args.threshold = 0.7
    output.write_bytes(b"records-v2")
    assert not cache_metadata_matches(
        metadata,
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        digests=FileDigestCache(),
    )


def test_build_cache_invalidates_when_declared_scientific_asset_changes(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "records.parquet"
    policy_asset = tmp_path / "measurement_semantics.json"
    source.write_bytes(b"source")
    output.write_bytes(b"records")
    policy_asset.write_text('{"version": 1}\n', encoding="utf-8")
    args = SimpleNamespace(threshold=0.7, workers=1)
    metadata = build_cache_metadata(
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        input_paths=[source],
        output_paths=[output],
        digests=FileDigestCache(),
        scientific_assets=[policy_asset],
    )
    assert cache_metadata_matches(
        metadata,
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        digests=FileDigestCache(),
        scientific_assets=[policy_asset],
    )
    policy_asset.write_text('{"version": 2}\n', encoding="utf-8")
    assert not cache_metadata_matches(
        metadata,
        task_id="runtime_test_task",
        completed_stage="organize",
        args=args,
        digests=FileDigestCache(),
        scientific_assets=[policy_asset],
    )


def test_full_index_cache_hashes_staged_outputs_under_published_paths(tmp_path) -> None:
    out_dir = tmp_path / "published"
    staged = tmp_path / "staged"
    args = SimpleNamespace(
        from_stage="clean",
        through_stage="index",
        out_dir=str(out_dir),
        cache_mode="auto",
        validation_level="strict",
        workers=8,
        progress_every=0,
    )
    index_files = [
        filename
        for filename in _stage_output_filenames(POLICY)["index"]
        if filename != MANIFEST_FILENAME
    ]
    for filename in index_files:
        path = staged / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"index:{filename}".encode())

    metadata = _record_build_cache(
        POLICY,
        out_dir,
        args,
        completed_stage="index",
        digests=FileDigestCache(),
        staged_stage_dir=staged,
    )
    assert metadata["completed_stage"] == "index"
    assert set(metadata["outputs"]) == {
        str(out_dir / filename) for filename in index_files
    }
    assert not any(str(staged) in path for path in metadata["outputs"])

    for filename in index_files:
        published = out_dir / filename
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_bytes((staged / filename).read_bytes())
    assert cache_metadata_matches(
        metadata,
        task_id=POLICY.task_id,
        completed_stage="index",
        args=args,
        digests=FileDigestCache(),
        scientific_assets=_scientific_assets(POLICY, args),
    )
