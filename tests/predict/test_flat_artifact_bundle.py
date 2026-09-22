from __future__ import annotations

import json
from pathlib import Path

import pytest

from predict.harnesses.branches.flat import resolve_default_cache_bundle
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.retrieval.assay_reranking.artifact_bundle import (
    ArtifactBundleError,
    build_bundle_manifest,
    canonical_cache_alias,
    canonical_cache_id,
    canonical_cache_path,
    package_archive,
    require_receipt,
    restore_archive,
    verify_bundle,
)


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    config = repo / "predict" / "cache.yaml"
    index = repo / "data" / "cache" / "bbb" / "RELEASE_INDEX.json"
    evidence_manifest = repo / "data" / "evidence" / "VERSION.json"
    records = evidence_manifest.with_name("records.parquet")
    config.parent.mkdir(parents=True)
    index.parent.mkdir(parents=True)
    evidence_manifest.parent.mkdir(parents=True)
    records.write_bytes(b"records")
    evidence_manifest.write_text(
        json.dumps({"schema_version": "ranked_evidence_projection.v1", "records": records.name}),
        encoding="utf-8",
    )
    index.write_text(
        json.dumps({
            "schema_version": "ranked_uid_task_release_index.v1",
            "evidence": {"manifest": "../../evidence/VERSION.json"},
            "splits": {"valid": {"levels": {
                "L1": {"manifest": "RELEASE_INDEX.json"},
            }}},
        }),
        encoding="utf-8",
    )
    config.write_text(
        "version: 20\ncaches:\n  bbb_martins: ../data/cache/bbb/RELEASE_INDEX.json\n",
        encoding="utf-8",
    )
    return repo, config


def test_inventory_and_receipt_are_reusable(tmp_path: Path) -> None:
    repo, config = _fixture_repo(tmp_path)
    manifest_path = repo / "bundle.json"
    manifest = build_bundle_manifest(
        task="bbb_martins",
        benchmark="gold_v1",
        cache_config=config,
        output=manifest_path,
        repo_root=repo,
    )
    assert manifest["status"] == "complete"
    receipt = verify_bundle(
        manifest_path,
        repo_root=repo,
        receipt_path=repo / "receipt.json",
    )
    assert receipt["status"] == "verified"
    assert require_receipt(
        manifest_path, receipt_path=repo / "receipt.json", repo_root=repo
    )["bundle_id"] == "flat_v5/gold_v1/bbb_martins"
    with pytest.raises(ArtifactBundleError, match="task differs"):
        require_receipt(
            manifest_path,
            receipt_path=repo / "receipt.json",
            repo_root=repo,
            expected_task="ames",
            expected_benchmark="gold",
        )
    with pytest.raises(ArtifactBundleError, match="benchmark differs"):
        require_receipt(
            manifest_path,
            receipt_path=repo / "receipt.json",
            repo_root=repo,
            expected_task="bbb_martins",
            expected_benchmark="tdc",
        )

    records = repo / "data" / "evidence" / "records.parquet"
    records.write_bytes(b"changed")
    with pytest.raises(ArtifactBundleError, match="changed after verification"):
        require_receipt(manifest_path, receipt_path=repo / "receipt.json", repo_root=repo)


def test_canonical_cache_identity_and_legacy_alias() -> None:
    cache_id = canonical_cache_id(
        benchmark="gold_v1",
        task="bbb_martins",
        role="l2plus",
        method="assay_transfer",
        release="v24_1",
        variant="morgan75",
    )
    assert cache_id == "flat_v5/gold_v1/bbb_martins/l2plus/assay_transfer/v24_1/morgan75"
    assert canonical_cache_path(cache_id, repo_root=Path("/repo")).parts[-7:] == (
        "flat_v5", "gold_v1", "bbb_martins", "l2plus", "assay_transfer", "v24_1", "morgan75"
    )
    assert canonical_cache_alias("v24_1_bbb_uid_levels_morgan75") == cache_id
    assert canonical_cache_alias("v25_oral_uid_levels_morgan75", benchmark="gold") == (
        "flat_v5/gold_v1/bioavailability_ma/l2plus/assay_transfer/v25/morgan75"
    )
    assert str(cache_profile_root(cache_id)).endswith(
        "data/caches/assay_reranking/active/flat_v5/gold_v1/"
        "bbb_martins/l2plus/assay_transfer/v24_1/morgan75"
    )


def test_default_cache_bundle_is_task_and_benchmark_scoped(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.yaml"
    assert resolve_default_cache_bundle("bbb_martins", "gold", explicit) == explicit
    tdc = resolve_default_cache_bundle(
        "bbb_martins",
        "tdc",
        Path("predict/retrieval/assay_reranking/ranked_level_retrieval_v4.yaml"),
    )
    assert tdc.name == "ranked_level_retrieval_tdc_v1.yaml"
    safety = resolve_default_cache_bundle(
        "ames",
        "gold",
        Path("predict/retrieval/assay_reranking/ranked_level_retrieval_v4.yaml"),
    )
    assert safety.name == "ranked_level_retrieval_gold_v1_all_tasks_v1.yaml"


def test_archive_round_trip_restores_relative_payloads(tmp_path: Path) -> None:
    repo, config = _fixture_repo(tmp_path)
    manifest_path = repo / "bundle.json"
    build_bundle_manifest(
        task="bbb_martins",
        benchmark="gold_v1",
        cache_config=config,
        output=manifest_path,
        repo_root=repo,
    )
    archive_root = tmp_path / "archive"
    package_archive(manifest_path, archive_root, repo_root=repo, part_size=64)

    restored = tmp_path / "restored"
    receipt = restore_archive(archive_root, repo_root=restored)
    assert receipt["status"] == "verified"
    assert (restored / "data/cache/bbb/RELEASE_INDEX.json").is_file()
