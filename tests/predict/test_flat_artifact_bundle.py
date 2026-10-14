from __future__ import annotations

import json
from pathlib import Path

import pytest

from predict.harnesses.branches.flat import resolve_default_cache_bundle
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.retrieval.assay_reranking.artifact_bundle import (
    ArtifactBundleError,
    benchmark_id,
    bundle_manifest_path,
    build_bundle_manifest,
    canonical_cache_alias,
    canonical_cache_id,
    canonical_cache_path,
    package_archive,
    require_receipt,
    restore_archive,
    verify_bundle,
)


def test_tdc_v2_artifacts_have_their_own_identity() -> None:
    assert benchmark_id("tdc_v2") == "tdc_v2"
    assert bundle_manifest_path("ames", "tdc_v2").parts[-2:] == ("tdc_v2", "ames.json")


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


def test_explicit_tdc_v1_inventory_preserves_benchmark_identity(tmp_path: Path) -> None:
    repo, config = _fixture_repo(tmp_path)
    path = repo / "tdc_v1.json"
    manifest = build_bundle_manifest(
        task="bbb_martins", benchmark="tdc_v1", cache_config=config, output=path, repo_root=repo,
    )
    assert manifest["benchmark"] == "tdc_v1"
    receipt = repo / "receipt.json"
    verify_bundle(path, repo_root=repo, receipt_path=receipt)
    assert require_receipt(path, repo_root=repo, receipt_path=receipt,
                           expected_benchmark="tdc_v1")["bundle_id"] == "flat_v5/tdc_v1/bbb_martins"
    with pytest.raises(ArtifactBundleError, match="benchmark differs"):
        require_receipt(path, repo_root=repo, receipt_path=receipt, expected_benchmark="tdc_v2")


def test_inventory_excludes_audit_only_score_and_source_provenance(tmp_path: Path) -> None:
    repo, config = _fixture_repo(tmp_path)
    evidence = repo / "data/evidence/VERSION.json"
    document = json.loads(evidence.read_text())
    document["source_paths"] = {"records": "raw.parquet"}
    evidence.write_text(json.dumps(document))
    (evidence.parent / "raw.parquet").write_bytes(b"raw")
    index = repo / "data/cache/bbb/RELEASE_INDEX.json"
    document = json.loads(index.read_text())
    document["score_reuse_sources"] = [{"path": "old-score-manifest.json"}]
    index.write_text(json.dumps(document))
    (index.parent / "old-score-manifest.json").write_text("{}")
    manifest = build_bundle_manifest(
        task="bbb_martins", benchmark="tdc_v2", cache_config=config, repo_root=repo,
    )
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert manifest["status"] == "complete"
    assert "data/evidence/raw.parquet" not in paths
    assert "data/cache/bbb/old-score-manifest.json" not in paths


def test_direct_only_bundle_does_not_require_later_cache(tmp_path: Path) -> None:
    repo, config = _fixture_repo(tmp_path)
    config.write_text(
        "version: 20\ncaches:\n  bbb_martins:\n"
        "    L1: ../data/cache/bbb/RELEASE_INDEX.json\n"
        "    later: ../data/cache/missing/RELEASE_INDEX.json\n",
        encoding="utf-8",
    )
    manifest_path = repo / "direct.json"
    manifest = build_bundle_manifest(
        task="bbb_martins", benchmark="gold_v1", cache_config=config,
        max_level=1, output=manifest_path, repo_root=repo,
    )
    assert manifest["status"] == "complete"
    verify_bundle(manifest_path, repo_root=repo, receipt_path=repo / "receipt.json")
    require_receipt(manifest_path, receipt_path=repo / "receipt.json",
                    repo_root=repo, expected_max_level=1)
    with pytest.raises(ArtifactBundleError, match="does not cover"):
        require_receipt(manifest_path, receipt_path=repo / "receipt.json",
                        repo_root=repo, expected_max_level=2)


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
    assert tdc == Path("data/gold_labels/TDC/manifest.json").resolve()
    with pytest.raises(TypeError):
        resolve_default_cache_bundle("bbb_martins", "tdc", Path("predict/retrieval/assay_reranking/ranked_level_retrieval_v4.yaml"), tdc_version="v1")
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
