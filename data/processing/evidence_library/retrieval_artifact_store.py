"""Package, restore, and verify the active all-task retrieval bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from data.processing.evidence_library.stage_artifact_store import (
    StageArtifactStoreProfile,
    package_stages,
    restore_stages,
    verify_local,
    verify_tracked,
)
from data.processing.paths import REPO_ROOT


BUNDLE_ID = "retrieval-gold-v1-all-tasks-v1"
PART_SIZE = 1_900_000_000
CACHE_CONFIG = Path(
    "predict/retrieval/assay_reranking/"
    "ranked_level_retrieval_gold_v1_all_tasks_v1.yaml"
)
BUNDLE_PATHS = (
    "data/caches/assay_reranking/active/ranked_level_retrieval_v4/bbb_martins",
    "data/caches/assay_reranking/active/ranked_level_retrieval_v4/bioavailability_ma",
    "data/caches/assay_reranking/active/"
    "ranked_level_retrieval_skin_gold_v1_l1_adapter_v2/skin_reaction",
    "data/caches/assay_reranking/active/"
    "ranked_level_retrieval_skin_v27_gold_v1/skin_reaction",
    "data/caches/assay_reranking/active/"
    "ranked_level_retrieval_gold_v1_addon_v2/ames",
    "data/caches/assay_reranking/active/"
    "ranked_level_retrieval_gold_v1_addon_v2/dili",
    "data/caches/assay_reranking/active/"
    "ranked_level_retrieval_gold_v1_addon_v2/carcinogens",
    "data/evidence_libraries/bbb_martins/v10_main_universe_v3/"
    "retrieval_projection/ranked_evidence_v2",
    "data/evidence_libraries/bioavailability_ma/v10_main_universe_v3/"
    "retrieval_projection/ranked_evidence_v2",
    "data/evidence_libraries/skin_reaction/v10_main_universe_v5/"
    "retrieval_projection/ranked_evidence_v1",
    "data/evidence_libraries/ames/v10_main_universe_v3/"
    "retrieval_projection/ranked_evidence_gold_v1_addon_v2",
    "data/evidence_libraries/dili/v10_main_universe_v3/"
    "retrieval_projection/ranked_evidence_gold_v1_addon_v2",
    "data/evidence_libraries/carcinogens/v10_main_universe_v3/"
    "retrieval_projection/ranked_evidence_gold_v1_addon_v2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile(repo_root: Path, archive_root: Path) -> StageArtifactStoreProfile:
    return StageArtifactStoreProfile(
        store_version="ranked_retrieval_bundle.v1",
        task_id="all_tasks",
        stages=BUNDLE_PATHS,
        local_root=repo_root,
        tracked_root=archive_root,
    )


def _source_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_release_references(repo_root: Path) -> None:
    roots = [(repo_root / path).resolve() for path in BUNDLE_PATHS]
    for root in roots:
        index_path = root / "RELEASE_INDEX.json"
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.get("evidence_sources") or [index["evidence"]]:
            manifest_path = (index_path.parent / entry["manifest"]).resolve()
            if not any(manifest_path.is_relative_to(candidate) for candidate in roots):
                raise ValueError(
                    f"retrieval evidence is outside the bundle: {manifest_path}"
                )
            if _sha256(manifest_path) != entry["manifest_sha256"]:
                raise ValueError(
                    f"retrieval evidence manifest hash mismatch: {manifest_path}"
                )


def _package(repo_root: Path, archive_root: Path, temporary_root: Path) -> None:
    if archive_root.exists() and any(archive_root.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty archive root: {archive_root}")
    missing = [path for path in BUNDLE_PATHS if not (repo_root / path).is_dir()]
    if missing:
        raise FileNotFoundError(f"retrieval bundle inputs are absent: {missing}")
    _validate_release_references(repo_root)

    profile = _profile(repo_root, archive_root)
    manifest = package_stages(
        profile,
        part_size=PART_SIZE,
        temporary_root=temporary_root,
    )
    part_number = 0
    for stage in BUNDLE_PATHS:
        for part in manifest["stages"][stage]["parts"]:
            part_number += 1
            source = archive_root / part["path"]
            target = archive_root / f"retrieval-bundle.part-{part_number:04d}"
            source.replace(target)
            part["path"] = target.name

    config = repo_root / CACHE_CONFIG
    manifest.update(
        {
            "schema_version": "ranked_retrieval_bundle.v1",
            "bundle_id": BUNDLE_ID,
            "cache_config": {
                "path": CACHE_CONFIG.as_posix(),
                "sha256": _sha256(config),
            },
            "source_git_commit": _source_commit(repo_root),
        }
    )
    manifest_path = archive_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (archive_root / "manifest.json.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    for path in sorted(archive_root.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()


def _verify_manifest(archive_root: Path, repo_root: Path | None = None) -> None:
    manifest_path = archive_root / "manifest.json"
    expected = (
        (archive_root / "manifest.json.sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    if _sha256(manifest_path) != expected:
        raise ValueError("retrieval bundle manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "ranked_retrieval_bundle.v1"
        or manifest.get("bundle_id") != BUNDLE_ID
    ):
        raise ValueError("unsupported retrieval bundle manifest")
    if repo_root is not None:
        config = repo_root / manifest["cache_config"]["path"]
        if _sha256(config) != manifest["cache_config"]["sha256"]:
            raise ValueError("cache configuration differs from the retrieval bundle")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("package", "restore", "verify-tracked", "verify-local")
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    archive_root = args.archive_root.resolve()
    temporary_root = (args.temporary_root or archive_root.parent).resolve()
    profile = _profile(repo_root, archive_root)

    if args.action == "package":
        _package(repo_root, archive_root, temporary_root)
    elif args.action == "restore":
        _verify_manifest(archive_root, repo_root)
        restore_stages(
            profile,
            force=args.force,
            temporary_root=temporary_root,
        )
    elif args.action == "verify-tracked":
        _verify_manifest(archive_root)
        verify_tracked(profile)
    else:
        _verify_manifest(archive_root, repo_root)
        verify_local(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
