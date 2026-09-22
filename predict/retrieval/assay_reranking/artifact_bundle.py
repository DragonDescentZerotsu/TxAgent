"""Inventory, verify, package, and restore flat-harness retrieval artifacts.

The bundle manifest is the small tracked contract; evidence libraries and cache
payloads remain under their canonical data owners. Verification hashes every
declared payload once and writes a receipt containing cheap stat fingerprints.
Inference checks that receipt instead of rescanning the payloads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

import yaml

from data.processing.paths import REPO_ROOT
from predict.utils.json import sha256_file, write_json_atomic


BUNDLE_SCHEMA = "flat_v5_artifact_bundle.v1"
RECEIPT_SCHEMA = "flat_v5_artifact_receipt.v1"
ARCHIVE_SCHEMA = "flat_v5_artifact_archive.v1"


class ArtifactBundleError(RuntimeError):
    """A required retrieval artifact is absent, stale, or incompatible."""


def _component(value: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid artifact component: {value!r}")
    return value


def benchmark_id(value: str) -> str:
    """Normalize CLI benchmark names to manifest identities."""
    value = {"gold": "gold_v1", "tdc": "tdc_v1"}.get(str(value), str(value))
    if value not in {"gold_v1", "tdc_v1"}:
        raise ValueError(f"unsupported artifact benchmark: {value!r}")
    return value


def canonical_cache_id(
    *,
    benchmark: str,
    task: str,
    role: str,
    method: str,
    release: str,
    variant: str = "default",
) -> str:
    """Build a stable cache identity without overloading a version suffix."""
    benchmark = benchmark_id(benchmark)
    return "/".join(
        (
            "flat_v5",
            _component(benchmark),
            _component(task),
            _component(role),
            _component(method),
            _component(release),
            _component(variant),
        )
    )


def canonical_cache_path(cache_id: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a canonical cache identity below the active cache owner."""
    parts = tuple(cache_id.split("/"))
    if len(parts) != 7 or parts[0] != "flat_v5":
        raise ValueError(f"invalid canonical cache id: {cache_id!r}")
    for value in parts[1:]:
        _component(value)
    return repo_root / "data/caches/assay_reranking/active" / Path(*parts)


LEGACY_CACHE_ALIASES: dict[str, tuple[str, str, str, str]] = {
    "v24_1_bbb_uid_levels_morgan75": (
        "bbb_martins", "l2plus", "assay_transfer", "v24_1/morgan75"
    ),
    "v25_oral_uid_levels_morgan75": (
        "bioavailability_ma", "l2plus", "assay_transfer", "v25/morgan75"
    ),
    "v25_oral_uid_levels_morgan75_l2": (
        "bioavailability_ma", "l2plus", "assay_transfer", "v25/morgan75_l2_builder"
    ),
    "v25_oral_uid_levels_morgan75_l3": (
        "bioavailability_ma", "l2plus", "assay_transfer", "v25/morgan75_l3_builder"
    ),
}


def canonical_cache_alias(
    profile: str, *, benchmark: str = "gold_v1"
) -> str | None:
    """Return the canonical successor for a known historical profile name."""
    value = LEGACY_CACHE_ALIASES.get(str(profile))
    if value is None:
        return None
    task, role, method, release_variant = value
    release, variant = release_variant.split("/", 1)
    return canonical_cache_id(
        benchmark=benchmark_id(benchmark),
        task=task,
        role=role,
        method=method,
        release=release,
        variant=variant,
    )


def bundle_manifest_path(
    task: str, benchmark: str, *, repo_root: Path = REPO_ROOT
) -> Path:
    """Return the tracked manifest location for one task/benchmark bundle."""
    return (
        repo_root
        / "predict/retrieval/assay_reranking/flat_v5_manifests"
        / _component(benchmark_id(benchmark))
        / f"{_component(task)}.json"
    )


def bundle_receipt_path(
    task: str, benchmark: str, *, repo_root: Path = REPO_ROOT
) -> Path:
    """Return the ignored local receipt location for one bundle."""
    return (
        repo_root
        / "data/caches/assay_reranking/active/flat_v5/receipts"
        / _component(benchmark_id(benchmark))
        / f"{_component(task)}.json"
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ArtifactBundleError(
            f"artifact is outside the repository root: {path}"
        ) from error


def _recorded_path(path: Path, repo_root: Path) -> str:
    """Use a portable repository path when possible, otherwise an absolute path."""
    try:
        return _repo_relative(path, repo_root)
    except ArtifactBundleError:
        return str(path.resolve())


def _repo_artifact_path(relative: str, repo_root: Path) -> Path:
    """Resolve a manifest-owned relative path without allowing traversal."""
    candidate = Path(relative)
    root = repo_root.resolve()
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ArtifactBundleError(f"artifact path must be repository-relative: {relative!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ArtifactBundleError(f"artifact path escapes repository root: {relative!r}")
    return resolved


def _archive_child(archive_root: Path, relative: str) -> Path:
    """Resolve an archive-owned child without allowing archive traversal."""
    candidate = Path(relative)
    root = archive_root.resolve()
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ArtifactBundleError(f"archive path is invalid: {relative!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ArtifactBundleError(f"archive path escapes archive root: {relative!r}")
    return resolved


def _resolve_reference(value: str, parent: Path, repo_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = parent / candidate
    return candidate.resolve()


def _add_path(paths: dict[str, Path], path: Path, repo_root: Path) -> None:
    relative = _repo_relative(path, repo_root)
    paths.setdefault(relative, path)


def _collect_known_files(
    document: Any, *, parent: Path, repo_root: Path, paths: dict[str, Path]
) -> None:
    """Collect payload files from release-index/manifest path fields."""
    if not isinstance(document, dict):
        return
    path_keys = {
        "database", "records", "record_file", "payload", "payload_path", "data",
        "file", "path", "parquet", "sqlite", "assignments", "prompts",
    }
    for key, value in document.items():
        if isinstance(value, str) and key in path_keys:
            candidate = _resolve_reference(value, parent, repo_root)
            if candidate.is_file() or candidate.suffix in {".parquet", ".sqlite3", ".json"}:
                _add_path(paths, candidate, repo_root)
        elif isinstance(value, dict):
            _collect_known_files(value, parent=parent, repo_root=repo_root, paths=paths)
        elif isinstance(value, list):
            for item in value:
                _collect_known_files(item, parent=parent, repo_root=repo_root, paths=paths)


def _collect_release_index(
    index_path: Path, *, repo_root: Path, paths: dict[str, Path]
) -> None:
    _add_path(paths, index_path, repo_root)
    if not index_path.is_file():
        return
    try:
        document = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactBundleError(f"invalid release index: {index_path}") from error
    manifests: list[Path] = []
    for entry in document.get("evidence_sources") or [document.get("evidence")]:
        if isinstance(entry, dict) and entry.get("manifest"):
            manifests.append(
                _resolve_reference(str(entry["manifest"]), index_path.parent, repo_root)
            )
    for split in (document.get("splits") or {}).values():
        for entry in (split.get("levels") or {}).values() if isinstance(split, dict) else ():
            if isinstance(entry, dict):
                for value in entry.values():
                    if isinstance(value, dict) and value.get("manifest"):
                        manifests.append(
                            _resolve_reference(str(value["manifest"]), index_path.parent, repo_root)
                        )
                    elif isinstance(value, str) and value.endswith((".json", "VERSION.json")):
                        manifests.append(_resolve_reference(value, index_path.parent, repo_root))
    _collect_known_files(document, parent=index_path.parent, repo_root=repo_root, paths=paths)
    for manifest_path in manifests:
        _add_path(paths, manifest_path, repo_root)
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactBundleError(f"invalid artifact manifest: {manifest_path}") from error
        _collect_known_files(
            manifest, parent=manifest_path.parent, repo_root=repo_root, paths=paths
        )


def _release_indexes(
    config: dict[str, Any], task: str, config_path: Path, repo_root: Path
) -> list[Path]:
    selected = (config.get("caches") or {}).get(task)
    if selected is None:
        raise ArtifactBundleError(f"cache configuration has no task entry: {task}")
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.endswith("RELEASE_INDEX.json"):
            values.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(selected)
    return [_resolve_reference(value, config_path.parent, repo_root) for value in values]


def build_bundle_manifest(
    *,
    task: str,
    benchmark: str,
    cache_config: Path,
    output: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Inventory one task/benchmark without requiring payloads to be tracked."""
    benchmark = benchmark_id(benchmark)
    cache_config = cache_config.resolve()
    config = yaml.safe_load(cache_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("caches"), dict):
        raise ArtifactBundleError(f"invalid cache configuration: {cache_config}")
    paths: dict[str, Path] = {}
    _add_path(paths, cache_config, repo_root)
    for index_path in _release_indexes(config, task, cache_config, repo_root):
        _collect_release_index(index_path, repo_root=repo_root, paths=paths)
    entries = []
    missing = []
    for relative, path in sorted(paths.items()):
        exists = path.is_file()
        if not exists:
            missing.append(relative)
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size if exists else None,
                "sha256": sha256_file(path) if exists else "",
            }
        )
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "status": "complete" if not missing else "incomplete",
        "bundle_id": f"flat_v5/{benchmark}/{task}",
        "task": task,
        "benchmark": benchmark,
        "harness_version": "full-flat-context-v5",
        "cache_config": {
            "path": _repo_relative(cache_config, repo_root),
            "sha256": sha256_file(cache_config),
        },
        "artifacts": entries,
        "missing": missing,
    }
    if output is not None:
        write_json_atomic(output, manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactBundleError(f"cannot read artifact bundle manifest: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != BUNDLE_SCHEMA:
        raise ArtifactBundleError(f"unsupported artifact bundle manifest: {path}")
    if not isinstance(document.get("artifacts"), list):
        raise ArtifactBundleError(f"artifact bundle has no artifact list: {path}")
    artifact_paths = [
        str(entry.get("path") or "")
        for entry in document["artifacts"]
        if isinstance(entry, dict)
    ]
    if len(artifact_paths) != len(document["artifacts"]) or not artifact_paths or len(
        artifact_paths
    ) != len(set(artifact_paths)):
        raise ArtifactBundleError(f"artifact bundle has invalid or duplicate paths: {path}")
    return document


def verify_bundle(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Hash every declared payload and write a reusable verification receipt."""
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    if manifest.get("status") != "complete":
        raise ArtifactBundleError(
            "artifact bundle is incomplete: "
            f"{manifest_path}; missing={manifest.get('missing') or []}"
        )
    verified = []
    for entry in manifest["artifacts"]:
        relative = str(entry.get("path") or "")
        path = _repo_artifact_path(relative, repo_root)
        if not path.is_file():
            raise ArtifactBundleError(f"required artifact is absent: {path}")
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            raise ArtifactBundleError(
                f"artifact hash mismatch: {path}; expected={entry.get('sha256')} observed={digest}"
            )
        stat = path.stat()
        if entry.get("size") is not None and stat.st_size != int(entry["size"]):
            raise ArtifactBundleError(
                "artifact size mismatch: "
                f"{path}; expected={entry['size']} observed={stat.st_size}"
            )
        verified.append(
            {
                "path": relative,
                "sha256": digest,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    receipt_path = receipt_path or bundle_receipt_path(
        str(manifest["task"]), str(manifest["benchmark"]), repo_root=repo_root
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "verified",
        "bundle_id": manifest["bundle_id"],
        "manifest": _recorded_path(manifest_path, repo_root),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": verified,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def require_receipt(
    manifest_path: Path,
    *,
    receipt_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    verify: bool = False,
    expected_cache_config: Path | None = None,
    expected_task: str | None = None,
    expected_benchmark: str | None = None,
) -> dict[str, Any]:
    """Perform the cheap launch check, or explicitly refresh the full receipt."""
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    if expected_task is not None and manifest.get("task") != expected_task:
        raise ArtifactBundleError(
            "artifact bundle task differs from the requested task: "
            f"manifest={manifest.get('task')!r} requested={expected_task!r}"
        )
    if expected_benchmark is not None:
        requested_benchmark = benchmark_id(expected_benchmark)
        if manifest.get("benchmark") != requested_benchmark:
            raise ArtifactBundleError(
                "artifact bundle benchmark differs from the requested benchmark: "
                f"manifest={manifest.get('benchmark')!r} "
                f"requested={requested_benchmark!r}"
            )
    if expected_cache_config is not None:
        config = expected_cache_config.resolve()
        recorded = manifest.get("cache_config") or {}
        try:
            config_path = _recorded_path(config, repo_root)
            config_hash = sha256_file(config)
        except (ArtifactBundleError, OSError) as error:
            raise ArtifactBundleError(
                f"configured cache bundle is absent or unreadable: {config}"
            ) from error
        if config_path != recorded.get("path") or config_hash != recorded.get("sha256"):
            raise ArtifactBundleError(
                f"artifact bundle cache configuration differs from {config}"
            )
    if verify:
        return verify_bundle(manifest_path, repo_root=repo_root, receipt_path=receipt_path)
    receipt_path = receipt_path or bundle_receipt_path(
        str(manifest.get("task") or ""), str(manifest.get("benchmark") or ""), repo_root=repo_root
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactBundleError(
            "artifact verification receipt is absent: "
            f"{receipt_path}; run `python -m predict.retrieval.assay_reranking.artifact_bundle "
            f"verify --manifest {manifest_path}`"
        ) from error
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("bundle_id") != manifest.get("bundle_id")
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ArtifactBundleError(f"artifact verification receipt is stale: {receipt_path}")
    expected = {
        str(entry.get("path")): str(entry.get("sha256"))
        for entry in manifest["artifacts"]
    }
    observed = {
        str(entry.get("path")): str(entry.get("sha256"))
        for entry in receipt.get("artifacts") or []
    }
    if observed != expected:
        raise ArtifactBundleError(f"artifact verification receipt does not match: {receipt_path}")
    for entry in receipt.get("artifacts") or []:
        path = _repo_artifact_path(str(entry.get("path") or ""), repo_root)
        if not path.is_file():
            raise ArtifactBundleError(f"verified artifact is absent: {path}")
        stat = path.stat()
        if (
            stat.st_size != int(entry.get("size", -1))
            or stat.st_mtime_ns != int(entry.get("mtime_ns", -1))
        ):
            raise ArtifactBundleError(
                "artifact changed after verification: "
                f"{path}; run `python -m predict.retrieval.assay_reranking.artifact_bundle "
                f"verify --manifest {manifest_path}`"
            )
    return receipt


def _zstd_binary() -> str:
    executable = shutil.which("zstd")
    if executable:
        return executable
    fallback = Path("/data1/joseph/miniconda3/bin/zstd")
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("zstd executable not found")


def _sha256_bytes(path: Path) -> str:
    return sha256_file(path)


def _safe_extract(tar_path: Path, destination: Path) -> None:
    with tarfile.open(tar_path, "r") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise ArtifactBundleError(f"archive member escapes restore root: {member.name}")
        archive.extractall(root, filter="data")


def package_archive(
    manifest_path: Path,
    archive_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    part_size: int = 1_900_000_000,
) -> dict[str, Any]:
    """Create a deterministic compressed archive from a verified manifest."""
    if part_size < 1:
        raise ValueError("part_size must be positive")
    if archive_root.exists() and any(archive_root.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty archive root: {archive_root}")
    archive_root.mkdir(parents=True, exist_ok=True)
    verify_bundle(manifest_path, repo_root=repo_root)
    manifest = _load_manifest(manifest_path)
    with tempfile.TemporaryDirectory(dir=archive_root.parent) as temporary_name:
        temporary = Path(temporary_name)
        tar_path = temporary / "payload.tar"
        compressed = temporary / "payload.tar.zst"
        with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as archive:
            for entry in sorted(manifest["artifacts"], key=lambda item: item["path"]):
                relative = str(entry["path"])
                source = _repo_artifact_path(relative, repo_root)
                info = archive.gettarinfo(str(source), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        subprocess.run(
            [_zstd_binary(), "-19", "--threads=0", "-f", str(tar_path), "-o", str(compressed)],
            check=True,
        )
        archive_sha256 = sha256_file(compressed)
        parts = []
        with compressed.open("rb") as source:
            number = 0
            while True:
                chunk = source.read(part_size)
                if not chunk:
                    break
                number += 1
                part = archive_root / f"payload.tar.zst.part-{number:04d}"
                part.write_bytes(chunk)
                parts.append({"path": part.name, "size": len(chunk), "sha256": _sha256_bytes(part)})
        shutil.copyfile(manifest_path, archive_root / "manifest.json")
    metadata = {
        "schema_version": ARCHIVE_SCHEMA,
        "bundle_id": manifest["bundle_id"],
        "manifest_sha256": sha256_file(archive_root / "manifest.json"),
        "archive_sha256": archive_sha256,
        "manifest_path": manifest_path.resolve().relative_to(repo_root.resolve()).as_posix(),
        "parts": parts,
    }
    write_json_atomic(archive_root / "archive.json", metadata)
    return metadata


def restore_archive(
    archive_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Verify and restore a compressed bundle into canonical repository paths."""
    try:
        metadata = json.loads(
            (archive_root / "archive.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactBundleError(f"cannot read artifact archive: {archive_root}") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != ARCHIVE_SCHEMA:
        raise ArtifactBundleError(f"unsupported artifact archive: {archive_root}")
    manifest_source = archive_root / "manifest.json"
    try:
        manifest_sha256 = sha256_file(manifest_source)
    except OSError as error:
        raise ArtifactBundleError(f"artifact archive manifest is absent: {manifest_source}") from error
    if manifest_sha256 != metadata.get("manifest_sha256"):
        raise ArtifactBundleError("artifact archive manifest hash mismatch")
    with tempfile.TemporaryDirectory(dir=archive_root.parent) as temporary_name:
        temporary = Path(temporary_name)
        compressed = temporary / "payload.tar.zst"
        with compressed.open("wb") as output:
            for part_info in metadata.get("parts") or []:
                part = _archive_child(archive_root, str(part_info["path"]))
                if sha256_file(part) != part_info["sha256"]:
                    raise ArtifactBundleError(f"artifact archive part hash mismatch: {part}")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if sha256_file(compressed) != metadata.get("archive_sha256"):
            raise ArtifactBundleError("reassembled artifact archive hash mismatch")
        tar_path = temporary / "payload.tar"
        subprocess.run(
            [_zstd_binary(), "-d", "-f", str(compressed), "-o", str(tar_path)],
            check=True,
        )
        extracted = temporary / "extracted"
        extracted.mkdir()
        _safe_extract(tar_path, extracted)
        manifest = _load_manifest(manifest_source)
        if manifest.get("bundle_id") != metadata.get("bundle_id"):
            raise ArtifactBundleError("artifact archive bundle identity mismatch")
        for entry in manifest["artifacts"]:
            relative = str(entry["path"])
            source = _archive_child(extracted, relative)
            if not source.is_file() or sha256_file(source) != entry["sha256"]:
                raise ArtifactBundleError(f"restored payload failed validation: {relative}")
            target = _repo_artifact_path(relative, repo_root)
            if target.exists():
                if sha256_file(target) == entry["sha256"]:
                    continue
                if not force:
                    raise FileExistsError(f"refusing to overwrite changed artifact: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_target = target.with_name(f".{target.name}.restore.tmp")
            shutil.copy2(source, temporary_target)
            os.replace(temporary_target, target)
    restored_manifest = _repo_artifact_path(
        str(metadata.get("manifest_path") or ""), repo_root
    )
    restored_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_differs = restored_manifest.exists() and (
        sha256_file(restored_manifest) != metadata["manifest_sha256"]
    )
    if manifest_differs and not force:
        raise FileExistsError(
            f"refusing to overwrite changed manifest: {restored_manifest}"
        )
    if not restored_manifest.exists() or manifest_differs:
        shutil.copyfile(manifest_source, restored_manifest)
    return verify_bundle(restored_manifest, repo_root=repo_root)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--task", required=True)
    inventory.add_argument("--benchmark", choices=("gold_v1", "tdc_v1"), required=True)
    inventory.add_argument("--cache-config", type=Path, required=True)
    inventory.add_argument("--output", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--receipt", type=Path)
    package = subparsers.add_parser("package")
    package.add_argument("--manifest", type=Path, required=True)
    package.add_argument("--archive-root", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--archive-root", type=Path, required=True)
    restore.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "inventory":
        manifest = build_bundle_manifest(
            task=args.task,
            benchmark=args.benchmark,
            cache_config=args.cache_config,
            output=args.output,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.action == "verify":
        print(json.dumps(verify_bundle(args.manifest, receipt_path=args.receipt), indent=2))
    elif args.action == "package":
        print(json.dumps(package_archive(args.manifest, args.archive_root), indent=2))
    else:
        print(json.dumps(restore_archive(args.archive_root, force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
