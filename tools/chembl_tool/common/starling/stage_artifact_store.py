"""Generic deterministic packaging for staged normalized evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PART_SIZE = 90_000_000


@dataclass(frozen=True)
class StageArtifactStoreProfile:
    store_version: str
    task_id: str
    stages: tuple[str, ...]
    local_root: Path
    tracked_root: Path


def package_stages(
    profile: StageArtifactStoreProfile,
    *,
    local_root: str | Path | None = None,
    tracked_root: str | Path | None = None,
    stages: Sequence[str] | None = None,
    part_size: int = DEFAULT_PART_SIZE,
) -> dict[str, Any]:
    source_root = Path(local_root or profile.local_root)
    destination_root = Path(tracked_root or profile.tracked_root)
    selected = _validated_stages(profile, stages)
    destination_root.mkdir(parents=True, exist_ok=True)
    existing = _read_json(destination_root / "manifest.json")
    manifests = dict(existing.get("stages") or {})
    for stage in selected:
        source = source_root / stage
        if not source.is_dir():
            raise FileNotFoundError(f"local stage is absent: {source}")
        target = destination_root / stage
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("stage.tar.zst.part-*"):
            old.unlink()
        with tempfile.TemporaryDirectory(prefix=f"{stage}-package-") as name:
            temporary = Path(name)
            tar_path = temporary / "stage.tar"
            zst_path = temporary / "stage.tar.zst"
            _write_deterministic_tar(source, tar_path)
            _zstd("-19", "--threads=0", "-f", str(tar_path), "-o", str(zst_path))
            parts = _split(zst_path, target, part_size)
            manifests[stage] = {
                "archive_sha256": _sha256(zst_path),
                "archive_size": zst_path.stat().st_size,
                "uncompressed_tar_size": tar_path.stat().st_size,
                "parts": [
                    {
                        "path": str(part.relative_to(destination_root)),
                        "size": part.stat().st_size,
                        "sha256": _sha256(part),
                    }
                    for part in parts
                ],
                "files": _inventory(source),
            }
    root_file = None
    local_manifest = source_root / "manifest.json"
    if local_manifest.is_file():
        tracked_manifest = destination_root / "build_manifest.json"
        shutil.copyfile(local_manifest, tracked_manifest)
        root_file = {
            "path": tracked_manifest.name,
            "restored_path": "manifest.json",
            "size": tracked_manifest.stat().st_size,
            "sha256": _sha256(tracked_manifest),
        }
    payload = {
        "store_version": profile.store_version,
        "task_id": profile.task_id,
        "archive_format": "deterministic tar + zstd",
        "part_size_limit": part_size,
        "local_layout": list(profile.stages),
        "root_file": root_file,
        "stages": {key: manifests[key] for key in sorted(manifests)},
    }
    _write_json(destination_root / "manifest.json", payload)
    return payload


def restore_stages(
    profile: StageArtifactStoreProfile,
    *,
    tracked_root: str | Path | None = None,
    local_root: str | Path | None = None,
    stages: Sequence[str] | None = None,
    force: bool = False,
) -> None:
    source_root = Path(tracked_root or profile.tracked_root)
    destination_root = Path(local_root or profile.local_root)
    manifest = _required_manifest(profile, source_root)
    selected = _validated_stages(profile, stages)
    for stage in selected:
        metadata = _stage_metadata(manifest, stage)
        destination = destination_root / stage
        if destination.exists() and any(destination.iterdir()):
            if not force:
                raise FileExistsError(f"refusing to overwrite non-empty stage: {destination}")
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{stage}-restore-") as name:
            temporary = Path(name)
            zst_path = temporary / "stage.tar.zst"
            tar_path = temporary / "stage.tar"
            with zst_path.open("wb") as output:
                for part_info in metadata["parts"]:
                    part = source_root / str(part_info["path"])
                    if _sha256(part) != part_info["sha256"]:
                        raise ValueError(f"tracked part hash mismatch: {part}")
                    with part.open("rb") as handle:
                        shutil.copyfileobj(handle, output)
            if _sha256(zst_path) != metadata["archive_sha256"]:
                raise ValueError(f"reassembled archive hash mismatch: {stage}")
            _zstd("-d", "-f", str(zst_path), "-o", str(tar_path))
            _safe_extract(tar_path, destination)
        _verify_inventory(destination, metadata["files"], stage)
    if set(selected) == set(profile.stages) and isinstance(manifest.get("root_file"), dict):
        root_file = manifest["root_file"]
        source = source_root / str(root_file["path"])
        destination = destination_root / str(root_file["restored_path"])
        if destination.exists() and not force and _sha256(destination) != root_file["sha256"]:
            raise FileExistsError(f"refusing to overwrite local root manifest: {destination}")
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256(destination) != root_file["sha256"]:
            raise ValueError("restored root manifest hash mismatch")


def verify_tracked(
    profile: StageArtifactStoreProfile,
    *,
    tracked_root: str | Path | None = None,
    stages: Sequence[str] | None = None,
) -> None:
    root = Path(tracked_root or profile.tracked_root)
    manifest = _required_manifest(profile, root)
    root_file = manifest.get("root_file")
    if isinstance(root_file, dict):
        path = root / str(root_file["path"])
        if path.stat().st_size != int(root_file["size"]) or _sha256(path) != root_file["sha256"]:
            raise ValueError(f"tracked root-file metadata mismatch: {path}")
    limit = int(manifest["part_size_limit"])
    for stage in _validated_stages(profile, stages):
        metadata = _stage_metadata(manifest, stage)
        digest = hashlib.sha256()
        total = 0
        for part_info in metadata["parts"]:
            part = root / str(part_info["path"])
            size = part.stat().st_size
            if size > limit or size != int(part_info["size"]) or _sha256(part) != part_info["sha256"]:
                raise ValueError(f"tracked part metadata mismatch: {part}")
            total += size
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        if total != int(metadata["archive_size"]) or digest.hexdigest() != metadata["archive_sha256"]:
            raise ValueError(f"tracked archive mismatch: {stage}")


def verify_local(
    profile: StageArtifactStoreProfile,
    *,
    local_root: str | Path | None = None,
    tracked_root: str | Path | None = None,
    stages: Sequence[str] | None = None,
) -> None:
    root = Path(local_root or profile.local_root)
    manifest = _required_manifest(profile, Path(tracked_root or profile.tracked_root))
    selected = _validated_stages(profile, stages)
    for stage in selected:
        _verify_inventory(root / stage, _stage_metadata(manifest, stage)["files"], stage)
    if set(selected) == set(profile.stages) and isinstance(manifest.get("root_file"), dict):
        root_file = manifest["root_file"]
        if _sha256(root / str(root_file["restored_path"])) != root_file["sha256"]:
            raise ValueError("local root manifest hash mismatch")


def _write_deterministic_tar(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w", format=tarfile.GNU_FORMAT) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(source).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def _safe_extract(tar_path: Path, destination: Path) -> None:
    resolved = destination.resolve()
    with tarfile.open(tar_path, "r") as archive:
        for member in archive.getmembers():
            candidate = (destination / member.name).resolve()
            if candidate != resolved and resolved not in candidate.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def _split(source: Path, target: Path, part_size: int) -> list[Path]:
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    parts: list[Path] = []
    with source.open("rb") as handle:
        number = 0
        while payload := handle.read(part_size):
            number += 1
            part = target / f"stage.tar.zst.part-{number:04d}"
            part.write_bytes(payload)
            parts.append(part)
    if not parts:
        raise ValueError(f"empty compressed archive: {source}")
    return parts


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ]


def _verify_inventory(root: Path, expected: list[dict[str, Any]], stage: str) -> None:
    if _inventory(root) != expected:
        raise ValueError(f"local restored-file inventory mismatch: {stage}")


def _required_manifest(profile: StageArtifactStoreProfile, root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("store_version") != profile.store_version or manifest.get("task_id") != profile.task_id:
        raise ValueError(f"unsupported or missing artifact-store manifest: {root}")
    return manifest


def _stage_metadata(manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    value = (manifest.get("stages") or {}).get(stage)
    if not isinstance(value, dict):
        raise FileNotFoundError(f"tracked stage is absent from manifest: {stage}")
    return value


def _validated_stages(profile: StageArtifactStoreProfile, stages: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(stages or profile.stages)
    unknown = sorted(set(selected) - set(profile.stages))
    if unknown:
        raise ValueError(f"unknown stage(s): {unknown}")
    return selected


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zstd(*arguments: str) -> None:
    executable = shutil.which("zstd")
    if executable is None:
        fallback = Path("/data1/joseph/miniconda3/bin/zstd")
        executable = str(fallback) if fallback.exists() else ""
    if not executable:
        raise FileNotFoundError("zstd executable not found")
    subprocess.run([executable, *arguments], check=True)


__all__ = [
    "DEFAULT_PART_SIZE",
    "StageArtifactStoreProfile",
    "package_stages",
    "restore_stages",
    "verify_local",
    "verify_tracked",
]
