"""Package, restore, and verify Git-trackable Skin_Reaction v6 stage bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

STORE_VERSION = "skin_reaction.normalized_v6_store.v2"
DEFAULT_LOCAL_ROOT = Path(
    "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
    "starling_normalized_v6"
)
DEFAULT_TRACKED_ROOT = Path(
    "artifacts/chembl_tool/tasks/skin_reaction/starling_normalized_v6"
)
STAGES = tuple(
    f"{number:02d}_{name}"
    for number, name in enumerate(
        (
            "cleaned",
            "normalized",
            "records",
            "pair_buckets",
            "assay_transfer_policy",
            "remove_heldout_overlap",
            "molecule_evidence",
            "neighbor_index",
            "audits",
        ),
        start=1,
    )
)
DEFAULT_PART_SIZE = 90_000_000


def package_stages(
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    *,
    stages: list[str] | tuple[str, ...] = STAGES,
    part_size: int = DEFAULT_PART_SIZE,
) -> dict[str, Any]:
    """Create deterministic tar.zst parts and a content-addressed manifest."""
    source_root = Path(local_root)
    destination_root = Path(tracked_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest(destination_root / "manifest.json")
    stage_manifests = dict(existing.get("stages") or {})
    for stage in _validate_stages(stages):
        source = source_root / stage
        if not source.is_dir():
            raise FileNotFoundError(f"local stage is absent: {source}")
        target = destination_root / stage
        target.mkdir(parents=True, exist_ok=True)
        for old_part in target.glob("stage.tar.zst.part-*"):
            old_part.unlink()
        with tempfile.TemporaryDirectory(prefix=f"{stage}-package-") as tmp_name:
            temporary = Path(tmp_name)
            tar_path = temporary / "stage.tar"
            zst_path = temporary / "stage.tar.zst"
            _write_deterministic_tar(source, tar_path)
            _run_zstd("-19", "--threads=0", "-f", str(tar_path), "-o", str(zst_path))
            parts = _split_file(zst_path, target, part_size)
            stage_manifests[stage] = {
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
                "files": _file_inventory(source),
            }
    root_file = None
    local_build_manifest = source_root / "manifest.json"
    if local_build_manifest.exists():
        tracked_build_manifest = destination_root / "build_manifest.json"
        shutil.copyfile(local_build_manifest, tracked_build_manifest)
        root_file = {
            "path": "build_manifest.json",
            "restored_path": "manifest.json",
            "size": tracked_build_manifest.stat().st_size,
            "sha256": _sha256(tracked_build_manifest),
        }
    manifest = {
        "store_version": STORE_VERSION,
        "task_id": "skin_reaction",
        "archive_format": "deterministic tar + zstd",
        "part_size_limit": part_size,
        "local_layout": list(STAGES),
        "root_file": root_file,
        "stages": {key: stage_manifests[key] for key in sorted(stage_manifests)},
    }
    _write_json(destination_root / "manifest.json", manifest)
    return manifest


def restore_stages(
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    *,
    stages: list[str] | tuple[str, ...] = STAGES,
    force: bool = False,
) -> None:
    """Restore selected stage directories without requiring a full rebuild."""
    source_root = Path(tracked_root)
    destination_root = Path(local_root)
    manifest = _required_manifest(source_root)
    for stage in _validate_stages(stages):
        stage_manifest = (manifest.get("stages") or {}).get(stage)
        if not isinstance(stage_manifest, dict):
            raise FileNotFoundError(f"tracked stage is absent from manifest: {stage}")
        destination = destination_root / stage
        if destination.exists() and any(destination.iterdir()):
            if not force:
                raise FileExistsError(
                    f"refusing to overwrite non-empty local stage: {destination}"
                )
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{stage}-restore-") as tmp_name:
            temporary = Path(tmp_name)
            zst_path = temporary / "stage.tar.zst"
            tar_path = temporary / "stage.tar"
            with zst_path.open("wb") as output:
                for part_info in stage_manifest["parts"]:
                    part = source_root / str(part_info["path"])
                    if _sha256(part) != part_info["sha256"]:
                        raise ValueError(f"tracked part hash mismatch: {part}")
                    with part.open("rb") as handle:
                        shutil.copyfileobj(handle, output)
            if _sha256(zst_path) != stage_manifest["archive_sha256"]:
                raise ValueError(f"reassembled archive hash mismatch: {stage}")
            _run_zstd("-d", "-f", str(zst_path), "-o", str(tar_path))
            _safe_extract(tar_path, destination)
        _verify_file_inventory(destination, stage_manifest["files"], stage)
    if set(stages) == set(STAGES) and isinstance(manifest.get("root_file"), dict):
        root_file = manifest["root_file"]
        source = source_root / str(root_file["path"])
        destination = destination_root / str(root_file["restored_path"])
        if (
            destination.exists()
            and not force
            and _sha256(destination) != root_file["sha256"]
        ):
            raise FileExistsError(
                f"refusing to overwrite local root manifest: {destination}"
            )
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256(destination) != root_file["sha256"]:
            raise ValueError("restored root manifest hash mismatch")


def verify_tracked(
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    *,
    stages: list[str] | tuple[str, ...] = STAGES,
) -> None:
    root = Path(tracked_root)
    manifest = _required_manifest(root)
    root_file = manifest.get("root_file")
    if isinstance(root_file, dict):
        path = root / str(root_file["path"])
        if (
            path.stat().st_size != int(root_file["size"])
            or _sha256(path) != root_file["sha256"]
        ):
            raise ValueError(f"tracked root-file metadata mismatch: {path}")
    limit = int(manifest["part_size_limit"])
    for stage in _validate_stages(stages):
        stage_manifest = (manifest.get("stages") or {}).get(stage)
        if not isinstance(stage_manifest, dict):
            raise FileNotFoundError(f"tracked stage is absent from manifest: {stage}")
        total = 0
        digest = hashlib.sha256()
        for part_info in stage_manifest["parts"]:
            part = root / str(part_info["path"])
            size = part.stat().st_size
            if size > limit:
                raise ValueError(f"part exceeds configured limit: {part}")
            if size != int(part_info["size"]) or _sha256(part) != part_info["sha256"]:
                raise ValueError(f"tracked part metadata mismatch: {part}")
            total += size
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        if total != int(stage_manifest["archive_size"]):
            raise ValueError(f"tracked archive size mismatch: {stage}")
        if digest.hexdigest() != stage_manifest["archive_sha256"]:
            raise ValueError(f"tracked archive hash mismatch: {stage}")


def verify_local(
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    *,
    stages: list[str] | tuple[str, ...] = STAGES,
) -> None:
    root = Path(local_root)
    manifest = _required_manifest(Path(tracked_root))
    for stage in _validate_stages(stages):
        stage_manifest = (manifest.get("stages") or {}).get(stage)
        if not isinstance(stage_manifest, dict):
            raise FileNotFoundError(f"tracked stage is absent from manifest: {stage}")
        _verify_file_inventory(root / stage, stage_manifest["files"], stage)
    if set(stages) == set(STAGES) and isinstance(manifest.get("root_file"), dict):
        root_file = manifest["root_file"]
        path = root / str(root_file["restored_path"])
        if _sha256(path) != root_file["sha256"]:
            raise ValueError("local root manifest hash mismatch")


def _write_deterministic_tar(source: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(source).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def _split_file(source: Path, target: Path, part_size: int) -> list[Path]:
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    parts: list[Path] = []
    with source.open("rb") as handle:
        number = 0
        while True:
            payload = handle.read(part_size)
            if not payload:
                break
            number += 1
            part = target / f"stage.tar.zst.part-{number:04d}"
            part.write_bytes(payload)
            parts.append(part)
    if not parts:
        raise ValueError(f"empty compressed archive: {source}")
    return parts


def _safe_extract(tar_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(tar_path, mode="r") as archive:
        for member in archive.getmembers():
            candidate = (destination / member.name).resolve()
            if (
                candidate != destination_resolved
                and destination_resolved not in candidate.parents
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ]


def _verify_file_inventory(
    root: Path, expected: list[dict[str, Any]], stage: str
) -> None:
    if _file_inventory(root) != expected:
        raise ValueError(f"local restored-file inventory mismatch: {stage}")


def _required_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_manifest(root / "manifest.json")
    if manifest.get("store_version") != STORE_VERSION:
        raise ValueError(f"unsupported or missing artifact-store manifest: {root}")
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zstd_binary() -> str:
    executable = shutil.which("zstd")
    if executable:
        return executable
    fallback = Path("/data1/joseph/miniconda3/bin/zstd")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("zstd executable not found")


def _run_zstd(*arguments: str) -> None:
    subprocess.run([_zstd_binary(), *arguments], check=True)


def _validate_stages(stages: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stage(s): {unknown}")
    return tuple(stages)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("package", "restore", "verify-tracked", "verify-local")
    )
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--tracked-root", default=str(DEFAULT_TRACKED_ROOT))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.action == "package":
        package_stages(
            args.local_root,
            args.tracked_root,
            stages=args.stages,
            part_size=args.part_size,
        )
    elif args.action == "restore":
        restore_stages(
            args.tracked_root,
            args.local_root,
            stages=args.stages,
            force=args.force,
        )
    elif args.action == "verify-tracked":
        verify_tracked(args.tracked_root, stages=args.stages)
    else:
        verify_local(args.local_root, args.tracked_root, stages=args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCAL_ROOT",
    "DEFAULT_PART_SIZE",
    "DEFAULT_TRACKED_ROOT",
    "STAGES",
    "STORE_VERSION",
    "package_stages",
    "restore_stages",
    "verify_local",
    "verify_tracked",
]
