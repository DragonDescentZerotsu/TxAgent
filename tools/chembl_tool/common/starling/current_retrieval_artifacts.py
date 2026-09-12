"""Restore and validate the frozen Starling records used by current retrieval.

The current assay-progressive experiments start from one immutable canonical
record snapshot per task, restored under the shared Stage-03 directory layout.  Git-trackable zstd parts live under ``artifacts/``;
this module restores them into the ignored ``outputs/`` tree.  No current
builder may fall back to another user's checkout or silently select an older
record version.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/chembl_tool/starling/current_records/manifest.json"
)
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "outputs/chembl_tool/starling/current_records"
TASKS = (
    "bbb_martins",
    "bioavailability_ma",
    "skin_reaction",
    "ames",
    "dili",
    "carcinogens",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"current Starling artifact manifest is absent: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "current_starling_records.v1":
        raise ValueError(f"unsupported current Starling manifest: {path}")
    if set(manifest.get("tasks") or {}) != set(TASKS):
        raise ValueError(f"current Starling manifest task set is incomplete: {path}")
    return manifest


def current_records_path(
    task: str,
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
) -> Path:
    """Return the sole project-local canonical-record path for ``task``."""

    if task not in TASKS:
        raise ValueError(f"unknown current Starling task {task!r}")
    return Path(local_root) / task / "03_records" / "records.parquet"


def require_current_records(
    task: str,
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
) -> Path:
    """Return a verified records path or explain how to restore it."""

    path = current_records_path(task, local_root=local_root)
    if not path.is_file():
        command = (
            "python -m tools.chembl_tool.paper_experiments."
            "rebuild_current_starling_retrieval restore-records"
        )
        raise FileNotFoundError(
            f"current Starling records are absent: {path}\nRun: {command}"
        )
    expected = _load_manifest()["tasks"][task]["records_sha256"]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"current Starling records hash mismatch for {task}: "
            f"expected {expected}, found {actual}"
        )
    return path


def _verify_part(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"current Starling bundle part is absent: {path}")
    if path.stat().st_size != int(expected["size"]):
        raise ValueError(f"current Starling bundle part size mismatch: {path}")
    if _sha256(path) != expected["sha256"]:
        raise ValueError(f"current Starling bundle part hash mismatch: {path}")


def verify_packaged(task: str, *, manifest_path: Path = MANIFEST_PATH) -> None:
    manifest = _load_manifest(manifest_path)
    root = manifest_path.resolve().parents[4]
    info = manifest["tasks"][task]
    total = 0
    digest = hashlib.sha256()
    for part in info["parts"]:
        path = root / part["path"]
        _verify_part(path, part)
        total += path.stat().st_size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if total != int(info["archive_size"]):
        raise ValueError(f"current Starling archive size mismatch for {task}")
    if digest.hexdigest() != info["archive_sha256"]:
        raise ValueError(f"current Starling archive hash mismatch for {task}")


def publish_snapshot(task: str, source: Path, bundle_root: Path) -> dict[str, Any]:
    """Freeze an explicitly reviewed source snapshot using the existing bundle format."""
    from tools.chembl_tool.common.build_runtime import (
        local_workdir,
        publish_file,
        sha256_file,
    )
    from tools.chembl_tool.common.json_utils import write_json_atomic

    if task not in TASKS:
        raise ValueError(task)
    files = [
        p for p in sorted(source.iterdir()) if p.is_file() and p.suffix not in {".lock"}
    ]
    inventory = [
        {"path": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)}
        for p in files
    ]
    if not any(p.name == "records.parquet" for p in files):
        raise ValueError("Missing records")
    bundle_root.mkdir(parents=True, exist_ok=True)
    with local_workdir() as temporary:
        archive = temporary / "stage.tar"
        with tarfile.open(archive, "w", dereference=True) as handle:
            for p in files:
                handle.add(p, arcname=p.name, recursive=False)
        compressed = temporary / "stage.tar.zst"
        subprocess.run(
            ["zstd", "-q", "-T8", "-3", str(archive), "-o", str(compressed)], check=True
        )
        parts = []
        with compressed.open("rb") as handle:
            while chunk := handle.read(90_000_000):
                local = temporary / f"stage.tar.zst.part-{len(parts) + 1:04d}"
                local.write_bytes(chunk)
                target = bundle_root / local.name
                publish_file(local, target)
                parts.append(
                    {
                        "path": str(target),
                        "size": len(chunk),
                        "sha256": hashlib.sha256(chunk).hexdigest(),
                    }
                )
        info = {
            "archive_sha256": sha256_file(compressed),
            "archive_size": compressed.stat().st_size,
            "records_sha256": sha256_file(source / "records.parquet"),
            "records_size": (source / "records.parquet").stat().st_size,
            "parts": parts,
            "files": inventory,
            "source_release": str(source),
        }
    for p, expected in zip(files, inventory):
        if sha256_file(p) != expected["sha256"]:
            raise ValueError("Snapshot changed during packaging")
    destination = current_records_path(task).parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=task + "-publish-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "03_records"
        staged.mkdir()
        for p in files:
            os.link(p, staged / p.name)
        # Old acquisition/review snapshots remain at their versioned source paths.
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    manifest = _load_manifest()
    manifest["tasks"][task] = info
    write_json_atomic(MANIFEST_PATH, manifest)
    return info


def _verify_inventory(
    stage_root: Path, files: Iterable[dict[str, Any]], task: str
) -> None:
    expected_paths = {str(row["path"]) for row in files}
    actual_paths = {
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"current Starling file inventory mismatch for {task}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for row in files:
        path = stage_root / str(row["path"])
        if path.stat().st_size != int(row["size"]) or _sha256(path) != row["sha256"]:
            raise ValueError(f"current Starling restored-file mismatch: {path}")


def verify_local(
    task: str,
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    info = _load_manifest(manifest_path)["tasks"][task]
    stage_root = Path(local_root) / task / "03_records"
    if not stage_root.is_dir():
        raise FileNotFoundError(
            f"current Starling restored stage is absent: {stage_root}"
        )
    _verify_inventory(stage_root, info["files"], task)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if (
                target != destination_resolved
                and destination_resolved not in target.parents
            ):
                raise ValueError(
                    f"unsafe path in current Starling archive: {member.name}"
                )
            if member.issym() or member.islnk():
                raise ValueError(
                    f"links are not allowed in current Starling archive: {member.name}"
                )
        archive.extractall(destination, filter="data")


def restore(
    task: str,
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    manifest_path: Path = MANIFEST_PATH,
    force: bool = False,
) -> Path:
    """Restore one exact Stage-03 bundle and validate every output file."""

    manifest = _load_manifest(manifest_path)
    root = manifest_path.resolve().parents[4]
    info = manifest["tasks"][task]
    destination = Path(local_root) / task / "03_records"
    if destination.is_dir() and any(destination.iterdir()):
        try:
            verify_local(task, local_root=local_root, manifest_path=manifest_path)
            return destination
        except (FileNotFoundError, ValueError):
            if not force:
                raise FileExistsError(
                    f"refusing to replace a nonmatching current Starling stage: {destination}"
                )

    verify_packaged(task, manifest_path=manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{task}-current-records-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        compressed = temporary / "stage.tar.zst"
        archive_path = temporary / "stage.tar"
        extracted = temporary / "extracted"
        extracted.mkdir()
        with compressed.open("wb") as output:
            for part in info["parts"]:
                with (root / part["path"]).open("rb") as handle:
                    shutil.copyfileobj(handle, output, length=1024 * 1024)
        if _sha256(compressed) != info["archive_sha256"]:
            raise ValueError(
                f"reassembled current Starling archive mismatch for {task}"
            )
        zstd = shutil.which("zstd")
        if not zstd:
            raise RuntimeError("zstd executable is required to restore current records")
        subprocess.run(
            [zstd, "-d", "-f", str(compressed), "-o", str(archive_path)],
            check=True,
        )
        _safe_extract(archive_path, extracted)
        _verify_inventory(extracted, info["files"], task)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(extracted, destination)
    verify_local(task, local_root=local_root, manifest_path=manifest_path)
    return destination


def restore_published_files(task: str, stage: Path) -> None:
    """Restore declared large audit files from the same verified source bundle.

    Ordinary Git archive parts keep every file below the hosting size limit;
    scientific inputs retain their original bytes and paths after restoration.
    """
    from tools.chembl_tool.common.json_utils import atomic_output_path

    info = _load_manifest()["tasks"][task]
    inventory = {item["path"]: item for item in info["files"]}
    for relative, member in info.get("published_file_aliases", {}).items():
        target = PROJECT_ROOT / relative
        if not target.resolve().is_relative_to(PROJECT_ROOT / "data"):
            raise ValueError(f"Published audit path must stay under data/: {relative}")
        if member not in inventory or Path(member).name != member:
            raise ValueError(f"Unknown source bundle member: {member}")
        source = stage / member
        expected = inventory[member]["sha256"]
        if _sha256(source) != expected:
            raise ValueError(f"Published audit member hash mismatch: {member}")
        if target.exists():
            if _sha256(target) != expected:
                raise ValueError(
                    f"Refusing to replace a changed published audit: {relative}"
                )
            continue
        with atomic_output_path(target) as temporary:
            shutil.copyfile(source, temporary)
