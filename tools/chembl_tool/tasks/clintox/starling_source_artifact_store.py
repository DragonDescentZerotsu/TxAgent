"""Package, verify, and restore the exact supplied ClinTox send_v2 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.chembl_tool.tasks.clintox.starling_source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    DEFAULT_DATA_ROOT,
    import_archive,
)


STORE_VERSION = "clintox.send_v2_source_archive.v1"
DEFAULT_PART_SIZE = 90_000_000
DEFAULT_TRACKED_ROOT = Path(
    "artifacts/chembl_tool/tasks/clintox/clintox_send_v2_source"
)
DEFAULT_RESTORED_ARCHIVE = Path(
    "outputs/chembl_tool/tasks/clintox/source_archives/clintox_send_v2.tar.gz"
)


def package_archive(
    archive: str | Path = DEFAULT_ARCHIVE,
    *,
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    part_size: int = DEFAULT_PART_SIZE,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Split the exact archive into repository-safe, checksummed parts."""
    source = Path(archive)
    expected = expected_sha256 or ARCHIVE_SHA256
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    if _sha256(source) != expected:
        raise ValueError(f"unexpected ClinTox source archive digest: {source}")
    root = Path(tracked_root)
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("clintox_send_v2.tar.gz.part-*"):
        old.unlink()
    parts = _split(source, root, part_size)
    payload = {
        "store_version": STORE_VERSION,
        "archive_name": "clintox_send_v2.tar.gz",
        "archive_size": source.stat().st_size,
        "archive_sha256": expected,
        "part_size_limit": part_size,
        "parts": [_part_receipt(path, root) for path in parts],
    }
    _write_json(root / "manifest.json", payload)
    return payload


def verify_tracked(
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every part and the reassembled archive digest without writing it."""
    root = Path(tracked_root)
    manifest = _load_manifest(root)
    expected = expected_sha256 or ARCHIVE_SHA256
    if manifest.get("archive_sha256") != expected:
        raise ValueError("tracked ClinTox archive digest does not match the source contract")
    digest = hashlib.sha256()
    total = 0
    limit = int(manifest["part_size_limit"])
    for receipt in manifest["parts"]:
        part = root / str(receipt["path"])
        size = part.stat().st_size
        if size > limit or size != int(receipt["size"]):
            raise ValueError(f"tracked source part size mismatch: {part}")
        if _sha256(part) != receipt["sha256"]:
            raise ValueError(f"tracked source part digest mismatch: {part}")
        total += size
        _update_digest(digest, part)
    if total != int(manifest["archive_size"]) or digest.hexdigest() != expected:
        raise ValueError("reassembled ClinTox source archive mismatch")
    return manifest


def restore_archive(
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    *,
    output: str | Path = DEFAULT_RESTORED_ARCHIVE,
    expected_sha256: str | None = None,
) -> Path:
    """Atomically restore the exact archive from verified tracked parts."""
    root = Path(tracked_root)
    manifest = verify_tracked(root, expected_sha256=expected_sha256)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            for receipt in manifest["parts"]:
                with (root / str(receipt["path"])).open("rb") as part:
                    while chunk := part.read(1024 * 1024):
                        handle.write(chunk)
        if _sha256(temporary) != manifest["archive_sha256"]:
            raise ValueError("restored ClinTox source archive digest mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def restore_source(
    *,
    tracked_root: str | Path = DEFAULT_TRACKED_ROOT,
    archive_output: str | Path = DEFAULT_RESTORED_ARCHIVE,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Restore the archive and atomically import its seven source directories."""
    archive = restore_archive(tracked_root, output=archive_output)
    return import_archive(archive, data_root=data_root, force=force)


def _split(source: Path, root: Path, part_size: int) -> list[Path]:
    parts: list[Path] = []
    with source.open("rb") as handle:
        number = 0
        while payload := handle.read(part_size):
            number += 1
            part = root / f"clintox_send_v2.tar.gz.part-{number:04d}"
            part.write_bytes(payload)
            parts.append(part)
    if not parts:
        raise ValueError("cannot package an empty ClinTox source archive")
    return parts


def _part_receipt(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("store_version") != STORE_VERSION:
        raise ValueError(f"unsupported ClinTox source artifact store: {root}")
    return manifest


def _update_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, path)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "verify-tracked", "restore-archive", "restore-source"))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--tracked-root", default=str(DEFAULT_TRACKED_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_RESTORED_ARCHIVE))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "package":
        package_archive(args.archive, tracked_root=args.tracked_root, part_size=args.part_size)
    elif args.action == "verify-tracked":
        verify_tracked(args.tracked_root)
    elif args.action == "restore-archive":
        restore_archive(args.tracked_root, output=args.output)
    else:
        restore_source(
            tracked_root=args.tracked_root,
            archive_output=args.output,
            data_root=args.data_root,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PART_SIZE",
    "DEFAULT_RESTORED_ARCHIVE",
    "DEFAULT_TRACKED_ROOT",
    "STORE_VERSION",
    "package_archive",
    "restore_archive",
    "restore_source",
    "verify_tracked",
]
