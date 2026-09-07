"""Bounded build workers, verified local input copies and atomic publication."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import gc
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import tempfile
import threading
import time

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    sha256_file as uncached_sha256,
    write_json_atomic,
)

BUILD_CACHE = Path(
    os.environ.get(
        "TXAGENT_BUILD_CACHE",
        (
            "/local/tmp/txagent-build-cache"
            if Path("/local/tmp").is_dir()
            else tempfile.gettempdir() + "/txagent-build-cache"
        ),
    )
)
_digests = {}
_digest_lock = threading.RLock()
try:
    _boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
except OSError:
    _boot_id = str(os.getpid())


def _digest_receipt(path):
    return (
        BUILD_CACHE
        / "digests"
        / (hashlib.sha256(str(path).encode()).hexdigest() + ".json")
    )


def _load_digest(path):
    try:
        receipt = json.loads(_digest_receipt(path).read_text())
        if receipt["boot_id"] == _boot_id and receipt["path"] == str(path):
            digest = receipt["sha256"]
            if len(digest) == 64 and int(digest, 16) >= 0:
                return tuple(receipt["stamp"]), digest, float(receipt["observed"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _remember(path, stamp, digest, observed=None):
    observed = time.monotonic() if observed is None else observed
    _digests[path] = stamp, digest, observed
    write_json_atomic(
        _digest_receipt(path),
        {
            "path": str(path),
            "stamp": stamp,
            "sha256": digest,
            "boot_id": _boot_id,
            "observed": observed,
        },
    )


@contextmanager
def worker_pool(workers):
    """Fork immutable build inputs without repeatedly scanning them in child GC."""
    already_frozen = bool(gc.get_freeze_count())
    if not already_frozen:
        gc.freeze()
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            yield pool
    finally:
        if not already_frozen:
            gc.unfreeze()


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def sha256_file(path):
    """Reuse a verified digest for an unchanged file on the same host boot.

    The observation window handles filesystems that coalesce rapid timestamps.
    Independent audits continue to use uncached_sha256 for their initial scan.
    """
    path = Path(path).resolve()
    with _digest_lock:
        before = _stamp(path)
        cached = _digests.get(path) or _load_digest(path)
        if cached and cached[0] == before and time.monotonic() - cached[2] >= 1.0:
            return cached[1]
        digest = uncached_sha256(path)
        if _stamp(path) != before:
            raise RuntimeError(f"Input changed while hashing: {path}")
        _remember(
            path,
            before,
            digest,
            cached[2] if cached and cached[:2] == (before, digest) else None,
        )
        return digest


def local_input(path):
    """Content-addressed NVMe copy; public provenance continues to use original paths."""
    path = Path(path)
    digest = sha256_file(path)
    target = BUILD_CACHE / "inputs" / (digest + path.suffix)
    if not target.is_file() or sha256_file(target) != digest:
        with atomic_output_path(target) as temporary:
            shutil.copyfile(path, temporary)
            if sha256_file(temporary) != digest:
                raise RuntimeError(f"Input changed while copying: {path}")
        with _digest_lock:
            _remember(target.resolve(), _stamp(target), digest)
    return target


@contextmanager
def local_workdir():
    BUILD_CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=BUILD_CACHE) as folder:
        yield Path(folder)


def publish_file(local_path, target):
    """Hash on NVMe; skip identical outputs, otherwise copy once and publish atomically."""
    local_path, target = Path(local_path), Path(target)
    value = uncached_sha256(local_path)
    if target.is_file() and sha256_file(target) == value:
        print(f"Reused identical output: {target}", flush=True)
    else:
        digest = hashlib.sha256()
        with atomic_output_path(target) as temporary:
            with local_path.open("rb") as source, temporary.open(
                "wb", buffering=8 * 1024 * 1024
            ) as output:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != value:
                raise RuntimeError(
                    f"Local output changed while publishing: {local_path}"
                )
        with _digest_lock:
            _remember(target.resolve(), _stamp(target), value)
    cached = BUILD_CACHE / "inputs" / (value + target.suffix)
    if not cached.is_file() or sha256_file(cached) != value:
        with atomic_output_path(cached) as temporary:
            temporary.unlink()
            os.link(local_path, temporary)
        with _digest_lock:
            _remember(cached.resolve(), _stamp(cached), value)
    return value


def build_signature(paths, options):
    """Conservative code/environment invalidation for reusable build stages."""
    import numpy, pandas, pyarrow
    from rdkit import rdBase

    common = Path(__file__).parent
    inputs = sorted({Path(p).resolve() for p in paths} | set(common.rglob("*.py")))
    return {
        "files": {str(p): sha256_file(p) for p in inputs},
        "options": options,
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }


def reusable(manifest_path, signature, outputs):
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        return manifest.get("build_inputs") == signature and all(
            sha256_file(Path(p)) == manifest[key] for p, key in outputs
        )
    except (OSError, ValueError, KeyError):
        return False


_json_rows = None


def _json_shard(task):
    start, stop, target = task
    with open(target, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as handle:
        for i in range(start, stop):
            handle.write(
                json.dumps(_json_rows[i], ensure_ascii=False, default=str) + "\n"
            )
    return target


def write_jsonl_local(path, rows, workers=1):
    """Parallel serialization with exact stdlib JSON bytes and original row order."""
    global _json_rows
    if workers < 2 or len(rows) < 10000:
        with Path(path).open(
            "w", encoding="utf-8", buffering=8 * 1024 * 1024
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return
    _json_rows = rows
    try:
        with local_workdir() as shards:
            count = min(workers, (len(rows) + 9999) // 10000)
            size = (len(rows) + count - 1) // count
            tasks = [
                (start, min(start + size, len(rows)), str(shards / f"{start}.jsonl"))
                for start in range(0, len(rows), size)
            ]
            with worker_pool(count) as pool:
                parts = list(pool.map(_json_shard, tasks))
            with Path(path).open("wb", buffering=8 * 1024 * 1024) as output:
                for part in parts:
                    with open(part, "rb") as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    finally:
        _json_rows = None
