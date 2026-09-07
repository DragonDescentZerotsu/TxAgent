from pathlib import Path
import json
import os

import pytest

from tools.chembl_tool.common import build_runtime as runtime


def test_local_cache_validates_content_and_recovers_from_corruption(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "BUILD_CACHE", tmp_path / "cache")
    source = tmp_path / "source.jsonl"
    source.write_text("original")
    cached = runtime.local_input(source)
    assert cached.read_text() == "original"
    cached.write_text("corrupt!")
    assert runtime.local_input(source).read_text() == "original"
    old_stat = source.stat()
    old_hash = runtime.sha256_file(source)
    source.write_text("changed!")
    os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert runtime.sha256_file(source) != old_hash
    assert runtime.local_input(source).read_text() == "changed!"


def test_parallel_json_preserves_exact_serial_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "BUILD_CACHE", tmp_path / "cache")
    rows = [
        {
            "i": i,
            "text": '中文\n"',
            "value": float("nan") if i % 2 else None,
            "path": Path("literal"),
        }
        for i in range(10001)
    ]
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    runtime.write_jsonl_local(serial, rows, workers=1)
    runtime.write_jsonl_local(parallel, rows, workers=2)
    assert parallel.read_bytes() == serial.read_bytes()
    assert runtime._json_rows is None
    assert not list((tmp_path / "cache").glob("build-*"))


def test_publication_is_atomic_and_digest_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "BUILD_CACHE", tmp_path / "cache")
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new payload")
    target.write_bytes(b"previous")
    digest = runtime.publish_file(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert digest == runtime.uncached_sha256(target)
    before = target.stat()
    assert runtime.publish_file(source, target) == digest
    after = target.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)

    class BrokenHash:
        def update(self, chunk):
            raise RuntimeError("interrupted copy")

    monkeypatch.setattr(runtime.hashlib, "sha256", BrokenHash)
    source.write_bytes(b"interrupted payload")
    with pytest.raises(RuntimeError, match="interrupted copy"):
        runtime.publish_file(source, target)
    assert target.read_bytes() == b"new payload"
    assert not list(tmp_path.glob(".*.tmp"))


def test_stage_reuse_requires_signature_and_output_digest(tmp_path):
    artifact = tmp_path / "output"
    artifact.write_text("verified")
    signature = {"input": "hash", "policy": "v1"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"build_inputs": signature, "output_sha256": runtime.sha256_file(artifact)}
        )
    )
    assert runtime.reusable(manifest, signature, [(artifact, "output_sha256")])
    assert not runtime.reusable(
        manifest, {"policy": "v2"}, [(artifact, "output_sha256")]
    )
    artifact.write_text("modified")
    assert not runtime.reusable(manifest, signature, [(artifact, "output_sha256")])


def test_persisted_digest_reuse_is_scoped_to_file_metadata_and_boot(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(runtime, "BUILD_CACHE", tmp_path / "cache")
    clock = [100.0]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    source = tmp_path / "source"
    source.write_text("original")
    digest = runtime.sha256_file(source)
    runtime._digests.clear()
    clock[0] += 2
    calls = []
    original = runtime.uncached_sha256
    monkeypatch.setattr(
        runtime, "uncached_sha256", lambda p: calls.append(p) or original(p)
    )
    assert runtime.sha256_file(source) == digest
    assert calls == []
    source.write_text("changed payload")
    assert runtime.sha256_file(source) != digest
    assert calls == [source.resolve()]
    runtime._digests.clear()
    monkeypatch.setattr(runtime, "_boot_id", "different boot")
    runtime.sha256_file(source)
    assert len(calls) == 2
