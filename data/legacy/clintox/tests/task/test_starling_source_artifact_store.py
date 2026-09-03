from __future__ import annotations

import hashlib

import pytest

from tools.chembl_tool.tasks.clintox.starling_source_artifact_store import (
    package_archive,
    restore_archive,
    verify_tracked,
)


def test_source_archive_store_round_trip_and_tamper_detection(tmp_path):
    source = tmp_path / "source.tar.gz"
    payload = b"clin-tox-source-archive" * 7
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    tracked = tmp_path / "tracked"

    manifest = package_archive(
        source,
        tracked_root=tracked,
        part_size=17,
        expected_sha256=digest,
    )

    assert len(manifest["parts"]) > 1
    assert verify_tracked(tracked, expected_sha256=digest) == manifest
    restored = restore_archive(
        tracked,
        output=tmp_path / "restored.tar.gz",
        expected_sha256=digest,
    )
    assert restored.read_bytes() == payload

    first = tracked / manifest["parts"][0]["path"]
    first.write_bytes(b"corrupt" + first.read_bytes())
    with pytest.raises(ValueError, match="size mismatch"):
        verify_tracked(tracked, expected_sha256=digest)
