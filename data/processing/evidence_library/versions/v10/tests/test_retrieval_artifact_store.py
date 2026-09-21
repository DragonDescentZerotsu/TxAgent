import json
from pathlib import Path

import pytest

from data.processing.evidence_library import retrieval_artifact_store as store


def _write_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = ("data/cache/example", "data/evidence/example")
    monkeypatch.setattr(store, "BUNDLE_PATHS", paths)
    monkeypatch.setattr(store, "PART_SIZE", 64)
    monkeypatch.setattr(store, "CACHE_CONFIG", Path("profile.yaml"))
    for number, path in enumerate(paths):
        target = root / path
        target.mkdir(parents=True)
        (target / "payload.bin").write_bytes(bytes([number]) * 256)
    (root / "profile.yaml").write_text("version: 1\n", encoding="utf-8")


def test_retrieval_bundle_round_trip_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    archive = tmp_path / "archive"
    restored = tmp_path / "restored"
    source.mkdir()
    restored.mkdir()
    _write_fixture(source, monkeypatch)
    (restored / "profile.yaml").write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setattr(store, "_source_commit", lambda root: "deadbeef")

    store._package(source, archive, tmp_path)
    store._verify_manifest(archive, restored)
    store.verify_tracked(store._profile(source, archive))
    store.restore_stages(store._profile(restored, archive), temporary_root=tmp_path)
    store.verify_local(store._profile(restored, archive))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        store.restore_stages(store._profile(restored, archive), temporary_root=tmp_path)

    part = next(archive.glob("retrieval-bundle.part-*"))
    part.write_bytes(part.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="part metadata mismatch"):
        store.verify_tracked(store._profile(source, archive))


def test_manifest_hash_rejects_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    archive = tmp_path / "archive"
    source.mkdir()
    _write_fixture(source, monkeypatch)
    monkeypatch.setattr(store, "_source_commit", lambda root: "deadbeef")
    store._package(source, archive, tmp_path)

    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    manifest["bundle_id"] = "changed"
    (archive / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        store._verify_manifest(archive)


def test_package_rejects_evidence_outside_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    archive = tmp_path / "archive"
    source.mkdir()
    _write_fixture(source, monkeypatch)
    cache = source / "data/cache/example"
    (cache / "RELEASE_INDEX.json").write_text(
        json.dumps(
            {
                "evidence": {
                    "manifest": "../../../outside/VERSION.json",
                    "manifest_sha256": "unused",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "_source_commit", lambda root: "deadbeef")

    with pytest.raises(ValueError, match="outside the bundle"):
        store._package(source, archive, tmp_path)
