from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.starling import build_current_release as release
from tools.chembl_tool.common.starling.build_runtime import (
    INCOMPLETE_BUILD_FILENAME,
    starling_build_session,
)


def test_task_locks_are_independent_and_same_root_fails_fast(tmp_path) -> None:
    roots = [tmp_path / task for task in release.TASKS]
    with starling_build_session(roots[0]):
        with starling_build_session(roots[1]):
            with starling_build_session(roots[2]):
                assert all(
                    (root / INCOMPLETE_BUILD_FILENAME).is_file() for root in roots
                )

        code = """
import sys
from tools.chembl_tool.common.starling.build_runtime import starling_build_session
try:
    with starling_build_session(sys.argv[1]):
        pass
except RuntimeError as error:
    print(error)
    raise SystemExit(7)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(roots[0])],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 7
        assert "Starling build already running" in result.stdout


def test_complete_session_clears_marker_only_after_success(tmp_path) -> None:
    root = tmp_path / "successful"
    with starling_build_session(root, complete=True):
        assert (root / INCOMPLETE_BUILD_FILENAME).is_file()
    assert not (root / INCOMPLETE_BUILD_FILENAME).exists()

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="stop"):
        with starling_build_session(failed, complete=True):
            raise RuntimeError("stop")
    assert (failed / INCOMPLETE_BUILD_FILENAME).is_file()


def test_view_build_rejects_an_incomplete_source(tmp_path) -> None:
    root = tmp_path / "incomplete"
    root.mkdir()
    (root / INCOMPLETE_BUILD_FILENAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="normalized source has an incomplete build"):
        release.build_v7_benchmark_view(
            policy=SimpleNamespace(),
            normalized_root=root,
            heldout_labels_jsonl=tmp_path / "heldout.jsonl",
            out_dir=tmp_path / "view",
            benchmark_split="scaffold",
        )


def test_current_release_orders_all_stages_and_then_clears_marker(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "bbb"
    heldout = tmp_path / "heldout.jsonl"
    view_out = tmp_path / "view"
    events: list[object] = []
    policy = SimpleNamespace(default_out_dir=str(root))

    monkeypatch.setattr(release, "load_task_policy", lambda task_id: policy)
    monkeypatch.setattr(
        release,
        "parse_record_args",
        lambda *args, **kwargs: SimpleNamespace(out_dir=str(root)),
    )
    monkeypatch.setattr(
        release, "run_with_args", lambda *args: events.append("stage2") or 0
    )
    downstream = SimpleNamespace(
        build_canonical_artifacts=lambda **kwargs: events.append(
            ("stage3", kwargs["apply_outlier_review"])
        )
    )
    monkeypatch.setattr(release.importlib, "import_module", lambda name: downstream)
    monkeypatch.setattr(
        release,
        "generate_final_endpoint_pruning",
        lambda **kwargs: events.append("pruning"),
    )
    monkeypatch.setattr(
        release,
        "build_v7_benchmark_view",
        lambda **kwargs: events.append("view") or {"view": "manifest"},
    )
    monkeypatch.setattr(
        release,
        "_validate_release",
        lambda *args: events.append("validate") or {"task_id": "bbb_martins"},
    )

    result = release.build_current_release(
        task_id="bbb_martins",
        normalized_root=root,
        heldout_labels=heldout,
        view_out=view_out,
    )

    assert result == {"task_id": "bbb_martins"}
    assert events == [
        "stage2",
        ("stage3", False),
        "pruning",
        ("stage3", True),
        "view",
        "validate",
    ]
    assert not (root / INCOMPLETE_BUILD_FILENAME).exists()


def test_current_release_failure_keeps_incomplete_marker(tmp_path, monkeypatch) -> None:
    root = tmp_path / "skin"
    policy = SimpleNamespace(default_out_dir=str(root))
    monkeypatch.setattr(release, "load_task_policy", lambda task_id: policy)
    monkeypatch.setattr(
        release,
        "parse_record_args",
        lambda *args, **kwargs: SimpleNamespace(out_dir=str(root)),
    )
    monkeypatch.setattr(release, "run_with_args", lambda *args: 0)
    downstream = SimpleNamespace(build_canonical_artifacts=lambda **kwargs: None)
    monkeypatch.setattr(release.importlib, "import_module", lambda name: downstream)

    def fail_pruning(**kwargs):
        raise RuntimeError("review failed")

    monkeypatch.setattr(release, "generate_final_endpoint_pruning", fail_pruning)
    with pytest.raises(RuntimeError, match="review failed"):
        release.build_current_release(task_id="skin_reaction", normalized_root=root)
    assert (root / INCOMPLETE_BUILD_FILENAME).is_file()


def test_release_validation_rejects_stale_review_bytes(tmp_path) -> None:
    root = tmp_path / "release"
    canonical_dir = root / "02_canonicalized"
    review_dir = root / release.ARTIFACT_DIR
    stage3_dir = root / "03_pair_buckets"
    for directory in (canonical_dir, review_dir, stage3_dir):
        directory.mkdir(parents=True)

    canonical = canonical_dir / "records.parquet"
    auxiliary = canonical_dir / "auxiliary_mapping_manifest.json"
    decisions = review_dir / "decisions.parquet"
    stage3 = stage3_dir / "records.parquet"
    canonical.write_bytes(b"stage2")
    auxiliary.write_bytes(b"auxiliary")
    decisions.write_bytes(b"review")
    stage3.write_bytes(b"stage3")
    canonical_sha = release.file_sha256(canonical)
    review_manifest = review_dir / release.REVIEW_MANIFEST_FILENAME
    review_manifest.write_text(
        json.dumps(
            {
                "task_id": "bbb_martins",
                "inputs": {"canonical_records": {"sha256": canonical_sha}},
                "files": {"decisions.parquet": release.file_sha256(decisions)},
            }
        ),
        encoding="utf-8",
    )
    (canonical_dir / "manifest.json").write_text(
        json.dumps({"output": {"sha256": canonical_sha}}), encoding="utf-8"
    )
    (stage3_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "bbb_martins",
                "input_hashes": {
                    "canonical_records": canonical_sha,
                    "auxiliary_mapping_manifest": release.file_sha256(auxiliary),
                    "assay_transfer_outlier_review": release.file_sha256(
                        review_manifest
                    ),
                },
                "outputs": {"records.parquet": release.file_sha256(stage3)},
            }
        ),
        encoding="utf-8",
    )

    release._validate_release("bbb_martins", root, None)
    decisions.write_bytes(b"stale")
    with pytest.raises(ValueError, match="pruning review output hash mismatch"):
        release._validate_release("bbb_martins", root, None)
