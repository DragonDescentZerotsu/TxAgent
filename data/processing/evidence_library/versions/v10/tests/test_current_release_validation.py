from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.versions.v10 import build_current_release as release
from data.processing.evidence_library.versions.v10 import (
    build_reference_semantics_mapping as reference_mapping,
)
from data.processing.paths import KNOWN_RELEASES, evidence_library_root


def _release_tree(tmp_path, *, pair: str, transfer: str):
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
    auxiliary.write_text("{}\n", encoding="utf-8")
    decisions.write_bytes(b"review")
    stage3.write_bytes(b"stage3")
    canonical_sha = release.file_sha256(canonical)
    pruning = review_dir / release.PRUNING_MANIFEST_FILENAME
    pruning.write_text(
        json.dumps(
            {
                "task_id": "skin_reaction",
                "inputs": {"canonical_records": {"sha256": canonical_sha}},
                "files": {"decisions.parquet": release.file_sha256(decisions)},
            }
        )
    )
    (canonical_dir / "manifest.json").write_text(
        json.dumps(
            {
                "output": {"sha256": canonical_sha},
                "validations": {
                    "measurement_unit_pair_validation": pair,
                    "final_assay_transfer_measurement_validation": transfer,
                },
            }
        )
    )
    (stage3_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "skin_reaction",
                "input_hashes": {
                    "canonical_records": canonical_sha,
                    "auxiliary_mapping_manifest": release.file_sha256(auxiliary),
                    "assay_transfer_record_pruning": release.file_sha256(pruning),
                },
                "outputs": {"records.parquet": release.file_sha256(stage3)},
            }
        )
    )
    return root


def test_release_validation_requires_full_scientific_checks(tmp_path) -> None:
    root = _release_tree(tmp_path, pair="not_run", transfer="not_run")
    with pytest.raises(ValueError, match="pair validation did not pass"):
        release._validate_release("skin_reaction", root)

    manifest = json.loads((root / "02_canonicalized/manifest.json").read_text())
    manifest["validations"]["measurement_unit_pair_validation"] = "passed"
    (root / "02_canonicalized/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="assay-transfer measurement validation"):
        release._validate_release("skin_reaction", root)

    manifest["validations"]["final_assay_transfer_measurement_validation"] = "passed"
    (root / "02_canonicalized/manifest.json").write_text(json.dumps(manifest))
    assert release._validate_release("skin_reaction", root)["task_id"] == "skin_reaction"


def test_release_validation_checks_bbb_level_mapping(tmp_path, monkeypatch) -> None:
    root = _release_tree(tmp_path, pair="passed", transfer="passed")
    uid = "sr_00000000000000000000000000000001"
    stage3 = root / "03_pair_buckets/records.parquet"
    pq.write_table(pa.Table.from_pylist([{"source_row_uid": uid}]), stage3)
    stage3_manifest_path = root / "03_pair_buckets/manifest.json"
    stage3_manifest = json.loads(stage3_manifest_path.read_text())
    stage3_manifest["task_id"] = "bbb_martins"
    stage3_manifest["outputs"]["records.parquet"] = release.file_sha256(stage3)
    stage3_manifest_path.write_text(json.dumps(stage3_manifest))
    pruning_path = root / release.ARTIFACT_DIR / release.PRUNING_MANIFEST_FILENAME
    pruning = json.loads(pruning_path.read_text())
    pruning["task_id"] = "bbb_martins"
    pruning_path.write_text(json.dumps(pruning))
    stage3_manifest["input_hashes"]["assay_transfer_record_pruning"] = release.file_sha256(
        pruning_path
    )
    stage3_manifest_path.write_text(json.dumps(stage3_manifest))

    voters = tmp_path / "voters.parquet"
    pq.write_table(pa.Table.from_pylist([{"source_row_uid": uid}]), voters)
    level_root = root / "level_mapping"
    level_root.mkdir()
    level_records = level_root / "records.parquet"
    eligibility = level_root / "assay_transfer_record_eligibility.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"source_row_uid": uid, "level": 1}]), level_records
    )
    pq.write_table(pa.Table.from_pylist([{"source_row_uid": uid}]), eligibility)
    level_manifest = {
        "version": "evidence_library_level_mapping.v1",
        "status": "complete",
        "task": "bbb_martins",
        "evidence_library_version": root.name,
        "inputs": {
            "stage3_records": {"sha256": release.file_sha256(stage3)},
            "voter_membership": {"sha256": release.file_sha256(voters)},
        },
        "output": {
            "sha256": release.file_sha256(level_records),
            "rows": 1,
        },
        "assay_transfer_record_eligibility": {
            "sha256": release.file_sha256(eligibility),
            "rows": 1,
        },
        "validation": {
            "stage3_uid_coverage": "exact",
            "l1_equals_gold_physical_voters": True,
        },
    }
    (level_root / "manifest.json").write_text(json.dumps(level_manifest))
    monkeypatch.setitem(release.RELEASE_VOTER_MEMBERSHIP, "bbb_martins", voters)
    monkeypatch.setattr(release, "_validate_published_manifest_paths", lambda *_: None)

    assert release._validate_release("bbb_martins", root)["task_id"] == "bbb_martins"
    level_manifest["inputs"]["stage3_records"]["sha256"] = "stale"
    (level_root / "manifest.json").write_text(json.dumps(level_manifest))
    with pytest.raises(ValueError, match="incomplete or stale"):
        release._validate_release("bbb_martins", root)


def test_publication_rejects_transient_manifest_paths(tmp_path) -> None:
    root = tmp_path / "v10_test"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"input_path": "/tmp/lost"}))
    with pytest.raises(ValueError, match="transient path"):
        release._validate_published_manifest_paths("bbb_martins", root)


def test_completed_release_rejects_strict_validation() -> None:
    with pytest.raises(ValueError, match="require validation_level='full'"):
        release.build_current_release(
            task_id="skin_reaction",
            normalized_root="unused",
            validation_level="strict",
        )


def test_existing_mapping_mode_validates_the_selected_registry() -> None:
    selected = release._validate_existing_mappings("bbb_martins")
    assert selected["mode"] == "existing"
    assert selected["registry_version"] == "bbb_martins_prebuilt_mapping_registry.v2"
    assert selected["mappings"]["measurement_resolution"]["sha256"] == (
        "74196a0cd1483e501f47a4124c2b763a0031060426a1cdf832b735b76bad5bf2"
    )


def test_accepted_mapping_is_limited_to_target_release() -> None:
    selected = release._validate_existing_mappings(
        "ames", target_release="v10_main_universe_v2"
    )
    assert selected["registry_version"] == "ames_prebuilt_mapping_registry.v3"
    with pytest.raises(ValueError, match="not valid for"):
        release._validate_existing_mappings(
            "ames", target_release="v10_main_universe_v4"
        )


def test_reference_semantics_low_reasoning_has_a_distinct_request_identity() -> None:
    config = reference_mapping._load_config("ames")
    row = {"id": "row-1", "source_id": "mutagenicity_outcomes"}
    low = reference_mapping._batches(
        [row], config=config, prompt="prompt", attempted=set(), reasoning_effort="low"
    )
    high = reference_mapping._batches(
        [row], config=config, prompt="prompt", attempted=set(), reasoning_effort="high"
    )
    assert low[0].request_id != high[0].request_id

    args = reference_mapping.parse_args(
        [
            "--task",
            "ames",
            "--provider-pool-config",
            "pool.json",
            "--parallelism",
            "512",
            "--reasoning-effort",
            "low",
        ]
    )
    assert args.reasoning_effort == "low"
    assert args.parallelism == 512


def test_reference_semantics_retries_only_failed_or_interrupted_rows() -> None:
    cache = SimpleNamespace(
        attempted={"valid", "invalid", "partial", "interrupted"},
        assignments={
            "valid": {"assignment_method": "model_single_pass"},
            "invalid": {"assignment_method": "invalid_response"},
            "partial": {
                "assignment_method": "invalid_row_response:endpoint_ratio_basis_mismatch"
            },
        },
    )
    assert reference_mapping.retryable_assignment_ids(cache) == {
        "invalid",
        "partial",
        "interrupted",
    }


def test_reference_semantics_normalizes_an_internally_inconsistent_pair() -> None:
    config = reference_mapping._load_config("dili")
    row = {"id": "row-1", "source_id": "dili_v1"}
    assignment, error = reference_mapping._validate_assignment(
        {
            "id": "0",
            "reference_scope": "endpoint_defined_ratio",
            "reference_basis": "baseline",
        },
        row,
        config=config,
        model_id="0",
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    assert error is None
    assert assignment["reference_scope"] == "comparator_relative"
    assert "normalized_scope_from_basis" in assignment["assignment_method"]


def test_existing_mapping_mode_rejects_an_unregistered_override(tmp_path) -> None:
    args = Namespace(
        measurement_resolution_mapping=tmp_path / "replacement.parquet",
        exact_unit_mapping=(
            release.import_task_module("bbb_martins", "mapping_registry").mapping_path(
                "exact_measurement_units"
            )
        ),
        auxiliary_mapping=(
            release.import_task_module("bbb_martins", "mapping_registry").mapping_path(
                "auxiliary_context"
            )
        ),
    )
    with pytest.raises(ValueError, match="registry-selected measurement_resolution"):
        release._require_selected_mapping_args("bbb_martins", args)


def test_skin_v6_is_registered_before_activation() -> None:
    version = "v10_main_universe_v6"
    assert version in KNOWN_RELEASES["skin_reaction"]
    assert evidence_library_root("skin_reaction", version).name == version


@pytest.mark.parametrize("task", ["bbb_martins", "bioavailability_ma"])
def test_validation_only_v3_is_registered_before_activation(task: str) -> None:
    version = "v10_main_universe_v3"
    assert version in KNOWN_RELEASES[task]
    assert evidence_library_root(task, version).name == version


@pytest.mark.parametrize("task", ["ames", "dili", "carcinogens"])
def test_imported_task_main_universe_stage1_is_registered(task: str) -> None:
    version = "v10_main_universe_v1"
    assert version in KNOWN_RELEASES[task]
    assert evidence_library_root(task, version).name == version


@pytest.mark.parametrize("task", ["ames", "dili", "carcinogens"])
def test_imported_task_final_v3_is_registered_before_activation(task: str) -> None:
    version = "v10_main_universe_v3"
    assert version in KNOWN_RELEASES[task]
    assert evidence_library_root(task, version).name == version
