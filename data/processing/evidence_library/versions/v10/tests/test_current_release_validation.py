from __future__ import annotations

import json
from argparse import Namespace

import pytest

from data.processing.evidence_library.versions.v10 import build_current_release as release
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
    auxiliary.write_bytes(b"auxiliary")
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


def test_skin_v5_is_registered_before_activation() -> None:
    version = "v10_main_universe_v5"
    assert version in KNOWN_RELEASES["skin_reaction"]
    assert evidence_library_root("skin_reaction", version).name == version


@pytest.mark.parametrize("task", ["bbb_martins", "bioavailability_ma"])
def test_validation_only_v2_is_registered_before_activation(task: str) -> None:
    version = "v10_main_universe_v2"
    assert version in KNOWN_RELEASES[task]
    assert evidence_library_root(task, version).name == version
