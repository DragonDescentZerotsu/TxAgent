from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.chembl_tool.common.starling import split_downstream
from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    _retain_valid_structures,
    _verify_stage_artifact,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.contracts import (
    NORMALIZATION_STAGE_VERSION,
)
from tools.chembl_tool.tasks.bbb_martins import (
    build_normalized_starling_evidence_library as bbb,
)
from tools.chembl_tool.tasks.bioavailability_ma import (
    build_normalized_starling_evidence_library as bio,
)
from tools.chembl_tool.tasks.skin_reaction import (
    build_normalized_starling_evidence_library as skin,
)


BUILDERS = (bbb, bio, skin)


def test_stage1_retains_only_resolved_structures_with_lineage():
    retained, rejected = _retain_valid_structures(
        [
            {
                "cleaned_record_id": "valid",
                "canonical_smiles": "CCO",
                "structure_status": "resolved",
            },
            {
                "cleaned_record_id": "missing",
                "canonical_smiles": None,
                "structure_status": "missing_structure",
            },
        ]
    )

    assert [row["cleaned_record_id"] for row in retained] == ["valid"]
    assert rejected[0]["cleaned_record_id"] == "missing"
    assert rejected[0]["stage1_rejection_reason"] == "missing_structure"


def test_normalize_restart_rejects_stale_assay_transfer_policy(tmp_path):
    cleaned = tmp_path / "01_cleaned/records.parquet"
    cleaned.parent.mkdir()
    cleaned.write_text("cleaned", encoding="utf-8")
    records = tmp_path / "02_canonicalized/records.parquet"
    records.parent.mkdir()
    records.write_text("canonical", encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"version":1}\n', encoding="utf-8")
    manifest = {
        "version": NORMALIZATION_STAGE_VERSION,
        "inputs": {
            "cleaned_records": {
                "path": str(cleaned),
                "sha256": file_sha256(cleaned),
            },
            "assay_transfer_measurement_policy": {
                "path": str(policy_path),
                "sha256": file_sha256(policy_path),
            },
        },
        "output": {"path": str(records), "sha256": file_sha256(records)},
    }
    (records.parent / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    policy = SimpleNamespace(
        record_contract=object(), assay_transfer_measurement_policy=policy_path
    )

    assert _verify_stage_artifact(tmp_path, "normalize", policy) == records
    policy_path.write_text('{"version":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="assay-transfer policy hash mismatch"):
        _verify_stage_artifact(tmp_path, "normalize", policy)


def _args(*, legacy: bool) -> SimpleNamespace:
    return SimpleNamespace(
        allow_missing_auxiliary_mapping=False,
        benchmark_split_root="unused-historical-splits",
        cache_mode="off",
        from_stage="pair-buckets",
        legacy_task_local_downstream=legacy,
        max_record_examples=6,
        out_dir="canonical-root",
        progress_every=0,
        through_stage="pair-buckets",
        validation_level="strict",
        workers=1,
    )


@pytest.mark.parametrize("module", BUILDERS)
def test_normalized_v7_builder_uses_active_canonical_builder(
    monkeypatch, module
):
    calls: list[str] = []
    monkeypatch.setattr(module, "parse_args", lambda *_args: _args_factory(False))
    monkeypatch.setattr(
        module,
        "build_canonical_artifacts",
        lambda **_kwargs: calls.append("canonical"),
    )
    assert module.main([]) == 0
    assert calls == ["canonical"]


@pytest.mark.parametrize("module", BUILDERS)
def test_normalized_v7_builder_does_not_route_through_legacy_views(
    monkeypatch, module
):
    calls: list[str] = []
    monkeypatch.setattr(module, "parse_args", lambda *_args: _args_factory(True))
    monkeypatch.setattr(
        module,
        "build_canonical_artifacts",
        lambda **_kwargs: calls.append("canonical"),
    )
    assert module.main([]) == 0
    assert calls == ["canonical"]


def _args_factory(legacy: bool) -> SimpleNamespace:
    return _args(legacy=legacy)


def test_canonical_publication_retires_task_local_lineage_views(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    for stage in (
        *split_downstream.CANONICAL_V7_STAGES,
        split_downstream.DISTANCE_CALIBRATION_STAGE,
        *split_downstream.LINEAGE_VIEW_STAGES,
    ):
        directory = root / stage
        directory.mkdir()
        (directory / "marker").write_text("old", encoding="utf-8")
    (root / "manifest.json").write_text('{"state":"old"}\n', encoding="utf-8")

    candidate = root / ".candidate"
    for stage in split_downstream.CANONICAL_V7_STAGES:
        directory = candidate / stage
        directory.mkdir(parents=True)
        (directory / "marker").write_text("new", encoding="utf-8")
    (candidate / "manifest.json").write_text(
        '{"artifact_scope":"canonical_split_independent"}\n',
        encoding="utf-8",
    )

    split_downstream._publish_canonical_candidate(root, candidate)

    for stage in split_downstream.CANONICAL_V7_STAGES:
        assert (root / stage / "marker").read_text(encoding="utf-8") == "new"
    assert all(
        not (root / stage).exists()
        for stage in split_downstream.LINEAGE_VIEW_STAGES
    )
    assert not (root / split_downstream.DISTANCE_CALIBRATION_STAGE).exists()
    assert not list(root.glob(".canonical-backup-*"))


def test_active_canonical_builder_publishes_only_stage_3(tmp_path):
    records_dir = tmp_path / "02_canonicalized"
    records_dir.mkdir()
    pd.DataFrame(
        [
            {
                "canonical_record_id": "record-1",
                "source_id": "source-1",
                "canonical_smiles": "CCO",
            }
        ]
    ).to_parquet(
        records_dir / "records.parquet", index=False
    )
    (records_dir / "auxiliary_mapping_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    for stage in split_downstream.LINEAGE_VIEW_STAGES:
        (tmp_path / stage).mkdir()

    def build_sidecar(*, out_dir, **_kwargs):
        output = out_dir / "pair_bucket_records.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "canonical_record_id": "record-1",
                    "source_id": "source-1",
                    "pair_bucket_key": '["source-1","endpoint","unit"]',
                    "canonical_pair_fields_json": "{}",
                    "assay_transfer_eligible": True,
                    "assay_transfer_ineligibility_reason": None,
                }
            ]
        ).to_parquet(
            output, index=False
        )
        return {
            "contract_version": "pair.v1",
            "stats": {"input_records": 1, "buckets": 1},
        }

    def build_transfer_policy(*, out_dir, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pair_bucket_distance_calibration.json.gz").write_bytes(
            gzip.compress(json.dumps({"entries": {}}).encode(), mtime=0)
        )
        return {
            "pair_bucket_version": "pair.v1",
            "inputs": {"records": {"path": str(out_dir / "records.parquet")}},
            "summary": {"calibration_valid_buckets": 1},
        }

    policy = SimpleNamespace(
        scientific_assets=(),
        manifest_versions=lambda: {"record_contract_version": "records.v7"},
    )
    spec = SimpleNamespace(
        task_id="test_task",
        policy=policy,
        pipeline_layout_version="test.layout.v1",
        pair_bucket_version="pair.v1",
        pair_bucket_records_filename="pair_bucket_records.parquet",
        pair_bucket_metadata_filename="pair_bucket_metadata.json",
        build_sidecar=build_sidecar,
        build_transfer_policy=build_transfer_policy,
        direct_mapping_builder=None,
    )

    manifest = split_downstream.build_canonical_artifacts(
        spec, normalized_root=tmp_path, cache_mode="off"
    )

    assert manifest["artifact_scope"] == "canonical_split_independent"
    assert manifest["completed_artifact_stages"][-1] == "03_pair_buckets"
    assert manifest["canonical_artifact_hashes"]
    assert (tmp_path / "03_pair_buckets").is_dir()
    with gzip.open(
        tmp_path / "03_pair_buckets/pair_bucket_distance_calibration.json.gz",
        "rt",
    ) as handle:
        calibration = json.load(handle)
    assert calibration["inputs"]["records"]["path"] == str(
        tmp_path / "03_pair_buckets/records.parquet"
    )
    assert not (tmp_path / "06_collapsed_records").exists()
    assert all(
        (tmp_path / stage).exists()
        for stage in split_downstream.LINEAGE_VIEW_STAGES
    )
