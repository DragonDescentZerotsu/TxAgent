from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tools.chembl_tool.common.starling import split_downstream
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


def _args(*, legacy: bool) -> SimpleNamespace:
    return SimpleNamespace(
        allow_missing_auxiliary_mapping=False,
        benchmark_split_root="unused-historical-splits",
        cache_mode="off",
        from_stage="index",
        legacy_task_local_downstream=legacy,
        max_record_examples=6,
        out_dir="canonical-root",
        progress_every=0,
        through_stage="index",
        validation_level="strict",
        workers=1,
    )


@pytest.mark.parametrize("module", BUILDERS)
def test_normalized_v7_builder_defaults_to_canonical_stage_05(
    monkeypatch, module
):
    calls: list[str] = []
    monkeypatch.setattr(module, "parse_args", lambda *_args: _args_factory(False))
    monkeypatch.setattr(
        module,
        "build_canonical_artifacts",
        lambda **_kwargs: calls.append("canonical"),
    )
    monkeypatch.setattr(
        module,
        "build_downstream_artifacts",
        lambda **_kwargs: calls.append("legacy"),
    )

    assert module.main([]) == 0
    assert calls == ["canonical"]


@pytest.mark.parametrize("module", BUILDERS)
def test_normalized_v7_builder_requires_explicit_legacy_flag_for_stage_06_09(
    monkeypatch, module
):
    calls: list[str] = []
    monkeypatch.setattr(module, "parse_args", lambda *_args: _args_factory(True))
    monkeypatch.setattr(
        module,
        "build_canonical_artifacts",
        lambda **_kwargs: calls.append("canonical"),
    )
    monkeypatch.setattr(
        module,
        "build_downstream_artifacts",
        lambda **_kwargs: calls.append("legacy"),
    )

    assert module.main([]) == 0
    assert calls == ["legacy"]


def _args_factory(legacy: bool) -> SimpleNamespace:
    return _args(legacy=legacy)


def test_canonical_publication_retires_task_local_lineage_views(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    for stage in (
        *split_downstream.CANONICAL_V7_STAGES,
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
    assert not list(root.glob(".canonical-backup-*"))


def test_canonical_builder_publishes_only_stage_04_05(tmp_path):
    records_dir = tmp_path / "03_records"
    records_dir.mkdir()
    pd.DataFrame([{"canonical_record_id": "record-1"}]).to_parquet(
        records_dir / "records.parquet", index=False
    )
    canonical_dir = tmp_path / "02_canonicalized"
    canonical_dir.mkdir()
    (canonical_dir / "auxiliary_mapping_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    for stage in split_downstream.LINEAGE_VIEW_STAGES:
        (tmp_path / stage).mkdir()

    def build_sidecar(*, out_dir, **_kwargs):
        output = out_dir / "pair_bucket_records.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"canonical_record_id": "record-1"}]).to_parquet(
            output, index=False
        )
        return {"contract_version": "pair.v1", "stats": {"buckets": 1}}

    def build_transfer_policy(*, out_dir, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "pair_bucket_version": "pair.v1",
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
        pair_bucket_records_filename="pair_bucket_records.parquet",
        pair_bucket_metadata_filename="pair_bucket_metadata.json",
        build_sidecar=build_sidecar,
        build_transfer_policy=build_transfer_policy,
    )

    manifest = split_downstream.build_canonical_artifacts(
        spec, normalized_root=tmp_path, cache_mode="off"
    )

    assert manifest["artifact_scope"] == "canonical_split_independent"
    assert manifest["completed_artifact_stages"][-2:] == [
        "04_pair_buckets",
        "05_distance_calibration",
    ]
    assert manifest["canonical_artifact_hashes"]
    assert all(
        (tmp_path / stage).is_dir()
        for stage in split_downstream.CANONICAL_V7_STAGES
    )
    assert all(
        not (tmp_path / stage).exists()
        for stage in split_downstream.LINEAGE_VIEW_STAGES
    )
