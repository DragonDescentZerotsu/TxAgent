from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.starling import current_retrieval_artifacts as artifacts
from tools.chembl_tool.paper_experiments import build_assay_family_catalog as catalog
from tools.chembl_tool.paper_experiments import (
    export_current_starling_level_records as share_export,
)
from tools.chembl_tool.paper_experiments import rebuild_current_starling_retrieval as rebuild
from tools.chembl_tool.tasks.bbb_martins import experiment_config as bbb_config
from tools.chembl_tool.tasks.bbb_martins.source_family_purity import DEFAULT_RECORDS
from tools.chembl_tool.tasks.bioavailability_ma.build_nondirect_assay_context import (
    DEFAULT_INPUT,
)


def test_current_record_contract_is_latest_only_and_repo_local() -> None:
    manifest = json.loads(artifacts.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "current_starling_records.v1"
    assert manifest["history"]["earlier_record_versions_retained"] is False
    assert set(manifest["tasks"]) == set(artifacts.TASKS)
    for task, info in manifest["tasks"].items():
        assert info["records_sha256"]
        assert info["parts"]
        assert all(int(part["size"]) <= 90_000_000 for part in info["parts"])
        assert all(str(part["path"]).startswith("artifacts/") for part in info["parts"])
        assert "/data1/" not in json.dumps(info)
        assert artifacts.current_records_path(task).is_relative_to(artifacts.PROJECT_ROOT)


def test_published_snapshot_restores_after_checkout_move(tmp_path, monkeypatch):
    import io
    import shutil
    import subprocess
    import tarfile

    project = tmp_path / "original"
    source = project / "data/source"
    source.mkdir(parents=True)
    (source / "records.parquet").write_bytes(b"frozen record bytes")
    manifest = project / "artifacts/chembl_tool/starling/current_records/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema_version": "current_starling_records.v1", "tasks": {}}))
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", project)
    monkeypatch.setattr(artifacts, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(artifacts, "_load_manifest", lambda path=manifest: json.loads(path.read_text()))
    monkeypatch.setattr(artifacts, "current_records_path", lambda task:
                        project / "outputs/current_records" / task / "03_records/records.parquet")
    info = artifacts.publish_snapshot("dili", source, project / "artifacts/dili/03_records")
    assert info["source_release"] == "data/source"
    assert all(part["path"].startswith("artifacts/") for part in info["parts"])
    compressed = b"".join((project / part["path"]).read_bytes() for part in info["parts"])
    unpacked = subprocess.run(
        ["zstd", "-q", "-d", "-c"], input=compressed, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(unpacked)) as archive:
        assert all(
            member.uid == member.gid == 0 and not member.uname and not member.gname
            for member in archive
        )

    moved = tmp_path / "moved"
    shutil.copytree(project / "artifacts", moved / "artifacts")
    shutil.rmtree(project)
    restored = artifacts.restore(
        "dili", local_root=moved / "outputs/current_records",
        manifest_path=moved / manifest.relative_to(project),
    )
    assert (restored / "records.parquet").read_bytes() == b"frozen record bytes"


def test_active_source_defaults_do_not_depend_on_external_checkout() -> None:
    assert DEFAULT_RECORDS == artifacts.current_records_path("bbb_martins")
    assert DEFAULT_INPUT == artifacts.current_records_path("bioavailability_ma")
    for task in ("dili", "carcinogens"):
        assert catalog.TASKS[task]["records"] == str(artifacts.current_records_path(task))
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        assert "/data1/" not in str(catalog.TASKS[task]["records"])
        assert "source_overlays" in str(catalog.TASKS[task]["records"])


def test_current_retrieval_contract_matches_task_set() -> None:
    contract = rebuild._contract()
    assert set(contract["tasks"]) == set(artifacts.TASKS)
    assert contract["canonical_records_manifest"] == str(
        artifacts.MANIFEST_PATH.relative_to(artifacts.PROJECT_ROOT)
    )
    for task, info in contract["tasks"].items():
        assert info["canonical_records_sha256"] == json.loads(
            artifacts.MANIFEST_PATH.read_text(encoding="utf-8")
        )["tasks"][task]["records_sha256"]
        assert set(info["indices"]) == {"scaffold", "random"}


def test_current_catalog_configs_are_explicit() -> None:
    contract = rebuild._contract()
    assert {
        task: info["catalog_config"] for task, info in contract["tasks"].items()
    } == {
        "bbb_martins": "STARLING_SOURCE_PURITY",
        "bioavailability_ma": "STARLING",
        "skin_reaction": "STARLING",
        "ames": "STARLING",
        "dili": "STARLING",
        "carcinogens": "STARLING",
    }


def test_bbb_progressive_level_descriptions_match_current_five_levels() -> None:
    assert set(bbb_config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS) == {1, 2, 3, 4, 5}
    assert "Near-direct" in bbb_config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[2]
    assert "Passive permeability" in bbb_config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[3]
    assert "Efflux" in bbb_config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[4]
    assert "Influx" in bbb_config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[5]


def test_current_results_registry_points_to_portable_rebuild() -> None:
    registry_path = (
        artifacts.PROJECT_ROOT
        / "tools/chembl_tool/paper_experiments/current_conditioned_results.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    build = registry["retrieval_build"]
    assert build["status"] == "current_latest_only_hash_verified"
    assert build["external_checkout_required"] is False
    assert build["canonical_records_manifest"] == str(
        artifacts.MANIFEST_PATH.relative_to(artifacts.PROJECT_ROOT)
    )
    assert rebuild._contract()["contract"]["overlay_parquet_batch_sizes"] == {
        "bbb_martins": 500,
        "bioavailability_ma": 10000,
        "skin_reaction": 20000,
    }


def test_unknown_current_record_task_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown current Starling task"):
        artifacts.current_records_path("unknown", local_root=tmp_path)


def test_ames_rebuild_preserves_both_outcome_levels_and_custom_records_root(
    tmp_path: Path, monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(rebuild, "_run", calls.append)
    records_root = tmp_path / "records"
    rebuild.build_indices(tmp_path / "indices", workers=2, records_root=records_root)
    assert len(calls) == 2 * len(artifacts.TASKS)
    for args in calls:
        task = args[args.index("--task") + 1]
        scope = args[args.index("--filter-scope-field") + 1]
        value = args[args.index("--filter-scope-value") + 1]
        if task == "ames":
            assert (scope, value) == ("heldout_filter_scope", "bacterial_outcome")
            assert Path(args[args.index("--records") + 1]) == artifacts.current_records_path(
                "ames", local_root=records_root
            )
        elif task in {"dili", "carcinogens"}:
            assert (scope, value) == (("group_id", "Group.dili_actual_voter") if task == "dili"
                                    else ("group_id", "Group.carcinogenicity_direct"))
            assert args[args.index("--identity-contract") + 1] == "new_task_tautomer_identity.v2"
            assert Path(args[args.index("--identity-cache") + 1]) == artifacts.current_records_path(task, local_root=records_root).with_name("retrieval_identity_cache.jsonl")
        else:
            assert scope == "group_id"


def test_partial_export_cannot_overwrite_complete_current_manifest() -> None:
    with pytest.raises(ValueError, match="Partial exports require"):
        share_export.export_dataset(
            artifact_root=rebuild.DEFAULT_ARTIFACT_ROOT,
            records_root=artifacts.DEFAULT_LOCAL_ROOT,
            output_dir=share_export.DEFAULT_OUTPUT_DIR,
            tasks=["ames"], splits=["scaffold", "random"],
        )


def _write_test_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level": 1,
                        "family_id": "Direct.family",
                        "endpoint_group": "direct",
                        "source_groups": ["source.direct"],
                    },
                    {
                        "level": 2,
                        "family_id": "Mechanism.family",
                        "endpoint_group": "mechanism",
                        "source_groups": ["source.mechanism"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_share_export_separates_source_membership_from_index_cards(
    tmp_path: Path,
) -> None:
    catalog_manifest = tmp_path / "catalog.json"
    _write_test_catalog(catalog_manifest)
    records = tmp_path / "records.parquet"
    source_rows = []
    for record_id, group_id, eligible in (
        ("r1", "source.direct", True),
        ("r2", "source.mechanism", True),
        ("r3", "source.direct", False),
    ):
        row = {name: "" for name in share_export._TEXT_COLUMNS}
        row.update(
            {
                "canonical_record_id": record_id,
                "group_id": group_id,
                "retrieval_eligible": eligible,
                "molecule_id": "molecule-1",
                "canonical_endpoint_name": (
                    "outcome" if group_id == "source.direct" else "transport"
                ),
                "canonical_measurement_text": (
                    "positive" if group_id == "source.direct" else "high"
                ),
                "canonical_assay_context": "example assay",
                "canonical_species_context": "human",
                "support_text": (
                    "direct support"
                    if group_id == "source.direct"
                    else "mechanism support"
                ),
            }
        )
        source_rows.append(row)
    pq.write_table(share_export.pa.Table.from_pylist(source_rows), records)
    source_output = tmp_path / "source_membership.parquet"
    source_report = share_export.export_source_membership(
        task="example",
        records_path=records,
        catalog_manifest=catalog_manifest,
        output_path=source_output,
    )
    assert source_report["n_records"] == 2
    assert source_report["records_by_level"] == {"1": 1, "2": 1}

    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "assay_chembl_id": "assay-1",
                "molecule_chembl_id": "molecule-1",
                "canonical_smiles": "CC",
                "source_record_count": 9,
                "assay_retrieval": {
                    "first_level": 1,
                    "assay_context": "example assay",
                },
                "source_record_examples": [
                    {
                        "endpoint_type": "outcome",
                        "reported_value": "positive",
                        "reported_units": "",
                        "assay_context": "example assay",
                        "species_context": "human",
                        "qualifying_conditions": "",
                        "support_text": "direct support",
                        "evidence_family": "direct",
                        "evidence_family_level": 1,
                    },
                    {
                        "endpoint_type": "transport",
                        "reported_value": "high",
                        "reported_units": "",
                        "assay_context": "example assay",
                        "species_context": "human",
                        "qualifying_conditions": "",
                        "support_text": "mechanism support",
                        "evidence_family": "mechanism",
                        "evidence_family_level": 2,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    card_output = tmp_path / "cards.parquet"
    card_report = share_export.export_indexed_cards(
        task="example",
        split_scheme="scaffold",
        evidence_path=evidence,
        catalog_manifest=catalog_manifest,
        output_path=card_output,
    )
    assert card_report["n_assay_molecule_rows"] == 1
    assert card_report["n_representative_cards"] == 2
    assert card_report["cards_by_level"] == {"1": 1, "2": 1}
    card_table = pq.read_table(card_output)
    assert "support_text" not in card_table.column_names
    source_table = pq.read_table(source_output)
    assert set(source_table.column("card_fingerprint_sha256").to_pylist()) == set(
        card_table.column("card_fingerprint_sha256").to_pylist()
    )
    assert share_export.validate_card_links(source_output, [card_output]) == {
        "n_unique_source_card_keys": 2,
        "n_indexed_cards_checked": 2,
        "n_missing_source_card_keys": 0,
    }

    # Ames assigns families during canonicalization and has no original group.
    without_original = pq.read_table(records).drop(["source_family_original_group_id"])
    pq.write_table(without_original, records)
    share_export.export_source_membership(
        task="example", records_path=records, catalog_manifest=catalog_manifest,
        output_path=source_output,
    )
    assert pq.read_table(source_output).schema == share_export.SOURCE_MEMBERSHIP_SCHEMA
    assert pq.read_table(source_output)["source_family_original_group_id"].null_count == 2
    # New source adapters expose the same review reason under their native name.
    native = without_original.rename_columns([
        "level_assignment_reason" if name == "source_family_purity_reason" else name
        for name in without_original.column_names
    ])
    pq.write_table(native, records)
    share_export.export_source_membership(
        task="example", records_path=records, catalog_manifest=catalog_manifest,
        output_path=source_output,
    )
    assert pq.read_table(source_output).schema == share_export.SOURCE_MEMBERSHIP_SCHEMA
    assert share_export.validate_card_links(source_output, [card_output])["n_missing_source_card_keys"] == 0


def test_large_source_ledger_shards_preserve_rows_schema_and_hashes(tmp_path, monkeypatch):
    table = share_export.pa.table({"id": range(131_072)})
    path = tmp_path / "source.parquet"
    pq.write_table(table, path, compression="zstd")
    original_size = path.stat().st_size
    monkeypatch.setattr(share_export, "MAX_PARQUET_BYTES", int(original_size * 0.75))
    for _ in range(2):
        pq.write_table(table, path, compression="zstd")
        report = share_export._source_file_inventory(path)
        assert not path.exists()
        assert len(report["parts"]) == 2
        assert pq.read_table(report["path"]).equals(table)
        for part in report["parts"]:
            assert part["size_bytes"] <= share_export.MAX_PARQUET_BYTES
            assert share_export.sha256_file(Path(part["path"])) == part["sha256"]


def test_selected_rebuild_does_not_launch_unrelated_tasks(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rebuild, "_run", calls.append)
    monkeypatch.setattr(rebuild, "require_current_records", lambda *a, **k: None)
    rebuild.build_overlays(tmp_path, tasks=("dili", "carcinogens"))
    assert calls == []
    rebuild.build_catalogs(tmp_path, tasks=("dili",))
    rebuild.build_indices(tmp_path, workers=2, tasks=("dili",))
    assert len(calls) == 3
    assert all("carcinogens" not in args and "ames" not in args for args in calls)


def test_restore_published_audits_preserves_bytes_and_rejects_drift(tmp_path, monkeypatch):
    import hashlib
    stage = tmp_path / 'stage'
    stage.mkdir()
    payload = b'unchanged frozen vote ledger\n'
    (stage / 'votes.jsonl').write_bytes(payload)
    info = {'files': [{'path': 'votes.jsonl', 'sha256': hashlib.sha256(payload).hexdigest()}],
            'published_file_aliases': {'data/votes.jsonl': 'votes.jsonl'}}
    monkeypatch.setattr(artifacts, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(artifacts, '_load_manifest', lambda: {'tasks': {'dili': info}})
    artifacts.restore_published_files('dili', stage)
    assert (tmp_path / 'data/votes.jsonl').read_bytes() == payload
    (tmp_path / 'data/votes.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed published audit'):
        artifacts.restore_published_files('dili', stage)
    info['published_file_aliases'] = {'../outside.jsonl': 'votes.jsonl'}
    with pytest.raises(ValueError, match='stay under data'):
        artifacts.restore_published_files('dili', stage)
