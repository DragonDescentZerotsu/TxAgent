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
        assert "/data1/joseph" not in json.dumps(info)
        assert artifacts.current_records_path(task).is_relative_to(artifacts.PROJECT_ROOT)


def test_active_source_defaults_do_not_depend_on_external_checkout() -> None:
    assert DEFAULT_RECORDS == artifacts.current_records_path("bbb_martins")
    assert DEFAULT_INPUT == artifacts.current_records_path("bioavailability_ma")
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        assert "/data1/joseph" not in str(catalog.TASKS[task]["records"])
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
