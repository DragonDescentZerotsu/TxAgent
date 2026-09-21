import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_source_rows,
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import (
    NormalizedSourceProfile,
)
from data.processing.evidence_library.import_main_source_universe import _attach_uids
from data.processing.evidence_library.build_release_level_mapping import (
    build_complete,
    build_subset,
)
from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    _apply_gold_v1_voter_protection,
    _load_source_universe,
    _load_uid_smiles_universe,
    _overlay_authoritative_smiles,
    _scientific_assets,
    _stage_output_filenames,
    load_task_policy,
)
from data.processing.evidence_library.versions.v10 import build_normalized_evidence_library as builder
from data.processing.evidence_library.versions.v10.stage1_exact_deduplication import (
    PROTECTION_VERSION,
    load_voter_protection,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.starling_measurement_resolution import (
    MAPPING_VERSION as BBB_MAPPING_VERSION,
    validate_mapping_provenance as validate_bbb_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_measurement_resolution import (
    MAPPING_VERSION as ORAL_MAPPING_VERSION,
    validate_mapping_provenance as validate_oral_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_measurement_resolution import (
    MAPPING_VERSION as SKIN_MAPPING_VERSION,
    validate_mapping_provenance as validate_skin_mapping,
)


@pytest.mark.parametrize("task", ["bbb_martins", "bioavailability_ma"])
def test_stage1_has_no_level_mapping_output(task):
    policy = load_task_policy(task)
    assert not any(
        "level_mapping" in path
        for path in _stage_output_filenames(policy)["clean"]
    )


@pytest.mark.parametrize(
    "task", ["bbb_martins", "bioavailability_ma", "ames", "dili", "carcinogens"]
)
def test_stage1_scientific_assets_have_no_level_mapping_dependency(task):
    policy = load_task_policy(task)
    args = type("Args", (), {"unit_mapping": None})()
    assert not any("source_uid_levels" in str(path) for path in _scientific_assets(policy, args))


@pytest.mark.parametrize(
    "task", ["skin_reaction", "ames", "carcinogens", "dili"]
)
def test_main_uid_smiles_universe_is_separate_from_downstream_levels(task):
    policy = load_task_policy(task)
    assert policy.source_universe_mapping is not None
    assert "gold_labels" in str(policy.source_universe_mapping)
    expected_version = "v2" if task in {"ames", "carcinogens", "dili"} else "v1"
    assert f"level_mappings/{expected_version}/level_mapping" in str(
        policy.source_universe_mapping
    )
    if task in {"ames", "carcinogens", "dili"}:
        assert policy.source_uid_universe_records is not None
        assert "main_universe_v1" in str(policy.source_uid_universe_records)
        assert policy.source_uid_universe_manifest is not None
    assert not any(
        "level_mapping" in path
        for path in _stage_output_filenames(policy)["clean"]
    )


def test_main_uid_smiles_loader_pins_only_membership_and_structure(tmp_path):
    task_dir = tmp_path / "fixture"
    task_dir.mkdir()
    records = task_dir / "records.parquet"
    rows = [
        {
            "source_row_uid": "sr_00000000000000000000000000000001",
            "canonical_smiles": "CCO",
            "canonical_assay_context": "must not be imported",
        },
        {
            "source_row_uid": "sr_00000000000000000000000000000002",
            "canonical_smiles": "",
            "canonical_assay_context": "must not be imported either",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), records)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "main_source_universe.v1",
                "upstream_commit": "fixture-commit",
                "tasks": {
                    "fixture": {
                        "path": "fixture/records.parquet",
                        "rows": 2,
                        "sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    policy = replace(
        load_task_policy("carcinogens"),
        task_id="fixture",
        source_uid_universe_records=records,
        source_uid_universe_manifest=manifest_path,
    )
    inputs = {}

    uids, structures, receipt = _load_uid_smiles_universe(policy, inputs)

    assert uids == {row["source_row_uid"] for row in rows}
    assert structures[rows[0]["source_row_uid"]] == "CCO"
    assert structures[rows[1]["source_row_uid"]] is None
    assert receipt["authoritative_fields"] == ["source_row_uid", "canonical_smiles"]
    assert receipt["local_source_fields_authoritative"] is True
    assert receipt["missing_canonical_smiles"] == 1
    assert set(inputs) == {
        "source_uid_universe_manifest",
        "source_uid_universe_records",
    }


def test_main_smiles_overlay_preserves_local_scientific_values():
    uid = "sr_00000000000000000000000000000001"
    frame = pd.DataFrame(
        [
            {
                "source_row_uid": uid,
                "SMILES": "bad-local-structure",
                "assay_method": "local assay",
                "result_value": "local result",
            }
        ]
    )
    profile = NormalizedSourceProfile(
        source_id="fixture",
        source_name="fixture",
        smiles_field="SMILES",
        structure_mode="direct",
    )

    overlaid = _overlay_authoritative_smiles(frame, profile, {uid: "CCO"})

    assert overlaid.to_dict("records") == [
        {
            "source_row_uid": uid,
            "SMILES": "CCO",
            "assay_method": "local assay",
            "result_value": "local result",
        }
    ]
    assert frame.loc[0, "SMILES"] == "bad-local-structure"


def test_source_universe_loader_validates_and_loads_uid_membership(tmp_path):
    mapping = tmp_path / "fixture"
    mapping.mkdir()
    part = mapping / "part-00000.parquet"
    rows = [
        {
            "source_row_uid": f"sr_{index:032x}",
            "canonical_record_id": f"record-{index}",
            "source_group_id": "group",
            "family_key": "family",
            "level": f"L{index}",
        }
        for index in (1, 2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), part)
    manifest = {
        "version": "source_uid_levels.v3",
        "tasks": {
            "fixture": {
                "path": mapping.name,
                "parts": [
                    {
                        "path": part.name,
                        "size_bytes": part.stat().st_size,
                        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                    }
                ],
                "mapped_rows": 2,
                "unique_uids": 2,
                "upstream_commit": "fixture-commit",
                "rows_by_level": {"L1": 1, "L2": 1},
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    policy = replace(
        load_task_policy("carcinogens"),
        task_id="fixture",
        source_universe_mapping=mapping,
    )
    inputs = {}

    uids, receipt = _load_source_universe(policy, inputs)

    assert uids == {row["source_row_uid"] for row in rows}
    assert receipt["levels_attached_to_records"] is False
    assert receipt["mapped_rows"] == 2
    assert inputs["source_universe_part_00000"] == part


def test_source_universe_loader_accepts_gold_owned_mapping_manifest(tmp_path):
    mapping = tmp_path / "level_mapping"
    mapping.mkdir()
    part = mapping / "part-00000.parquet"
    row = {
        "source_row_uid": "sr_00000000000000000000000000000001",
        "canonical_record_id": "record-1",
        "source_group_id": "group",
        "family_key": "family",
        "level": "L1",
    }
    pq.write_table(pa.Table.from_pylist([row]), part)
    manifest = {
        "version": "gold_original_level_mapping.v1",
        "task": "fixture",
        "upstream": {"upstream_commit": "fixture-commit"},
        "outputs": {
            "level_mapping": {
                "path": mapping.name,
                "rows": 1,
                "rows_by_level": {"L1": 1},
                "parts": [
                    {
                        "path": part.name,
                        "size_bytes": part.stat().st_size,
                        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                    }
                ],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    policy = replace(
        load_task_policy("carcinogens"),
        task_id="fixture",
        source_universe_mapping=mapping,
    )

    uids, receipt = _load_source_universe(policy, {})

    assert uids == {row["source_row_uid"]}
    assert receipt["upstream_commit"] == "fixture-commit"


def test_filtered_source_rows_preserve_their_original_row_number():
    profile = NormalizedSourceProfile(
        source_id="fixture",
        source_name="fixture",
        endpoint_constant="endpoint",
    )
    record = clean_source_rows(
        [
            {
                "source_row_uid": "sr_00000000000000000000000000000001",
                "_source_row_number": 42,
            }
        ],
        profile,
        smiles_mapping=None,
    )[0]

    assert record["source_row_number"] == 42
    assert "_source_row_number" not in json.loads(record["source_payload_json"])


def test_main_source_import_keeps_full_universe_without_reordering(tmp_path, monkeypatch):
    source = tmp_path / "records.parquet"
    output = tmp_path / "imported.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_id": "source", "source_row_number": 1, "source_record_id": "one", "canonical_record_id": "record-1", "value": "first"},
                {"source_id": "source", "source_row_number": 2, "source_record_id": "two", "canonical_record_id": "record-2", "value": "second"},
            ]
        ),
        source,
    )
    mapping = {
        ("source", 1, "one"): "sr_00000000000000000000000000000001",
        ("source", 2, "two"): "sr_00000000000000000000000000000002",
    }
    monkeypatch.setattr(
        "data.processing.evidence_library.import_main_source_universe._uid_mapping",
        lambda task: (mapping, {}),
    )

    assert _attach_uids("skin_reaction", source, tmp_path / "duplicates.parquet", output) == (2, 0, 0)
    assert pq.read_table(output).to_pylist() == [
        {
            "source_id": "source",
            "source_row_number": 1,
            "source_record_id": "one",
            "canonical_record_id": "record-1",
            "value": "first",
            "source_row_uid": "sr_00000000000000000000000000000001",
        },
        {
            "source_id": "source",
            "source_row_number": 2,
            "source_record_id": "two",
            "canonical_record_id": "record-2",
            "value": "second",
            "source_row_uid": "sr_00000000000000000000000000000002",
        },
    ]


def test_voter_protection_does_not_overwrite_authoritative_source_values(
    tmp_path, monkeypatch
):
    contract = tmp_path / "voters.parquet"
    contract.write_bytes(b"fixture")
    uid = "sr_00000000000000000000000000000001"
    monkeypatch.setattr(
        builder,
        "load_voter_protection",
        lambda _args, _task: ({uid: {"canonical_smiles": "N"}}, contract),
    )
    record = {
        "source_row_uid": uid,
        "canonical_smiles": "CCO",
        "structure_status": "resolved",
    }
    state = SimpleNamespace(
        args=SimpleNamespace(),
        policy=SimpleNamespace(task_id="fixture"),
        cleaned=[record],
        source_inventory={"authoritative_source_seed": {"version": "fixture"}},
        source_value_cleaning_manifest={},
        cleaning_inputs={},
    )

    _apply_gold_v1_voter_protection(state)

    assert record["canonical_smiles"] == "CCO"
    assert state.source_value_cleaning_manifest["gold_v1_voter_protection"][
        "source_values_mutated"
    ] is False


def test_voter_protection_loader_requires_membership_not_structure(tmp_path):
    uid = "sr_00000000000000000000000000000001"
    table = pa.Table.from_pylist([{"task": "fixture", "source_row_uid": uid}])
    table = table.replace_schema_metadata(
        {b"schema_version": PROTECTION_VERSION.encode("utf-8")}
    )
    contract = tmp_path / "voters.parquet"
    pq.write_table(table, contract)

    protected, path = load_voter_protection(
        SimpleNamespace(stage1_protected_voter_contract=str(contract)), "fixture"
    )

    assert protected == {uid: {"task": "fixture", "source_row_uid": uid}}
    assert path == contract


def test_skin_stage1_protection_is_exactly_gold_v1_l1():
    contract = (
        "data/gold_labels/Skin_Reaction/level_mappings/v1/"
        "stage1_voter_protection.parquet"
    )
    protected, _ = load_voter_protection(
        SimpleNamespace(stage1_protected_voter_contract=contract), "skin_reaction"
    )
    mapping = pq.read_table(
        "data/gold_labels/Skin_Reaction/level_mappings/v1/level_mapping",
        columns=["source_row_uid", "level"],
    ).to_pylist()
    expected = {str(row["source_row_uid"]) for row in mapping if row["level"] == 1}
    assert set(protected) == expected
    assert len(protected) == 42_435


def test_complete_level_mapping_publishes_record_eligibility_sidecar(tmp_path):
    uids = [
        "sr_00000000000000000000000000000001",
        "sr_00000000000000000000000000000002",
    ]
    records = tmp_path / "records.parquet"
    prior = tmp_path / "prior"
    voters = tmp_path / "voters.parquet"
    output = tmp_path / "level_mapping"
    published = tmp_path / "published/level_mapping"
    archived_prior = tmp_path / "archive/records.parquet"
    prior.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": uids[0],
                    "canonical_record_id": "record-1",
                    "pair_bucket_key": '["direct"]',
                    "measurement_kind": "ordinal",
                    "assay_transfer_eligible": True,
                    "assay_transfer_ineligibility_reason": None,
                },
                {
                    "source_row_uid": uids[1],
                    "canonical_record_id": "record-2",
                    "pair_bucket_key": '["exposure"]',
                    "measurement_kind": "continuous",
                    "assay_transfer_eligible": False,
                    "assay_transfer_ineligibility_reason": "reference_scope_comparator_relative",
                },
            ]
        ),
        records,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": uid,
                    "canonical_record_id": f"old-{index}",
                    "source_group_id": group,
                    "family_key": family,
                    "level": level,
                }
                for index, (uid, group, family, level) in enumerate(
                    [
                        (uids[0], "Observed.direct_oral_bioavailability", "direct_oral_bioavailability", 1),
                        (uids[1], "Observed.nondirect_oral_bioavailability", "nondirect_oral_bioavailability", 2),
                    ],
                    start=1,
                )
            ]
        ),
        prior / "part-00000.parquet",
    )
    pq.write_table(pa.Table.from_pylist([{"source_row_uid": uids[0]}]), voters)

    build_complete(
        "bioavailability_ma",
        records,
        prior,
        voters,
        output,
        "v10",
        None,
        published,
        archived_prior,
    )

    sidecar = pq.read_table(
        output / "assay_transfer_record_eligibility.parquet"
    ).to_pylist()
    assert [row["level"] for row in sidecar] == [1, 2]
    assert [row["assay_transfer_eligible"] for row in sidecar] == [True, False]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["assay_transfer_record_eligibility"]["rows"] == 2
    assert manifest["output"]["path"] == str(published / "records.parquet")
    assert manifest["inputs"]["prior_mapping"]["path"] == str(archived_prior)


def test_subset_level_mapping_resolves_stage1_duplicate_lineage_and_reviewed_drop(
    tmp_path,
):
    retained = "sr_00000000000000000000000000000001"
    duplicate = "sr_00000000000000000000000000000002"
    reviewed_drop = "sr_00000000000000000000000000000003"
    records = tmp_path / "records.parquet"
    prior = tmp_path / "prior.parquet"
    audit = tmp_path / "audit.parquet"
    output = tmp_path / "levels"
    pq.write_table(
        pa.Table.from_pylist(
            [{
                "source_row_uid": retained,
                "canonical_record_id": "record-1",
                "pair_bucket_key": '["skin"]',
                "measurement_kind": "continuous",
                "assay_transfer_eligible": True,
                "assay_transfer_ineligibility_reason": None,
            }]
        ),
        records,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_row_uid": duplicate, "level": 2},
                {"source_row_uid": reviewed_drop, "level": 3},
            ]
        ),
        prior,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": duplicate,
                    "field": "record",
                    "after": "dropped",
                    "cleaning_version": "skin_reaction_stage1_exact_deduplication.v1",
                    "rule_id": "skin_reaction_stage1_exact_deduplication.v1",
                    "retained_source_row_uid": retained,
                },
                {
                    "source_row_uid": reviewed_drop,
                    "field": "record",
                    "after": "dropped",
                    "cleaning_version": "starling_source_value_cleaning.v7",
                    "rule_id": "reviewed_drop:fixture",
                    "retained_source_row_uid": None,
                },
            ]
        ),
        audit,
    )

    build_subset("skin_reaction", records, prior, output, audit)

    assert pq.read_table(output / "records.parquet").to_pylist() == [
        {"source_row_uid": retained, "canonical_record_id": "record-1", "level": 2}
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["version"] == "gold_owned_level_mapping.complete_stage3_v4"
    assert manifest["stage1_lineage_reconciliation"] == {
        "policy": (
            "stage3_or_stage1_exact_duplicate_to_retained_uid_or_reviewed_source_drop"
        ),
        "missing_gold_mapping_uids": 2,
        "stage3_exact_duplicate_uids_remapped": 0,
        "stage1_exact_duplicate_uids_remapped": 1,
        "unique_retained_uid_targets": 1,
        "reviewed_source_row_drops": 1,
    }
    assert pq.read_table(output / "assay_transfer_record_eligibility.parquet").to_pylist() == [
        {
            "source_row_uid": retained,
            "canonical_record_id": "record-1",
            "pair_bucket_key": '["skin"]',
            "measurement_kind": "continuous",
            "assay_transfer_eligible": True,
            "assay_transfer_ineligibility_reason": None,
            "level": 2,
        }
    ]


def test_subset_level_mapping_accepts_reviewed_endpoint_exclusion_audit(tmp_path):
    retained = "sr_00000000000000000000000000000001"
    excluded = "sr_00000000000000000000000000000002"
    records = tmp_path / "records.parquet"
    prior = tmp_path / "prior.parquet"
    audit = tmp_path / "audit.parquet"
    output = tmp_path / "levels"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": retained,
                    "canonical_record_id": "record-1",
                    "pair_bucket_key": '["ames"]',
                    "measurement_kind": "binary",
                    "assay_transfer_eligible": False,
                    "assay_transfer_ineligibility_reason": (
                        "missing_controlled_categorical_scale"
                    ),
                }
            ]
        ),
        records,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_row_uid": retained, "level": 1},
                {"source_row_uid": excluded, "level": 2},
            ]
        ),
        prior,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": excluded,
                    "field": "record",
                    "after": "dropped",
                    "action": "dropped",
                    "audit_type": "endpoint_excluded",
                }
            ]
        ),
        audit,
    )

    build_subset("ames", records, prior, output, audit)

    assert pq.read_table(output / "records.parquet").num_rows == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stage1_lineage_reconciliation"][
        "reviewed_source_row_drops"
    ] == 1


def test_subset_level_mapping_uses_stage3_lineage_and_level_zero(tmp_path):
    retained = "sr_00000000000000000000000000000001"
    discarded = "sr_00000000000000000000000000000002"
    unreviewed = "sr_00000000000000000000000000000003"
    records = tmp_path / "records.parquet"
    prior = tmp_path / "prior.parquet"
    duplicates = tmp_path / "duplicates.parquet"
    output = tmp_path / "levels"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": uid,
                    "canonical_record_id": record_id,
                    "pair_bucket_key": '["ames"]',
                    "measurement_kind": "continuous",
                    "assay_transfer_eligible": True,
                    "assay_transfer_ineligibility_reason": None,
                }
                for uid, record_id in (
                    (retained, "record-1"),
                    (unreviewed, "record-3"),
                )
            ]
        ),
        records,
    )
    pq.write_table(
        pa.Table.from_pylist([{"source_row_uid": discarded, "level": 4}]),
        prior,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": discarded,
                    "retained_source_row_uid": retained,
                }
            ]
        ),
        duplicates,
    )

    build_subset(
        "ames",
        records,
        prior,
        output,
        stage3_duplicate_lineage=duplicates,
    )

    assert pq.read_table(output / "records.parquet").to_pylist() == [
        {"source_row_uid": retained, "canonical_record_id": "record-1", "level": 4},
        {"source_row_uid": unreviewed, "canonical_record_id": "record-3", "level": 0},
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["unmapped_stage3_rows"] == 0
    assert manifest["level_zero_stage3_rows"] == 1
    assert manifest["stage1_lineage_reconciliation"][
        "stage3_exact_duplicate_uids_remapped"
    ] == 1


def test_subset_level_mapping_validates_exact_current_l1_without_rewriting(tmp_path):
    former_voter = "sr_00000000000000000000000000000001"
    current_voter = "sr_00000000000000000000000000000002"
    other = "sr_00000000000000000000000000000003"
    records = tmp_path / "records.parquet"
    prior = tmp_path / "prior.parquet"
    voters = tmp_path / "voters.parquet"
    output = tmp_path / "levels"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_row_uid": uid,
                    "canonical_record_id": f"record-{index}",
                    "pair_bucket_key": '["ames"]',
                    "measurement_kind": "continuous",
                    "assay_transfer_eligible": True,
                    "assay_transfer_ineligibility_reason": None,
                }
                for index, uid in enumerate(
                    (former_voter, current_voter, other), start=1
                )
            ]
        ),
        records,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_row_uid": former_voter, "level": 2},
                {"source_row_uid": current_voter, "level": 1},
                {"source_row_uid": other, "level": 4},
            ]
        ),
        prior,
    )
    pq.write_table(
        pa.Table.from_pylist([{"source_row_uid": current_voter}]), voters
    )

    build_subset(
        "ames",
        records,
        prior,
        output,
        current_voter_membership=voters,
    )

    levels = {
        row["source_row_uid"]: row["level"]
        for row in pq.read_table(output / "records.parquet").to_pylist()
    }
    assert levels == {former_voter: 2, current_voter: 1, other: 4}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["current_l1_validation"] == {
        "policy": "reviewed_levels_preserved; voter_membership_is_validation_only",
        "physical_voters": 1,
        "l1_equals_current_physical_voters": True,
    }
    assert manifest["validations"]["l1_equals_current_physical_voters"] is True


def test_subset_level_mapping_rejects_voter_level_mismatch(tmp_path):
    voter = "sr_00000000000000000000000000000001"
    records = tmp_path / "records.parquet"
    prior = tmp_path / "prior.parquet"
    voters = tmp_path / "voters.parquet"
    output = tmp_path / "levels"
    pq.write_table(
        pa.Table.from_pylist(
            [{
                "source_row_uid": voter,
                "canonical_record_id": "record-1",
                "pair_bucket_key": '["ames"]',
                "measurement_kind": "continuous",
                "assay_transfer_eligible": True,
                "assay_transfer_ineligibility_reason": None,
            }]
        ),
        records,
    )
    pq.write_table(pa.Table.from_pylist([{"source_row_uid": voter, "level": 2}]), prior)
    pq.write_table(pa.Table.from_pylist([{"source_row_uid": voter}]), voters)

    with pytest.raises(ValueError, match="differs from current voter membership"):
        build_subset(
            "ames",
            records,
            prior,
            output,
            current_voter_membership=voters,
        )


@pytest.mark.parametrize(
    ("task", "version", "validator"),
    [
        ("bbb_martins", BBB_MAPPING_VERSION, validate_bbb_mapping),
        ("bioavailability_ma", ORAL_MAPPING_VERSION, validate_oral_mapping),
        ("skin_reaction", SKIN_MAPPING_VERSION, validate_skin_mapping),
    ],
)
def test_measurement_replay_requires_hash_pinned_sources(
    tmp_path, task, version, validator
):
    mapping = tmp_path / "measurement_resolution.parquet"
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"cleaned_record_id": "record", "status": "relative", "measurements_json": None}]
        ),
        mapping,
    )
    source.write_bytes(b"frozen source")
    manifest = {
        "generation_version": "stage1_measurement_resolution_replay.v1",
        "task_id": task,
        "mapping_version": version,
        "mapping_sha256": file_sha256(mapping),
        "mapping_rows": 1,
        "record_ids_sha256": hashlib.sha256(b"record").hexdigest(),
        "sources": {"frozen": {"path": str(source), "sha256": file_sha256(source)}},
        "validations": {"unique": True, "no_new_llm_requests": True},
    }
    mapping.with_suffix(".manifest.json").write_text(json.dumps(manifest))

    validator(mapping)
    source.write_bytes(b"drift")
    with pytest.raises(ValueError, match="provenance mismatch"):
        validator(mapping)
