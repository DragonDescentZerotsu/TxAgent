"""New task families must survive catalog aggregation and shared prompt rendering."""

import importlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.progressive_assay_reasoning import build_progressive_messages
from tools.chembl_tool.paper_experiments.build_assay_family_catalog import (
    DEFAULT_OUTPUT_ROOT,
    TASKS,
    build_catalog,
)


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
def test_catalog_retains_each_record_family_within_one_physical_assay(task, tmp_path):
    config = importlib.import_module(TASKS[task]["config_module"])
    families = config.STARLING.mechanism_groups
    assert len(families) == 7
    assert config.STARLING.direct_groups == families[:1]
    assert TASKS[task]["output_root"] == str(DEFAULT_OUTPUT_ROOT)
    assert TASKS[task]["output_name"] == task
    from tools.chembl_tool.common.starling.current_retrieval_artifacts import current_records_path
    assert TASKS[task]["records"] == str(current_records_path(task))
    path = tmp_path / "records.parquet"
    rows = [
        {
            "group_id": group,
            "canonical_assay_context": "one shared physical study assay",
            "canonical_endpoint_name": family.family_key,
            "retrieval_eligible": True,
        }
        for family in families
        for group in family.source_group_ids
    ]
    # An excluded source record must not create another family or assay.
    rows.append({**rows[0], "group_id": "Group.excluded", "retrieval_eligible": False})
    pq.write_table(pa.Table.from_pylist(rows), path)
    catalog, manifest = build_catalog(task, path)
    assert len(catalog) == 1
    assert catalog[0]["first_level"] == 1
    assert catalog[0]["family_levels"] == list(range(1, 8))
    assert catalog[0]["record_count"] == 7
    assert {
        row["source_group_id"]: (row["level"], row["endpoint_group"])
        for row in catalog[0]["source_families"]
    } == {
        f"Group.{family.family_key}": (level, family.family_key)
        for level, family in enumerate(families, 1)
    }
    assert manifest["family_assignment_unit"] == "source_record"


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
def test_contract_renders_shared_progressive_prompt_without_external_condition(task):
    config = importlib.import_module(TASKS[task]["config_module"])
    contract = config.get_progressive_task_contract()
    assert contract.task == task
    assert len(contract.prediction_values) == 2
    assert set(config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS) == set(range(1, 8))
    levels = [
        {
            "level": level,
            "endpoint_group": family.family_key,
            "description": config.PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[level],
        }
        for level, family in enumerate(config.STARLING.mechanism_groups, 1)
    ]
    messages = build_progressive_messages(
        contract=contract,
        levels=levels,
        current_level=1,
        query_smiles="CCO",
        condition_sentence="",
        query_prior={},
        query_tool_summary=None,
        active={},
        prior_state=None,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert contract.prediction_field in rendered
    assert all(family.family_key in rendered for family in config.STARLING.mechanism_groups)
    assert "no_reported_external_condition" not in rendered


def _validation_shard(tmp_path, monkeypatch, *, review=None, change_raw=False):
    from tools.chembl_tool.common.starling import validate_progressive_sources as validator
    task = "carcinogens"
    family = "epigenetic_intercellular_growth_restraint"
    row = {
        "source_row_uid": "uid1", "canonical_record_id": "uid1",
        "source_id": "carcinogens_v4", "molecule_id": "mol1",
        "source_smiles": "CCO", "molecule_name": "ethanol",
        "molecule_identity_key": "parent1", "group_id": "Source.carcinogens_v4",
        "retrieval_eligible": True, "is_gold_voter": False,
        "canonical_endpoint_name": "DNA methylation", "canonical_measurement_text": "increased",
        "canonical_unit_text": "", "canonical_assay_context": "hepatocytes",
        "canonical_species_context": "rat", "qualifying_conditions": "Known carcinogen exposure.",
        "support_text": "Histone methylation increased.",
        "raw_record_json": json.dumps({"support_text": "Known carcinogen exposure."}),
    }
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([row]), source)
    over = {**row, "group_id": "Group." + family, "progressive_level": 6,
            "family_key": family, "level_assignment_reason": "test_review" if review else "endpoint",
            "heldout_filter_scope": "", "direct_signal": False, "placement_reviewed": bool(review)}
    if change_raw:
        over["raw_record_json"] = "{}"
    overlay = tmp_path / "overlay.parquet"
    pq.write_table(pa.Table.from_pylist([over]), overlay)
    audit = tmp_path / "audit.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "source_row_uid": "uid1", "previous_group_id": row["group_id"], "level": 6,
        "family_key": family, "reason": over["level_assignment_reason"],
        "is_gold_voter": False, "reviewed": bool(review), "direct_signal": False,
    }]), audit)
    monkeypatch.setattr(validator, "_state", (task, [source, overlay, audit],
                       {6: family}, set(), {"uid1": review} if review else {}, True, {}))
    return validator


def test_independent_guard_checks_direct_content_in_qualifiers(tmp_path, monkeypatch):
    validator = _validation_shard(tmp_path, monkeypatch)
    result = validator._scan_group(0)
    assert result[3][0]["source_row_uid"] == "uid1"
    assert "hazard_assertion" in result[3][0]["patterns"]


def test_source_validator_rejects_raw_record_edit(tmp_path, monkeypatch):
    validator = _validation_shard(tmp_path, monkeypatch, change_raw=True)
    with pytest.raises(AssertionError, match="changed original column"):
        validator._scan_group(0)


def test_direct_guard_exemption_is_bound_to_original_payload(tmp_path, monkeypatch):
    from tools.chembl_tool.common.starling.source_gold_review import payload_hash
    review = {"level": 6, "direct_guard_exemption": True,
              "source_payload_sha256": payload_hash({"support_text": "Known carcinogen exposure."})}
    validator = _validation_shard(tmp_path, monkeypatch, review=review)
    assert validator._scan_group(0)[3] == []
    review["source_payload_sha256"] = "stale"
    with pytest.raises(AssertionError, match="stale review"):
        validator._scan_group(0)


def test_column_conservation_distinguishes_nan_from_null():
    from tools.chembl_tool.common.starling.validate_progressive_sources import columns_equal
    original = pa.chunked_array([[1.0, float("nan"), None]])
    assert columns_equal(original, pa.chunked_array([[1.0, float("nan"), None]]))
    assert not columns_equal(original, pa.chunked_array([[1.0, None, None]]))


def test_independent_guard_preserves_field_boundaries():
    from tools.chembl_tool.common.starling.validate_progressive_sources import mentions
    assert not mentions('carcinogens', ['no numeric value reported', 'NCI60 human tumor cell assay'])
    assert not mentions('carcinogens', ['apoptosis induced', 'breast cancer cell culture'])
    assert 'tumor_action' in mentions('carcinogens', ['', 'No tumors were observed in exposed rats.'])
