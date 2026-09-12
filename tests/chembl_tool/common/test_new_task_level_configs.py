"""New task families must survive catalog aggregation and shared prompt rendering."""

import importlib

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
