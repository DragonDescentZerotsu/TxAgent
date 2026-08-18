import json
from pathlib import Path

from tools.chembl_tool.tasks.clintox.clintox_base import (
    DIRECT_GROUP,
    clean_text,
)
from tools.chembl_tool.tasks.clintox.clintox_base_artifact_store import (
    ARTIFACT_STAGES,
    PROFILE,
)
from tools.chembl_tool.tasks.clintox.experiment_config import get_source_config


def test_basic_cleaning_is_null_and_whitespace_only():
    for value in (None, "", "  ", "NaN", "none", "NULL", "n/a", "NA"):
        assert clean_text(value) is None
    assert clean_text(" unspecified ") == "unspecified"
    assert clean_text(" - ") == "-"
    assert clean_text("a\n  b") == "a b"


def test_clintox_base_is_explicit_direct_only_source():
    config = get_source_config("clintox_base")
    assert config.source_name == "clintox_base"
    assert [group.group_id for group in config.direct_groups] == [DIRECT_GROUP]
    assert [group.group_id for group in config.mechanism_groups] == [DIRECT_GROUP]


def test_artifact_profile_covers_all_stages():
    assert PROFILE.task_id == "clintox"
    assert PROFILE.stages == ARTIFACT_STAGES
    assert ARTIFACT_STAGES[0] == "01_cleaned"
    assert ARTIFACT_STAGES[-1] == "06_record_supported_v2_scaffold_view"


def test_tracked_source_manifest_discloses_missing_qualifier():
    root = Path("data/starling_data/clintox/send_v2")
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    source = next(
        item for item in manifest["sources"]
        if item["source_id"] == "human_clinical_toxicity"
    )
    assert source["rows"] == 584_307
    assert "qualifying_conditions" not in source["columns"]
    assert manifest["qualifying_conditions"]["human_clinical_toxicity"] == (
        "unavailable_in_source_schema"
    )
