from tools.chembl_tool.tasks.bbb_martins.experiment_config import (
    CHEMBL,
    STARLING,
    STARLING_V6,
    get_source_config,
)


def test_v6_source_reuses_family_views_but_has_distinct_provenance():
    assert STARLING_V6.source_name == "starling_v6"
    assert STARLING_V6.direct_groups == STARLING.direct_groups
    assert STARLING_V6.mechanism_groups == STARLING.mechanism_groups
    assert get_source_config("starling_v6") is STARLING_V6


def test_legacy_source_configs_are_unchanged():
    assert CHEMBL.source_name == "chembl"
    assert STARLING.source_name == "starling"
