import pytest

from tools.chembl_tool.common.task_workflows.retrieve_neighbors import (
    _parse_args as parse_retrieve_args,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    _parse_args as parse_bbb_args,
)
from tools.chembl_tool.tasks.dili.run_reasoning_pipeline import (
    _parse_args as parse_dili_args,
)
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline import (
    _parse_args as parse_skin_args,
)


@pytest.mark.parametrize(
    "parse_args",
    [
        parse_bbb_args,
        parse_skin_args,
        parse_dili_args,
    ],
)
def test_task_pipeline_accepts_strict_morgan_neighbor_selector(parse_args):
    args = parse_args(
        ["--morgan-neighbor-selector", "query_feature_coverage"],
    )

    assert args.morgan_neighbor_selector == "query_feature_coverage"


@pytest.mark.parametrize(
    "parse_args",
    [
        parse_bbb_args,
        parse_skin_args,
        parse_dili_args,
    ],
)
def test_task_pipeline_rejects_legacy_neighbor_selector(parse_args):
    with pytest.raises(SystemExit):
        parse_args(["--neighbor-selector", "similarity"])


def test_standalone_retrieval_uses_strict_morgan_neighbor_selector():
    args = parse_retrieve_args(
        "index.pkl",
        "test",
        ["--morgan-neighbor-selector", "query_feature_coverage"],
    )

    assert args.morgan_neighbor_selector == "query_feature_coverage"
    with pytest.raises(SystemExit):
        parse_retrieve_args(
            "index.pkl",
            "test",
            ["--neighbor-selector", "similarity"],
        )
