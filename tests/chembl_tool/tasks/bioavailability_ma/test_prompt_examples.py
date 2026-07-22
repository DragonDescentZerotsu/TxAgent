"""Golden-file test: committed compiled-prompt examples must match the code.

Regenerates every stage's compiled prompt from the frozen fixture and diffs it against
the committed prompt_examples/*.txt. If this fails after a prompt change, refresh with:
  python -m tools.chembl_tool.tasks.bioavailability_ma.prompt_audit.dump_prompt_examples --write
"""

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.prompt_audit.dump_prompt_examples import (
    EXAMPLES_DIR,
    load_fixture,
    render_examples,
)

_EXAMPLES = render_examples(load_fixture())


@pytest.mark.parametrize("name", sorted(_EXAMPLES))
def test_committed_example_matches_compiled_prompt(name):
    committed = (EXAMPLES_DIR / name).read_text(encoding="utf-8")
    assert committed == _EXAMPLES[name], (
        f"{name} is stale; run "
        "`python -m tools.chembl_tool.tasks.bioavailability_ma.prompt_audit.dump_prompt_examples --write`"
    )


def test_all_stage_examples_present():
    assert set(_EXAMPLES) == {
        "single.prompt.txt",
        "group.legacy.prompt.txt",
        "group.morganfingerprint.prompt.txt",
        "group.assay_transfer_tool.prompt.txt",
        "final.prompt.txt",
        "assay_transfer_scoring_pair.v6_5.prompt.txt",
    }


def test_scoring_pair_is_in_distribution():
    pair = _EXAMPLES["assay_transfer_scoring_pair.v6_5.prompt.txt"]
    assert "endpoint: q3." in pair          # real canonical key, not index.*
    assert "index." not in pair
    assert "known value:" in pair           # retrieval value shown
    assert "(A) transfer" in pair and "(B) not transfer" in pair
    # query record has value hidden (only one "known value:" line, on the retrieval side)
    assert pair.count("known value:") == 1


def test_group_text_formats_are_not_json():
    # legacy is JSON; the two new formats are human-readable text.
    for name in ("group.morganfingerprint.prompt.txt", "group.assay_transfer_tool.prompt.txt"):
        assert "NEIGHBOR ANALOGS" in _EXAMPLES[name]
        assert '"evidence_rows"' not in _EXAMPLES[name]
