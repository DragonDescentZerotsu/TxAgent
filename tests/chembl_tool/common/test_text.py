from tools.chembl_tool.common.text import contains_phrase, normalize_text


def test_normalize_common_bbb_spellings():
    assert normalize_text("blood-brain barrier") == "blood brain barrier"
    assert normalize_text("K(p,uu,brain)") == "k p uu brain"
    assert normalize_text("CSF/plasma") == "csf plasma"


def test_contains_phrase_is_token_bounded():
    assert contains_phrase("P-glycoprotein efflux", "p glycoprotein")
    assert not contains_phrase("scaffold", "aff")
