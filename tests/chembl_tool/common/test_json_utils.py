from tools.chembl_tool.common.json_utils import parse_json_content


def test_parse_json_content_never_raises_for_malformed_embedded_object():
    parsed = parse_json_content('prefix {"value": [1, 2} suffix')

    assert parsed["unparsed_text"].startswith("prefix")
    assert parsed["parse_error"]


def test_parse_json_content_extracts_valid_embedded_object():
    assert parse_json_content('prefix {"ok": true} suffix') == {"ok": True}
