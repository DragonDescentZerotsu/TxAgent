import hashlib
import json

import pytest

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    canonical_json_bytes,
    parse_json_content,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_canonical_json_and_streaming_hash_are_deterministic(tmp_path):
    first = canonical_json_bytes({"b": 2, "a": "可审计"})
    second = canonical_json_bytes({"a": "可审计", "b": 2})
    assert first == second

    path = tmp_path / "payload.json"
    path.write_bytes(first)
    assert sha256_file(path) == hashlib.sha256(first).hexdigest()


def test_parse_json_content_never_raises_for_malformed_embedded_object():
    parsed = parse_json_content('prefix {"value": [1, 2} suffix')

    assert parsed["unparsed_text"].startswith("prefix")
    assert parsed["parse_error"]


def test_parse_json_content_extracts_valid_embedded_object():
    assert parse_json_content('prefix {"ok": true} suffix') == {"ok": True}


def test_parse_json_content_recovers_complete_object_after_extra_open_brace():
    assert parse_json_content('{\n{"prediction": "pass"}') == {
        "prediction": "pass"
    }


def test_atomic_json_writers_publish_complete_files(tmp_path):
    json_path = tmp_path / "nested" / "payload.json"
    jsonl_path = tmp_path / "rows.jsonl"

    write_json_atomic(json_path, {"label": "可审计", "value": 2})
    write_jsonl_atomic(jsonl_path, [{"row": 1}, {"row": 2}])

    assert json.loads(json_path.read_text(encoding="utf-8"))["label"] == "可审计"
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [
        {"row": 1},
        {"row": 2},
    ]


def test_atomic_output_does_not_replace_destination_after_failure(tmp_path):
    output = tmp_path / "state.json"
    output.write_text('{"state": "old"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError):
        with atomic_output_path(output) as temporary:
            temporary.write_text('{"state": "partial"}\n', encoding="utf-8")
            raise RuntimeError("serialization failed")

    assert json.loads(output.read_text(encoding="utf-8")) == {"state": "old"}
    assert not list(tmp_path.glob(".state.json.*.tmp"))
