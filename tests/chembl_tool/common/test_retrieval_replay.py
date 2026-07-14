import json

import pytest

from tools.chembl_tool.common.retrieval_replay import load_retrieval_replay


def test_load_retrieval_replay_returns_frozen_payload(tmp_path):
    payload = {"status": "ok", "query": {"input_smiles": "CCO"}, "groups": []}
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    assert load_retrieval_replay(str(tmp_path), "CCO") == payload


def test_load_retrieval_replay_rejects_query_mismatch(tmp_path):
    payload = {"status": "ok", "query": {"input_smiles": "CCO"}, "groups": []}
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="query mismatch"):
        load_retrieval_replay(str(tmp_path), "CCN")
