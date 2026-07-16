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


def test_load_retrieval_replay_validates_reranker_provenance(tmp_path):
    provenance = {
        "name": "assay_transfer",
        "model": "model",
        "model_revision": "a" * 40,
        "scoring_contract_version": "score.v1",
        "template_hash": "templates",
        "catalog_version": "catalog",
    }
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "experiment": {"retrieval_reranker": provenance},
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    assert load_retrieval_replay(
        str(tmp_path), "CCO", expected_reranker_provenance=provenance
    ) == payload
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_reranker_provenance={**provenance, "catalog_version": "different"},
        )
