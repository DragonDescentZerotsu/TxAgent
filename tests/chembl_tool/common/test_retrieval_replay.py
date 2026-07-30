import json

import pytest

from tools.chembl_tool.common.neighbor_selection import selector_metadata
from tools.chembl_tool.common.assay_transfer_selection import assay_transfer_selection_policy
from tools.chembl_tool.common.retrieval_replay import load_retrieval_replay


def test_load_retrieval_replay_returns_frozen_payload(tmp_path):
    payload = {"status": "ok", "query": {"input_smiles": "CCO"}, "groups": []}
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    assert load_retrieval_replay(
        str(tmp_path),
        "CCO",
        expected_neighbor_selector="similarity",
    ) == payload


def test_load_retrieval_replay_rejects_query_mismatch(tmp_path):
    payload = {"status": "ok", "query": {"input_smiles": "CCO"}, "groups": []}
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="query mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCN",
            expected_neighbor_selector="similarity",
        )


def test_load_retrieval_replay_validates_neighbor_selector_provenance(tmp_path):
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "experiment": {
            "neighbor_selector": selector_metadata("query_feature_coverage"),
        },
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    assert load_retrieval_replay(
        str(tmp_path),
        "CCO",
        expected_neighbor_selector="query_feature_coverage",
    ) == payload
    with pytest.raises(ValueError, match="neighbor-selector provenance mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="similarity",
        )


def test_load_retrieval_replay_accepts_native_selector_provenance(tmp_path):
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "retrieval_policy": {
            "neighbor_selector": selector_metadata("query_feature_coverage"),
        },
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    assert load_retrieval_replay(
        str(tmp_path),
        "CCO",
        expected_neighbor_selector="query_feature_coverage",
    ) == payload


def test_load_retrieval_replay_accepts_matching_dual_selector_provenance(tmp_path):
    selector = selector_metadata("query_feature_coverage")
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "retrieval_policy": {"neighbor_selector": selector},
        "experiment": {"neighbor_selector": selector},
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    assert load_retrieval_replay(
        str(tmp_path),
        "CCO",
        expected_neighbor_selector="query_feature_coverage",
    ) == payload


def test_load_retrieval_replay_rejects_conflicting_dual_selector_provenance(tmp_path):
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "retrieval_policy": {
            "neighbor_selector": selector_metadata("query_feature_coverage"),
        },
        "experiment": {"neighbor_selector": selector_metadata("similarity")},
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="provenance conflicts"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="query_feature_coverage",
        )


def test_load_retrieval_replay_treats_absent_selector_as_similarity(tmp_path):
    payload = {"status": "ok", "query": {"input_smiles": "CCO"}, "groups": []}
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="neighbor-selector provenance mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="query_feature_coverage",
        )


def test_load_retrieval_replay_rejects_selector_version_mismatch(tmp_path):
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "experiment": {
            "neighbor_selector": {
                **selector_metadata("query_feature_coverage"),
                "version": "query_feature_coverage.v0",
            },
        },
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="neighbor-selector provenance mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="query_feature_coverage",
        )


@pytest.mark.parametrize(
    "metadata",
    [
        "query_feature_coverage",
        {},
        {"name": "unknown", "version": "unknown.v1"},
        {"name": "query_feature_coverage"},
    ],
)
def test_load_retrieval_replay_rejects_malformed_selector_metadata(
    tmp_path,
    metadata,
):
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "experiment": {"neighbor_selector": metadata},
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="neighbor-selector provenance is malformed"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="query_feature_coverage",
        )


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
        str(tmp_path),
        "CCO",
        expected_neighbor_selector="similarity",
        expected_reranker_provenance=provenance,
    ) == payload
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="similarity",
            expected_reranker_provenance={**provenance, "catalog_version": "different"},
        )


def test_load_retrieval_replay_validates_assay_transfer_diversity_policy(tmp_path):
    structural = assay_transfer_selection_policy(mode="structural", score_slack=0.1)
    payload = {
        "status": "ok",
        "query": {"input_smiles": "CCO"},
        "experiment": {
            "assay_transfer_selection_policy": {
                "min_score": 0.5,
                "diversity": structural,
            }
        },
        "groups": [],
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(payload))

    expected = {"min_score": 0.5, "diversity": structural}
    assert load_retrieval_replay(
        str(tmp_path),
        "CCO",
        expected_neighbor_selector="similarity",
        expected_assay_transfer_selection_policy=expected,
    ) == payload
    with pytest.raises(ValueError, match="selection-policy mismatch"):
        load_retrieval_replay(
            str(tmp_path),
            "CCO",
            expected_neighbor_selector="similarity",
            expected_assay_transfer_selection_policy={
                "min_score": 0.5,
                "diversity": assay_transfer_selection_policy(mode="assay", score_slack=0.1),
            },
        )
