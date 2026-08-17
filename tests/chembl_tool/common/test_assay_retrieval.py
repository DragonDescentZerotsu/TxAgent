from tools.chembl_tool.common.assay_retrieval import (
    FLAT_GROUP_ID,
    geometric_assay_prefixes,
    retrieve_assay_prefix,
    retrieve_assay_prefixes,
)
from tools.chembl_tool.common.evidence_contract import ASSAY_COMPACT_PROMPT_PROFILE
from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index


def _row(molecule_id, smiles, assay_id, endpoint):
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "assay_chembl_id": assay_id,
        "assay_tier": "Assay",
        "endpoint_group": "assay_ranked_evidence",
        "group_id": f"Assay.{assay_id}",
        "standard_type": endpoint,
        "standard_value": "1",
        "standard_units": "unit",
        "evidence_source": "Starling",
        "assay_description": f"evidence for {endpoint}",
        "source_record_count": 1,
    }


def _index():
    rows = [
        _row("shared", "CCCO", "a1", "endpoint-1"),
        _row("only-a1", "CCCCO", "a1", "endpoint-1"),
        _row("shared", "CCCO", "a2", "endpoint-2"),
        _row("only-a2", "CCCN", "a2", "endpoint-2"),
    ]
    index = build_neighbor_index(rows, index_version="test", workers=1)
    index["assay_ranking"] = [
        {"assay_id": "a1", "assay_context": "assay one", "group_id": "Assay.a1"},
        {"assay_id": "a2", "assay_context": "assay two", "group_id": "Assay.a2"},
    ]
    index["source"] = {
        "type": "starling_assay_ranked_flat",
        "relevance_scores_visible_to_llm": False,
    }
    return index


def test_assay_prefix_is_flat_and_merges_same_molecule_across_assays():
    result = retrieve_assay_prefix(
        "CC",
        _index(),
        assay_prefix=2,
        top_k_per_assay=3,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    assert result["status"] == "ok"
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["group_id"] == FLAT_GROUP_ID
    assert group["evidence_prompt_profile"] == ASSAY_COMPACT_PROMPT_PROFILE
    assert group["n_selected_assays"] == 2
    assert group["n_assay_neighbor_slots"] == 4
    assert group["n_unique_neighbor_molecules"] == 3
    shared = next(row for row in group["neighbors"] if row["molecule_chembl_id"] == "shared")
    assert len(shared["evidence_rows"]) == 2
    assert {row["standard_type"] for row in shared["evidence_rows"]} == {
        "endpoint-1",
        "endpoint-2",
    }


def test_assay_prefix_does_not_expose_relevance_scores_to_evidence_prompt_surface():
    result = retrieve_assay_prefix(
        "CC",
        _index(),
        assay_prefix=1,
        top_k_per_assay=3,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    serialized_group = str(result["groups"])
    assert "relevance_score" not in serialized_group
    assert "relevance_rank" not in serialized_group
    assert result["experiment"]["relevance_scores_visible_to_llm"] is False


def test_prefix_selection_is_strictly_nested():
    index = _index()
    top_one = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=1,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    top_two = retrieve_assay_prefix(
        "CC",
        index,
        assay_prefix=2,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    assert top_one["experiment"]["selected_assay_ids"] == ["a1"]
    assert top_two["experiment"]["selected_assay_ids"] == ["a1", "a2"]
    top_one_evidence = {
        row["standard_type"]
        for neighbor in top_one["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    }
    top_two_evidence = {
        row["standard_type"]
        for neighbor in top_two["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    }
    assert top_one_evidence < top_two_evidence


def test_multi_prefix_retrieval_matches_independent_prefix_results():
    index = _index()
    combined = retrieve_assay_prefixes(
        "CC",
        index,
        prefixes=[1, 2],
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    for prefix in (1, 2):
        independent = retrieve_assay_prefix(
            "CC",
            index,
            assay_prefix=prefix,
            min_similarity=0.0,
            neighbor_identity_policy="operational",
        )
        assert combined[prefix] == independent


def test_geometric_assay_prefixes_always_ends_at_full_catalog():
    assert geometric_assay_prefixes(22820) == (5, 20, 80, 320, 1280, 5120, 20480, 22820)
    assert geometric_assay_prefixes(1840) == (5, 20, 80, 320, 1280, 1840)
    assert geometric_assay_prefixes(3) == (3,)
