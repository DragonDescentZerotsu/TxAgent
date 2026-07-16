from tools.chembl_tool.common.experiment_retrieval import (
    EvidenceGroupSpec,
    SourceExperimentConfig,
    _rank_group_candidates,
    retrieve_experiment_view,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index


def _index():
    rows = [
        _row("exact", "CCO", "Tier 1.direct", 1),
        _row("amine", "CCN", "Tier 1.direct", 2),
        _row("amine", "CCN", "Tier 2.mechanism", 3),
        _row("alkane", "CCC", "Tier 2.mechanism", 4),
    ]
    index = build_neighbor_index(rows, index_version="test.v1")
    index["source"] = {"dataset": "test-source"}
    return index


def _row(molecule_id, smiles, group_id, value):
    tier, endpoint = group_id.split(".", 1)
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "group_id": group_id,
        "assay_tier": tier,
        "endpoint_group": endpoint,
        "standard_type": endpoint,
        "standard_value": value,
        "standard_units": "%",
    }


CONFIG = SourceExperimentConfig(
    source_name="test",
    direct_groups=(
        EvidenceGroupSpec("Direct.outcome", "Direct", "outcome", source_groups=("Tier 1.direct",)),
    ),
    mechanism_groups=(
        EvidenceGroupSpec("Mechanism.direct", "Tier 1", "direct", source_groups=("Tier 1.direct",)),
        EvidenceGroupSpec(
            "Mechanism.factor",
            "Tier 2",
            "factor",
            source_group_prefixes=("Tier 2.",),
        ),
    ),
)


def test_none_mode_does_not_need_an_index():
    result = retrieve_experiment_view(
        "CCO",
        None,
        mode="none",
        config=None,
        top_k_per_group=3,
        min_similarity=0.3,
    )

    assert result["status"] == "ok"
    assert result["groups"] == []
    assert result["coverage"]["n_neighbors_total"] == 0


def test_direct_mode_excludes_exact_query_and_uses_declared_groups():
    result = retrieve_experiment_view(
        "CCO",
        _index(),
        mode="direct",
        config=CONFIG,
        top_k_per_group=3,
        min_similarity=0.0,
    )

    assert [item["molecule_chembl_id"] for item in result["groups"][0]["neighbors"]] == ["amine"]
    assert result["experiment"]["resolved_group_mapping"] == {"Direct.outcome": ["Tier 1.direct"]}


def test_flat_and_mechanism_views_contain_the_same_evidence_rows():
    index = _index()
    mechanism = retrieve_experiment_view(
        "CO",
        index,
        mode="full_mechanism",
        config=CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
    )
    flat = retrieve_experiment_view(
        "CO",
        index,
        mode="full_flat",
        config=CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
    )

    mechanism_values = sorted(
        row["standard_value"]
        for group in mechanism["groups"]
        for neighbor in group["neighbors"]
        for row in neighbor["evidence_rows"]
    )
    flat_values = sorted(
        row["standard_value"]
        for neighbor in flat["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    )
    assert mechanism_values == flat_values
    assert flat["groups"][0]["group_id"] == "Flat.all_evidence"


def test_parent_disjoint_excludes_salt_and_backfills_only_eligible_analogs():
    rows = [
        _row("salt", "CC[NH3+].[Cl-]", "Tier 1.direct", 1),
        _row("analog", "CCCN", "Tier 1.direct", 2),
        _row("below_threshold", "c1ccccc1", "Tier 1.direct", 3),
    ]
    index = build_neighbor_index(rows, index_version="test.parent.v1")

    operational = retrieve_experiment_view(
        "CCN",
        index,
        mode="direct",
        config=CONFIG,
        top_k_per_group=3,
        min_similarity=0.3,
        neighbor_identity_policy="operational",
    )
    disjoint = retrieve_experiment_view(
        "CCN",
        index,
        mode="direct",
        config=CONFIG,
        top_k_per_group=3,
        min_similarity=0.3,
        neighbor_identity_policy="parent_disjoint",
    )

    operational_by_id = {
        row["molecule_chembl_id"]: row for row in operational["groups"][0]["neighbors"]
    }
    assert operational_by_id["salt"]["molecule_relation"] == "same_parent"
    assert [row["molecule_chembl_id"] for row in disjoint["groups"][0]["neighbors"]] == ["analog"]
    assert disjoint["coverage"]["top_k_per_group"] == 3


def test_index_stores_versioned_parent_identity_metadata():
    index = build_neighbor_index([_row("salt", "CC[NH3+].[Cl-]", "Tier 1.direct", 1)], index_version="test")
    identity = index["molecules"][0]["molecule_identity"]

    assert identity["normalizer_version"] == "rdkit_fragment_parent.v1"
    assert identity["parent_inchi_key"]


class _ReverseScoreReranker:
    name = "test_reverse"

    def rerank(self, *, query_smiles, group_id, candidates):
        output = [
            {**candidate, "transfer_selection_score": float(index)}
            for index, candidate in enumerate(candidates)
        ]
        return sorted(output, key=lambda row: -row["transfer_selection_score"])

    def provenance(self):
        return {"name": self.name, "version": "test.v1"}


def test_rerank_contract_truncates_raw_pool_before_exclusion_and_does_not_backfill():
    molecules = [
        {"molecule_chembl_id": "exact", "canonical_smiles": "CCO"},
        {"molecule_chembl_id": "a", "canonical_smiles": "CCN"},
        {"molecule_chembl_id": "b", "canonical_smiles": "CCC"},
        {"molecule_chembl_id": "below_raw_pool", "canonical_smiles": "CCCC"},
    ]
    index = {
        "molecules": molecules,
        "evidence_by_molecule_group": {
            row["molecule_chembl_id"]: {"Tier 1.direct": [{"id": row["molecule_chembl_id"]}]}
            for row in molecules
        },
    }
    neighbors = _rank_group_candidates(
        index,
        [0, 1, 2, 3],
        source_groups=("Tier 1.direct",),
        similarities=[1.0, 0.9, 0.8, 0.7],
        query_canonical_smiles="CCO",
        query_inchi_key="",
        top_k=3,
        min_similarity=0.0,
        query_identity=normalize_molecule_identity("CCO"),
        neighbor_identity_policy="operational",
        query_smiles="CCO",
        group_id="Direct.outcome",
        reranker=_ReverseScoreReranker(),
        rerank_raw_pool_size=3,
        rerank_candidate_size=3,
    )

    assert [row["molecule_chembl_id"] for row in neighbors] == ["b", "a"]
    assert "below_raw_pool" not in {row["molecule_chembl_id"] for row in neighbors}
    assert [row["structural_rank"] for row in neighbors] == [3, 2]


def test_reranker_disabled_preserves_structural_selection_order():
    baseline = retrieve_experiment_view(
        "CO",
        _index(),
        mode="full_mechanism",
        config=CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
    )
    explicitly_disabled = retrieve_experiment_view(
        "CO",
        _index(),
        mode="full_mechanism",
        config=CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
        reranker=None,
    )

    assert baseline == explicitly_disabled
    assert "retrieval_reranker" not in baseline["experiment"]
    assert all("transfer_neighbor_selection" not in group for group in baseline["groups"])
    assert all(
        "structural_rank" not in neighbor
        for group in baseline["groups"]
        for neighbor in group["neighbors"]
    )
