import json
import pickle

import numpy as np
import pytest

from tools.chembl_tool.common.experiment_retrieval import (
    EvidenceGroupSpec,
    SourceExperimentConfig,
    _rank_group_candidates,
    flatten_retrieval_groups,
    retrieve_experiment_view,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_features import (
    DESCRIPTOR_TYPE,
    candidate_order_sha256,
)
from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import load_index


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


def test_scaffold_disjoint_excludes_same_scaffold_and_backfills():
    rows = [
        _row("same_scaffold", "CCc1ccccc1", "Tier 1.direct", 1),
        _row("different_scaffold", "c1ccncc1", "Tier 1.direct", 2),
    ]
    index = build_neighbor_index(rows, index_version="test.scaffold.v1")

    operational = retrieve_experiment_view(
        "Cc1ccccc1",
        index,
        mode="direct",
        config=CONFIG,
        top_k_per_group=1,
        min_similarity=0.0,
        neighbor_identity_policy="operational",
    )
    disjoint = retrieve_experiment_view(
        "Cc1ccccc1",
        index,
        mode="direct",
        config=CONFIG,
        top_k_per_group=1,
        min_similarity=0.0,
        neighbor_identity_policy="scaffold_disjoint",
    )

    assert operational["groups"][0]["neighbors"][0]["molecule_chembl_id"] == "same_scaffold"
    assert disjoint["groups"][0]["neighbors"][0]["molecule_chembl_id"] == "different_scaffold"


def test_query_feature_coverage_selector_keeps_retrieval_payload_contract():
    similarity = retrieve_experiment_view(
        "CO",
        _index(),
        mode="full_mechanism",
        config=CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
    )
    coverage = retrieve_experiment_view(
        "CO",
        _index(),
        mode="full_mechanism",
        config=CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
        neighbor_selector="query_feature_coverage",
    )

    similarity_neighbor = next(
        neighbor
        for group in similarity["groups"]
        for neighbor in group["neighbors"]
    )
    coverage_neighbor = next(
        neighbor
        for group in coverage["groups"]
        for neighbor in group["neighbors"]
    )
    assert set(coverage_neighbor) == set(similarity_neighbor)
    assert similarity["retrieval_policy"]["neighbor_selector"]["name"] == "similarity"
    assert coverage["experiment"]["neighbor_selector"]["name"] == "query_feature_coverage"
    assert coverage["experiment"]["neighbor_selector"]["rank1_forced"] is False
    assert (
        coverage["retrieval_policy"]["neighbor_selector"]
        == coverage["experiment"]["neighbor_selector"]
    )


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


class _EligiblePoolReranker(_ReverseScoreReranker):
    def provenance(self):
        return {
            "name": self.name,
            "version": "test.v1",
            "candidate_contract": "tanimoto_identity_exclusion_then_eligible_pool.v1",
        }


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
        assay_transfer_initial_morgan_filter=3,
    )

    assert [row["molecule_chembl_id"] for row in neighbors] == ["b", "a"]
    assert "below_raw_pool" not in {row["molecule_chembl_id"] for row in neighbors}
    assert [row["structural_rank"] for row in neighbors] == [3, 2]


def test_compact_v11_contract_backfills_to_fifty_after_identity_exclusion():
    molecules = [
        {"molecule_chembl_id": "exact", "canonical_smiles": "CCO"},
        {"molecule_chembl_id": "a", "canonical_smiles": "CCN"},
        {"molecule_chembl_id": "b", "canonical_smiles": "CCC"},
        {"molecule_chembl_id": "backfill", "canonical_smiles": "CCCC"},
    ]
    index = {
        "molecules": molecules,
        "evidence_by_molecule_group": {
            row["molecule_chembl_id"]: {
                "Tier 1.direct": [{"id": row["molecule_chembl_id"]}]
            }
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
        reranker=_EligiblePoolReranker(),
        assay_transfer_initial_morgan_filter=3,
    )

    assert {row["molecule_chembl_id"] for row in neighbors} == {"a", "b", "backfill"}
    assert {row["structural_rank"] for row in neighbors} == {2, 3, 4}


class _ThresholdReranker:
    name = "test_threshold"

    def rerank_records(self, *, query_smiles, group_id, candidates):
        scores = {"a": 0.7, "b": 0.5, "c": 0.4999}
        return sorted(
            [
                {**candidate, "transfer_selection_score": scores[candidate["molecule_chembl_id"]]}
                for candidate in candidates
            ],
            key=lambda row: -row["transfer_selection_score"],
        )

    def provenance(self):
        return {"name": self.name, "version": "test.v1"}


def test_assay_transfer_threshold_is_inclusive_and_applied_before_top_k():
    molecules = [
        {"molecule_chembl_id": "a", "canonical_smiles": "CCN"},
        {"molecule_chembl_id": "b", "canonical_smiles": "CCC"},
        {"molecule_chembl_id": "c", "canonical_smiles": "CCCl"},
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
        [0, 1, 2],
        source_groups=("Tier 1.direct",),
        similarities=[0.9, 0.8, 0.7],
        query_canonical_smiles="CCO",
        query_inchi_key="",
        top_k=3,
        min_similarity=0.0,
        query_identity=normalize_molecule_identity("CCO"),
        neighbor_identity_policy="operational",
        query_smiles="CCO",
        group_id="Direct.outcome",
        reranker=_ThresholdReranker(),
        assay_transfer_initial_morgan_filter=3,
        assay_transfer_min_score=0.5,
    )

    assert [row["molecule_chembl_id"] for row in neighbors] == ["a", "b"]
    assert neighbors.selection_metadata["n_below_min_score_dropped"] == 1


class _MultiRecordReranker:
    name = "test_multi_record"

    def rerank_records(self, *, query_smiles, group_id, candidates):
        by_id = {row["molecule_chembl_id"]: row for row in candidates}
        records = [
            ("a", "endpoint_one", 0.90),
            ("a", "endpoint_one", 0.89),
            ("a", "endpoint_two", 0.88),
            ("b", "endpoint_three", 0.80),
            ("c", "endpoint_four", 0.70),
        ]
        return [
            {
                **by_id[molecule_id],
                "transfer_selection_score": score,
                "transfer_winning_record_id": f"{molecule_id}-{endpoint}-{score}",
                "transfer_winning_record": {
                    "record_id": f"{molecule_id}-{endpoint}-{score}",
                    "canonical_endpoint_key": endpoint,
                    "source_contract": {},
                    "source_fields": {"endpoint_name": endpoint},
                },
            }
            for molecule_id, endpoint, score in records
        ]

    def provenance(self):
        return {"name": self.name, "version": "test.v1"}


def test_unique_molecule_top_k_is_unchanged_by_multi_record_presentation():
    molecules = [
        {"molecule_chembl_id": "a", "canonical_smiles": "CCN"},
        {"molecule_chembl_id": "b", "canonical_smiles": "CCC"},
        {"molecule_chembl_id": "c", "canonical_smiles": "CCCl"},
    ]
    index = {
        "molecules": molecules,
        "evidence_by_molecule_group": {
            row["molecule_chembl_id"]: {
                "Tier 1.direct": [{"id": row["molecule_chembl_id"]}]
            }
            for row in molecules
        },
    }

    neighbors = _rank_group_candidates(
        index,
        [0, 1, 2],
        source_groups=("Tier 1.direct",),
        similarities=[0.9, 0.8, 0.7],
        query_canonical_smiles="CCO",
        query_inchi_key="",
        top_k=2,
        min_similarity=0.0,
        query_identity=normalize_molecule_identity("CCO"),
        neighbor_identity_policy="operational",
        query_smiles="CCO",
        group_id="Direct.outcome",
        reranker=_MultiRecordReranker(),
        assay_transfer_initial_morgan_filter=3,
        assay_transfer_selection_unit="unique_molecule",
        assay_transfer_records_per_molecule=2,
    )

    assert [row["molecule_chembl_id"] for row in neighbors] == ["a", "b"]
    assert neighbors[0]["transfer_selection_score"] == 0.90
    assert [
        record["canonical_endpoint_key"]
        for record in neighbors[0]["transfer_selected_records"]
    ] == ["endpoint_one", "endpoint_two"]
    assert neighbors[1]["transfer_selected_record_count"] == 1
    assert neighbors.selection_metadata["selected_record_display"] == {
        "records_per_molecule": 2,
        "n_selected_molecules": 2,
        "n_selected_records_displayed": 3,
        "n_underfilled_selected_molecules": 1,
        "n_duplicate_endpoint_records_skipped": 1,
        "duplicate_endpoint_backfill": False,
    }


def test_flat_assay_transfer_merge_preserves_family_order_and_bundles():
    def neighbor(molecule_id, similarity, score, record_id):
        return {
            "rank": 1,
            "molecule_chembl_id": molecule_id,
            "similarity": similarity,
            "transfer_selection_score": score,
            "transfer_winning_record": {"record_id": record_id},
            "evidence_rows": [{"evidence_id": record_id, "group_id": record_id[0]}],
            "source_group_ids": [record_id[0]],
        }

    flat = flatten_retrieval_groups(
        [
            {
                "group_id": "family_a",
                "tier": "A",
                "endpoint_group": "a",
                "source_group_ids": ["a"],
                "transfer_neighbor_selection": {"diversity": {"selection_unit": "scored_record"}},
                "neighbors": [neighbor("shared", 0.4, 0.9, "a1"), neighbor("first", 0.2, 0.8, "a2")],
            },
            {
                "group_id": "family_b",
                "tier": "B",
                "endpoint_group": "b",
                "source_group_ids": ["b"],
                "transfer_neighbor_selection": {"diversity": {"selection_unit": "scored_record"}},
                "neighbors": [neighbor("shared", 0.4, 0.7, "b1"), neighbor("higher_morgan", 0.99, 0.6, "b2")],
            },
        ]
    )

    assert [row["molecule_chembl_id"] for row in flat["neighbors"]] == [
        "shared", "first", "higher_morgan"
    ]
    assert [
        family["group_id"]
        for family in flat["neighbors"][0]["transfer_family_selections"]
    ] == ["family_a", "family_b"]
    assert flat["transfer_neighbor_selection"]["flat_merge_policy"] == (
        "stable_family_rank_merge_preserve_family_bundles.v1"
    )


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


def test_minimol_descriptor_replaces_morgan_ranking_and_preserves_provenance(tmp_path):
    index = _index()
    base_index = tmp_path / "index.pkl"
    with base_index.open("wb") as handle:
        pickle.dump(index, handle)

    # Candidate order is exact, amine, alkane. The query is closest to alkane
    # in embedding space even though the Morgan ranking differs.
    vector_by_id = {
        "exact": [0.0, 1.0],
        "amine": [0.6, 0.8],
        "alkane": [1.0, 0.0],
    }
    candidate_embeddings = np.asarray(
        [vector_by_id[item["molecule_chembl_id"]] for item in index["molecules"]],
        dtype=np.float32,
    )
    query_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    candidate_path = tmp_path / "candidate.npy"
    query_path = tmp_path / "query.npy"
    np.save(candidate_path, candidate_embeddings)
    np.save(query_path, query_embeddings)
    query_manifest = tmp_path / "query.json"
    query_manifest.write_text(
        json.dumps({"canonical_smiles_to_row": {"CO": 0}}),
        encoding="utf-8",
    )
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "type": DESCRIPTOR_TYPE,
                "model": "MiniMol",
                "model_version": "minimol_v1",
                "base_index_path": str(base_index),
                "candidate_embeddings_path": str(candidate_path),
                "candidate_order_sha256": candidate_order_sha256(index),
                "query_embeddings_path": str(query_path),
                "query_manifest_path": str(query_manifest),
            }
        ),
        encoding="utf-8",
    )

    result = retrieve_experiment_view(
        "CO",
        load_index(descriptor),
        mode="full_mechanism",
        config=CONFIG,
        top_k_per_group=1,
        min_similarity=0.0,
    )

    factor_neighbor = result["groups"][1]["neighbors"][0]
    assert factor_neighbor["molecule_chembl_id"] == "alkane"
    assert factor_neighbor["similarity"] == 1.0
    assert factor_neighbor["similarity_metric"] == "cosine"
    assert factor_neighbor["similarity_bucket"] == (
        "minimol_embedding_cosine_not_structural_similarity"
    )
    assert result["experiment"]["retrieval_feature"] == {
        "feature": "minimol_embedding",
        "model": "MiniMol",
        "model_version": "minimol_v1",
        "normalization": "L2",
        "similarity": "cosine",
    }


def test_minimol_descriptor_rejects_morgan_bit_coverage_selector(tmp_path):
    index = _index()
    base_index = tmp_path / "index.pkl"
    with base_index.open("wb") as handle:
        pickle.dump(index, handle)
    candidate_path = tmp_path / "candidate.npy"
    query_path = tmp_path / "query.npy"
    np.save(candidate_path, np.ones((len(index["molecules"]), 2), dtype=np.float32))
    np.save(query_path, np.ones((1, 2), dtype=np.float32))
    query_manifest = tmp_path / "query.json"
    query_manifest.write_text(
        json.dumps({"canonical_smiles_to_row": {"CO": 0}}),
        encoding="utf-8",
    )
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "type": DESCRIPTOR_TYPE,
                "base_index_path": str(base_index),
                "candidate_embeddings_path": str(candidate_path),
                "candidate_order_sha256": candidate_order_sha256(index),
                "query_embeddings_path": str(query_path),
                "query_manifest_path": str(query_manifest),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Morgan-bit-specific"):
        retrieve_experiment_view(
            "CO",
            load_index(descriptor),
            mode="full_mechanism",
            config=CONFIG,
            top_k_per_group=1,
            min_similarity=0.0,
            neighbor_selector="query_feature_coverage",
        )
