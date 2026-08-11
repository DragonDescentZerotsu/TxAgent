import json
import pickle

import numpy as np
import pytest

from tools.chembl_tool.common.experiment_retrieval import (
    EvidenceGroupSpec,
    SourceExperimentConfig,
    retrieve_experiment_view,
)
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
    assert coverage["experiment"]["neighbor_selector"]["name"] == "query_feature_coverage"
    assert coverage["experiment"]["neighbor_selector"]["rank1_forced"] is False


def test_index_stores_versioned_parent_identity_metadata():
    index = build_neighbor_index([_row("salt", "CC[NH3+].[Cl-]", "Tier 1.direct", 1)], index_version="test")
    identity = index["molecules"][0]["molecule_identity"]

    assert identity["normalizer_version"] == "rdkit_fragment_parent.v1"
    assert identity["parent_inchi_key"]


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
