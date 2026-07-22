import hashlib

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.reranking.analyze_assay_transfer_rerank_overlap import (
    _collapse_candidates,
    _overlap_metrics,
    _rank_molecules,
    _score_candidate_flat,
    _selection_row,
    _spearman_between_rankings,
    _structural_statistics,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import flat_score_key


def _candidate(molecule_id, smiles, similarity, *, record_ids=(), structural_rank=1):
    return {
        "molecule_id": molecule_id,
        "canonical_smiles": smiles,
        "similarity": similarity,
        "structural_rank": structural_rank,
        "record_ids": list(record_ids),
    }


def _scored(molecule_id, smiles, similarity, probability, winning_record_id="record"):
    return {
        **_candidate(molecule_id, smiles, similarity),
        "transfer_probability": probability,
        "winning_record_id": winning_record_id,
        "scored_record_count": 1,
    }


class _Catalog:
    def __init__(self, records):
        self.records = {record["record_id"]: record for record in records}

    def records_by_id(self, record_ids):
        return [self.records[record_id] for record_id in record_ids]


class _Renderer:
    @staticmethod
    def render(record, query_smiles):
        return f"{record['record_id']}|{query_smiles}"


def test_molecule_collapse_merges_records_and_uses_max_record_score():
    collapsed = _collapse_candidates(
        [
            _candidate("A", "CCO", 0.70, record_ids=("r1",), structural_rank=4),
            _candidate("A", "CCO", 0.75, record_ids=("r2", "r1"), structural_rank=3),
        ]
    )
    assert len(collapsed) == 1
    assert collapsed[0]["record_ids"] == ["r1", "r2"]
    assert collapsed[0]["similarity"] == 0.75
    assert collapsed[0]["structural_rank"] == 3

    provenance = {
        "model": "model",
        "model_revision": "a" * 40,
        "scoring_contract_version": "contract",
        "template_hash": "template",
    }
    scores = {}
    for record_id, probability in (("r1", 0.2), ("r2", 0.9)):
        prompt = _Renderer.render({"record_id": record_id}, "CCN")
        scores[
            flat_score_key(
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                **provenance,
            )
        ] = probability
    scored, missing = _score_candidate_flat(
        collapsed[0],
        "CCN",
        "Fa.absorption_solubility_permeability",
        _Catalog([{"record_id": "r1"}, {"record_id": "r2"}]),
        _Renderer(),
        provenance,
        scores,
    )
    assert missing == 0
    assert scored["scored_record_count"] == 2
    assert scored["transfer_probability"] == 0.9
    assert scored["winning_record_id"] == "r2"


def test_morgan_ranks_each_molecule_once_without_inheriting_a_record():
    rows = [
        _scored("A", "CCO", 0.9, 0.1, "a-record"),
        _scored("B", "CCN", 0.8, 0.9, "b-record"),
    ]
    assert [row["molecule_id"] for row in _rank_molecules(rows, method="morgan")] == ["A", "B"]
    assert [row["molecule_id"] for row in _rank_molecules(rows, method="assay_transfer")] == ["B", "A"]
    morgan_output = _selection_row(rows[0], 1, method="morgan")
    assert "winning_record_id" not in morgan_output
    assert "transfer_probability" not in morgan_output
    with pytest.raises(ValueError, match="one row per molecule"):
        _rank_molecules([rows[0], rows[0]], method="morgan")


def test_known_full_pool_and_shared_top_spearman_permutations():
    assert _spearman_between_rankings(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _spearman_between_rankings(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert _spearman_between_rankings(["a", "b", "c", "d"], ["a", "c", "b", "d"]) == pytest.approx(0.8)

    left = [{"molecule_id": value} for value in ("a", "b", "c", "d", "e")]
    right = [{"molecule_id": value} for value in ("x", "d", "y", "b", "z")]
    overlap = _overlap_metrics(left, right, top_k=5, pool_size=8)
    assert overlap["overlap_count"] == 2
    assert overlap["overlap_percentage"] == 40.0
    assert overlap["shared_order_spearman"] == -1.0


def test_overlap_denominators_for_complete_short_and_empty_pools():
    full = [{"molecule_id": value} for value in "abcde"]
    assert _overlap_metrics(full, full[:4] + [{"molecule_id": "x"}], top_k=5, pool_size=6)[
        "overlap_denominator"
    ] == 5

    short_left = [{"molecule_id": value} for value in "abc"]
    short_right = [{"molecule_id": value} for value in "abx"]
    short = _overlap_metrics(short_left, short_right, top_k=5, pool_size=3)
    assert short["overlap_denominator"] == 3
    assert short["overlap_percentage"] == pytest.approx(200 / 3)

    empty = _overlap_metrics([], [], top_k=5, pool_size=0)
    assert empty["overlap_denominator"] == 0
    assert empty["overlap_percentage"] is None
    assert empty["exact_set_match"] is None


def test_pairwise_diversity_for_identical_and_dissimilar_structures_is_bounded():
    identical = _structural_statistics(
        [_candidate("A", "CCO", 1.0), _candidate("B", "CCO", 1.0)],
        "CCO",
    )
    assert identical["mean_pairwise_tanimoto"] == 1.0
    assert identical["structural_diversity"] == 0.0
    assert identical["fraction_pairs_tanimoto_ge_0_80"] == 1.0

    dissimilar = _structural_statistics(
        [_candidate("A", "CCO", 0.5), _candidate("B", "c1ccccc1", 0.5)],
        "CCN",
    )
    for key in (
        "mean_pairwise_tanimoto",
        "median_pairwise_tanimoto",
        "min_pairwise_tanimoto",
        "max_pairwise_tanimoto",
        "structural_diversity",
        "fraction_pairs_tanimoto_ge_0_80",
        "mean_similarity_to_query",
    ):
        assert 0.0 <= dissimilar[key] <= 1.0
    assert dissimilar["mean_pairwise_tanimoto"] < 1.0
    assert dissimilar["structural_diversity"] > 0.0
