from tools.chembl_tool.paper_experiments.skin_canonical_gold_sensitivity.run import (
    classify_evidence_family,
    collapse_to_study_votes,
    consensus_vote,
    direct_quality_gate,
)


def test_classify_skin_evidence_families() -> None:
    assert classify_evidence_family(
        {"assay_or_test": "LLNA", "species_or_population": "mouse"}
    )[0] == "predictive_animal"
    assert classify_evidence_family(
        {"assay_or_test": "HRIPT", "species_or_population": "human volunteers"}
    )[0] == "predictive_human"
    assert classify_evidence_family(
        {
            "assay_or_test": "sensitization assessment",
            "species_or_population": "healthy human volunteers",
        }
    )[0] == "predictive_human"
    assert classify_evidence_family(
        {"assay_or_test": "diagnostic patch test", "species_or_population": "patients"}
    )[0] == "diagnostic_human"
    assert classify_evidence_family(
        {
            "assay_or_test": "patch test",
            "species_or_population": "human hairdressers",
            "support_text": "occupational dermatitis case series",
        }
    )[0] == "case_occupational"
    assert classify_evidence_family(
        {"assay_or_test": "skin sensitization test", "species_or_population": ""}
    )[0] == "other_direct_ambiguous"


def test_consensus_vote_requires_seventy_percent() -> None:
    assert consensus_vote([1] * 7 + [0] * 3)["label"] == 1
    assert consensus_vote([1] * 6 + [0] * 4)["label"] is None
    assert consensus_vote([1, 0])["status"] == "exact_tie"
    assert consensus_vote([])["status"] == "no_usable_votes"


def test_direct_quality_gate_rejects_prediction_only_manifest_leak() -> None:
    assert direct_quality_gate(
        {"assay_or_test": "computational ADMET prediction"}
    ) == (False, "reject_prediction_only_assay")
    assert direct_quality_gate(
        {"assay_or_test": "LLNA supported computational model"}
    ) == (False, "reject_prediction_only_assay")
    assert direct_quality_gate(
        {"assay_or_test": "predictive human skin sensitization study"}
    ) == (True, "accepted_canonical_direct_candidate")
    assert direct_quality_gate({"assay_or_test": "LLNA"}) == (
        True,
        "accepted_canonical_direct_candidate",
    )


def test_study_vote_deduplicates_records_and_abstains_on_conflict() -> None:
    records = [
        {
            "source_partition": "direct_skin_reaction",
            "pmid": "1",
            "source_record_id": "a",
            "outcome_label": "positive",
        },
        {
            "source_partition": "direct_skin_reaction",
            "pmid": "1",
            "source_record_id": "b",
            "outcome_label": "positive",
        },
        {
            "source_partition": "direct_skin_reaction",
            "pmid": "2",
            "source_record_id": "c",
            "outcome_label": "positive",
        },
        {
            "source_partition": "direct_skin_reaction",
            "pmid": "2",
            "source_record_id": "d",
            "outcome_label": "negative",
        },
    ]
    votes, status = collapse_to_study_votes(records)
    assert len(votes) == 2
    assert votes[0]["label"] == 1
    assert votes[0]["n_records"] == 2
    assert votes[1]["label"] is None
    assert status == {
        "abstain_within_study_conflict": 1,
        "accepted_unanimous_study": 1,
    }


def test_study_vote_merges_same_pmid_across_acquisition_sources() -> None:
    records = [
        {
            "source_partition": "direct_skin_reaction",
            "pmid": "7",
            "source_record_id": "a",
            "outcome_label": "positive",
        },
        {
            "source_partition": "sensitization_aop",
            "pmid": "7",
            "source_record_id": "b",
            "outcome_label": "negative",
        },
    ]
    votes, status = collapse_to_study_votes(records)
    assert len(votes) == 1
    assert votes[0]["label"] is None
    assert votes[0]["source_partitions"] == [
        "direct_skin_reaction",
        "sensitization_aop",
    ]
    assert status == {"abstain_within_study_conflict": 1}
