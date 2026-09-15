import pytest

from tools.chembl_tool.tasks.ames.build_conditioned_source import collapse_study_votes
from tools.chembl_tool.tasks.ames.reviewed_source import apply_review, payload_hash
from tools.chembl_tool.tasks.ames.source_contract import DIRECT, NEAR, Decision


def fixture_review(action="accept"):
    raw = {"pmid": "123", "SMILES": "CCO", "support_text": "Measured TA100 outcome"}
    review = {
        "source_record_id": "ames_v3:7",
        "pmid": "123",
        "raw_sha256": payload_hash(raw),
        "parent_inchi_key": "parent",
        "action": action,
        "Y": 1,
        "condition_atoms": ["strain_panel=TA100", "metabolic_activation=present"],
    }
    return review, raw


def test_review_can_accept_a_direct_claim_from_an_indirect_run():
    review, raw = fixture_review()
    decision = apply_review(review, raw, "parent")
    assert decision.group == DIRECT and decision.label == 1
    assert decision.condition_atoms == tuple(sorted(review["condition_atoms"]))


def test_review_never_authorizes_changed_payload_or_parent():
    review, raw = fixture_review()
    with pytest.raises(ValueError, match="payload changed"):
        apply_review(
            review, {**raw, "support_text": "predicted TA100 outcome"}, "parent"
        )
    with pytest.raises(ValueError, match="parent changed"):
        apply_review(review, raw, "another molecule")


def test_pending_direct_claim_cannot_enter_a_later_family_or_vote():
    review, raw = fixture_review("withhold")
    decision = apply_review(review, raw, "parent")
    assert decision.group == NEAR and decision.label is None
    review["action"] = "exclude"
    decision = apply_review(review, raw, "parent")
    assert not decision.group and decision.label is None


def test_reviewed_cross_source_description_does_not_add_an_independent_vote():
    common = {
        "pmid": "123",
        "molecule_identity_key": "parent",
        "condition_group": "TA100+S9",
        "Y": 1,
    }
    votes, audit = collapse_study_votes(
        [
            {**common, "source_record_id": "ames_base:8"},
            {**common, "source_record_id": "ames_v3:7"},
        ]
    )
    assert len(votes) == 1
    assert votes[0]["source_record_id"] == "ames_base:8"
    assert len(votes[0]["supporting_source_record_ids"]) == 2
    assert audit[1]["reason"] == "same_study_parent_condition_repeated_description"


def test_structure_correction_preserves_nonvoter_and_cannot_rewrite_gold():
    review, raw = fixture_review("correct_structure")
    original = Decision(NEAR, "nonvoter")
    assert apply_review(review, raw, "parent", original_decision=original) == original
    for decision in (None, Decision(DIRECT, "voter", 1), Decision("", "excluded")):
        with pytest.raises(ValueError, match="eligible nonvoter"):
            apply_review(review, raw, "parent", original_decision=decision)


@pytest.mark.parametrize("smiles", ["CN(N=O)C1CCCCC1", "[2H]C([2H])([2H])N(N=O)C1CCCCC1"])
def test_corrected_nitrosamine_preserves_connectivity_and_isotope(smiles):
    from rdkit import Chem
    from tools.chembl_tool.tasks.ames.build_conditioned_source import _identity

    identity = _identity(smiles)[1]
    mol = Chem.MolFromSmiles(identity["canonical_smiles"])
    assert mol.HasSubstructMatch(Chem.MolFromSmarts("[N]-[N]=[O]"))
    assert sum(a.GetIsotope() == 2 for a in mol.GetAtoms()) == (3 if "[2H]" in smiles else 0)


def test_structure_review_updates_retrieval_identity_without_creating_a_vote():
    from tools.chembl_tool.tasks.ames.build_conditioned_source import _identity, _prepare_record

    raw = {"SMILES": "O=NCNC1CCCCC1", "pmid": "3319273", "endpoint_class": "dna_adduct",
           "result_status": "damage_or_response_increased", "support_text": "Nitrosomethylcyclohexylamine methylates DNA."}
    identity = _identity(raw["SMILES"])[1]
    corrected = _identity("CN(N=O)C1CCCCC1")[1]
    review = {"action": "correct_structure", "source_record_id": "ames_v2:1", "pmid": raw["pmid"],
              "raw_sha256": payload_hash(raw), "parent_inchi_key": identity["parent_inchi_key"],
              "corrected_smiles": corrected["canonical_smiles"], "corrected_parent_inchi_key": corrected["parent_inchi_key"]}
    record, vote, audit, *_ = _prepare_record("ames_v2", 1, raw, {raw["SMILES"]: identity}, {}, {"ames_v2:1": review}, {})
    assert record["canonical_smiles"] == corrected["canonical_smiles"]
    assert record["source_smiles"] == raw["SMILES"] == "O=NCNC1CCCCC1"
    assert record["support_text"] == raw["support_text"]
    assert record["molecule_identity_key"] == corrected["parent_inchi_key"]
    assert vote is None and not audit["is_voter"]
    assert record["identity_verification"] == "payload_pinned_structure_correction"
