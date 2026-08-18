from tools.chembl_tool.common.molecule_identity import (
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)
from tools.chembl_tool.common.retrieval_policy import (
    MoleculeRelation,
    classify_molecule_relation,
    decide_candidate,
    policy_metadata,
)


def test_salt_and_free_base_share_a_parent_but_not_a_whole_record():
    free_base = normalize_molecule_identity("CCN")
    hydrochloride = normalize_molecule_identity("CC[NH3+].[Cl-]")

    assert free_base.standard_inchi_key != hydrochloride.standard_inchi_key
    assert free_base.parent_inchi_key == hydrochloride.parent_inchi_key
    assert classify_molecule_relation(free_base, hydrochloride) is MoleculeRelation.SAME_PARENT


def test_stereoisomers_are_connectivity_variants_not_the_same_parent():
    left = normalize_molecule_identity("F[C@H](Cl)Br")
    right = normalize_molecule_identity("F[C@@H](Cl)Br")

    assert left.parent_inchi_key != right.parent_inchi_key
    assert classify_molecule_relation(left, right) is MoleculeRelation.SAME_CONNECTIVITY_VARIANT


def test_covalent_analog_is_not_collapsed_into_the_same_parent():
    ethanol = normalize_molecule_identity("CCO")
    ethyl_acetate = normalize_molecule_identity("CCOC(C)=O")

    assert classify_molecule_relation(ethanol, ethyl_acetate) is MoleculeRelation.STRUCTURAL_ANALOG


def test_shared_counterion_does_not_make_two_salts_the_same_parent():
    ethylamine_hcl = normalize_molecule_identity("CC[NH3+].[Cl-]")
    propylamine_hcl = normalize_molecule_identity("CCC[NH3+].[Cl-]")

    assert classify_molecule_relation(ethylamine_hcl, propylamine_hcl) is MoleculeRelation.STRUCTURAL_ANALOG


def test_invalid_smiles_is_auditable_and_unresolved():
    valid = normalize_molecule_identity("CCO")
    invalid = normalize_molecule_identity("not a smiles")

    assert invalid.status == "invalid_smiles"
    assert classify_molecule_relation(valid, invalid) is MoleculeRelation.UNRESOLVED


def test_scaffold_disjoint_excludes_only_matching_nonempty_parent_scaffolds():
    query = normalize_molecule_identity("Cc1ccccc1")
    same_scaffold = {"canonical_smiles": "CCc1ccccc1"}
    different_scaffold = {"canonical_smiles": "c1ccncc1"}

    assert not decide_candidate(query, same_scaffold, "parent_disjoint").excluded
    decision = decide_candidate(query, same_scaffold, "scaffold_disjoint")
    assert decision.excluded
    assert decision.relation is MoleculeRelation.SAME_SCAFFOLD
    assert not decide_candidate(query, different_scaffold, "scaffold_disjoint").excluded


def test_scaffold_disjoint_does_not_collapse_acyclic_molecules():
    query = normalize_molecule_identity("CCO")
    decision = decide_candidate(
        query,
        {"canonical_smiles": "CCCO"},
        "scaffold_disjoint",
    )

    assert bemis_murcko_scaffold(query.parent_smiles) == ""
    assert not decision.excluded
    assert decision.relation is MoleculeRelation.STRUCTURAL_ANALOG


def test_scaffold_ignores_inconsistent_double_bond_stereo():
    smiles = "Oc1cc(C/N=C(S)/C=C/c2cc(O)c(O)c(Br)c2)cc(O)c1O"

    scaffold = bemis_murcko_scaffold(smiles)

    assert scaffold
    assert scaffold == bemis_murcko_scaffold(smiles.replace("/", ""))


def test_scaffold_policy_metadata_is_a_strict_parent_disjoint_superset():
    parent_relations = set(policy_metadata("parent_disjoint")["excluded_relations"])
    scaffold_relations = set(policy_metadata("scaffold_disjoint")["excluded_relations"])

    assert scaffold_relations == parent_relations | {"same_scaffold"}
