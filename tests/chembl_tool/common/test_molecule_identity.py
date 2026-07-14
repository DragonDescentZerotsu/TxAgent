from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import (
    MoleculeRelation,
    classify_molecule_relation,
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
