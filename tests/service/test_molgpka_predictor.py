from rdkit import Chem

from molgpka import MolGpKa
from tools.service.molgpka_predictor import ResidentMolGpKa


def test_resident_molgpka_matches_package_predictor():
    molecule = Chem.MolFromSmiles("CCN(CC)CC")
    expected = MolGpKa(uncharged=True).predict(molecule)
    actual = ResidentMolGpKa(uncharged=True, max_concurrency=2).predict(molecule)

    assert actual.base_sites == expected.base_sites
    assert actual.acid_sites == expected.acid_sites
    assert Chem.MolToSmiles(actual.mol) == Chem.MolToSmiles(expected.mol)
