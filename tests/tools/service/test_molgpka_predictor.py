from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem

from molgpka import MolGpKa
from tools.service.molgpka_predictor import ResidentMolGpKa


def test_resident_molgpka_matches_package_predictor():
    molecule = Chem.MolFromSmiles("CCN(CC)CC")
    expected = MolGpKa(uncharged=True).predict(molecule)
    actual = ResidentMolGpKa(uncharged=True).predict(molecule)

    assert actual.base_sites == expected.base_sites
    assert actual.acid_sites == expected.acid_sites
    assert Chem.MolToSmiles(actual.mol) == Chem.MolToSmiles(expected.mol)


def test_resident_molgpka_serializes_shared_model_inference():
    predictor = ResidentMolGpKa(uncharged=True)
    molecules = [
        Chem.MolFromSmiles("O=C(N[C@@H]1CC[C@@H](c2cccc(F)c2F)CN(CC(F)(F)F)C1=O)N1CCC(n2c(=O)[nH]c3ncccc32)CC1"),
        Chem.MolFromSmiles("Nc1nc2c(c(=O)[nH]1)N(C=O)[C@@H](CNc1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1)CN2"),
    ]
    expected = [predictor.predict(molecule) for molecule in molecules]
    with ThreadPoolExecutor(max_workers=16) as pool:
        observed = list(pool.map(lambda index: predictor.predict(molecules[index % 2]), range(32)))
    for index, prediction in enumerate(observed):
        reference = expected[index % 2]
        assert prediction.base_sites == reference.base_sites
        assert prediction.acid_sites == reference.acid_sites
