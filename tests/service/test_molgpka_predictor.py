from rdkit import Chem
from concurrent.futures import ThreadPoolExecutor

from molgpka import MolGpKa
from tools.service.molgpka_predictor import ResidentMolGpKa


def test_resident_molgpka_matches_package_predictor():
    molecule = Chem.MolFromSmiles("CCN(CC)CC")
    expected = MolGpKa(uncharged=True).predict(molecule)
    actual = ResidentMolGpKa(uncharged=True).predict(molecule)

    assert actual.base_sites == expected.base_sites
    assert actual.acid_sites == expected.acid_sites
    assert Chem.MolToSmiles(actual.mol) == Chem.MolToSmiles(expected.mol)


def test_shared_predictor_concurrent_different_graphs_matches_serial():
    predictor = ResidentMolGpKa()
    smiles = ["CCN(CC)CC", "CC(=O)O", "Oc1ccccc1", "NCC(=O)O",
              "CCCCN1CCC(COC(=O)c2ccc(N)c(OC)c2)CC1"]

    def predict(smiles):
        result = predictor.predict(Chem.MolFromSmiles(smiles))
        return result.base_sites, result.acid_sites

    expected = {s: predict(s) for s in smiles}
    inputs = smiles * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(predict, inputs))
    assert actual == [expected[s] for s in inputs]
