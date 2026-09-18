from __future__ import annotations

from importlib import resources
from threading import Lock

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


class ResidentMolGpKa:
    """MolGpKa-compatible predictor that loads the two GNN weights once."""

    def __init__(self, *, uncharged: bool = True) -> None:
        from molgpka.predict_pka import _resource_path, load_model

        self.uncharged = uncharged
        # MolGpKa's GCN layers reuse mutable normalization buffers even when
        # configured with cached=False. A shared model therefore cannot run
        # two molecule graphs concurrently without mixing their edge indices.
        self._model_lock = Lock()
        with resources.as_file(_resource_path("models", "weight_base.pth")) as path:
            self._base_model = load_model(path)
        with resources.as_file(_resource_path("models", "weight_acid.pth")) as path:
            self._acid_model = load_model(path)

    def predict(self, mol: Chem.Mol):
        from molgpka.api import PkaPrediction
        from molgpka.predict_pka import model_pred
        from molgpka.utils.ionization_group import get_ionization_aid

        if self.uncharged:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        mol = AllChem.AddHs(mol)
        with self._model_lock:
            base = {
                index + 1: model_pred(mol, index, self._base_model)
                for index in get_ionization_aid(mol, acid_or_base="base")
            }
            acid = {
                index + 1: model_pred(mol, index, self._acid_model)
                for index in get_ionization_aid(mol, acid_or_base="acid")
            }
        numbered = Chem.Mol(mol)
        for atom in numbered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        return PkaPrediction(base_sites=base, acid_sites=acid, mol=numbered)
