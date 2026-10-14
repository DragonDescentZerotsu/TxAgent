from __future__ import annotations

from tools.service.config import ServiceSettings
from tools.service.tools.rdkit_properties import MoleculePropertiesTool


def test_molecule_properties_returns_core_rdkit_features_without_pka_initialization():
    tool = MoleculePropertiesTool()
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))

    output = tool.invoke({"query_smiles": "CCO"})
    features = {feature["feature_name"]: feature for feature in output["features"]}

    assert output["query"]["canonical_smiles"] == "CCO"
    assert features["rdkit__MolWt"]["display_name"] == "molecular weight"
    assert features["rdkit__MolWt"]["feature_value"] == 46.07
    assert features["rdkit__NumHDonors"]["feature_value"] == 1.0
    assert features["pka__most_acidic_pka"]["feature_value_text"] == "not applicable (pKa predictor unavailable)"
    assert any(group["name"] == "primary hydroxyl" for group in output["functional_groups"])
    assert output["present_functional_groups"] == output["functional_groups"]
    assert "functional groups:" in output["text"]
    assert "molecular weight:" in output["text"]


def test_molecule_properties_reuses_canonical_result_inside_pair_comparisons(monkeypatch):
    tool = MoleculePropertiesTool()
    tool.initialize(
        ServiceSettings(
            enable_molgpka=False,
            prewarm_molgpka=False,
            cache_memory_entries=100,
        )
    )
    original = tool._compute_rdkit_features
    calls = 0

    def counted(mol):
        nonlocal calls
        calls += 1
        return original(mol)

    monkeypatch.setattr(tool, "_compute_rdkit_features", counted)
    first = tool.invoke({"query_smiles": "CCO"})
    second = tool.invoke({"query_smiles": "OCC"})

    assert calls == 1
    assert first["raw_features"] == second["raw_features"]
    assert second["query"]["input_smiles"] == "OCC"
