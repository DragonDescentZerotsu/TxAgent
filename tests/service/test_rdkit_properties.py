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
    assert "[functional_group_tree]" in output["text"]
    assert "primary hydroxyl: 1" in output["functional_group_tree"]
    assert "functional groups:" not in output["text"]
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


def test_pka_domain_failure_preserves_descriptors_and_does_not_poison_predictor():
    tool = MoleculePropertiesTool()
    tool.initialize(ServiceSettings(prewarm_molgpka=False))
    try:
        output = tool.invoke({
            "query_smiles": "CN(C)Cc1ccc[c-]1CNc1ccnc2cc(Cl)ccc12.[Fe+2].c1cc[cH-]c1",
        }, return_debug=True)
        features = {row["feature_name"]: row for row in output["features"]}
        assert output["raw_features"]["rdkit__MolWt"] > 0
        assert output["debug"]["pka"]["available"] is False
        assert output["_warnings"]
        for name, row in features.items():
            if name.startswith("pka__"):
                assert row["feature_value"] is None
                assert row["feature_value_missing_reason"] == "pka_unavailable"
        assert tool._pka_predictor is not None
        assert tool._pka_error is None
        ordinary = tool.invoke({"query_smiles": "CC(=O)O"}, return_debug=True)
        assert ordinary["debug"]["pka"]["available"] is True
        assert ordinary["raw_features"]["pka__most_acidic_pka"] is not None
    finally:
        tool.close()


def test_unexpected_pka_error_is_not_converted_to_missing(monkeypatch):
    import pytest
    tool = MoleculePropertiesTool()
    tool._pka_predictor = object()

    def fail(*args):
        raise IndexError("invalid graph indices")

    monkeypatch.setattr(tool, "_predict_pka", fail)
    with pytest.raises(IndexError, match="invalid graph indices"):
        tool._compute_pka_features("CCO", 7.4)
