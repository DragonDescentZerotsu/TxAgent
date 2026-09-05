from __future__ import annotations

from rdkit import Chem

from tools.service.config import ServiceSettings
from tools.service.tools.mmp_structure_compare import MmpStructureCompareTool


def test_mmp_structure_compare_returns_similarity_mcs_without_property_deltas():
    tool = MmpStructureCompareTool()
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))

    output = tool.invoke({"query_smiles": "CCN", "reference_smiles": "CCO"})

    assert output["query"]["canonical_smiles"] == "CCN"
    assert output["reference"]["canonical_smiles"] == "CCO"
    assert output["similarity"]["fingerprint"]["type"] == "Morgan"
    assert output["similarity"]["similarity_bucket"] == "distant_analog"
    assert output["similarity"]["tanimoto"] == 0.33
    assert output["mcs"]["num_common_atoms"] >= 2
    assert "descriptor_deltas" not in output
    assert "Morgan fingerprint Tanimoto similarity" in output["text"]
    assert "descriptor deltas" not in output["text"]


def test_mmp_structure_compare_reuses_molecule_fragmentations(monkeypatch):
    tool = MmpStructureCompareTool()
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))
    original = tool._fragmentations_uncached
    calls = 0

    def counted(mol, *, limit):
        nonlocal calls
        calls += 1
        return original(mol, limit=limit)

    monkeypatch.setattr(tool, "_fragmentations_uncached", counted)
    tool.invoke({"query_smiles": "CCN", "reference_smiles": "CCO"})
    tool.invoke({"query_smiles": "CCN", "reference_smiles": "CCC"})

    assert calls == 3


def test_disconnected_reference_preserves_similarity_and_mcs():
    tool = MmpStructureCompareTool()
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))
    try:
        output = tool.invoke({
            "query_smiles": "O=C(O)CCCC[C@@H]1CCSS1",
            "reference_smiles": "O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21.O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21",
        })
        assert output["similarity"]["tanimoto"] > 0
        assert output["mcs"]["num_common_atoms"] > 0
        assert output["matched_pair"]["matched_pair_found"] is False
        assert "not applicable" in output["text"]
    finally:
        tool.close()


def test_fragment_component_gate_runs_after_salt_normalization():
    tool = MmpStructureCompareTool()
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))
    try:
        single = tool._fragmentations(Chem.MolFromSmiles("CCO"))
        salted = tool._fragmentations(Chem.MolFromSmiles("CCO.[Na+]"))
        assert single
        assert salted == single
    finally:
        tool.close()
