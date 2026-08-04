from __future__ import annotations

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
