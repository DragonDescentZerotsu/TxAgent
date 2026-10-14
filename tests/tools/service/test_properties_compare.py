from __future__ import annotations

from tools.service.config import ServiceSettings
from tools.service.tools.properties_compare import PropertiesCompareTool
from tools.service.tools.rdkit_properties import CORE_FEATURE_COLUMNS, MoleculePropertiesTool


def test_properties_compare_compares_all_non_fg_molecule_properties():
    properties_tool = MoleculePropertiesTool()
    properties_tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))
    tool = PropertiesCompareTool(properties_tool)
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))

    output = tool.invoke({"query_smiles": "CCN", "reference_smiles": "CCO"})
    comparisons = {row["feature_name"]: row for row in output["feature_comparisons"]}

    assert len(output["feature_comparisons"]) == len(CORE_FEATURE_COLUMNS)
    assert "functional_groups" not in output
    assert comparisons["rdkit__MolWt"]["query_value"] == 45.08
    assert comparisons["rdkit__MolWt"]["reference_value"] == 46.07
    assert comparisons["rdkit__MolWt"]["delta_value"] == -0.99
    assert comparisons["rdkit__MolWt"]["delta_text"] == "-0.99"
    assert comparisons["pka__most_acidic_pka"]["delta_text"] == "not applicable"
    assert "molecular weight: query=45.08 | reference=46.07 | delta=-0.99" in output["text"]

