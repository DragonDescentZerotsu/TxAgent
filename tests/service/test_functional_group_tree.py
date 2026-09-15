from concurrent.futures import ThreadPoolExecutor

from tools.service.config import ServiceSettings
from tools.service.registry import ToolRegistry
from tools.service.schemas import ToolRequest
from tools.service.tools.properties_compare import PropertiesCompareTool
from tools.service.tools.rdkit_properties import MoleculePropertiesTool


def test_other_instances_cannot_hide_a_nested_match():
    import networkx as nx
    from tools.service.functional_group_tree import render_functional_group_tree

    graph = nx.DiGraph()
    graph.add_node("A", mapped_atoms=[(0, 1, 2)])
    graph.add_node("B", mapped_atoms=[(0, 1), (3, 4)])
    graph.add_node("C", mapped_atoms=[(2,), (3,)])
    graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    # B inside A does not contain C. A different B elsewhere does contain C.
    tree = render_functional_group_tree({"A": [(0, 1, 2)], "B": [(3, 4)]}, graph)
    assert tree == "├──A: 1\n│  ├──B: 1\n│  └──C: 1\n└──B: 1\n   └──C: 1"


def test_child_counts_exclude_matches_outside_the_parent_branch():
    from accfg import AccFG
    from tools.service.functional_group_tree import render_functional_group_tree

    detector = AccFG(print_load_info=False)
    groups, graph = detector.run("CC(=O)OCCOCC", show_atoms=True, show_graph=True, canonical=True)
    tree = render_functional_group_tree(groups, graph)
    assert "carboxylic ester: 1" in tree
    assert "dialkyl ether: 1" in tree
    assert "│  ├──ether: 1" in tree
    assert "ether: 2" not in tree
    groups, graph = detector.run("CCC", show_atoms=True, show_graph=True, canonical=True)
    assert render_functional_group_tree(groups, graph).startswith("No functional groups matched")


def test_nested_epoxide_survives_and_parallel_batch_results_stay_molecule_specific(tmp_path):
    properties = MoleculePropertiesTool()
    registry = ToolRegistry([properties, PropertiesCompareTool(properties)])
    registry.initialize_all(ServiceSettings(
        enable_molgpka=False, prewarm_molgpka=False, batch_workers=4,
        cache_path=tmp_path / "cache.sqlite3", cache_memory_entries=100,
    ))
    epoxide = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C[C@@H]4O[C@@]45C[C@@H](O)CC[C@]5(C)[C@H]3CC[C@]12C"
    smiles = ["CCO", "CCN", epoxide]
    requests = [ToolRequest(tool_name="molecule_properties", input={"query_smiles": s}) for s in smiles]
    requests.append(ToolRequest(tool_name="properties_compare", input={"query_smiles": "CCN", "reference_smiles": epoxide}))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(lambda _: registry.invoke_many(requests), range(8)))
        expected = [r.output for r in batches[0]]
        for batch in batches:
            assert all(r.status == "ok" for r in batch)
            assert [r.output for r in batch] == expected
        alcohol, amine, epoxy, comparison = expected
        assert "primary hydroxyl: 1" in alcohol["functional_group_tree"]
        assert "primary aliphatic amine: 1" in amine["functional_group_tree"]
        assert "epoxide: 1" in epoxy["functional_group_tree"]
        assert "oxepane: 1" in epoxy["functional_group_tree"]
        assert "((" not in epoxy["functional_group_tree"]
        assert comparison["reference_functional_group_tree"] == epoxy["functional_group_tree"]
        assert all(r.metadata.cache_hit for r in registry.invoke_many(requests))
    finally:
        registry.close()


def test_tree_failure_is_explicit_and_does_not_poison_other_molecules(monkeypatch):
    tool = MoleculePropertiesTool()
    tool.initialize(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))
    original = tool._fg_detector.run

    def fail_one(smiles, **kwargs):
        if smiles == "CCO":
            raise ValueError("deliberate test failure")
        return original(smiles, **kwargs)

    monkeypatch.setattr(tool._fg_detector, "run", fail_one)
    try:
        failed = tool.invoke({"query_smiles": "CCO"}, return_debug=True)
        assert failed["raw_features"]["rdkit__MolWt"] > 0
        assert failed["functional_group_tree"].startswith("Unavailable")
        assert failed["debug"]["functional_groups_available"] is False
        assert any("deliberate test failure" in w for w in failed["_warnings"])
        good = tool.invoke({"query_smiles": "CCN"})
        assert "amine: 1" in good["functional_group_tree"]
        assert not any("deliberate test failure" in w for w in good["_warnings"])
    finally:
        tool.close()
