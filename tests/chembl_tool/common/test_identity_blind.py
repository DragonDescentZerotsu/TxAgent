import json

from tools.chembl_tool.common.identity_blind import (
    find_identity_blind_leaks,
    prepare_identity_blind_final_retrieval,
    prepare_identity_blind_retrieval,
    sanitize_identity_blind_branch_outputs,
)


class FakeToolService:
    def invoke(self, tool_name, arguments):
        return {
            "tool_name": tool_name,
            "status": "ok",
            "content": f"{tool_name}: " + " | ".join(str(value) for value in arguments.values()),
            "warnings": [f"warning for {str(arguments.get('query_smiles', '')).lower()}"],
        }


def test_identity_blind_prefetch_removes_query_and_neighbor_structures():
    retrieval = {
        "status": "ok",
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "experiment": {"mode": "direct"},
        "groups": [
            {
                "group_id": "Direct.outcome",
                "tier": "Direct",
                "endpoint_group": "outcome",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                        "similarity": 0.7,
                        "similarity_bucket": "moderate_analog",
                        "evidence_rows": [
                            {
                                "molecule_chembl_id": "CHEMBL1",
                                "canonical_smiles": "CCN",
                                "group_id": "Direct.outcome",
                                "standard_type": "outcome",
                                "standard_value": 1,
                                "source_molecule_names": ["ExampleDrug", "ED"],
                                "assay_description": "ExampleDrug (ED) had a measured outcome.",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    output = prepare_identity_blind_retrieval(retrieval, FakeToolService())
    serialized = json.dumps(output)

    assert output["query"]["identity_hidden"] is True
    assert output["groups"][0]["identity_blind"] is True
    assert output["groups"][0]["neighbors"][0]["molecule_chembl_id"] == "neighbor_1_1"
    assert "CCO" not in serialized
    assert "CCN" not in serialized
    assert "CHEMBL1" not in serialized
    assert "ExampleDrug" not in serialized
    assert '"ED"' not in serialized
    assert "[neighbor] ([neighbor]) had a measured outcome" in serialized


def test_final_only_redaction_reuses_saved_property_result():
    retrieval = {"query": {"input_smiles": "CCO"}, "groups": [], "coverage": {}}
    single_output = {"llm": {"tool_results": [{"tool_name": "molecule_properties", "content": "hidden"}]}}

    output = prepare_identity_blind_final_retrieval(retrieval, single_output)

    assert output["query"]["identity_hidden"] is True
    assert output["query"]["prefetched_molecule_properties"]["content"] == "hidden"
    assert "CCO" not in json.dumps(output)


def test_branch_sanitization_removes_model_inferred_neighbor_name():
    retrieval = {
        "query": {"input_smiles": "CCO"},
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                        "evidence_rows": [
                            {
                                "group_id": "Direct.outcome",
                                "standard_type": "outcome",
                                "standard_value": 1,
                                "source_molecule_names": ["ExampleDrug", "ED"],
                            }
                        ],
                    }
                ]
            }
        ],
    }
    raw = [{"status": "ok", "llm": {"content": {"reasoning_summary": "This is ExampleDrug (ED)."}}}]

    sanitized = sanitize_identity_blind_branch_outputs(raw, retrieval)

    assert "ExampleDrug" in json.dumps(raw)
    assert "ExampleDrug" not in json.dumps(sanitized)
    assert "(ED)" not in json.dumps(sanitized)
    assert sanitized[0]["identity_blind_sanitization"]["changed"] is True


def test_identity_blind_leak_finder_reports_categories():
    retrieval = {
        "query": {
            "input_smiles": "C[C@H](O)Cl",
            "canonical_smiles": "C[C@H](O)Cl",
            "standard_inchi_key": "QUERY-INCHIKEY",
        },
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": "CHEMBL123",
                        "canonical_smiles": "CCN(CC)CC",
                        "standard_inchi_key": "NEIGHBOR-INCHIKEY",
                        "evidence_rows": [
                            {
                                "molecule_chembl_id": "CHEMBL123",
                                "canonical_smiles": "CCN(CC)CC",
                                "group_id": "Direct.outcome",
                                "standard_type": "outcome",
                                "source_molecule_names": ["ExampleDrug"],
                            }
                        ],
                    }
                ]
            }
        ],
    }
    payload = "C[C@H](O)Cl CHEMBL123 ExampleDrug"

    leaks = find_identity_blind_leaks(retrieval, payload)

    assert leaks == {
        "structures": ["C[C@H](O)Cl"],
        "identifiers": ["CHEMBL123"],
        "names": ["ExampleDrug"],
    }


def test_identity_blind_leak_finder_uses_identifier_boundaries_and_ignores_generic_net():
    retrieval = {
        "query": {},
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": "CHEMBL8",
                        "canonical_smiles": "CCN",
                        "evidence_rows": [
                            {
                                "molecule_chembl_id": "CHEMBL8",
                                "canonical_smiles": "CCN",
                                "group_id": "Direct.outcome",
                                "standard_type": "outcome",
                                "source_molecule_names": ["NET"],
                            }
                        ],
                    }
                ]
            }
        ],
    }

    leaks = find_identity_blind_leaks(retrieval, "assay CHEMBL876624; net oral effect")

    assert leaks == {"structures": [], "identifiers": [], "names": []}


def test_identity_blind_leak_finder_does_not_match_short_smiles_inside_words():
    retrieval = {
        "query": {"input_smiles": "CCC(C)(C)O"},
        "groups": [{"neighbors": [{"canonical_smiles": "CCO", "evidence_rows": []}]}],
    }

    assert find_identity_blind_leaks(retrieval, "according to the evidence") == {
        "structures": [],
        "identifiers": [],
        "names": [],
    }
    assert find_identity_blind_leaks(retrieval, 'structure "CCO"') ["structures"] == ["CCO"]
