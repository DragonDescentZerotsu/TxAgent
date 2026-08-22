import json

from tools.chembl_tool.common.identity_blind import (
    find_identity_blind_leaks,
    prepare_harness_prefetched_retrieval,
    prepare_identity_blind_final_retrieval,
    prepare_identity_blind_retrieval,
    prepare_replayed_prefetched_retrieval,
    prepare_reasoning_retrieval,
    query_without_prefetched_tools,
    sanitize_identity_blind_branch_outputs,
)


class FakeToolService:
    def __init__(self):
        self.calls = []
        self.batch_calls = 0

    def invoke(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {
            "tool_name": tool_name,
            "status": "ok",
            "content": f"{tool_name}: " + " | ".join(str(value) for value in arguments.values()),
            "warnings": [f"warning for {str(arguments.get('query_smiles', '')).lower()}"],
        }

    def invoke_many(self, calls):
        self.batch_calls += 1
        return [self.invoke(tool_name, arguments) for tool_name, arguments in calls]


def test_identity_blind_prefetch_removes_query_and_neighbor_structures():
    retrieval = {
        "status": "ok",
        "query": {
            "input_smiles": "CCO",
            "canonical_smiles": "CCO",
            "external_condition": "This prediction concerns a fasted state.",
        },
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
                        "transfer_selected_records": [
                            {
                                "record_rank": 1,
                                "transfer_selection_score": 0.9,
                                "transfer_winning_record_id": "record-identity",
                                "transfer_winning_record": {
                                    "canonical_smiles": "CCN",
                                    "source_fields": {
                                        "compound_name": "ExampleDrug",
                                        "endpoint_name": "outcome",
                                    },
                                },
                            }
                        ],
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
    assert output["query"]["external_condition"] == (
        "This prediction concerns a fasted state."
    )
    assert output["groups"][0]["identity_blind"] is True
    assert output["groups"][0]["neighbors"][0]["molecule_chembl_id"] == "neighbor_1_1"
    assert "CCO" not in serialized
    assert "CCN" not in serialized
    assert "CHEMBL1" not in serialized
    assert "ExampleDrug" not in serialized
    assert '"ED"' not in serialized
    assert "[neighbor] ([neighbor]) had a measured outcome" in serialized


def test_visible_prefetch_preserves_identity_but_matches_blind_tool_calls():
    retrieval = {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "groups": [
            {
                "group_id": "Direct.outcome",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                        "evidence_rows": [{"source_molecule_names": ["VisibleNeighbor"]}],
                    }
                ],
            }
        ],
    }
    blind_service = FakeToolService()
    visible_service = FakeToolService()

    blind = prepare_harness_prefetched_retrieval(
        retrieval, blind_service, identity_blind=True
    )
    visible = prepare_harness_prefetched_retrieval(
        retrieval, visible_service, identity_blind=False
    )

    assert blind_service.calls == visible_service.calls
    assert blind_service.batch_calls == visible_service.batch_calls == 1
    assert [name for name, _ in visible_service.calls] == [
        "molecule_properties",
        "mmp_structure_compare",
        "properties_compare",
    ]
    assert visible["query"]["canonical_smiles"] == "CCO"
    assert visible["query"]["tools_prefetched"] is True
    assert visible["groups"][0]["tools_prefetched"] is True
    assert visible["groups"][0]["neighbors"][0]["canonical_smiles"] == "CCN"
    assert visible["groups"][0]["neighbors"][0]["molecule_chembl_id"] == "CHEMBL1"
    assert "VisibleNeighbor" in json.dumps(visible)
    assert blind["query"]["identity_hidden"] is True


def test_identity_blind_prefetch_redacts_backend_error_details():
    class ErrorToolService(FakeToolService):
        def invoke(self, tool_name, arguments):
            self.calls.append((tool_name, arguments))
            if tool_name == "mmp_structure_compare":
                return {
                    "tool_name": tool_name,
                    "status": "error",
                    "content": "AssertionError: *CCO.*CCN",
                    "warnings": ["raw backend warning for CCO"],
                    "errors": [
                        {
                            "code": "TOOL_RUNTIME_ERROR",
                            "message": "AssertionError: *CCO.*CCN",
                            "recoverable": False,
                        }
                    ],
                }
            return super().invoke(tool_name, arguments)

    retrieval = {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "groups": [
            {
                "group_id": "Direct.outcome",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                        "evidence_rows": [],
                    }
                ],
            }
        ],
    }

    output = prepare_identity_blind_retrieval(retrieval, ErrorToolService())
    receipt = output["groups"][0]["neighbors"][0]["prefetched_comparisons"][0]
    serialized = json.dumps(receipt)
    assert receipt["status"] == "error"
    assert receipt["errors"][0]["code"] == "TOOL_RESULT_UNAVAILABLE"
    assert "AssertionError" not in serialized
    assert "CCO" not in serialized
    assert "CCN" not in serialized


def test_query_tool_omission_never_contacts_tool_service():
    retrieval = {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "groups": [
            {
                "group_id": "Mechanism.absorption",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                        "evidence_rows": [],
                    }
                ],
            }
        ],
    }

    class NoToolService:
        def invoke(self, *args, **kwargs):
            raise AssertionError((args, kwargs))

        def invoke_many(self, *args, **kwargs):
            raise AssertionError((args, kwargs))

    visible = prepare_reasoning_retrieval(
        retrieval,
        NoToolService(),
        identity_blind=False,
        harness_prefetch_tools=True,
        include_query_tools=False,
    )
    blind = prepare_reasoning_retrieval(
        retrieval,
        NoToolService(),
        identity_blind=True,
        harness_prefetch_tools=True,
        include_query_tools=False,
    )

    assert visible["experiment"]["tool_execution_mode"] == "omitted"
    assert "prefetched_molecule_properties" not in visible["query"]
    assert "prefetched_comparisons" not in visible["groups"][0]["neighbors"][0]
    assert blind["experiment"]["tool_execution_mode"] == "omitted"
    assert blind["query"] == {"molecule_id": "query", "identity_hidden": True}


def test_flat_tool_omission_keeps_query_properties_but_skips_neighbor_comparisons():
    retrieval = {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "experiment": {"mode": "full_flat"},
        "groups": [
            {
                "group_id": "Flat.all_evidence",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                        "evidence_rows": [],
                    }
                ],
            }
        ],
    }
    service = FakeToolService()
    output = prepare_reasoning_retrieval(
        retrieval,
        service,
        identity_blind=True,
        harness_prefetch_tools=False,
        include_query_tools=True,
        include_neighbor_tools=False,
    )

    assert [name for name, _ in service.calls] == ["molecule_properties"]
    assert output["query"]["tools_prefetched"] is True
    assert output["groups"][0].get("tools_prefetched") is None
    assert "prefetched_comparisons" not in output["groups"][0]["neighbors"][0]
    assert output["experiment"]["tool_execution_mode"] == "harness_prefetch_query_only"
    group_query = query_without_prefetched_tools(output["query"])
    assert group_query == {"molecule_id": "query", "identity_hidden": True}


def test_prefetched_tool_replay_keeps_visible_identity(tmp_path):
    retrieval = {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "experiment": {},
        "groups": [
            {
                "group_id": "Direct.outcome",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCN",
                    }
                ],
            }
        ],
    }
    (tmp_path / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"llm": {"tool_results": [{"tool_name": "molecule_properties", "content": "frozen"}]}})
    )
    payload = {
        "group": {"group_id": "Direct.outcome"},
        "neighbors": [
            {
                "rank": 1,
                "prefetched_comparisons": [
                    {"tool_name": "mmp_structure_compare", "content": "frozen comparison"}
                ],
            }
        ],
    }
    branch = {"llm": {"messages": [{"role": "user", "content": "Input JSON:\n" + json.dumps(payload)}]}}
    (tmp_path / "group_reasoning_outputs.jsonl").write_text(json.dumps(branch) + "\n")

    output = prepare_replayed_prefetched_retrieval(retrieval, str(tmp_path))

    neighbor = output["groups"][0]["neighbors"][0]
    assert neighbor["molecule_chembl_id"] == "CHEMBL1"
    assert neighbor["canonical_smiles"] == "CCN"
    assert neighbor["prefetched_comparisons"][0]["content"] == "frozen comparison"
    assert output["query"]["prefetched_molecule_properties"]["content"] == "frozen"


def test_final_only_redaction_reuses_saved_property_result():
    retrieval = {
        "query": {
            "input_smiles": "CCO",
            "external_condition": "This prediction concerns a fasted state.",
        },
        "groups": [],
        "coverage": {},
    }
    single_output = {"llm": {"tool_results": [{"tool_name": "molecule_properties", "content": "hidden"}]}}

    output = prepare_identity_blind_final_retrieval(retrieval, single_output)

    assert output["query"]["identity_hidden"] is True
    assert output["query"]["prefetched_molecule_properties"]["content"] == "hidden"
    assert output["query"]["external_condition"] == (
        "This prediction concerns a fasted state."
    )
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


def test_branch_sanitization_scales_to_dense_assay_trace_without_mutating_source():
    names = [f"SourceDrug{index:03d}" for index in range(250)]
    retrieval = {
        "query": {"input_smiles": "CCO"},
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": f"CHEMBL{index}",
                        "canonical_smiles": f"CCN{'C' * index}",
                        "evidence_rows": [
                            {
                                "source_molecule_names": [name],
                                "standard_type": "outcome",
                            }
                        ],
                    }
                    for index, name in enumerate(names, start=1)
                ]
            }
        ],
    }
    trace = " | ".join(names) * 20
    raw = [{"status": "ok", "llm": {"messages": [trace], "content": {"summary": trace}}}]

    sanitized = sanitize_identity_blind_branch_outputs(raw, retrieval)

    assert names[0] in raw[0]["llm"]["messages"][0]
    assert not any(name in json.dumps(sanitized) for name in names)
    assert sanitized[0]["identity_blind_sanitization"] == {
        "applied": True,
        "changed": True,
        "n_sensitive_terms": 751,
    }


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


def test_identity_blind_leak_finder_uses_identifier_boundaries_and_ignores_generic_names():
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
                                "source_molecule_names": ["NET", "PER"],
                            }
                        ],
                    }
                ]
            }
        ],
    }

    leaks = find_identity_blind_leaks(
        retrieval,
        "assay CHEMBL876624; net effect per oral dose",
    )

    assert leaks == {"structures": [], "identifiers": [], "names": []}


def test_identity_blind_leak_finder_ignores_four_letter_name_inside_common_word():
    """Regression guard: a 4-letter abbreviation like "SCOP" (e.g. scopolamine)
    must not falsely match as a substring of an unrelated word like "scope"."""
    retrieval = {
        "query": {},
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": "CHEMBL9",
                        "canonical_smiles": "CCO",
                        "evidence_rows": [
                            {
                                "molecule_chembl_id": "CHEMBL9",
                                "canonical_smiles": "CCO",
                                "group_id": "Direct.outcome",
                                "standard_type": "outcome",
                                "source_molecule_names": ["SCOP"],
                            }
                        ],
                    }
                ]
            }
        ],
    }

    leaks = find_identity_blind_leaks(
        retrieval, "the analysis is beyond the scope of available evidence"
    )

    assert leaks == {"structures": [], "identifiers": [], "names": []}
    assert find_identity_blind_leaks(retrieval, "SCOP was administered")["names"] == ["SCOP"]


def test_identity_blind_leak_finder_ignores_schema_key_alias_collision():
    retrieval = {
        "query": {},
        "groups": [
            {
                "neighbors": [
                    {
                        "evidence_rows": [
                            {
                                "source_molecule_names": ["FA"],
                            }
                        ]
                    }
                ]
            }
        ],
    }
    payload = {
        "mechanism_family_mapping": {
            "Fa.absorption_solubility_permeability": [
                "[neighbor].absorption_solubility_permeability"
            ]
        }
    }
    assert find_identity_blind_leaks(retrieval, payload)["names"] == []

    payload["evidence_text"] = "The result was reported for FA."
    assert find_identity_blind_leaks(retrieval, payload)["names"] == ["FA"]


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


def test_identity_blind_single_atom_smiles_does_not_corrupt_tool_text():
    class DescriptiveToolService(FakeToolService):
        def invoke(self, tool_name, arguments):
            self.calls.append((tool_name, arguments))
            return {
                "tool_name": tool_name,
                "status": "ok",
                "content": "Neutral fraction: 1\nNumber of nitrogen atoms: 1",
                "warnings": [],
            }

    retrieval = {
        "query": {
            "input_smiles": "N",
            "canonical_smiles": "N",
            "standard_inchi_key": "QGZKDVFQNNGYKY-UHFFFAOYSA-N",
        },
        "groups": [],
    }

    output = prepare_identity_blind_retrieval(retrieval, DescriptiveToolService())
    content = output["query"]["prefetched_molecule_properties"]["content"]

    assert output["query"]["identity_hidden"] is True
    assert content == "Neutral fraction: 1\nNumber of nitrogen atoms: 1"
    assert find_identity_blind_leaks(retrieval, output)["structures"] == []


def test_identity_blind_two_character_single_atom_smiles_does_not_flag_assay_text():
    retrieval = {
        "query": {"input_smiles": "C[Se]", "canonical_smiles": "C[Se]"},
        "groups": [
            {
                "neighbors": [
                    {
                        "canonical_smiles": "Cl",
                        "evidence_rows": [],
                    }
                ]
            }
        ],
    }

    assert find_identity_blind_leaks(
        retrieval, "The rate of 36Cl influx was measured."
    ) == {"structures": [], "identifiers": [], "names": []}
    assert find_identity_blind_leaks(
        retrieval, 'query structure "C[Se]"'
    )["structures"] == ["C[Se]"]
