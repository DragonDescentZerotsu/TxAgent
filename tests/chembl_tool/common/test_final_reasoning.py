from tools.chembl_tool.common.final_reasoning import (
    compact_group_reasoning_outputs,
    prepare_resumed_final_inputs,
)


def test_compact_group_reasoning_outputs_keeps_only_validated_content():
    valid = {
        "group_id": "direct",
        "status": "ok",
        "llm": {
            "content": {"reasoning_summary": "supported"},
            "structured_output_validation": {"valid": True},
        },
        "internal": "not prompt visible",
    }
    invalid = {
        "group_id": "mechanism",
        "status": "error",
        "llm": {"content": {"reasoning_summary": "ignore"}},
    }

    assert compact_group_reasoning_outputs([valid, invalid]) == [
        {
            "group_id": "direct",
            "status": "ok",
            "content": {"reasoning_summary": "supported"},
        },
        {"group_id": "mechanism", "status": "error", "content": None},
    ]


def test_prepare_resumed_visible_inputs_is_noop_without_prefetch():
    retrieval = {"query": {"canonical_smiles": "CCO"}, "groups": []}
    groups = [{"group_id": "direct", "status": "ok"}]

    prepared_retrieval, prepared_groups = prepare_resumed_final_inputs(
        retrieval,
        {},
        groups,
        {"identity_blind": False, "harness_prefetch_tools": False},
    )

    assert prepared_retrieval is retrieval
    assert prepared_groups is groups


def test_prepare_resumed_prefetched_inputs_restores_single_tool_result():
    retrieval = {"query": {"canonical_smiles": "CCO"}, "groups": []}
    single = {"llm": {"tool_results": [{"text": "properties"}]}}

    prepared, _ = prepare_resumed_final_inputs(
        retrieval,
        single,
        [],
        {"identity_blind": False, "harness_prefetch_tools": True},
    )

    assert prepared["query"]["canonical_smiles"] == "CCO"
    assert prepared["query"]["prefetched_molecule_properties"] == {
        "text": "properties"
    }
    assert prepared["experiment"]["tool_execution_mode"] == "harness_prefetch"
