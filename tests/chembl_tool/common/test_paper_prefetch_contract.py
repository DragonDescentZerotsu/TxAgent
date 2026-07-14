import json

from tools.chembl_tool.paper_experiments.audit_prefetch_contract import (
    _condition_coverage,
    prefetch_contract,
)


def test_prefetch_contract_ignores_visible_identity_but_compares_tool_results(tmp_path):
    def make_run(name, molecule_id):
        run = tmp_path / name
        run.mkdir()
        (run / "single_molecule_reasoning_output.json").write_text(
            json.dumps({"llm": {"tool_results": [{"tool_name": "molecule_properties", "status": "ok"}]}})
        )
        payload = {
            "group": {"group_id": "Direct.outcome"},
            "neighbors": [
                {
                    "rank": 1,
                    "molecule_chembl_id": molecule_id,
                    "canonical_smiles": "CCN",
                    "prefetched_comparisons": [
                        {"tool_name": "mmp_structure_compare", "status": "ok"},
                        {"tool_name": "properties_compare", "status": "ok"},
                    ],
                }
            ],
        }
        branch = {"group_id": "Direct.outcome", "llm": {"messages": [{"role": "user", "content": json.dumps(payload)}]}}
        (run / "group_reasoning_outputs.jsonl").write_text(json.dumps(branch) + "\n")
        return run

    blind = make_run("blind", "neighbor_1_1")
    visible = make_run("visible", "CHEMBL1")

    assert prefetch_contract(blind) == prefetch_contract(visible)


def test_prefetch_contract_detects_tool_output_difference(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "single_molecule_reasoning_output.json"
    path.write_text(json.dumps({"llm": {"tool_results": [{"tool_name": "molecule_properties", "content": "a"}]}}))
    (run / "group_reasoning_outputs.jsonl").write_text("")
    first = prefetch_contract(run)
    path.write_text(json.dumps({"llm": {"tool_results": [{"tool_name": "molecule_properties", "content": "b"}]}}))

    assert first != prefetch_contract(run)


def test_prefetch_contract_detects_retrieval_difference(tmp_path):
    def make_run(name, neighbor):
        run = tmp_path / name
        run.mkdir()
        (run / "retrieval.json").write_text(json.dumps({"status": "ok", "neighbor": neighbor}))
        (run / "single_molecule_reasoning_output.json").write_text(json.dumps({"llm": {}}))
        (run / "group_reasoning_outputs.jsonl").write_text("")
        return run

    assert prefetch_contract(make_run("a", "CHEMBL1")) != prefetch_contract(
        make_run("b", "CHEMBL2")
    )


def test_prefetch_contract_normalizes_identity_redaction_in_tool_text(tmp_path):
    retrieval = {
        "status": "ok",
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": "CHEMBL1",
                        "canonical_smiles": "CCCC(=O)O",
                        "evidence_rows": [],
                    }
                ]
            }
        ],
    }

    def make_run(name, content):
        run = tmp_path / name
        run.mkdir()
        (run / "retrieval.json").write_text(json.dumps(retrieval))
        (run / "single_molecule_reasoning_output.json").write_text(json.dumps({"llm": {}}))
        payload = {
            "group": {"group_id": "Flat.all_evidence"},
            "neighbors": [
                {
                    "rank": 1,
                    "prefetched_comparisons": [{"tool_name": "mmp_structure_compare", "content": content}],
                }
            ],
        }
        branch = {"llm": {"messages": [{"role": "user", "content": json.dumps(payload)}]}}
        (run / "group_reasoning_outputs.jsonl").write_text(json.dumps(branch) + "\n")
        return run

    blind = make_run("blind_redacted", "transformation: *[neighbor] -> *N")
    visible = make_run("visible_structure", "transformation: *CCCC(=O)O -> *N")

    assert prefetch_contract(blind) == prefetch_contract(visible)


def test_prefetch_contract_parses_legacy_input_json_prefix(tmp_path):
    run = tmp_path / "legacy_prompt"
    run.mkdir()
    (run / "single_molecule_reasoning_output.json").write_text(json.dumps({"llm": {}}))
    payload = {
        "group": {"group_id": "Direct.outcome"},
        "neighbors": [
            {
                "rank": 1,
                "prefetched_comparisons": [{"tool_name": "properties_compare", "status": "ok"}],
            }
        ],
    }
    branch = {
        "llm": {
            "messages": [
                {"role": "user", "content": "Input JSON:\n" + json.dumps(payload)},
                {"role": "user", "content": "Retry and return JSON."},
            ]
        }
    }
    (run / "group_reasoning_outputs.jsonl").write_text(json.dumps(branch) + "\n")

    contract = prefetch_contract(run)

    assert contract["groups"][0]["neighbors"][0]["prefetched_comparisons"] == [
        {"tool_name": "properties_compare", "status": "ok"}
    ]


def test_condition_coverage_rejects_missing_extra_and_mismatched_indices():
    row = _condition_coverage(
        "task__condition",
        "task",
        expected={0, 1, 2},
        blind={0, 1, 2, 4},
        visible_prefetched={0, 2},
        mismatched={2},
    )

    assert row["n_missing_blind"] == 0
    assert row["n_missing_visible_prefetched"] == 1
    assert row["n_extra_blind"] == 1
    assert row["n_mismatched"] == 1
    assert row["missing_visible_prefetched_indices"] == "1"
    assert row["mismatched_indices"] == "2"
    assert row["complete"] is False


def test_condition_coverage_accepts_full_matched_sets():
    row = _condition_coverage(
        "task__condition",
        "task",
        expected={0, 1},
        blind={0, 1},
        visible_prefetched={0, 1},
        mismatched=set(),
    )

    assert row["n_audited"] == 2
    assert row["complete"] is True
