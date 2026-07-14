import json

from tools.chembl_tool.paper_experiments.summarize_results import (
    _add_holm_adjusted_p,
    _count_identity_leaks,
    _audit_deployment_visibility,
    _llm_prompt_payloads,
    build_visibility_comparisons,
    macro_f1,
    mcnemar_exact_p,
    paired_bootstrap_delta_ci,
)


def test_macro_f1_binary():
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == (2 / 3 + 0.8) / 2


def test_mcnemar_exact_p_is_two_sided():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 5) == 0.0625
    assert mcnemar_exact_p(2, 2) == 1.0


def test_paired_bootstrap_delta_ci_preserves_pairing():
    labels = [0, 0, 1, 1]
    left = [1, 1, 0, 0]
    right = labels
    low, high = paired_bootstrap_delta_ci(labels, left, right, 200)
    assert low >= 0.5
    assert high <= 1.0


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    rows = [{"mcnemar_exact_p": value} for value in (0.01, 0.04, 0.03)]
    _add_holm_adjusted_p(rows)
    assert [row["mcnemar_holm_p"] for row in rows] == [0.03, 0.06, 0.06]


def test_llm_prompt_payloads_excludes_assistant_messages(tmp_path):
    output = {
        "llm": {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "invalid"},
                {"role": "user", "content": "retry"},
                {"role": "assistant", "content": "valid"},
            ]
        }
    }
    (tmp_path / "final_reasoning_output.json").write_text(json.dumps(output))

    [messages] = _llm_prompt_payloads(tmp_path)

    assert [message["content"] for message in messages] == ["system", "first", "retry"]


def test_trace_identity_audit_does_not_match_short_smiles_inside_words(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"content":"according to the evidence"}\n')

    assert _count_identity_leaks([{"smiles": "CCO", "trace_messages": str(trace)}]) == 0
    trace.write_text('{"content":"structure CCO was provided"}\n')
    assert _count_identity_leaks([{"smiles": "CCO", "trace_messages": str(trace)}]) == 1


def test_visibility_comparison_is_paired_by_query_index():
    blind_name = "bbb_martins__none"
    prefetched_name = "deployment_visible_prefetched__bbb_martins__none"
    visible_name = "deployment_visible__bbb_martins__none"
    prediction_sets = {
        blind_name: {
            0: {"label": 0, "pred_label": 1},
            1: {"label": 1, "pred_label": 0},
        },
        prefetched_name: {
            0: {"label": 0, "pred_label": 0},
            1: {"label": 1, "pred_label": 1},
        },
        visible_name: {
            0: {"label": 0, "pred_label": 1},
            1: {"label": 1, "pred_label": 1},
        },
    }

    rows = build_visibility_comparisons(prediction_sets, 200)
    row = next(
        item
        for item in rows
        if item["comparison_type"] == "identity_blind_vs_deployment_visible_prefetched"
    )

    assert row["condition"] == blind_name
    assert row["n_paired"] == 2
    assert row["delta_macro_f1"] == 1.0
    assert any(
        item["comparison_type"] == "deployment_visible_prefetched_vs_agentic"
        for item in rows
    )


def test_deployment_visibility_audit_checks_query_and_neighbor_contract(tmp_path):
    run_dir = tmp_path / "runs" / "run_0"
    run_dir.mkdir(parents=True)
    retrieval = {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "groups": [
            {
                "neighbors": [
                    {
                        "canonical_smiles": "CCN",
                        "molecule_chembl_id": "CHEMBL1",
                        "evidence_rows": [
                            {
                                "minimal_evidence": {
                                    "schema_version": "minimal_evidence.v1",
                                    "molecule": {
                                        "id": "CHEMBL1",
                                        "canonical_smiles": "CCN",
                                        "names": ["example drug"],
                                    },
                                    "group": {},
                                    "endpoint": {},
                                    "measurement": {},
                                    "text": {},
                                    "annotations": {},
                                    "quality": {},
                                    "provenance": {},
                                    "examples": [],
                                }
                            }
                        ],
                    }
                ]
            }
        ],
    }
    (run_dir / "retrieval.json").write_text(json.dumps(retrieval))
    output = {
        "llm": {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": {"canonical_smiles": "CCO"},
                            "neighbor": {
                                "canonical_smiles": "CCN",
                                "id": "CHEMBL1",
                                "name": "example drug",
                            },
                        }
                    ),
                }
            ]
        }
    }
    (run_dir / "final_reasoning_output.json").write_text(json.dumps(output))
    predictions = [{"run_dir": str(run_dir), "run_id": "run_0"}]

    audit = _audit_deployment_visibility(tmp_path, predictions)

    assert audit["deployment_visibility_audited_runs"] == 1
    assert audit["deployment_contract_satisfied_runs"] == 1
    assert audit["deployment_contract_failed_runs"] == 0
