import json

from tools.chembl_tool.paper_experiments.summarize_results import (
    _add_holm_adjusted_p,
    _count_identity_leaks,
    _llm_prompt_payloads,
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
