import json

import pytest

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    _parse_args as parse_batch_args,
    _single_run_command,
    _validate_analogous_reasoning_only as validate_batch_mode,
)
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_render import (
    build_group_messages,
    final_prompt_provenance,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch import CONFIG
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    _parse_args as parse_pipeline_args,
    _group_prompt_payload,
    _validate_analogous_reasoning_only as validate_pipeline_mode,
    final_messages,
)


@pytest.mark.parametrize(
    "extra_args",
    (
        [],
        [
            "--retrieval-strategy",
            "assay_transfer_tool",
            "--group-prompt-format",
            "assay_transfer_tool",
        ],
        [
            "--retrieval-strategy",
            "assay_transfer_tool",
            "--group-prompt-format",
            "assay_transfer_tool",
            "--assay-transfer-profile",
            "v9_direct_gold",
        ],
    ),
)
def test_mode_accepts_morgan_and_v9_transfer_profiles(extra_args):
    args = parse_pipeline_args(
        [
            "--experiment-mode",
            "full_mechanism",
            "--analogous-reasoning-only",
            *extra_args,
        ]
    )

    validate_pipeline_mode(args)


def test_batch_forwards_mode_and_allows_raw_retrieval_replay(tmp_path):
    replay_batch = tmp_path / "retrieval-source"
    args = parse_batch_args(
        CONFIG,
        [
            "--experiment-mode",
            "full_mechanism",
            "--analogous-reasoning-only",
            "--retrieval-replay-source-batch",
            str(replay_batch),
            "--skip-existing",
        ],
    )

    validate_batch_mode(CONFIG, args)
    command = _single_run_command(CONFIG, args, 3, "run-3", tmp_path)
    assert "--analogous-reasoning-only" in command
    assert "--retrieval-replay-run-dir" in command


@pytest.mark.parametrize(
    "reuse_args",
    (
        ["--single-analysis-source-batch", "old-batch"],
        ["--group-analysis-source-batch", "old-batch"],
        ["--prefetched-tool-replay-source-batch", "old-batch"],
        ["--final-only-source-batch", "old-batch"],
    ),
)
def test_batch_rejects_prior_reasoning_or_tool_replay(reuse_args):
    args = parse_batch_args(
        CONFIG,
        [
            "--experiment-mode",
            "full_mechanism",
            "--analogous-reasoning-only",
            *reuse_args,
        ],
    )

    with pytest.raises(SystemExit):
        validate_batch_mode(CONFIG, args)


@pytest.mark.parametrize(
    "forbidden_args",
    (
        ["--experiment-mode", "direct"],
        ["--single-analysis-source-run-dir", "old-run"],
        ["--group-analysis-source-run-dir", "old-run"],
        ["--prefetched-tool-replay-run-dir", "old-run"],
        ["--neighbor-context-profile", "coverage_mmp_ledger"],
        ["--enable-chembl-exact-context"],
        ["--final-evidence-surface", "summary_plus_cards"],
        ["--final-decision-profile", "train_ratio_tiebreak_v1"],
    ),
)
def test_pipeline_rejects_non_analog_or_prior_reasoning_inputs(forbidden_args):
    args = parse_pipeline_args(
        [
            "--experiment-mode",
            "full_mechanism",
            "--analogous-reasoning-only",
            *forbidden_args,
        ]
    )

    with pytest.raises(SystemExit):
        validate_pipeline_mode(args)


def test_final_prompt_contains_only_mechanism_branch_analyses():
    retrieval = {
        "query": {"input_smiles": "QUERY_SECRET_SMILES"},
        "coverage": {"n_neighbors_total": 7},
    }
    single = {
        "status": "omitted",
        "reason": "analogous_reasoning_only",
        "secret": "SINGLE_SECRET_PRIOR",
    }
    groups = [
        {
            "group_id": "Mechanism.absorption",
            "status": "ok",
            "llm": {
                "content": {"reasoning_summary": "ANALOG_BRANCH_TEXT"},
                "structured_output_validation": {"valid": True},
            },
        }
    ]

    messages = final_messages(
        retrieval,
        single,
        groups,
        analogous_reasoning_only=True,
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "ANALOG_BRANCH_TEXT" in prompt
    assert "QUERY_SECRET_SMILES" not in prompt
    assert "SINGLE_SECRET_PRIOR" not in prompt
    assert "n_neighbors_total" not in prompt
    assert "single_molecule_assessment" not in prompt
    assert "SINGLE-MOLECULE" not in prompt


def test_group_prompts_do_not_request_or_embed_query_tool_outputs():
    query = {"input_smiles": "CCO", "canonical_smiles": "CCO"}
    group = {
        "group_id": "Mechanism.absorption",
        "tier": "Mechanism",
        "endpoint_group": "absorption",
        "neighbors": [],
    }
    legacy = _group_prompt_payload(
        query,
        group,
        include_query_tool_guidance=False,
    )
    _, morgan_user = build_group_messages(
        query,
        group,
        prompt_format="morganfingerprint",
        options={"omit_query_tools": True},
    )
    rendered = json.dumps(legacy) + morgan_user["content"]

    assert "mmp_structure_compare" not in rendered
    assert "properties_compare" not in rendered
    assert "molecule_properties" not in rendered
    assert "prefetched_comparisons" not in rendered


def test_final_prompt_provenance_uses_distinct_immutable_contracts():
    standard = final_prompt_provenance(analogous_reasoning_only=False)
    analogous = final_prompt_provenance(analogous_reasoning_only=True)

    assert standard["profile"] == "standard"
    assert analogous["profile"] == "analogous_reasoning_only"
    assert standard["contract_sha256"] != analogous["contract_sha256"]
    assert len(analogous["instructions_sha256"]) == 64
    assert len(analogous["template_sha256"]) == 64
    assert len(analogous["schema_sha256"]) == 64
