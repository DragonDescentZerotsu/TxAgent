import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.final_decision_prior import (
    STANDARD_FINAL_DECISION,
    TRAIN_RATIO_TIEBREAK_V1,
    TrainRatioPrior,
    build_final_decision_prompt,
    final_decision_validation_errors,
)
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    _validate_final_decision_profile_contract,
)
from tools.chembl_tool.paper_experiments.run_train_ratio_prior_experiment import (
    TaskSpec,
    _batch_command,
)
from tools.chembl_tool.paper_experiments.train_ratio_prior_analysis import (
    TrainRatioAnalysisSpec,
    analyze_train_ratio_prior,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    _run_final_reasoning as run_bbb_final,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    _run_final_reasoning as run_bio_final,
)
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    LEGACY_BIOAVAILABILITY_V1,
)


PRIOR = TrainRatioPrior(
    dataset_lineage="lineage",
    split="scaffold/train",
    positive_count=7,
    negative_count=3,
    positive_label="positive",
    negative_label="negative",
)


class CapturingClient:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def chat_json(self, messages):
        self.messages = messages
        return {"content": dict(self.content)}


def _retrieval():
    return {"query": {"identity_hidden": True}, "coverage": {}, "groups": []}


def test_standard_final_decision_profile_is_a_strict_prompt_noop():
    prompt = build_final_decision_prompt(STANDARD_FINAL_DECISION, PRIOR)
    assert prompt.fields == {}
    assert prompt.instructions == ()
    assert prompt.schema == {}
    assert prompt.required_fields == ()


def test_train_ratio_profile_is_strong_but_narrow():
    prompt = build_final_decision_prompt(TRAIN_RATIO_TIEBREAK_V1, PRIOR)
    assert prompt.fields["training_label_prior"]["positive_count"] == 7
    assert prompt.fields["training_label_prior"]["negative_count"] == 3
    assert prompt.fields["training_label_prior"]["policy"] == "genuine_evidence_tie_only"
    instructions = " ".join(prompt.instructions)
    assert "MUST break the genuine tie" in instructions
    assert "not a target batch quota" in instructions


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            {
                "prediction": "positive",
                "evidence_state": "consistent_positive",
                "prior_used": False,
            },
            [],
        ),
        (
            {
                "prediction": "negative",
                "evidence_state": "consistent_positive",
                "prior_used": False,
            },
            ["inconsistent_decision:consistent_positive"],
        ),
        (
            {
                "prediction": "negative",
                "evidence_state": "mixed",
                "prior_used": True,
            },
            ["inconsistent_prior_tiebreak:majority_label_required"],
        ),
        (
            {
                "prediction": "positive",
                "evidence_state": "insufficient",
                "prior_used": "true",
            },
            ["invalid_type:prior_used:expected_boolean"],
        ),
    ],
)
def test_final_decision_cross_field_validation(content, expected):
    assert final_decision_validation_errors(
        content,
        profile=TRAIN_RATIO_TIEBREAK_V1,
        prior=PRIOR,
        prediction_field="prediction",
    ) == expected


def test_bbb_and_bio_final_prompts_receive_their_frozen_train_priors():
    bbb_client = CapturingClient(
        {
            "bbb_prediction": "pass",
            "evidence_state": "insufficient",
            "prior_used": True,
        }
    )
    bio_client = CapturingClient(
        {
            "bioavailability_prediction": "high",
            "evidence_state": "mixed",
            "prior_used": True,
        }
    )
    bbb = run_bbb_final(
        bbb_client,
        _retrieval(),
        {"status": "error"},
        [],
        final_decision_profile=TRAIN_RATIO_TIEBREAK_V1,
    )
    bio = run_bio_final(
        bio_client,
        _retrieval(),
        {"status": "error"},
        [],
        final_decision_profile=TRAIN_RATIO_TIEBREAK_V1,
    )
    assert bbb["status"] == "ok"
    assert bio["status"] == "ok"
    bbb_payload = json.loads(bbb_client.messages[-1]["content"])
    bio_payload = json.loads(bio_client.messages[-1]["content"])
    assert bbb_payload["training_label_prior"]["positive_count"] == 2162
    assert bbb_payload["training_label_prior"]["negative_count"] == 773
    assert bio_payload["training_label_prior"]["positive_count"] == 1213
    assert bio_payload["training_label_prior"]["negative_count"] == 461
    assert bbb_payload["required_json_schema"]["prior_used"] == "boolean"


def test_final_decision_profile_requires_final_only_and_cannot_mix_batch(tmp_path):
    args = SimpleNamespace(
        final_decision_profile=TRAIN_RATIO_TIEBREAK_V1,
        final_only_source_batch="",
    )
    with pytest.raises(ValueError, match="requires --final-only-source-batch"):
        _validate_final_decision_profile_contract(args, tmp_path)

    args.final_only_source_batch = "source"
    (tmp_path / "manifest.json").write_text(
        json.dumps({"final_decision_profile": STANDARD_FINAL_DECISION}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="profile mismatch"):
        _validate_final_decision_profile_contract(args, tmp_path)


def test_experiment_command_is_final_only_versioned_and_isolated(tmp_path):
    source = tmp_path / "source"
    source_run = source / "runs" / "run"
    source_run.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "input_jsonl": "valid.jsonl",
                "experiment_mode": "full_flat",
                "retrieval_source": "starling",
            }
        ),
        encoding="utf-8",
    )
    (source_run / "manifest.json").write_text(
        json.dumps(
            {
                "neighbor_index": "index.pkl",
                "tool_service_url": "http://localhost:8765",
                "max_tool_rounds": 3,
            }
        ),
        encoding="utf-8",
    )
    spec = TaskSpec(
        task="task",
        batch_module="task.batch",
        prediction_field="prediction",
        source_batch=source,
        train_jsonl=tmp_path / "train.jsonl",
        expected_valid_rows=1,
        prior=PRIOR,
        prompt_profile_args=(
            "--bioavailability-prompt-profile",
            LEGACY_BIOAVAILABILITY_V1,
        ),
    )
    args = SimpleNamespace(
        output_root=tmp_path / "output",
        python_executable="python",
        api_key_env="KEY",
        base_url="http://localhost:9001/v1",
        model="model",
        reasoning_effort="",
        timeout_s=300,
        parallelism=8,
    )
    command = _batch_command(spec, args)
    joined = " ".join(str(value) for value in command)
    assert "--final-only-source-batch" in command
    assert f"--final-decision-profile {TRAIN_RATIO_TIEBREAK_V1}" in joined
    assert "--identity-blind" in command
    profile_position = command.index("--bioavailability-prompt-profile")
    assert command[profile_position + 1] == LEGACY_BIOAVAILABILITY_V1
    assert str(tmp_path / "output" / "runs_identity_blind_parent_disjoint") in joined


def test_offline_analysis_checks_copied_artifact_parity(tmp_path):
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source_rows = []
    candidate_rows = []
    cases = [
        (0, 0, 0, "negative", "consistent_negative", False),
        (1, 1, 0, "positive", "mixed", True),
    ]
    for index, label, source_prediction, prediction, state, prior_used in cases:
        source_run = source / "runs" / f"source_idx{index:05d}"
        candidate_run = candidate / "runs" / f"candidate_idx{index:05d}"
        source_run.mkdir(parents=True)
        candidate_run.mkdir(parents=True)
        for name in (
            "retrieval.json",
            "single_molecule_reasoning_output.json",
            "group_reasoning_outputs.jsonl",
        ):
            content = f"artifact-{index}-{name}\n"
            (source_run / name).write_text(content, encoding="utf-8")
            (candidate_run / name).write_text(content, encoding="utf-8")
        (candidate_run / "manifest.json").write_text(
            json.dumps(
                {
                    "final_only_source_batch": str(source),
                    "final_decision_profile": TRAIN_RATIO_TIEBREAK_V1,
                }
            ),
            encoding="utf-8",
        )
        source_rows.append(
            {
                "query_index": index,
                "label": label,
                "pred_label": source_prediction,
                "run_dir": str(source_run),
                "status": "ok",
            }
        )
        candidate_rows.append(
            {
                "query_index": index,
                "label": label,
                "pred_label": 1 if prediction == "positive" else 0,
                "prediction": prediction,
                "evidence_state": state,
                "prior_used": prior_used,
                "run_dir": str(candidate_run),
                "status": "ok",
            }
        )
    for batch, rows in ((source, source_rows), (candidate, candidate_rows)):
        (batch / "metrics.json").write_text(
            json.dumps({"n_failed_runs": 0}), encoding="utf-8"
        )
        (batch / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    summary, samples = analyze_train_ratio_prior(
        [
            TrainRatioAnalysisSpec(
                task="task",
                prediction_field="prediction",
                source_batch=source,
                candidate_batch=candidate,
                prior=PRIOR,
            )
        ]
    )
    assert len(samples) == 2
    assert summary["tasks"]["task"]["reuse_audit"] == {
        "status": "pass",
        "artifact_names": [
            "retrieval.json",
            "single_molecule_reasoning_output.json",
            "group_reasoning_outputs.jsonl",
        ],
        "n_artifact_pairs_checked": 6,
        "n_candidate_manifests_checked": 2,
        "n_mismatches": 0,
    }
