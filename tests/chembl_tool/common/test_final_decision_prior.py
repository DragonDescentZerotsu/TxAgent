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
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    _run_final_reasoning as run_bbb_final,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    _run_final_reasoning as run_bio_final,
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
    assert (
        prompt.fields["training_label_prior"]["policy"] == "genuine_evidence_tie_only"
    )
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
    assert (
        final_decision_validation_errors(
            content,
            profile=TRAIN_RATIO_TIEBREAK_V1,
            prior=PRIOR,
            prediction_field="prediction",
        )
        == expected
    )


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
