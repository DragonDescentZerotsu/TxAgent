import argparse
import json

import pytest

from tools.chembl_tool.common.final_decision_prior import (
    TrainRatioPrior,
    add_final_decision_profile_argument,
)
from tools.chembl_tool.tasks.bbb_martins.final_decision_profiles import (
    BBB_FINAL_DECISION_PROFILES,
    DIRECT_ANCHORED_RESIDUAL_V1,
    DIRECT_OVERRIDE_RECHECK_V1,
    bbb_final_decision_allowed_values,
    bbb_final_decision_validation_errors,
    build_bbb_final_decision_prompt,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    _run_final_reasoning,
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


def test_direct_anchored_profile_is_task_label_aware_without_exposing_a_prior():
    prompt = build_bbb_final_decision_prompt(DIRECT_ANCHORED_RESIDUAL_V1, PRIOR)
    assert prompt.fields == {}
    assert prompt.schema["direct_anchor_prediction"] == "positive | negative"
    assert "direct_anchor_prediction" in prompt.required_fields
    assert "training_label_prior" not in " ".join(prompt.instructions)
    assert "same evidence surface as the Direct condition" in " ".join(
        prompt.instructions
    )
    assert bbb_final_decision_allowed_values(DIRECT_ANCHORED_RESIDUAL_V1) == {
        "direct_anchor_state": {
            "supports_positive",
            "supports_negative",
            "mixed",
            "insufficient",
        },
        "mechanism_override": {
            "no_change",
            "override_to_positive",
            "override_to_negative",
        },
        "mechanism_evidence_weight": {
            "decisive",
            "supporting",
            "context_only",
            "none",
        },
    }


def test_bbb_only_profiles_are_not_exposed_by_the_general_cli():
    general = argparse.ArgumentParser()
    add_final_decision_profile_argument(general)
    with pytest.raises(SystemExit):
        general.parse_args(
            ["--final-decision-profile", DIRECT_ANCHORED_RESIDUAL_V1]
        )

    bbb = argparse.ArgumentParser()
    add_final_decision_profile_argument(
        bbb,
        choices=BBB_FINAL_DECISION_PROFILES,
    )
    args = bbb.parse_args(
        ["--final-decision-profile", DIRECT_ANCHORED_RESIDUAL_V1]
    )
    assert args.final_decision_profile == DIRECT_ANCHORED_RESIDUAL_V1


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            {
                "prediction": "positive",
                "direct_anchor_state": "supports_positive",
                "direct_anchor_prediction": "positive",
                "mechanism_evidence_weight": "supporting",
                "mechanism_override": "no_change",
                "override_supported": False,
            },
            [],
        ),
        (
            {
                "prediction": "negative",
                "direct_anchor_state": "supports_positive",
                "direct_anchor_prediction": "positive",
                "mechanism_evidence_weight": "decisive",
                "mechanism_override": "override_to_negative",
                "override_supported": True,
            },
            [],
        ),
        (
            {
                "prediction": "negative",
                "direct_anchor_state": "supports_positive",
                "direct_anchor_prediction": "positive",
                "mechanism_evidence_weight": "supporting",
                "mechanism_override": "override_to_negative",
                "override_supported": True,
            },
            ["insufficient_override_support:override_to_negative"],
        ),
        (
            {
                "prediction": "negative",
                "direct_anchor_state": "supports_positive",
                "direct_anchor_prediction": "positive",
                "mechanism_evidence_weight": "decisive",
                "mechanism_override": "override_to_negative",
                "override_supported": True,
                "override_verdict": "reject",
            },
            ["inconsistent_recheck:reject_requires_no_change"],
        ),
    ],
)
def test_direct_anchored_cross_field_validation(content, expected):
    profile = (
        DIRECT_OVERRIDE_RECHECK_V1
        if "override_verdict" in content
        else DIRECT_ANCHORED_RESIDUAL_V1
    )
    assert bbb_final_decision_validation_errors(
        content,
        profile=profile,
        prior=PRIOR,
        prediction_field="prediction",
    ) == expected


def test_bbb_final_prompt_accepts_direct_anchored_contract():
    client = CapturingClient(
        {
            "bbb_prediction": "pass",
            "direct_anchor_state": "supports_positive",
            "direct_anchor_prediction": "pass",
            "mechanism_evidence_weight": "supporting",
            "mechanism_override": "no_change",
            "override_supported": False,
            "decisive_mechanism_evidence": [],
            "competing_direct_evidence": ["direct outcome branch"],
            "anchor_reasoning": "Direct evidence supports pass.",
        }
    )
    result = _run_final_reasoning(
        client,
        {"query": {"identity_hidden": True}, "coverage": {}, "groups": []},
        {"status": "error"},
        [],
        final_decision_profile=DIRECT_ANCHORED_RESIDUAL_V1,
    )
    assert result["status"] == "ok"
    payload = json.loads(client.messages[-1]["content"])
    assert "training_label_prior" not in payload
    assert payload["required_json_schema"]["mechanism_override"].startswith(
        "no_change"
    )
