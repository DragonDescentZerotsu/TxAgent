"""Provider-independent frozen settings for the current one-pass GRPO line.

This module is intentionally dependency-light.  Hosted Tinker and local NeMo
backends import the same object instead of maintaining matching magic numbers.
Backend topology, context capacity, optimizer implementation, and checkpoint
storage remain backend/model-specific and do not belong here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ONE_PASS_REWARD_VERSION = "one_pass_hierarchical_reward.v2"
ONE_PASS_TRAINING_PROFILE = "one_pass_bio_grpo.v1"


@dataclass(frozen=True)
class OnePassRewardWeights:
    final_correct: float = 1.0
    final_wrong: float = -1.0
    single_correct: float = 0.15
    analog_correct: float = 0.15
    resolved_conflict: float = 0.15
    parsed_json: float = 0.02
    full_schema: float = 0.03


@dataclass(frozen=True)
class OnePassTrainingRecipe:
    """Settings that must match across providers for a backend comparison."""

    profile: str = ONE_PASS_TRAINING_PROFILE
    reward_version: str = ONE_PASS_REWARD_VERSION
    groups_per_batch: int = 4
    generations_per_prompt: int = 8
    max_completion_tokens: int = 6144
    temperature: float = 1.0
    top_p: float = 1.0
    learning_rate: float = 1.0e-4
    lora_rank: int = 32
    lora_target: str = "all"
    seed: int = 42
    reference_kl_penalty: float = 0.0
    normalize_rewards: bool = True
    leave_one_out_baseline: bool = True

    @property
    def rollout_batch_size(self) -> int:
        return self.groups_per_batch * self.generations_per_prompt

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def deviations(self, observed: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Return explicit shared-setting differences for manifests and gates."""

        expected = self.as_dict()
        return {
            key: {"expected": expected[key], "observed": observed.get(key)}
            for key in expected
            if key not in {"profile", "reward_version"}
            and observed.get(key) != expected[key]
        }


ONE_PASS_REWARD_WEIGHTS = OnePassRewardWeights()
ONE_PASS_TRAINING_RECIPE = OnePassTrainingRecipe()
