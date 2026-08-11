"""NeMo RL processor and one-turn environment for Starling classification."""

from __future__ import annotations

from typing import Any

import ray
import torch

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt

from .training_contract import (
    StarlingRewardMetadata,
    reward_metadata_from_row,
    score_training_response,
    training_result_final_correct,
    training_result_final_prediction,
)


ADAPTER_VERSION = "starling_nemo_adapter.v1"


def starling_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    del task_data_spec
    raw_messages = datum_dict["messages"]
    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in raw_messages
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    token_ids = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    length = len(token_ids)
    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        token_ids = token_ids[: min(4, max_seq_length)]
        loss_multiplier = 0.0

    # The provider-independent contract owns the private metadata projection.
    # Tinker calls the same function before scoring, preventing backend drift.
    metadata = reward_metadata_from_row({**datum_dict, "messages": messages})
    return {
        "message_log": [{"role": "user", "content": formatted, "token_ids": token_ids}],
        "length": length,
        "extra_env_info": metadata,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": str(datum_dict.get("task_name", datum_dict["source_task"])),
    }


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)
class StarlingClassificationEnvironment(EnvironmentInterface[StarlingRewardMetadata]):
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[StarlingRewardMetadata],
    ) -> EnvironmentReturn[StarlingRewardMetadata]:
        results = []
        for conversation, sample_metadata in zip(message_log_batch, metadata):
            response = "".join(
                str(message["content"])
                for message in conversation
                if message["role"] == "assistant"
            )
            results.append(score_training_response(response, sample_metadata))
        rewards = torch.tensor(
            [result.reward for result in results], dtype=torch.float32
        )
        terminateds = torch.ones_like(rewards)
        return EnvironmentReturn(
            observations=[
                {
                    "role": "environment",
                    "content": (
                        "Environment: correct"
                        if training_result_final_correct(result)
                        else "Environment: incorrect"
                    ),
                }
                for result in results
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(results),
            rewards=rewards,
            terminateds=terminateds,
            answers=[
                None
                if training_result_final_prediction(result) is None
                else str(training_result_final_prediction(result))
                for result in results
            ],
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        rewards = batch["rewards"]
        ended = batch["is_end"].bool()
        correct = (rewards >= 1.0) & ended
        metrics = {
            "accuracy": correct.float().mean().item(),
            "mean_reward": (rewards * ended).float().mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], correct.float()
            ),
            "fraction_of_samples_properly_ended": ended.float().mean().item(),
            "num_problems_in_batch": int(ended.shape[0]),
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics


def register_starling_components() -> None:
    from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY, register_env

    if "starling_final_processor" not in PROCESSOR_REGISTRY:
        register_processor("starling_final_processor", starling_data_processor)
    actor_fqn = "rl_lora.nemo_adapter.StarlingClassificationEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY.setdefault(actor_fqn, PY_EXECUTABLES.SYSTEM)
    if "starling_binary" not in ENV_REGISTRY:
        register_env("starling_binary", actor_fqn)
