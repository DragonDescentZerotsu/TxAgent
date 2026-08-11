"""Run one-task Starling GRPO with Tinker's hosted GPT-OSS-120B LoRA service."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
from numbers import Real
from typing import Any

import chz
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.hyperparam_utils import get_lora_param_count
from tinker_cookbook.rl.train import Config, main as train_main
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tools.chembl_tool.paper_experiments.rl_lora.experiment_contract import (
    ONE_PASS_REWARD_VERSION,
    ONE_PASS_TRAINING_RECIPE,
)
from tools.chembl_tool.paper_experiments.rl_lora.one_pass_contract import (
    is_one_pass_contract,
)
from tools.chembl_tool.paper_experiments.rl_lora.one_pass_runtime import (
    load_fresh_one_pass_audit,
)
from tools.chembl_tool.paper_experiments.rl_lora.training_contract import (
    score_training_response,
    training_result_logs,
    training_result_metrics,
)
from tools.chembl_tool.common.json_utils import sha256_file


MODEL = "openai/gpt-oss-120b:peft:131072"
VALID_RENDERER = "gpt_oss_medium_reasoning"
ONE_PASS_DEFAULT_LORA_RANK = ONE_PASS_TRAINING_RECIPE.lora_rank
ONE_PASS_DEFAULT_LORA_TARGET = ONE_PASS_TRAINING_RECIPE.lora_target
HISTORICAL_FINAL_LORA_RANK = 16
HISTORICAL_FINAL_LORA_TARGET = "attention_only"


class StarlingTinkerEnv(Env):
    """Single-turn final-synthesis environment with the frozen reward contract."""

    def __init__(self, row: dict[str, Any], renderer: renderers.Renderer):
        self.row = row
        self.renderer = renderer

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self.row["messages"]),
            self.renderer.get_stop_sequences(),
        )

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        del extra
        message, termination = self.renderer.parse_response(action)
        response = renderers.get_text_content(message)
        result = score_training_response(response, self.row)
        metrics = training_result_metrics(result, clean_stop=termination.is_clean)
        logs = training_result_logs(result, self.row)
        return StepResult(
            reward=result.reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
            logs=logs,
        )


@dataclass(frozen=True)
class StarlingGroupBuilder(EnvGroupBuilder):
    row: dict[str, Any]
    renderer: renderers.Renderer
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            StarlingTinkerEnv(self.row, self.renderer) for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        del env_group
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [str(self.row["source_task"])]


class StarlingDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        renderer: renderers.Renderer,
        groups_per_batch: int,
        group_size: int,
        seed: int,
    ):
        self.rows = list(rows)
        random.Random(seed).shuffle(self.rows)
        self.renderer = renderer
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        rows = self.rows[start : start + self.groups_per_batch]
        return [
            StarlingGroupBuilder(row, self.renderer, self.group_size) for row in rows
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.groups_per_batch)


@chz.chz
class StarlingDatasetBuilder(RLDatasetBuilder):
    data_path: str
    model_name: str
    renderer_name: str
    groups_per_batch: int
    group_size: int
    seed: int = 42

    async def __call__(self) -> tuple[RLDataset, None]:
        path = Path(self.data_path)
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            StarlingDataset(
                rows,
                renderer=renderer,
                groups_per_batch=self.groups_per_batch,
                group_size=self.group_size,
                seed=self.seed,
            ),
            None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--renderer", default=VALID_RENDERER)
    parser.add_argument(
        "--lora-rank",
        type=int,
        help="Override the contract default (one-pass: 32; historical final-only: 16).",
    )
    parser.add_argument(
        "--lora-target",
        choices=("attention_only", "all"),
        help=(
            "Override the contract default (one-pass: SDK-default all modules; "
            "historical final-only: attention_only)."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=ONE_PASS_TRAINING_RECIPE.learning_rate,
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=ONE_PASS_TRAINING_RECIPE.generations_per_prompt,
    )
    parser.add_argument(
        "--groups-per-batch",
        type=int,
        default=ONE_PASS_TRAINING_RECIPE.groups_per_batch,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=(
            "Completion cap (one-pass default comes from the shared profile; "
            "historical final-only default remains 20,480)."
        ),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--num-groups-to-log",
        type=int,
        default=1,
        help="Full trajectory groups logged per step; keep bounded for long prompts.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Export a persistent, validation-ready sampler checkpoint every N steps.",
    )
    parser.add_argument(
        "--rolling-save-every",
        type=int,
        default=0,
        help="Save resume-only rolling state every N steps (no sampler export).",
    )
    parser.add_argument("--seed", type=int, default=ONE_PASS_TRAINING_RECIPE.seed)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the audited data and materialize the backend contract/config "
            "without creating a hosted training client or incurring training calls."
        ),
    )
    parser.add_argument(
        "--allow-contract-override",
        action="store_true",
        help=(
            "Allow one-pass shared scientific settings to differ from the frozen "
            "cross-backend profile; every difference is still written to the manifest."
        ),
    )
    return parser.parse_args()


async def _train_step_with_forward_metrics(
    data_D: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: str,
    loss_fn_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[Any]:
    """Cookbook train_step plus scalar forward/backward metrics.

    The pinned cookbook only forwards optimizer metrics to ``metrics`` and
    discards ``ForwardBackwardOutput.metrics``.  Preserve its pipelining while
    exposing the service-reported loss for local/W&B curve auditing.
    """

    batches = rl_train.split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return []
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    training_logprobs_D: list[Any] = []
    optim_result: Any | None = None
    forward_metric_sums: dict[str, float] = {}
    forward_metric_counts: dict[str, int] = {}
    fwd_bwd_future = await training_client.forward_backward_async(
        [rl_train._remove_mask(datum) for datum in batches[0]],
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    )
    optim_future = await training_client.optim_step_async(adam_params)
    for index in range(len(batches)):
        if index + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [rl_train._remove_mask(datum) for datum in batches[index + 1]],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None
        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs_D.extend(
            rl_train._training_logprobs_from_fwd_bwd(fwd_bwd_result)
        )
        for key, value in (fwd_bwd_result.metrics or {}).items():
            if isinstance(value, Real) and not isinstance(value, bool):
                name = str(key)
                forward_metric_sums[name] = forward_metric_sums.get(name, 0.0) + float(
                    value
                )
                forward_metric_counts[name] = forward_metric_counts.get(name, 0) + 1
        optim_result = await optim_future.result_async()
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future
    if metrics is not None:
        if optim_result is not None and optim_result.metrics:
            metrics.update(optim_result.metrics)
        for key, total in forward_metric_sums.items():
            safe_key = key.replace(":", "_")
            metrics[f"forward_backward/{safe_key}_sum"] = total
            metrics[f"forward_backward/{safe_key}_mean"] = (
                total / forward_metric_counts[key]
            )
        if "loss:sum" in forward_metric_sums:
            metrics["train/loss_sum"] = forward_metric_sums["loss:sum"]
    return training_logprobs_D


async def async_main() -> None:
    args = parse_args()
    first_row = json.loads(args.data.open(encoding="utf-8").readline())
    one_pass = is_one_pass_contract(first_row.get("contract_version"))
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = (
            ONE_PASS_TRAINING_RECIPE.max_completion_tokens if one_pass else 20_480
        )
    lora_rank = args.lora_rank
    if lora_rank is None:
        lora_rank = (
            ONE_PASS_DEFAULT_LORA_RANK if one_pass else HISTORICAL_FINAL_LORA_RANK
        )
    lora_target = args.lora_target
    if lora_target is None:
        lora_target = (
            ONE_PASS_DEFAULT_LORA_TARGET if one_pass else HISTORICAL_FINAL_LORA_TARGET
        )
    data_sha256 = sha256_file(args.data)
    data_audit = (
        load_fresh_one_pass_audit(
            args.data,
            minimum_contract="one_pass_data_audit.v2",
        )
        if one_pass
        else None
    )
    n_rows = int(data_audit["n_rows"]) if data_audit is not None else None
    expected_steps = (
        math.ceil(n_rows / args.groups_per_batch) if n_rows is not None else None
    )
    if args.max_steps is not None and expected_steps is not None:
        if args.max_steps <= 0 or args.max_steps > expected_steps:
            raise ValueError(
                f"--max-steps must be in [1, {expected_steps}], got {args.max_steps}"
            )
    renderer_name = args.renderer
    parameter_count_model = args.model.split(":peft:", 1)[0]
    train_mlp = lora_target == "all"
    train_attn = True
    train_unembed = lora_target == "all"
    trainable_parameters = get_lora_param_count(
        parameter_count_model,
        lora_rank=lora_rank,
        train_mlp=train_mlp,
        train_attn=train_attn,
        train_unembed=train_unembed,
    )
    shared_observed = {
        "groups_per_batch": args.groups_per_batch,
        "generations_per_prompt": args.group_size,
        "max_completion_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "learning_rate": args.learning_rate,
        "lora_rank": lora_rank,
        "lora_target": lora_target,
        "seed": args.seed,
        "reference_kl_penalty": 0.0,
        "normalize_rewards": True,
        "leave_one_out_baseline": True,
    }
    shared_deviations = (
        ONE_PASS_TRAINING_RECIPE.deviations(shared_observed) if one_pass else {}
    )
    if shared_deviations and not args.allow_contract_override:
        raise ValueError(
            "one-pass Tinker settings differ from the frozen shared contract; "
            f"pass --allow-contract-override only for an explicit ablation: {shared_deviations}"
        )
    args.log_path.mkdir(parents=True, exist_ok=True)
    (args.log_path / "backend_contract.json").write_text(
        json.dumps(
            {
                "backend": "tinker",
                "tinker_sdk_version": version("tinker"),
                "model": args.model,
                "parameter_count_model": parameter_count_model,
                "renderer": renderer_name,
                "lora_rank": lora_rank,
                "lora_target": lora_target,
                "lora_default_profile": (
                    "one_pass_rank32_all"
                    if one_pass
                    else "historical_final_rank16_attention_only"
                ),
                "train_mlp": train_mlp,
                "train_attn": train_attn,
                "train_unembed": train_unembed,
                "trainable_parameters": trainable_parameters,
                "data": str(args.data),
                "data_sha256": data_sha256,
                "data_audit": data_audit["audit_path"] if data_audit else "",
                "data_audit_sha256": data_audit["audit_sha256"] if data_audit else "",
                "data_n_rows": n_rows,
                "data_label_counts": data_audit.get("label_counts", {})
                if data_audit
                else {},
                "data_source_tasks": data_audit.get("source_tasks", {})
                if data_audit
                else {},
                "data_source_subsets": data_audit.get("source_subsets", {})
                if data_audit
                else {},
                "contract_version": first_row.get("contract_version", "final_only"),
                "reward_version": (
                    ONE_PASS_REWARD_VERSION if one_pass else "final_only_reward.v1"
                ),
                "shared_training_profile": (
                    ONE_PASS_TRAINING_RECIPE.profile if one_pass else ""
                ),
                "shared_training_contract": (
                    ONE_PASS_TRAINING_RECIPE.as_dict() if one_pass else {}
                ),
                "shared_training_deviations": shared_deviations,
                "max_tokens": max_tokens,
                "learning_rate": args.learning_rate,
                "group_size": args.group_size,
                "groups_per_batch": args.groups_per_batch,
                "expected_full_steps": expected_steps,
                "max_steps": args.max_steps,
                "save_every": args.save_every,
                "rolling_save_every": args.rolling_save_every,
                "num_groups_to_log": args.num_groups_to_log,
                "seed": args.seed,
                "wandb_project": args.wandb_project,
                "wandb_name": args.wandb_name,
                "dry_run": args.dry_run,
                "training_metrics_contract": {
                    "loss": "train/loss_sum",
                    "forward_backward": "forward_backward/*",
                    "reward": "env/all/reward/mean",
                    "kl": "kl/*",
                    "entropy": "sampling/entropy",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config = Config(
        learning_rate=args.learning_rate,
        dataset_builder=StarlingDatasetBuilder(
            data_path=str(args.data),
            model_name=args.model,
            renderer_name=renderer_name,
            groups_per_batch=args.groups_per_batch,
            group_size=args.group_size,
            seed=args.seed,
        ),
        model_name=args.model,
        recipe_name=(
            "txagent_starling_one_pass_full_flat_grpo"
            if one_pass
            else "txagent_starling_full_flat_grpo"
        ),
        renderer_name=renderer_name,
        lora_rank=lora_rank,
        max_tokens=max_tokens,
        temperature=1.0,
        loss_fn="importance_sampling",
        kl_penalty_coef=0.0,
        remove_constant_reward_groups=True,
        log_path=str(args.log_path),
        eval_every=0,
        save_every=args.save_every,
        rolling_save_every=args.rolling_save_every,
        ttl_seconds=None,
        num_groups_to_log=args.num_groups_to_log,
        rollout_json_export=True,
        max_steps=args.max_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    if args.dry_run:
        (args.log_path / "dry_run.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "hosted_training_client_created": False,
                    "hosted_training_calls_made": False,
                    "backend_contract": "backend_contract.json",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    original_create = tinker.ServiceClient.create_lora_training_client_async
    original_train_step = rl_train.train_step

    async def create_with_frozen_targets(
        service_client: tinker.ServiceClient,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> Any:
        del train_mlp, train_attn, train_unembed
        return await original_create(
            service_client,
            base_model,
            rank=rank,
            seed=seed,
            train_mlp=lora_target == "all",
            train_attn=True,
            train_unembed=lora_target == "all",
            user_metadata=user_metadata,
        )

    # The pinned cookbook Config exposes rank but not the SDK component flags.
    # Limit this override to the current process and restore it after training.
    setattr(
        tinker.ServiceClient,
        "create_lora_training_client_async",
        create_with_frozen_targets,
    )
    rl_train.train_step = _train_step_with_forward_metrics
    try:
        await train_main(config)
    finally:
        rl_train.train_step = original_train_step
        setattr(
            tinker.ServiceClient,
            "create_lora_training_client_async",
            original_create,
        )


if __name__ == "__main__":
    asyncio.run(async_main())
