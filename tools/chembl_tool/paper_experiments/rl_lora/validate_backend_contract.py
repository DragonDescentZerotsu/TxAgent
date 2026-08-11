"""Validate a NeMo one-pass config against the provider-independent recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .experiment_contract import ONE_PASS_TRAINING_RECIPE


NEMO_ALL_MODULE_TARGETS = (
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
    "output_layer",
)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "_override_":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_nemo_config(path: Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Resolve the single-file ``defaults`` inheritance used by these configs."""

    resolved = path.expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if resolved in seen:
        raise ValueError(f"cyclic NeMo config defaults: {resolved}")
    seen.add(resolved)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"NeMo config root must be a mapping: {resolved}")
    defaults = raw.pop("defaults", None)
    if defaults is None:
        return raw
    if not isinstance(defaults, str):
        raise TypeError(f"only one string defaults path is supported: {resolved}")
    default_path = Path(defaults)
    if not default_path.is_absolute():
        default_path = resolved.parent / default_path
    return _deep_merge(load_nemo_config(default_path, _seen=seen), raw)


def _get(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def nemo_shared_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    targets = tuple(_get(config, "policy", "megatron_cfg", "peft", "target_modules"))
    target_profile = (
        "all"
        if len(targets) == len(NEMO_ALL_MODULE_TARGETS)
        and set(targets) == set(NEMO_ALL_MODULE_TARGETS)
        else list(targets)
    )
    return {
        "groups_per_batch": int(_get(config, "grpo", "num_prompts_per_step")),
        "generations_per_prompt": int(
            _get(config, "grpo", "num_generations_per_prompt")
        ),
        "max_completion_tokens": int(
            _get(config, "policy", "generation", "max_new_tokens")
        ),
        "temperature": float(_get(config, "policy", "generation", "temperature")),
        "top_p": float(_get(config, "policy", "generation", "top_p")),
        "learning_rate": float(
            _get(config, "policy", "megatron_cfg", "optimizer", "lr")
        ),
        "lora_rank": int(_get(config, "policy", "megatron_cfg", "peft", "dim")),
        "lora_target": target_profile,
        "seed": int(_get(config, "grpo", "seed")),
        "reference_kl_penalty": float(
            _get(config, "loss_fn", "reference_policy_kl_penalty")
        ),
        "normalize_rewards": bool(_get(config, "grpo", "normalize_rewards")),
        "leave_one_out_baseline": bool(
            _get(config, "grpo", "use_leave_one_out_baseline")
        ),
    }


def validate_nemo_config(path: Path) -> dict[str, Any]:
    config = load_nemo_config(path)
    observed = nemo_shared_settings(config)
    deviations = ONE_PASS_TRAINING_RECIPE.deviations(observed)
    rollout_batch_size = (
        observed["groups_per_batch"] * observed["generations_per_prompt"]
    )
    train_global_batch_size = int(_get(config, "policy", "train_global_batch_size"))
    invariants: dict[str, dict[str, Any]] = {}
    if train_global_batch_size != rollout_batch_size:
        invariants["policy.train_global_batch_size"] = {
            "expected": rollout_batch_size,
            "observed": train_global_batch_size,
        }
    if not bool(_get(config, "grpo", "skip_reference_policy_logprobs_calculation")):
        invariants["grpo.skip_reference_policy_logprobs_calculation"] = {
            "expected": True,
            "observed": False,
        }
    return {
        "status": "pass" if not deviations and not invariants else "fail",
        "backend": "nemo_rl",
        "config": str(path.expanduser().resolve()),
        "shared_training_profile": ONE_PASS_TRAINING_RECIPE.profile,
        "shared_training_contract": ONE_PASS_TRAINING_RECIPE.as_dict(),
        "observed": observed,
        "deviations": deviations,
        "backend_invariant_failures": invariants,
        "model": str(_get(config, "policy", "model_name")),
        "log_dir": str(_get(config, "logger", "log_dir")),
        "data_path": str(_get(config, "data", "train", "data_path")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_nemo_config(args.nemo_config)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
