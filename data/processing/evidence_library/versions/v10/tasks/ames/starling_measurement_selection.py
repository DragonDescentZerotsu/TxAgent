"""Constrained model-selection config for AMES measurement candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_measurement_resolution as resolution,
)

TASK_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = TASK_ROOT / "data_processing/measurement_resolution_v2"
PROMPT_ROOT = TASK_ROOT / "prompts"

PROMPT_VERSION = "ames_measurement_resolution_prompt.v3"
MAPPING_VERSION = "ames_measurement_candidate_selection.v1"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
STRATIFY_BATCHES = True
ALLOW_REBATCH_UNATTEMPTED = True
REQUIRE_SOURCE_ROW_UID = True
REASONING_EFFORT = "low"

OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_CREDENTIAL_ENVS = ("OPENAI_API_KEY_ONE", "OPENAI_API_KEY_TWO")
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_BASE_URL = "http://dgx027:50001/v1"
DEEPSEEK_PROVIDER = "local"
DEEPSEEK_CREDENTIAL_ENV = ""
ENDPOINT_CONCURRENCY_BUDGET = 512
MAX_COMPLETION_TOKENS = 8_192

SOURCE_IDS = resolution.SOURCE_IDS
DEFAULT_CLEANED_RECORDS = ASSET_ROOT / "measurement_candidates.parquet"
DEFAULT_CANONICAL_RECORDS = resolution.DEFAULT_CANONICAL_RECORDS
DEFAULT_PROFILE_PATH = resolution.DEFAULT_PROFILE_PATH
DEFAULT_MAPPING_PATH = ASSET_ROOT / "measurement_selection.parquet"
DEFAULT_BASE_MAPPING_PATH = None
TEMPLATE_PATH = PROMPT_ROOT / "measurement_resolution_v3.jinja"

source_routing_rules = resolution.source_routing_rules
canonical_endpoint_name = resolution.canonical_endpoint_name


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    """Expose source evidence plus the frozen row-local candidate inventory."""
    return (
        *resolution.prompt_row_fields(source_id),
        "measurement_candidates_json",
        "candidate_set_sha256",
        "candidate_generation_disposition",
        "candidate_count",
    )


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPT_ROOT)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    """Render the candidate-selection prompt; profiles cannot add candidates."""
    del endpoint_profiles
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return (
        _environment()
        .get_template(TEMPLATE_PATH.name)
        .render(
            source_id=source_id,
            batch_size=batch_size,
            row_fields=list(prompt_row_fields(source_id)),
        )
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    """Pin the candidate-selection template and every rendered source prompt."""
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(TEMPLATE_PATH),
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "source_row_fields": {
            source_id: list(prompt_row_fields(source_id)) for source_id in SOURCE_IDS
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode("utf-8")
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


def validate_generation_args(args: Any) -> None:
    """Allow only the two frozen OpenAI keys or the local DeepSeek route."""
    common = {
        "task": "ames",
        "config_module": __name__,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "provider_only": None,
        "two_key_baidu_run": False,
        "require_complete": True,
    }
    if args.provider == "openai" and args.api_key_env in OPENAI_CREDENTIAL_ENVS:
        expected = {
            **common,
            "base_url": OPENAI_BASE_URL,
            "model": OPENAI_MODEL,
            "no_token_ledger": False,
        }
    elif args.provider == DEEPSEEK_PROVIDER:
        expected = {
            **common,
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
            "api_key_env": None,
            "workers": ENDPOINT_CONCURRENCY_BUDGET,
            "no_token_ledger": True,
        }
    else:
        expected = {"provider": "openai or local AMES route"}
    mismatches = {
        name: {"expected": value, "found": getattr(args, name, None)}
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if mismatches:
        raise SystemExit(f"AMES V10 generation contract mismatch: {mismatches}")


__all__ = [
    "ALLOW_REBATCH_UNATTEMPTED",
    "BATCH_SIZE",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_CREDENTIAL_ENV",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_PROVIDER",
    "DEFAULT_BASE_MAPPING_PATH",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "ENDPOINT_CONCURRENCY_BUDGET",
    "MAPPING_VERSION",
    "MAX_COMPLETION_TOKENS",
    "MAX_MEASUREMENTS_PER_ROW",
    "OPENAI_BASE_URL",
    "OPENAI_CREDENTIAL_ENVS",
    "OPENAI_MODEL",
    "PROMPT_VERSION",
    "REASONING_EFFORT",
    "REQUIRE_SOURCE_ROW_UID",
    "SOURCE_IDS",
    "STRATIFY_BATCHES",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "source_routing_rules",
    "validate_generation_args",
]
