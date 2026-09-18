"""DILI execution overlay for the reviewed three-endpoint local pool."""

from pathlib import Path
from typing import Any

from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution import *  # noqa: F403
from data.processing.paths import REPO_ROOT


PROVIDER_POOL_CONFIG = (
    REPO_ROOT
    / "predict/api_client/providers/measurement_resolution_v10_three_endpoint_256.json"
)
PROVIDER_POOL_PARALLELISM = 768


def validate_generation_args(
    args: Any,
    provider_pool_config: Path = PROVIDER_POOL_CONFIG,
    parallelism: int = PROVIDER_POOL_PARALLELISM,
    model: str = DEEPSEEK_MODEL,  # noqa: F405
) -> None:
    """Permit only the full, low-reasoning successor execution contract."""
    expected = {
        "task": "dili",
        "model": model,
        "base_url": None,
        "api_key_env": None,
        "provider": None,
        "provider_only": None,
        "provider_pool_config": provider_pool_config,
        "parallelism": parallelism,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,  # noqa: F405
        "no_token_ledger": True,
        "require_complete": True,
        "base_mapping": None,
    }
    mismatches = {}
    for name, value in expected.items():
        found = getattr(args, name)
        matches = found == value
        if name == "provider_pool_config" and found is not None:
            matches = Path(found).resolve() == Path(value).resolve()
        if not matches:
            mismatches[name] = {"expected": value, "found": found}
    if args.gold_replay or args.limit is not None or args.source is not None:
        mismatches["full_run"] = "the combined queue accepts only the full DILI run"
    if args.two_key_baidu_run:
        mismatches["two_key_baidu_run"] = {"expected": False, "found": True}
    if mismatches:
        raise SystemExit(f"DILI combined-queue contract mismatch: {mismatches}")
