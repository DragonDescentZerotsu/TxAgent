"""DILI incremental extraction over the pinned three-endpoint DeepSeek pool."""

from pathlib import Path
from typing import Any

from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution import *  # noqa: F403
from data.processing.paths import REPO_ROOT


PROVIDER_POOL_CONFIG = REPO_ROOT / (
    "predict/api_client/providers/"
    "measurement_resolution_v10_dgx005_dgx017_dgx020_512.json"
)
PROVIDER_POOL_PARALLELISM = 1_536


def validate_generation_args(args: Any) -> None:
    """Require UID-stable base reuse and the approved low-reasoning pool."""
    expected = {
        "task": "dili",
        "model": DEEPSEEK_MODEL,  # noqa: F405
        "base_url": None,
        "api_key_env": None,
        "provider": None,
        "provider_only": None,
        "provider_pool_config": PROVIDER_POOL_CONFIG,
        "parallelism": PROVIDER_POOL_PARALLELISM,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,  # noqa: F405
        "no_token_ledger": True,
        "require_complete": True,
        "base_mapping": DEFAULT_MAPPING_PATH,  # noqa: F405
    }
    mismatches = {}
    for name, value in expected.items():
        found = getattr(args, name)
        matches = found == value
        if name in {"provider_pool_config", "base_mapping"} and found is not None:
            matches = Path(found).resolve() == Path(value).resolve()
        if not matches:
            mismatches[name] = {"expected": str(value), "found": str(found)}
    if args.gold_replay or args.limit is not None or args.source is not None:
        mismatches["full_incremental_run"] = "filters are not permitted"
    if args.two_key_baidu_run:
        mismatches["two_key_baidu_run"] = {"expected": False, "found": True}
    if mismatches:
        raise SystemExit(f"DILI incremental queue contract mismatch: {mismatches}")
