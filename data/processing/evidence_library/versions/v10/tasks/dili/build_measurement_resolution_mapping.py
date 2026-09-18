"""Run DILI V10 measurement resolution on the approved DeepSeek provider pool."""

from __future__ import annotations

import sys

from data.processing.evidence_library.versions.v10 import (
    build_measurement_resolution_mapping as generator,
)
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution as config,
)


_FIXED_FLAGS = {
    "--task",
    "--base-url",
    "--model",
    "--api-key-env",
    "--provider",
    "--workers",
    "--max-completion-tokens",
    "--provider-only",
    "--provider-pool-config",
    "--parallelism",
    "--two-key-baidu-run",
    "--config-module",
    "--cleaned-records",
    "--endpoint-profile",
    "--base-mapping",
    "--gold-fixture",
}


def _uses_flag(argv: list[str], flag: str) -> bool:
    return any(value == flag or value.startswith(f"{flag}=") for value in argv)


def fixed_argv(argv: list[str]) -> list[str]:
    """Add the release's provider-pool contract and reject routing overrides."""
    supplied = sorted(flag for flag in _FIXED_FLAGS if _uses_flag(argv, flag))
    if supplied:
        raise SystemExit(
            "DILI V10 fixes task, provider pool, model, concurrency, and token limit; "
            f"remove these override flags: {', '.join(supplied)}"
        )
    fixed = [
        *argv,
        "--task",
        "dili",
        "--provider-pool-config",
        str(config.PROVIDER_POOL_CONFIG),
        "--parallelism",
        str(config.PROVIDER_POOL_PARALLELISM),
        "--model",
        config.DEEPSEEK_MODEL,
        "--max-completion-tokens",
        str(config.MAX_COMPLETION_TOKENS),
        "--no-token-ledger",
        "--require-complete",
    ]
    if _uses_flag(argv, "--gold-replay"):
        fixed.extend(["--gold-fixture", str(config.DEFAULT_GOLD_FIXTURE)])
    return fixed


def main(argv: list[str] | None = None) -> int:
    effective = fixed_argv(list(sys.argv[1:] if argv is None else argv))
    return generator.main(effective)


if __name__ == "__main__":
    raise SystemExit(main())
