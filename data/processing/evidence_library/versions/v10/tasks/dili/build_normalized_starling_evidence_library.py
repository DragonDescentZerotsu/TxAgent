"""Build the source and clean stages of the DILI V10 evidence library."""

from __future__ import annotations

from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    parse_args,
    run_with_args,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_policy import (
    POLICY,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(POLICY, argv, default_through_stage="clean", core_only=True)
    return run_with_args(POLICY, args)


if __name__ == "__main__":
    raise SystemExit(main())
