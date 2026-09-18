"""Build the source and clean stages of the Carcinogens v10 library."""

from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import parse_args, run_with_args
from .starling_policy import POLICY


def main(argv=None):
    return run_with_args(POLICY, parse_args(POLICY, argv, default_through_stage="clean", core_only=True))


if __name__ == "__main__":
    raise SystemExit(main())
