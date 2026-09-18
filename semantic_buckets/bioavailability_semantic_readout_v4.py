"""Replay Oral V10 semantic refinement with pair-bucket-only sample fields."""

from pathlib import Path

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import bioavailability_semantic_readout_v3 as v3
from semantic_buckets.artifacts import generation_root


VERSION = "bioavailability_semantic_readout.v4"
DEFAULT_OUTPUT = generation_root(
    "bioavailability_ma", "bioavailability_semantic_readout_v10_v4"
)
DEFAULT_REVIEW = Path("/tmp/bioavailability_semantic_readout_v4_prompt_review")


def configure_core() -> None:
    v3.configure_core()
    core.VERSION = VERSION
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.DEFAULT_REVIEW = DEFAULT_REVIEW


def main() -> None:
    configure_core()
    core.main()


if __name__ == "__main__":
    main()
