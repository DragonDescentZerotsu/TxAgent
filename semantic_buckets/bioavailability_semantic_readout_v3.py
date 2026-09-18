"""Finish the repaired V10 Oral semantic buckets by exhausting eligible columns."""

from pathlib import Path

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import bioavailability_semantic_readout_v2 as v2
from semantic_buckets.artifacts import generation_root


VERSION = "bioavailability_semantic_readout.v3"
DEFAULT_OUTPUT = generation_root(
    "bioavailability_ma", "bioavailability_semantic_readout_v10_v3"
)
DEFAULT_REVIEW = Path("/tmp/bioavailability_semantic_readout_v3_prompt_review")
MAX_SOURCE_ROUNDS = 1 + max(
    len(v2.REFINEMENT_COLUMNS[source]) - len(core.INITIAL_COLUMNS[source])
    for source in v2.REFINEMENT_COLUMNS
)


def configure_core() -> None:
    v2.configure_core()
    core.VERSION = VERSION
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.DEFAULT_REVIEW = DEFAULT_REVIEW
    core.MAX_SOURCE_ROUNDS = MAX_SOURCE_ROUNDS


def main() -> None:
    configure_core()
    core.main()


if __name__ == "__main__":
    main()
