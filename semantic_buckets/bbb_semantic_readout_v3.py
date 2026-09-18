"""Finish the repaired V10 BBB semantic buckets by exhausting eligible columns."""

from semantic_buckets import bbb_semantic_readout_v1 as workflow
from semantic_buckets import bbb_semantic_readout_v2 as v2
from semantic_buckets.artifacts import generation_root


VERSION = "bbb_semantic_readout.v3"
DEFAULT_OUTPUT = generation_root("bbb_martins", "bbb_semantic_readout_v10_v3")
MAX_SOURCE_ROUNDS = 1 + max(
    len(v2.REFINEMENT_COLUMNS[source]) - len(workflow.INITIAL_COLUMNS[source])
    for source in v2.REFINEMENT_COLUMNS
)


def configure_workflow() -> None:
    v2.configure_workflow()
    workflow.VERSION = VERSION
    workflow.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    workflow.core.MAX_SOURCE_ROUNDS = MAX_SOURCE_ROUNDS


def main() -> None:
    configure_workflow()
    workflow.main()


if __name__ == "__main__":
    main()
