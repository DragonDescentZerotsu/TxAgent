"""Replay BBB V10 semantic refinement with pair-bucket-only sample fields."""

from pathlib import Path

from semantic_buckets import bbb_semantic_readout_v1 as workflow
from semantic_buckets import bbb_semantic_readout_v3 as v3
from semantic_buckets.artifacts import generation_root


VERSION = "bbb_semantic_readout.v4"
DEFAULT_OUTPUT = generation_root("bbb_martins", "bbb_semantic_readout_v10_v4")
DEFAULT_REVIEW = Path("/tmp/bbb_semantic_readout_v4_prompt_review")


def configure_workflow() -> None:
    v3.configure_workflow()
    workflow.VERSION = VERSION
    workflow.DEFAULT_OUTPUT = DEFAULT_OUTPUT


def main() -> None:
    configure_workflow()
    workflow.main()


if __name__ == "__main__":
    main()
