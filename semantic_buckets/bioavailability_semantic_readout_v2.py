"""Run the repaired V10 Oral semantic-bucket workflow."""

from pathlib import Path

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets.artifacts import generation_root


VERSION = "bioavailability_semantic_readout.v2"
BASE_URL = "http://dgx014:50001/v1"
ENDPOINTS = tuple(
    {
        "name": name,
        "base_url": f"http://{name}:50001/v1",
        "provider": "local",
        "credential_env": "",
    }
    for name in ("dgx014", "dgx015", "dgx018")
)
IGNORED_COLUMNS = frozenset({"canonical_direct_condition_group"})
REFINEMENT_COLUMNS = {
    source: tuple(column for column in columns if column not in IGNORED_COLUMNS)
    for source, columns in core.PAIR_COLUMNS.items()
}
SAMPLE_CARD_COLUMNS = {
    source: tuple(column for column in columns if column not in IGNORED_COLUMNS)
    for source, columns in core.SAMPLE_CARD_COLUMNS.items()
}
DEFAULT_OUTPUT = generation_root(
    "bioavailability_ma", "bioavailability_semantic_readout_v10_v2"
)
DEFAULT_REVIEW = Path("/tmp/bioavailability_semantic_readout_v2_prompt_review")


def configure_core() -> None:
    core.VERSION = VERSION
    core.BASE_URL = BASE_URL
    core.ENDPOINTS = ENDPOINTS
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.DEFAULT_REVIEW = DEFAULT_REVIEW
    core.REFINEMENT_COLUMNS = REFINEMENT_COLUMNS
    core.PROMPT_DIMENSION_COLUMNS = REFINEMENT_COLUMNS
    core.SAMPLE_CARD_COLUMNS = SAMPLE_CARD_COLUMNS
    core.DEFER_TECHNICAL_BRANCHES = True
    core.SEMANTIC_SIZE_REVIEW_REQUIRED = True
    core.INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS = False
    core.INCLUDE_DOWNSTREAM_PROMPT_REVIEW = False


def main() -> None:
    configure_core()
    core.main()


if __name__ == "__main__":
    main()
