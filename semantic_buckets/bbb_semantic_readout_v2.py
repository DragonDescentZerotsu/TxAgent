"""Run the repaired V10 BBB semantic-bucket workflow."""

from pathlib import Path

from semantic_buckets import bbb_semantic_readout_v1 as workflow
from semantic_buckets.artifacts import generation_root


IGNORED_COLUMNS = frozenset({"condition_group"})
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
REFINEMENT_COLUMNS = {
    source: tuple(column for column in columns if column not in IGNORED_COLUMNS)
    for source, columns in workflow.PAIR_COLUMNS.items()
}
SAMPLE_CARD_COLUMNS = {
    source: tuple(column for column in columns if column not in IGNORED_COLUMNS)
    for source, columns in workflow.SAMPLE_CARD_COLUMNS.items()
}


def configure_workflow() -> None:
    workflow.VERSION = "bbb_semantic_readout.v2"
    workflow.BASE_URL = BASE_URL
    workflow.ENDPOINTS = ENDPOINTS
    workflow.PROMPT_ROOT = Path(__file__).with_name("prompts") / "bbb_semantic_readout_v2"
    workflow.DEFAULT_OUTPUT = generation_root(
        "bbb_martins", "bbb_semantic_readout_v10_v2"
    )
    workflow.REFINEMENT_COLUMNS = REFINEMENT_COLUMNS
    workflow.PROMPT_DIMENSION_COLUMNS = workflow.REFINEMENT_COLUMNS
    workflow.SAMPLE_CARD_COLUMNS = SAMPLE_CARD_COLUMNS
    workflow.DEFER_TECHNICAL_BRANCHES = True
    workflow.SEMANTIC_SIZE_REVIEW_REQUIRED = True
    workflow.INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS = False
    workflow.INCLUDE_DOWNSTREAM_PROMPT_REVIEW = False


def main() -> None:
    configure_workflow()
    workflow.main()


if __name__ == "__main__":
    main()
