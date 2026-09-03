"""Run the paper scalar-KNN baseline for Bioavailability_Ma."""

from tools.chembl_tool.common.scalar_knn import ScalarKnnConfig, main as run_knn
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING


CONFIG = ScalarKnnConfig(
    description=__doc__ or "",
    default_input="data/gold_labels/legacy/processed/Bioavailability_Ma/test.jsonl",
    default_index=(
        "outputs/paper/molecular_evidence_agent/evidence/"
        "bioavailability_starling_direct_numeric/starling_factor_neighbor_index.pkl"
    ),
    default_output_dir="outputs/paper/molecular_evidence_agent/bioavailability_ma/starling_direct_scalar_knn",
    threshold=20.0,
    positive_prediction="high",
    negative_prediction="low",
    source_config=STARLING,
)


def main(argv: list[str] | None = None) -> int:
    return run_knn(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
