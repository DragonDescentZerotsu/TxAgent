"""Degree-20, five-sample oral V10 semantic relevance ranking."""
from pathlib import Path

from analysis.evidence_library import bioavailability_relevance_bucket_levels as workflow


workflow.VERSION = "bioavailability_relevance_bucket_levels.v3"
workflow.ENTRYPOINT = Path(__file__)
workflow.DEFAULT_OUTPUT = (
    workflow.ROOT
    / "outputs/analysis/evidence_library/bioavailability_relevance_levels_v10_degree20_v3"
)
workflow.DEGREE = 20
workflow.SAMPLE_LIMIT = 5
workflow.PILOT_AGREEMENT_REQUIRED = False
workflow.EXPECTED_LEVELS = {
    "L2": {"buckets": 64, "comparisons": 640},
    "L3": {"buckets": 159, "comparisons": 1_590},
    "L4": {"buckets": 17, "comparisons": 136},
    "L5": {"buckets": 11, "comparisons": 55},
    "L6": {"buckets": 17, "comparisons": 136},
}


if __name__ == "__main__":
    workflow.main()
