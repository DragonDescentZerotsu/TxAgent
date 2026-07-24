"""Build the BBB H1/H2 ChEMBL assay mapping and audit artifacts."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.distance_assay_manifest import (
    DistanceAssayManifestConfig,
    main as run_manifest_builder,
)

from .distance_assay_rules import classify_base_measured_states, classify_distance_assay
from .distance_config import DISTANCE_CONFIG
from .distance_self_relevance import SELF_RELEVANCE_AUDIT


CONFIG = DistanceAssayManifestConfig(
    description=__doc__ or "",
    default_base_assays_csv=(
        "outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_assay_candidates.csv"
    ),
    default_out_dir="outputs/chembl_tool/tasks/bbb_martins/distance_expansion/assay_manifest/v3",
    distance_config=DISTANCE_CONFIG,
    self_relevance_audit=SELF_RELEVANCE_AUDIT,
    classify_assay=classify_distance_assay,
    classify_base_states=classify_base_measured_states,
)


def main(argv: list[str] | None = None) -> int:
    return run_manifest_builder(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
