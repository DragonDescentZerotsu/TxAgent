"""Build the BBB H1/H2-only molecule evidence library and neighbor index."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.evidence_library import (
    EvidenceLibraryConfig,
    main as run_builder,
)

from .distance_assay_rules import assign_distance_endpoint_group


CONFIG = EvidenceLibraryConfig(
    description=__doc__ or "",
    default_assays_csv=(
        "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/assay_manifest/v3/assay_to_family.tsv"
    ),
    default_activities_csv=(
        "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/assay_manifest/v3/"
        "distance_activity_evidence.csv"
    ),
    default_out_dir=(
        "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension"
    ),
    evidence_filename="bbb_distance_extension_evidence.jsonl",
    index_filename="bbb_distance_extension_index.pkl",
    meta_filename="bbb_distance_extension_index.meta.json",
    index_version="bbb_distance_extension_index.v3",
    assign_endpoint_group=assign_distance_endpoint_group,
)


def main(argv: list[str] | None = None) -> int:
    return run_builder(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
