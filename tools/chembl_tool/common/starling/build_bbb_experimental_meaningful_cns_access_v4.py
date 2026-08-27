"""Build BBB v4 after the frozen experimental metric-direction review."""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.common.starling.build_bbb_experimental_meaningful_cns_access import (
    BBBGoldBuildSpec,
    _parse_args,
    _validate_build_scope,
    run_build,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    SOURCE_REVISION,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v4 import (
    CONTRACT_VERSION,
    load_label_decisions_from_arrow,
)


BUILD_SPEC = BBBGoldBuildSpec(
    lineage="experimental_meaningful_cns_access_v4",
    protocol_version="starling_experimental_meaningful_cns_access_benchmark.v4",
    contract_version=CONTRACT_VERSION,
    default_output_root=Path(
        "data/processed_starling_experimental_meaningful_cns_access_v4"
    ),
    preserve_split_root=Path(
        "data/processed_starling_experimental_meaningful_cns_access_v3/"
        "BBB_Martins/scaffold"
    ),
    allow_new_parents_in_preserved_split=True,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(
        argv,
        default_output_root=BUILD_SPEC.default_output_root,
        allow_source_arrow=True,
    )
    _validate_build_scope(args, BUILD_SPEC)
    if not args.source_arrow:
        raise ValueError(
            "BBB v4 requires --source-arrow for the pinned source copy used by the "
            "row-level manual review"
        )
    decisions, metadata = load_label_decisions_from_arrow(args.source_arrow)
    metadata["benchmark_lineage_source_revision"] = SOURCE_REVISION
    return run_build(args, decisions, metadata, spec=BUILD_SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
