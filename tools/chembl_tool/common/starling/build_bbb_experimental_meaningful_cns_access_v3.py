"""Build the corrected v3 BBB meaningful-CNS-access benchmark lineage."""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.common.starling.build_bbb_experimental_meaningful_cns_access import (
    BBBGoldBuildSpec,
    _parse_args,
    _validate_build_scope,
    run_build,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v3 import (
    CONTRACT_VERSION,
    load_label_decisions,
)


BUILD_SPEC = BBBGoldBuildSpec(
    lineage="experimental_meaningful_cns_access_v3",
    protocol_version="starling_experimental_meaningful_cns_access_benchmark.v3",
    contract_version=CONTRACT_VERSION,
    default_output_root=Path(
        "data/processed_starling_experimental_meaningful_cns_access_v3"
    ),
    preserve_split_root=Path(
        "data/processed_starling_experimental_meaningful_cns_access_v2/"
        "BBB_Martins/scaffold"
    ),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv, default_output_root=BUILD_SPEC.default_output_root)
    _validate_build_scope(args, BUILD_SPEC)
    decisions, metadata = load_label_decisions(
        revision=args.bbb_revision,
        max_rows=args.max_rows,
    )
    return run_build(args, decisions, metadata, spec=BUILD_SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
