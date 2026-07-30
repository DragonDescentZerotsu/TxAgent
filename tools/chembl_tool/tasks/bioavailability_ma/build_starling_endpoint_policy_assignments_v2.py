"""Build standalone endpoint-specific assay-transfer policy v2 assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.audit import (
    read_parquet_records,
    write_parquet,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.tasks.bioavailability_ma.starling_endpoint_policies_v2 import (
    assign_endpoint_policies_v2,
)


DEFAULT_V5_DIR = (
    Path("outputs/chembl_tool/tasks/bioavailability_ma/evidence_library")
    / "starling_normalized_v5"
)
DEFAULT_OUT_DIR = DEFAULT_V5_DIR / "endpoint_policies" / "v2"
ASSIGNMENTS_FILENAME = "endpoint_policy_assignments.parquet"
REGISTRY_FILENAME = "endpoint_policy_registry.json"
METADATA_FILENAME = "endpoint_policy_metadata.json"


def build_endpoint_policy_assignments_v2(
    *,
    records_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    records_path = Path(records_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    records = read_parquet_records(records_path)
    assignments, registry, metadata = assign_endpoint_policies_v2(records)
    if not all(metadata["validations"].values()):
        raise ValueError(f"endpoint-policy v2 assignment audit failed: {metadata}")

    assignments_path = target / ASSIGNMENTS_FILENAME
    registry_path = target / REGISTRY_FILENAME
    write_parquet(assignments_path, assignments)
    _write_json(registry_path, registry)
    metadata.update(
        {
            "input": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "records": len(records),
            },
            "outputs": {
                "assignments": {
                    "path": str(assignments_path),
                    "sha256": file_sha256(assignments_path),
                },
                "registry": {
                    "path": str(registry_path),
                    "sha256": file_sha256(registry_path),
                },
            },
        }
    )
    _write_json(target / METADATA_FILENAME, metadata)
    return metadata


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_V5_DIR / "records.parquet"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = build_endpoint_policy_assignments_v2(
        records_path=args.records,
        out_dir=args.out_dir,
    )
    print(
        "[build_starling_endpoint_policy_assignments_v2] "
        f"records={metadata['stats']['assignment_records']:,} "
        f"assigned={metadata['stats']['assigned_records']:,} "
        f"policies={metadata['stats']['endpoint_policies']:,} out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
