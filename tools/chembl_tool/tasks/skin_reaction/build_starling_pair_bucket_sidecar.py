"""Build the Skin_Reaction source-aware pair-bucket sidecar.

The command reads normalized v6 ``records.parquet`` and never rewrites it.
It materializes bucket membership and audits only; it does not enumerate or
label molecular pairs.

All four sources declare comparison fields.  The two qualitative sources
reach buckets only when their categorical encoder produced a finite value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_buckets import (
    materialize_pair_buckets,
    read_pair_bucket_input,
)
from tools.chembl_tool.common.starling.reference_semantics import (
    ReferenceEligibilitySpec,
)
from tools.chembl_tool.tasks.skin_reaction.starling_pair_buckets import (
    ENDPOINT_FIELD_BY_SOURCE,
    SKIN_REACTION_PAIR_BUCKET_VERSION,
    SKIN_REACTION_V7_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import DEFAULT_OUT_DIR
from tools.chembl_tool.tasks.skin_reaction.starling_schema import RECORD_CONTRACT


DEFAULT_NORMALIZED_DIR = Path(DEFAULT_OUT_DIR)
DEFAULT_PAIR_BUCKET_DIR = DEFAULT_NORMALIZED_DIR / "04_pair_buckets"
PAIR_BUCKET_RECORDS_FILENAME = "pair_bucket_records.parquet"
PAIR_BUCKET_METADATA_FILENAME = "pair_bucket_metadata.json"


def build_sidecar(
    *,
    records_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    records_path = Path(records_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    records, v7, pair_fields = read_pair_bucket_input(
        records_path,
        v7_source_fields={
            source: spec.additional_dimensions
            for source, spec in RECORD_CONTRACT.pair_buckets.items()
        },
        legacy_source_fields=SOURCE_PAIR_FIELDS,
        legacy_endpoint_field_by_source=ENDPOINT_FIELD_BY_SOURCE,
    )
    sidecar_rows, metadata = materialize_pair_buckets(
        records,
        source_required_fields=pair_fields,
        contract_version=(
            SKIN_REACTION_V7_PAIR_BUCKET_VERSION
            if v7
            else SKIN_REACTION_PAIR_BUCKET_VERSION
        ),
        endpoint_field_by_source=None if v7 else ENDPOINT_FIELD_BY_SOURCE,
        reference_eligibility_by_source=(
            {
                source: ReferenceEligibilitySpec(
                    spec.eligible_reference_scopes,
                    spec.reference_basis_required,
                )
                for source, spec in RECORD_CONTRACT.pair_buckets.items()
            }
            if v7
            else None
        ),
        required_known_fields_by_source=(
            {
                source: spec.required_known_dimensions
                for source, spec in RECORD_CONTRACT.pair_buckets.items()
                if spec.required_known_dimensions
            }
            if v7
            else None
        ),
    )
    if not all(metadata["validations"].values()):
        raise ValueError(
            f"pair-bucket identity/coverage audit failed: {metadata['validations']}"
        )

    records_output = target / PAIR_BUCKET_RECORDS_FILENAME
    write_parquet(records_output, sidecar_rows)
    metadata.update(
        {
            "input": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "records": len(records),
            },
            "output": {
                "path": str(records_output),
                "sha256": file_sha256(records_output),
            },
            "scope": {
                "pair_enumeration": False,
                "pair_labels": False,
                "comparison_thresholds": False,
                "modeling_dataset": False,
            },
        }
    )
    _write_json(target / PAIR_BUCKET_METADATA_FILENAME, metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = build_sidecar(records_path=args.records, out_dir=args.out_dir)
    print(
        "[build_starling_pair_bucket_sidecar] "
        f"records={metadata['stats']['sidecar_records']:,} "
        f"eligible={metadata['stats']['eligible_records']:,} "
        f"buckets={metadata['stats']['buckets']:,} out={args.out_dir}",
        flush=True,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default=str(DEFAULT_NORMALIZED_DIR / "03_records/records.parquet")
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_PAIR_BUCKET_DIR))
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
