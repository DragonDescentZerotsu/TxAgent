"""Build the leakage-filtered BBB Martins pair-bucket transfer policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    TransferPolicyBuildSpec,
    build_pair_bucket_transfer_policy as _build,
)
from tools.chembl_tool.common.starling.heldout_index import load_heldout_identity_keys
from tools.chembl_tool.tasks.bbb_martins.build_starling_pair_bucket_sidecar import (
    DEFAULT_NORMALIZED_DIR,
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bbb_martins.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bbb_martins.starling_pair_bucket_transfer_policy import (
    TRANSFER_POLICY_PROFILE,
)
from tools.chembl_tool.tasks.bbb_martins.starling_pair_buckets import (
    BBB_MARTINS_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)


HELDOUT_LABEL_PATHS = (
    Path("data/processed_starling/BBB_Martins/random/heldout_molecule_labels.jsonl"),
    Path("data/processed_starling/BBB_Martins/scaffold/heldout_molecule_labels.jsonl"),
)


def load_union_heldout_identity_keys() -> set[str]:
    keys: set[str] = set()
    for path in HELDOUT_LABEL_PATHS:
        keys |= load_heldout_identity_keys(path)
    return keys


BUILD_SPEC = TransferPolicyBuildSpec(
    profile=TRANSFER_POLICY_PROFILE,
    pair_bucket_version=BBB_MARTINS_PAIR_BUCKET_VERSION,
    source_pair_fields=SOURCE_PAIR_FIELDS,
    auxiliary_mapping_version=MAPPING_VERSION,
    auxiliary_attachment_version=AUXILIARY_ATTACHMENT_VERSION,
    include_soft_transfer_contract=True,
    heldout_key_loader=load_union_heldout_identity_keys,
    heldout_identity_column="canonical_smiles",
)


def build_pair_bucket_transfer_policy(
    *,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    return _build(
        spec=BUILD_SPEC,
        records_path=records_path,
        pair_bucket_records_path=pair_bucket_records_path,
        pair_bucket_metadata_path=pair_bucket_metadata_path,
        auxiliary_manifest_path=auxiliary_manifest_path,
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default=str(DEFAULT_NORMALIZED_DIR / "03_records/records.parquet")
    )
    parser.add_argument(
        "--pair-bucket-records",
        default=str(DEFAULT_NORMALIZED_DIR / "04_pair_buckets" / PAIR_BUCKET_RECORDS_FILENAME),
    )
    parser.add_argument(
        "--pair-bucket-metadata",
        default=str(DEFAULT_NORMALIZED_DIR / "04_pair_buckets" / PAIR_BUCKET_METADATA_FILENAME),
    )
    parser.add_argument(
        "--auxiliary-manifest",
        default=str(DEFAULT_NORMALIZED_DIR / "02_normalized/auxiliary_mapping_manifest.json"),
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_NORMALIZED_DIR / "05_assay_transfer_policy")
    )
    args = parser.parse_args(argv)
    payload = build_pair_bucket_transfer_policy(
        records_path=args.records,
        pair_bucket_records_path=args.pair_bucket_records,
        pair_bucket_metadata_path=args.pair_bucket_metadata,
        auxiliary_manifest_path=args.auxiliary_manifest,
        out_dir=args.out_dir,
    )
    print(payload["summary"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "BUILD_SPEC",
    "HELDOUT_LABEL_PATHS",
    "POLICY_FILENAME",
    "build_pair_bucket_transfer_policy",
    "load_union_heldout_identity_keys",
]

