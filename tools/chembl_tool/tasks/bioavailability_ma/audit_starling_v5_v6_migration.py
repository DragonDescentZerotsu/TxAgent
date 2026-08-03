"""Audit the immutable v5 to globally reconciled v6 Starling migration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


EVIDENCE_ROOT = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library"
)
DEFAULT_V5_RECORDS = EVIDENCE_ROOT / "starling_normalized_v5/records.parquet"
DEFAULT_V6_RECORDS = EVIDENCE_ROOT / "starling_normalized_v6/03_records/records.parquet"
DEFAULT_AUXILIARY_MANIFEST = (
    EVIDENCE_ROOT
    / "starling_normalized_v6/02_normalized/auxiliary_mapping_manifest.json"
)
DEFAULT_OUTPUT = (
    EVIDENCE_ROOT
    / "starling_normalized_v6/08_audits/v5_v6_migration/migration_summary.json"
)

HEURISTIC_FIELDS = (
    "canonical_dose_key",
    "canonical_assay_system",
    "canonical_species",
)
GLOBAL_FIELDS = (
    "global_context",
    "global_species_context",
    "auxiliary_mapping_status",
)
APPLICABLE_SOURCES = frozenset({"fa", "fg", "fh"})


def audit_migration(
    *,
    v5_records: str | Path,
    v6_records: str | Path,
    auxiliary_manifest: str | Path,
) -> dict[str, Any]:
    v5_path, v6_path = Path(v5_records), Path(v6_records)
    auxiliary_path = Path(auxiliary_manifest)
    v5 = pd.read_parquet(v5_path)
    v6 = pd.read_parquet(v6_path)
    required = {"normalized_record_id", "source_id", *GLOBAL_FIELDS}
    missing = sorted(required - set(v6.columns))
    if missing:
        raise ValueError(f"v6 migration audit missing columns: {missing}")

    v5_ids = set(v5["normalized_record_id"].astype(str))
    v6_ids = set(v6["normalized_record_id"].astype(str))
    status_counts = {
        str(source): dict(sorted(Counter(group["auxiliary_mapping_status"].astype(str)).items()))
        for source, group in v6.groupby("source_id", dropna=False)
    }
    applicable = v6["source_id"].isin(APPLICABLE_SOURCES)
    inapplicable = ~applicable
    auxiliary_payload = json.loads(auxiliary_path.read_text(encoding="utf-8"))

    validations = {
        "v5_and_v6_ids_unique": bool(
            v5["normalized_record_id"].is_unique
            and v6["normalized_record_id"].is_unique
        ),
        "v6_heuristic_fields_absent": not any(
            field in v6.columns for field in HEURISTIC_FIELDS
        ),
        "v6_global_fields_present": True,
        "auxiliary_attachment_version_available": bool(
            auxiliary_payload.get("attachment_version")
            or auxiliary_payload.get("auxiliary_attachment_version")
            or (
                "auxiliary_attachment_version" in v6.columns
                and v6["auxiliary_attachment_version"].notna().all()
            )
        ),
        "all_fa_fg_fh_rows_mapped": bool(
            (v6.loc[applicable, "auxiliary_mapping_status"] == "mapped").all()
        ),
        "all_other_rows_explicitly_not_applicable": bool(
            (
                v6.loc[inapplicable, "auxiliary_mapping_status"]
                == "not_applicable"
            ).all()
        ),
        "auxiliary_manifest_validations_pass": all(
            auxiliary_payload.get("coverage", {})
            .get("validations", {})
            .values()
        ),
    }
    if not all(validations.values()):
        raise ValueError(f"v5-v6 migration validation failed: {validations}")

    return {
        "audit_version": "bioavailability_ma_starling_v5_v6_migration.v1",
        "scope": {
            "v5_is_read_only": True,
            "v6_is_read_only": True,
            "record_rewrite": False,
            "pair_enumeration": False,
        },
        "inputs": {
            "v5_records": {"path": str(v5_path), "sha256": file_sha256(v5_path)},
            "v6_records": {"path": str(v6_path), "sha256": file_sha256(v6_path)},
            "auxiliary_manifest": {
                "path": str(auxiliary_path),
                "sha256": file_sha256(auxiliary_path),
                "mapping_sha256": auxiliary_payload.get("mapping_sha256"),
                "mapping_version": auxiliary_payload.get("mapping_version"),
            },
        },
        "records": {
            "v5": len(v5),
            "v6": len(v6),
            "common_normalized_record_ids": len(v5_ids & v6_ids),
            "v5_only_normalized_record_ids": len(v5_ids - v6_ids),
            "v6_only_normalized_record_ids": len(v6_ids - v5_ids),
            "v6_by_source": {
                str(key): int(value)
                for key, value in sorted(v6["source_id"].value_counts(dropna=False).items())
            },
        },
        "schema": {
            "heuristic_fields": list(HEURISTIC_FIELDS),
            "heuristic_fields_present_in_v5": [
                field for field in HEURISTIC_FIELDS if field in v5.columns
            ],
            "heuristic_fields_present_in_v6": [
                field for field in HEURISTIC_FIELDS if field in v6.columns
            ],
            "global_fields": list(GLOBAL_FIELDS),
            "auxiliary_status_by_source": status_counts,
        },
        "validations": validations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-records", default=str(DEFAULT_V5_RECORDS))
    parser.add_argument("--v6-records", default=str(DEFAULT_V6_RECORDS))
    parser.add_argument("--auxiliary-manifest", default=str(DEFAULT_AUXILIARY_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = audit_migration(
        v5_records=args.v5_records,
        v6_records=args.v6_records,
        auxiliary_manifest=args.auxiliary_manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[audit_starling_v5_v6_migration] "
        f"v5={result['records']['v5']:,} v6={result['records']['v6']:,} "
        f"common={result['records']['common_normalized_record_ids']:,} out={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
