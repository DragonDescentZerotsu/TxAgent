"""Build the versioned canonical direct-F source and residual exposure source."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from data.processing.gold_labels.benchmark_dataset import parse_numeric_interval
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    clean_source_values,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.canonical_source_v3 import (
    CANONICAL_SOURCE_DIR,
    CANONICAL_VERSION,
    DEDUP_AUDIT_PATH,
    DIRECT_CLAIMS_PATH,
    DIRECT_REJECTED_ROWS_PATH,
    DIRECT_SOURCE_ROWS_PATH,
    HF_NONDIRECT_RECORDS_PATH,
    HF_SNAPSHOT_PATH,
    LOCAL_PARTITION_AUDIT_PATH,
    MANIFEST_PATH,
    PRIOR_DIRECT_CLAIMS_PATH,
    RESIDUAL_MANIFEST_PATH,
    RESIDUAL_RECORDS_PATH,
    RESIDUAL_SOURCE_DIR,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.canonical_source import (
    RAW_HF_SOURCE_PATH,
    REVIEWED_NAME_SMILES_CONFLICTS_PATH,
    REVIEWED_SOURCE_DROPS_PATH,
    REVIEWED_SOURCE_REPAIRS_PATH,
    SMILES_IDENTITY_AUDIT_PATH,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.canonical_source import (
    DIRECT_REPORT_TYPES,
    HF_SOURCE_DATASET,
    HF_SOURCE_REVISION,
    LOCAL_PARTITION_DIRECT,
    RAW_LOCAL_SOURCE_PATH,
    classify_local_record,
    local_classification_signals,
    local_value_percent,
    nondirect_measurement_fields,
)


MATCH_VALUE_TOLERANCE_PERCENT = 1.0
PAPER_DEDUP_VERSION = "bioavailability_paper_direct_claim_dedup.v2"
SOURCE_VALUE_AUDIT_FILENAME = "source_value_cleaning_audit.parquet"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    import pandas as pd

    hf_path = Path(args.hf_source)
    hf_frame = pd.read_parquet(hf_path)
    if "source_index" not in hf_frame:
        hf_frame.insert(0, "source_index", range(len(hf_frame)))
    local_path = Path(args.local_source)
    local_frame = pd.read_parquet(local_path)
    if "source_index" not in local_frame:
        local_frame.insert(0, "source_index", range(len(local_frame)))
    repaired_hf, repaired_local, cleaning = _apply_reviewed_source_edits(
        hf_frame,
        local_frame,
        hf_source_path=hf_path,
        local_source_path=local_path,
    )
    prior_claim_ids = _load_prior_claim_ids(Path(args.prior_direct_claims))

    outputs = build_canonical_frames(
        repaired_hf,
        repaired_local,
        hf_dataset=args.hf_dataset,
        hf_revision=args.hf_revision,
        local_source_path=local_path,
        value_tolerance_percent=args.match_value_tolerance_percent,
        prior_claim_ids=prior_claim_ids,
    )
    if outputs["stats"]["n_direct_source_rows_with_uid"] != outputs["stats"][
        "n_direct_source_rows_before_dedup"
    ]:
        raise ValueError("canonical direct source rows require complete source_row_uid")

    canonical_dir = Path(args.canonical_dir)
    residual_dir = Path(args.residual_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hf_snapshot": canonical_dir / HF_SNAPSHOT_PATH.name,
        "hf_nondirect_records": canonical_dir / HF_NONDIRECT_RECORDS_PATH.name,
        "direct_source_rows": canonical_dir / DIRECT_SOURCE_ROWS_PATH.name,
        "direct_claims": canonical_dir / DIRECT_CLAIMS_PATH.name,
        "direct_rejected_rows": canonical_dir / DIRECT_REJECTED_ROWS_PATH.name,
        "dedup_audit": canonical_dir / DEDUP_AUDIT_PATH.name,
        "local_partition_audit": canonical_dir / LOCAL_PARTITION_AUDIT_PATH.name,
        "source_value_cleaning_audit": canonical_dir / SOURCE_VALUE_AUDIT_FILENAME,
        "residual_records": residual_dir / RESIDUAL_RECORDS_PATH.name,
    }
    # Preserve the immutable input snapshot; reviewed edits are recorded in the
    # audit and appear only in the derived direct/nondirect/residual views.
    hf_frame.to_parquet(paths["hf_snapshot"], index=False)
    outputs["hf_nondirect_records"].to_parquet(
        paths["hf_nondirect_records"], index=False
    )
    outputs["direct_source_rows"].to_parquet(paths["direct_source_rows"], index=False)
    outputs["direct_claims"].to_parquet(paths["direct_claims"], index=False)
    outputs["direct_rejected_rows"].to_parquet(paths["direct_rejected_rows"], index=False)
    outputs["dedup_audit"].to_parquet(paths["dedup_audit"], index=False)
    outputs["local_partition_audit"].to_parquet(paths["local_partition_audit"], index=False)
    pd.DataFrame(cleaning.audit_rows).to_parquet(
        paths["source_value_cleaning_audit"], index=False
    )
    outputs["residual_records"].to_parquet(paths["residual_records"], index=False)

    stats = outputs["stats"]
    manifest = {
        "contract_version": CANONICAL_VERSION,
        "hf_source": {
            "dataset": args.hf_dataset,
            "revision": args.hf_revision,
            "split": "train",
            "path": str(hf_path),
            "sha256": _sha256_file(hf_path),
            "snapshot_path": str(paths["hf_snapshot"]),
        },
        "local_source": {
            "path": str(local_path),
            "sha256": _sha256_file(local_path),
        },
        "classification_policy": {
            "direct_local_partition": LOCAL_PARTITION_DIRECT,
            "rule": "explicit absolute-bioavailability wording or oral/IV anchor, with relative signals taking precedence",
            "ambiguous_bioavailability_is_residual": True,
        },
        "deduplication_policy": {
            "grain": "cross-source one-to-one claim match within parent identity and PMID",
            "numeric_match": "complete intervals must be on the same 20% threshold side and within tolerance",
            "text_fallback": "identical normalized support text only when either side lacks a parsed numeric value",
            "value_tolerance_percent": args.match_value_tolerance_percent,
            "same-source_rows_are_not_collapsed": True,
        },
        "reviewed_source_edits": cleaning.manifest,
        "claim_id_lineage": {
            "prior_claims_path": str(args.prior_direct_claims),
            "prior_claims_sha256": _sha256_file(Path(args.prior_direct_claims)),
            **stats["claim_id_lineage"],
        },
        "stats": stats,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    for key, path in paths.items():
        manifest["paths"][f"{key}_sha256"] = _sha256_file(path)
    manifest_path = canonical_dir / MANIFEST_PATH.name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    residual_manifest = {
        "contract_version": CANONICAL_VERSION,
        "source_path": str(local_path),
        "source_sha256": _sha256_file(local_path),
        "reviewed_source_edits": cleaning.manifest,
        "residual_records_path": str(paths["residual_records"]),
        "residual_records_sha256": _sha256_file(paths["residual_records"]),
        "partition_reconciliation": stats["local_partition_reconciliation"],
        "absolute_direct_rows_in_residual": 0,
    }
    (residual_dir / RESIDUAL_MANIFEST_PATH.name).write_text(
        json.dumps(residual_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def _apply_reviewed_source_edits(
    hf_frame: Any,
    local_frame: Any,
    *,
    hf_source_path: Path,
    local_source_path: Path,
) -> tuple[Any, Any, Any]:
    """Apply the shared reviewed ledger to copied raw source frames."""
    specs = {
        "hf_bioavailability": {
            "frame": hf_frame.copy(),
            "path": hf_source_path,
            "record_id": "source_index",
            "measurement": "oral_bioavailability_value",
            "unit": None,
        },
        "oral_exposure": {
            "frame": local_frame.copy(),
            "path": local_source_path,
            "record_id": "extraction_id",
            "measurement": "parameter_value",
            "unit": "parameter_units",
        },
    }
    carriers: list[dict[str, Any]] = []
    for source_id, spec in specs.items():
        frame = spec["frame"].astype(object).where(spec["frame"].notna(), None)
        spec["frame"] = frame
        if frame["source_index"].astype(int).tolist() != list(range(len(frame))):
            raise ValueError(f"{source_id} source_index must be contiguous from zero")
        source_sha256 = _sha256_file(spec["path"])
        for source_index, row in enumerate(frame.to_dict(orient="records")):
            carriers.append(
                {
                    "cleaned_record_id": f"{source_id}:{source_index}",
                    "source_row_uid": str(row.get("source_row_uid") or ""),
                    "source_id": source_id,
                    "source_sha256": source_sha256,
                    "source_row_number": source_index + 1,
                    "source_record_id": _text(row.get(spec["record_id"])),
                    "source_smiles": _raw_value(row.get("smiles")),
                    "canonical_smiles": _raw_value(row.get("smiles")),
                    "measurement_text": _raw_value(row.get(spec["measurement"])),
                    "unit_text": (
                        _raw_value(row.get(spec["unit"])) if spec["unit"] else None
                    ),
                    "pmid": _raw_value(row.get("pmid")),
                    "support_text": _raw_value(row.get("support_text")),
                }
            )
    result = clean_source_values(
        carriers,
        task_id="bioavailability_ma",
        reviewed_repairs_path=REVIEWED_SOURCE_REPAIRS_PATH,
        reviewed_drops_path=REVIEWED_SOURCE_DROPS_PATH,
        smiles_identity_audit_path=SMILES_IDENTITY_AUDIT_PATH,
        reviewed_smiles_conflicts_path=REVIEWED_NAME_SMILES_CONFLICTS_PATH,
        source_ids=set(specs),
    )
    dropped: dict[str, set[int]] = defaultdict(set)
    field_map = {
        source_id: {
            "measurement_text": spec["measurement"],
            "unit_text": spec["unit"],
            "smiles": "smiles",
            "pmid": "pmid",
        }
        for source_id, spec in specs.items()
    }
    for audit in result.audit_rows:
        source_id = str(audit["source_id"])
        source_index = int(audit["source_row_number"]) - 1
        if audit["field"] == "record":
            dropped[source_id].add(source_index)
            continue
        column = field_map[source_id].get(str(audit["field"]))
        if column is not None:
            specs[source_id]["frame"].at[source_index, column] = audit["after"]
    for source_id, indices in dropped.items():
        specs[source_id]["frame"] = specs[source_id]["frame"].drop(index=sorted(indices))
    return specs["hf_bioavailability"]["frame"], specs["oral_exposure"]["frame"], result


def _load_prior_claim_ids(path: Path) -> dict[tuple[str, ...], str]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["canonical_claim_id", "source_record_ids"])
    output: dict[tuple[str, ...], str] = {}
    used_ids: set[str] = set()
    for row in frame.to_dict(orient="records"):
        members = tuple(sorted(str(value) for value in row["source_record_ids"]))
        claim_id = str(row["canonical_claim_id"])
        if not members or members in output or not claim_id or claim_id in used_ids:
            raise ValueError("prior canonical claims must have unique member sets and IDs")
        output[members] = claim_id
        used_ids.add(claim_id)
    return output


def build_canonical_frames(
    hf_frame: Any,
    local_frame: Any,
    *,
    hf_dataset: str = HF_SOURCE_DATASET,
    hf_revision: str = HF_SOURCE_REVISION,
    local_source_path: str | Path = RAW_LOCAL_SOURCE_PATH,
    value_tolerance_percent: float = MATCH_VALUE_TOLERANCE_PERCENT,
    prior_claim_ids: Mapping[tuple[str, ...], str] | None = None,
) -> dict[str, Any]:
    """Return inspectable derived frames without writing or mutating inputs."""
    import pandas as pd

    hf_rows, hf_rejections = _normalize_hf_rows(hf_frame, hf_dataset, hf_revision)
    hf_nondirect_frame = _hf_nondirect_records(hf_frame)
    local_rows: list[dict[str, Any]] = []
    local_identity_cache: dict[str, tuple[str, str] | None] = {}
    local_source = Path(local_source_path)
    local_source_revision = _sha256_file(local_source) if local_source.exists() else ""
    direct_rejected_indices: list[int] = []
    residual_indices: list[int] = []
    partition_audit: list[dict[str, Any]] = []
    partition_counts: Counter[str] = Counter()
    local_records = local_frame.to_dict(orient="records")
    for fallback_index, row in enumerate(local_records):
        source_index = int(row.get("source_index", fallback_index))
        partition, reason = classify_local_record(row)
        signals = local_classification_signals(row)
        partition_counts[partition] += 1
        partition_audit.append(
            {
                "source_index": source_index,
                "extraction_id": _text(row.get("extraction_id")),
                "pmid": _text(row.get("pmid")),
                "smiles": _text(row.get("smiles")),
                "exposure_measure": _text(row.get("exposure_measure")),
                "partition": partition,
                "partition_reason": reason,
                **signals,
            }
        )
        if partition == LOCAL_PARTITION_DIRECT:
            normalized = _normalize_local_row(
                row,
                source_index,
                local_source,
                local_source_revision,
                reason,
                identity_cache=local_identity_cache,
            )
            if normalized is not None:
                local_rows.append(normalized)
            else:
                direct_rejected_indices.append(fallback_index)
                partition_counts[partition] -= 1
                partition_counts["canonical_absolute_rejected_structure"] += 1
                partition_audit[-1]["partition"] = "canonical_absolute_rejected_structure"
                partition_audit[-1]["partition_reason"] = "absolute_row_has_invalid_or_missing_structure"
        else:
            residual_indices.append(fallback_index)

    direct_rows = [*hf_rows, *local_rows]
    claims, dedup_rows = _deduplicate_cross_source_claims(
        direct_rows,
        value_tolerance_percent=value_tolerance_percent,
        prior_claim_ids=prior_claim_ids,
    )
    residual_frame = local_frame.iloc[residual_indices].copy()
    direct_rejected_frame = local_frame.iloc[direct_rejected_indices].copy()
    audit_by_index = {int(row["source_index"]): row for row in partition_audit}
    residual_frame["partition"] = [audit_by_index[int(index)]["partition"] for index in residual_frame["source_index"]]
    residual_frame["partition_reason"] = [
        audit_by_index[int(index)]["partition_reason"] for index in residual_frame["source_index"]
    ]

    direct_frame = pd.DataFrame(direct_rows)
    claims_frame = pd.DataFrame(claims)
    dedup_frame = pd.DataFrame(dedup_rows, columns=_dedup_columns())
    partition_frame = pd.DataFrame(partition_audit)
    n_local = len(local_frame)
    n_direct_local = len(local_rows)
    n_direct_rejected = len(direct_rejected_frame)
    n_direct_partition = n_direct_local + n_direct_rejected
    n_residual = len(residual_frame)
    if n_direct_partition + n_residual != n_local:
        raise AssertionError("local partition does not reconcile to the immutable input")
    if LOCAL_PARTITION_DIRECT in set(residual_frame.get("partition", [])):
        raise AssertionError("absolute direct rows leaked into the residual source")

    stats = {
        "n_hf_snapshot_rows": len(hf_frame),
        "n_hf_direct_source_rows": len(hf_rows),
        "n_hf_nondirect_rows": len(hf_nondirect_frame),
        "hf_nondirect_measurement_unit_status_counts": dict(
            sorted(
                Counter(
                    hf_nondirect_frame["measurement_unit_extraction_status"]
                ).items()
            )
        ),
        "n_hf_direct_rejected_invalid_structure": hf_rejections,
        "hf_snapshot_reconciliation": {
            "input_rows": len(hf_frame),
            "direct_rows_with_valid_structure": len(hf_rows),
            "direct_rows_rejected_structure": hf_rejections,
            "nondirect_rows_retained": len(hf_nondirect_frame),
            "reconciles": (
                len(hf_rows) + hf_rejections + len(hf_nondirect_frame)
                == len(hf_frame)
            ),
        },
        "n_local_input_rows": n_local,
        "n_local_absolute_rows_removed_from_residual": n_direct_partition,
        "n_local_absolute_rows_with_valid_structure": n_direct_local,
        "n_local_absolute_rows_rejected_structure": n_direct_rejected,
        "n_local_residual_rows": n_residual,
        "local_partition_counts": dict(sorted(partition_counts.items())),
        "n_direct_source_rows_before_dedup": len(direct_rows),
        "n_direct_source_rows_with_uid": sum(
            bool(_text(row.get("source_row_uid"))) for row in direct_rows
        ),
        "n_cross_source_matches": len(dedup_rows),
        "n_canonical_direct_claims": len(claims),
        "claim_id_lineage": _claim_id_lineage(claims, prior_claim_ids),
        "claim_source_origin_counts": dict(
            Counter("+".join(row["source_origins"]) for row in claims)
        ),
        "local_partition_reconciliation": {
            "input_rows": n_local,
            "absolute_rows_removed_from_residual": n_direct_partition,
            "absolute_rows_with_valid_structure": n_direct_local,
            "absolute_rows_rejected_structure": n_direct_rejected,
            "residual_rows": n_residual,
            "reconciles": n_direct_partition + n_residual == n_local,
        },
    }
    return {
        "hf_nondirect_records": hf_nondirect_frame,
        "direct_source_rows": direct_frame,
        "direct_claims": claims_frame,
        "direct_rejected_rows": direct_rejected_frame,
        "dedup_audit": dedup_frame,
        "local_partition_audit": partition_frame,
        "residual_records": residual_frame,
        "stats": stats,
    }


def _hf_nondirect_records(frame: Any) -> Any:
    """Retain every non-direct HF row as a separate inference evidence source."""
    report_types = (
        frame["bioavailability_report_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    nondirect = frame.loc[~report_types.isin(DIRECT_REPORT_TYPES)].copy()
    nondirect["endpoint_name"] = "oral_bioavailability"
    extracted = nondirect["oral_bioavailability_value"].map(
        nondirect_measurement_fields
    )
    for field in (
        "measurement_text",
        "numeric_value",
        "value_units",
        "measurement_unit_extraction_status",
    ):
        nondirect[field] = [item[field] for item in extracted]
    return nondirect


def _normalize_hf_rows(frame: Any, dataset: str, revision: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    identity_cache: dict[str, tuple[str, str] | None] = {}
    for fallback_index, row in enumerate(frame.to_dict(orient="records")):
        report_type = _text(row.get("bioavailability_report_type")).lower()
        if report_type not in DIRECT_REPORT_TYPES:
            continue
        source_index = int(row.get("source_index", fallback_index))
        smiles = _text(row.get("smiles"))
        identity = _parent_identity(smiles, identity_cache)
        if identity is None:
            invalid += 1
            continue
        raw_value = _text(row.get("oral_bioavailability_value"))
        value = _hf_value_summary(raw_value)
        rows.append(
            {
                "source_origin": "hf",
                "source_dataset": dataset,
                "source_revision": revision,
                "source_record_id": f"hf:{source_index}",
                "source_row_uid": _text(row.get("source_row_uid")),
                "source_index": source_index,
                "pmid": _text(row.get("pmid")),
                "molecule_name": _text(row.get("molecule_name")),
                "smiles": smiles,
                "parent_smiles": identity[0],
                "parent_identity_key": identity[1],
                "bioavailability_report_type": report_type,
                "oral_bioavailability_value": raw_value,
                "value_percent": value[0],
                "value_lower_percent": value[1],
                "value_upper_percent": value[2],
                "value_parse_method": value[3],
                "value_units": "%" if value[0] is not None else "",
                "support_text": _text(row.get("support_text")),
                "species_or_population": _text(row.get("species_or_population")),
                "dose": _text(row.get("dose")),
                "oral_exposure_mode": _text(row.get("oral_exposure_mode")),
                "qualifying_conditions": _text(row.get("qualifying_conditions")),
                "comparator": _text(row.get("comparator")),
                "extra_details": _text(row.get("extra_details")),
                "classification_reason": "hf_direct_report_type",
            }
        )
    return rows, invalid


def _normalize_local_row(
    row: Mapping[str, Any],
    source_index: int,
    source_path: Path,
    source_revision: str,
    reason: str,
    *,
    identity_cache: dict[str, tuple[str, str] | None],
) -> dict[str, Any] | None:
    identity = _parent_identity(_text(row.get("smiles")), identity_cache)
    if identity is None:
        return None
    value = local_value_percent(row)
    raw_value = " ".join(
        part for part in (_text(row.get("parameter_value")), _text(row.get("parameter_units"))) if part
    )
    return {
        "source_origin": "local",
        "source_dataset": str(source_path),
        "source_revision": source_revision,
        "source_record_id": f"local:{source_index}:{_text(row.get('extraction_id')) or 'row'}",
        "source_row_uid": _text(row.get("source_row_uid")),
        "source_index": source_index,
        "pmid": _text(row.get("pmid")),
        "molecule_name": _text(row.get("global_identifier")),
        "smiles": _text(row.get("smiles")),
        "parent_smiles": identity[0],
        "parent_identity_key": identity[1],
        "bioavailability_report_type": "absolute",
        "oral_bioavailability_value": raw_value,
        "value_percent": value,
        "value_lower_percent": value,
        "value_upper_percent": value,
        "value_parse_method": "local_normalized_point" if value is not None else "",
        "value_units": "%" if value is not None else "",
        "support_text": _text(row.get("support_text")),
        "species_or_population": _text(row.get("study_context")),
        "dose": _text(row.get("oral_dose")),
        "oral_exposure_mode": "oral",
        "qualifying_conditions": _text(row.get("qualifying_conditions")),
        "comparator": _text(row.get("comparator_exposure")),
        "extra_details": _text(row.get("extra_details")),
        "classification_reason": reason,
    }


def _deduplicate_cross_source_claims(
    source_rows: list[dict[str, Any]],
    *,
    value_tolerance_percent: float,
    prior_claim_ids: Mapping[tuple[str, ...], str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_parent_pmid: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source_rows):
        by_parent_pmid[(row["parent_identity_key"], row["pmid"])].append(index)
    matched: dict[int, int] = {}
    audits: list[dict[str, Any]] = []
    for (parent_key, pmid), indices in sorted(by_parent_pmid.items()):
        if not pmid:
            continue
        hf_indices = [index for index in indices if source_rows[index]["source_origin"] == "hf"]
        local_indices = [index for index in indices if source_rows[index]["source_origin"] == "local"]
        candidates: list[tuple[float, float, int, int, str]] = []
        for local_index in local_indices:
            for hf_index in hf_indices:
                match = _claim_match(source_rows[hf_index], source_rows[local_index], value_tolerance_percent)
                if match is None:
                    continue
                value_distance, context_similarity, method = match
                candidates.append((value_distance, -context_similarity, local_index, hf_index, method))
        used_hf: set[int] = set()
        used_local: set[int] = set()
        for value_distance, negative_similarity, local_index, hf_index, method in sorted(candidates):
            if hf_index in used_hf or local_index in used_local:
                continue
            used_hf.add(hf_index)
            used_local.add(local_index)
            matched[local_index] = hf_index
            audits.append(
                {
                    "parent_identity_key": parent_key,
                    "pmid": pmid,
                    "hf_source_record_id": source_rows[hf_index]["source_record_id"],
                    "local_source_record_id": source_rows[local_index]["source_record_id"],
                    "hf_value_percent": source_rows[hf_index]["value_percent"],
                    "local_value_percent": source_rows[local_index]["value_percent"],
                    "value_distance_percent": value_distance,
                    "context_token_jaccard": -negative_similarity,
                    "match_method": method,
                }
            )

    reverse = {hf_index: local_index for local_index, hf_index in matched.items()}
    prior_claim_ids = prior_claim_ids or {}
    claims: list[dict[str, Any]] = []
    assigned_claim_ids: set[str] = set()
    for index, row in enumerate(source_rows):
        if index in matched:
            continue
        merged_indices = [index]
        if index in reverse:
            merged_indices.append(reverse[index])
        merged_rows = [source_rows[item] for item in merged_indices]
        representative = dict(merged_rows[0])
        members = sorted(
            (
                item["source_record_id"],
                _text(item.get("source_row_uid")),
            )
            for item in merged_rows
        )
        source_record_ids = [source_record_id for source_record_id, _ in members]
        source_row_uids = [source_row_uid for _, source_row_uid in members]
        member_key = tuple(source_record_ids)
        claim_id = prior_claim_ids.get(member_key) or (
            "BIOAVAIL_CLAIM_"
            + hashlib.sha256(
                (CANONICAL_VERSION + "|" + "|".join(source_record_ids)).encode(
                    "utf-8"
                )
            ).hexdigest()[:20].upper()
        )
        if claim_id in assigned_claim_ids:
            raise ValueError(f"canonical claim ID collision: {claim_id}")
        assigned_claim_ids.add(claim_id)
        representative.update(
            {
                "canonical_claim_id": claim_id,
                "source_origins": sorted({item["source_origin"] for item in merged_rows}),
                "source_record_ids": source_record_ids,
                "source_row_uids": source_row_uids,
                "source_datasets": sorted({item["source_dataset"] for item in merged_rows}),
                "n_source_records": len(merged_rows),
                "cross_source_deduplicated": len(merged_rows) > 1,
            }
        )
        claims.append(representative)
    claims.sort(key=lambda row: (row["parent_identity_key"], row["pmid"], row["canonical_claim_id"]))
    return claims, audits


def _claim_id_lineage(
    claims: list[dict[str, Any]],
    prior_claim_ids: Mapping[tuple[str, ...], str] | None,
) -> dict[str, int]:
    prior = prior_claim_ids or {}
    current = {
        tuple(sorted(str(value) for value in claim["source_record_ids"]))
        for claim in claims
    }
    return {
        "prior_claims": len(prior),
        "reused_claim_ids": len(current & set(prior)),
        "new_claim_ids": len(current - set(prior)),
        "retired_claim_ids": len(set(prior) - current),
    }


def deduplicate_paper_direct_claims(
    source_rows: list[dict[str, Any]],
    *,
    value_tolerance_percent: float = MATCH_VALUE_TOLERANCE_PERCENT,
) -> tuple[set[str], list[dict[str, Any]], dict[str, Any]]:
    """Collapse same-paper direct-F duplicates within and across sources."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(str(row["parent_identity_key"]), _text(row.get("pmid")))].append(row)

    retained: set[str] = set()
    audit: list[dict[str, Any]] = []
    for (_, pmid), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=_paper_representative_key)
        if not pmid:
            retained.update(str(row["source_record_id"]) for row in ordered)
            continue
        clusters: list[list[dict[str, Any]]] = []
        for row in ordered:
            compatible: list[tuple[float, float, list[dict[str, Any]], str]] = []
            for cluster in clusters:
                matches = [
                    _paper_claim_match(
                        member,
                        row,
                        tolerance=value_tolerance_percent,
                    )
                    for member in cluster
                ]
                if all(match is not None for match in matches):
                    compatible.append(
                        (
                            max(float(match[0]) for match in matches if match),
                            -min(float(match[1]) for match in matches if match),
                            cluster,
                            str(matches[0][2]),
                        )
                    )
            if compatible:
                compatible.sort(key=lambda item: (item[0], item[1], item[2][0]["source_record_id"]))
                compatible[0][2].append(row)
            else:
                clusters.append([row])

        for cluster in clusters:
            representative = cluster[0]
            representative_id = str(representative["source_record_id"])
            retained.add(representative_id)
            for duplicate in cluster[1:]:
                match = _paper_claim_match(
                    representative,
                    duplicate,
                    tolerance=value_tolerance_percent,
                )
                if match is None:
                    raise AssertionError("paper claim cluster lost complete-link compatibility")
                audit.append(
                    {
                        "retained_source_record_id": representative_id,
                        "discarded_source_record_id": str(duplicate["source_record_id"]),
                        "retained_source_origin": str(representative["source_origin"]),
                        "discarded_source_origin": str(duplicate["source_origin"]),
                        "parent_identity_key": str(representative["parent_identity_key"]),
                        "pmid": pmid,
                        "value_distance_percent": match[0],
                        "support_token_jaccard": match[1],
                        "match_method": match[2],
                    }
                )

    stats = {
        "version": PAPER_DEDUP_VERSION,
        "input_direct_source_rows": len(source_rows),
        "retained_direct_claims": len(retained),
        "discarded_duplicate_rows": len(audit),
        "cross_source_duplicates": sum(
            row["retained_source_origin"] != row["discarded_source_origin"]
            for row in audit
        ),
        "within_source_duplicates": sum(
            row["retained_source_origin"] == row["discarded_source_origin"]
            for row in audit
        ),
        "value_tolerance_percent": value_tolerance_percent,
        "dedup_key": "same_parent_pmid_compatible_value",
    }
    return retained, audit, stats


def apply_paper_direct_contract(
    records: list[dict[str, Any]],
    *,
    direct_source_rows_path: str | Path = DIRECT_SOURCE_ROWS_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split local direct/residual evidence and apply guarded claim deduplication."""
    import pandas as pd

    path = Path(direct_source_rows_path)
    frame = pd.read_parquet(path)
    source_rows = frame.astype(object).where(pd.notna(frame), None).to_dict(
        orient="records"
    )
    direct_ids = {str(row["source_record_id"]) for row in source_rows}
    canonical_by_source: dict[str, str] = {}
    source_id_by_record: dict[str, str] = {}
    for record in records:
        source_id = str(record.get("source_id") or "")
        source_record_id = str(record.get("source_record_id") or "")
        canonical_source_id = ""
        if source_id == "hf_bioavailability":
            canonical_source_id = f"hf:{source_record_id}"
        elif source_id == "oral_exposure":
            source_row = int(record.get("source_row_number") or 0) - 1
            canonical_source_id = f"local:{source_row}:{source_record_id}"
        if canonical_source_id in direct_ids:
            canonical_by_source[canonical_source_id] = str(record["canonical_record_id"])
        source_id_by_record[str(record["canonical_record_id"])] = canonical_source_id

    mapped_direct_ids = set(canonical_by_source)
    available_source_rows = [
        row for row in source_rows if str(row["source_record_id"]) in mapped_direct_ids
    ]
    retained_ids, audit, stats = deduplicate_paper_direct_claims(available_source_rows)
    prepared: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id") or "")
        canonical_source_id = source_id_by_record[str(record["canonical_record_id"])]
        if canonical_source_id in direct_ids:
            if canonical_source_id not in retained_ids:
                continue
        if source_id == "oral_exposure":
            record = dict(record)
            direct = canonical_source_id in direct_ids
            record["canonical_paper_direct_scope"] = "direct" if direct else "residual"
            record["group_id"] = (
                "Observed.direct_oral_bioavailability"
                if direct
                else "Observed.oral_auc_cmax_exposure"
            )
        prepared.append(record)

    missing = direct_ids - mapped_direct_ids
    for row in audit:
        row["retained_canonical_record_id"] = canonical_by_source[
            row["retained_source_record_id"]
        ]
        row["discarded_canonical_record_id"] = canonical_by_source[
            row["discarded_source_record_id"]
        ]
    stats.update(
        {
            "input_v7_records": len(records),
            "output_v7_records": len(prepared),
            "input_direct_source_rows": len(source_rows),
            "dedup_input_mapped_direct_source_rows": len(available_source_rows),
            "mapped_direct_source_rows": len(mapped_direct_ids),
            "unmapped_direct_source_rows": len(missing),
            "unmapped_direct_source_record_ids": sorted(missing),
            "direct_source_rows_path": str(path),
        }
    )
    return prepared, audit, stats


def _paper_claim_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[float, float, str] | None:
    left_value = _float_or_none(left.get("value_percent"))
    right_value = _float_or_none(right.get("value_percent"))
    left_support = _normalized_text(left.get("support_text"))
    right_support = _normalized_text(right.get("support_text"))
    support_similarity = _token_jaccard(left_support, right_support)
    if left_value is None or right_value is None:
        if left_support and left_support == right_support:
            return 0.0, 1.0, "identical_normalized_support_text"
        return None
    left_label = _claim_threshold_label(left)
    right_label = _claim_threshold_label(right)
    if left_label is None or left_label != right_label:
        return None
    distance = min(
        _interval_distance(left_value, _float_or_none(right.get("value_lower_percent")), _float_or_none(right.get("value_upper_percent"))),
        _interval_distance(right_value, _float_or_none(left.get("value_lower_percent")), _float_or_none(left.get("value_upper_percent"))),
    )
    if distance > tolerance:
        return None
    return distance, support_similarity, "numeric_parent_pmid_value_match"


def _paper_representative_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    labelable = (
        _claim_threshold_label(row) is not None
        and not _text(row.get("qualifying_conditions"))
    )
    lower = _float_or_none(row.get("value_lower_percent"))
    upper = _float_or_none(row.get("value_upper_percent"))
    width = upper - lower if lower is not None and upper is not None else math.inf
    context = sum(bool(_text(row.get(field))) for field in (
        "support_text", "species_or_population", "dose", "oral_exposure_mode",
        "qualifying_conditions", "comparator", "extra_details",
    ))
    return (
        str(row.get("source_origin")) != "hf",
        not labelable,
        width,
        -context,
        str(row.get("source_record_id")),
    )


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9.]+", left))
    right_tokens = set(re.findall(r"[a-z0-9.]+", right))
    return (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )


def _claim_match(
    hf_row: Mapping[str, Any],
    local_row: Mapping[str, Any],
    tolerance: float,
) -> tuple[float, float, str] | None:
    hf_value = _float_or_none(hf_row.get("value_percent"))
    local_value = _float_or_none(local_row.get("value_percent"))
    similarity = _context_similarity(hf_row, local_row)
    if hf_value is not None and local_value is not None:
        lower = _float_or_none(hf_row.get("value_lower_percent"))
        upper = _float_or_none(hf_row.get("value_upper_percent"))
        interval_distance = _interval_distance(local_value, lower, upper)
        hf_label = _claim_threshold_label(hf_row)
        local_label = _claim_threshold_label(local_row)
        if (
            hf_label is not None
            and hf_label == local_label
            and interval_distance <= tolerance
        ):
            return interval_distance, similarity, "numeric_value_or_interval_match"
    hf_support = _normalized_text(hf_row.get("support_text"))
    local_support = _normalized_text(local_row.get("support_text"))
    if (
        (hf_value is None or local_value is None)
        and hf_support
        and hf_support == local_support
    ):
        return 0.0, similarity, "identical_normalized_support_text"
    return None


def _hf_value_summary(raw_value: str) -> tuple[float | None, float | None, float | None, str]:
    interval = parse_numeric_interval(raw_value, fraction_to_percent=True)
    if interval is None:
        return None, None, None, ""
    lower = None if not math.isfinite(interval.lower) else float(interval.lower)
    upper = None if not math.isfinite(interval.upper) else float(interval.upper)
    if lower is not None and upper is not None:
        value = (lower + upper) / 2.0
    else:
        value = lower if lower is not None else upper
    return value, lower, upper, interval.method


def _parent_identity(
    smiles: str,
    cache: dict[str, tuple[str, str] | None],
) -> tuple[str, str] | None:
    if smiles in cache:
        return cache[smiles]
    identity = normalize_molecule_identity(smiles)
    if identity.status != "ok" or not identity.parent_smiles:
        cache[smiles] = None
    else:
        cache[smiles] = (
            identity.parent_smiles,
            identity.parent_inchi_key or identity.parent_smiles,
        )
    return cache[smiles]


def _context_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    fields = ("species_or_population", "dose", "oral_exposure_mode", "qualifying_conditions", "comparator")
    left_tokens = set(re.findall(r"[a-z0-9.]+", " ".join(_text(left.get(field)).lower() for field in fields)))
    right_tokens = set(re.findall(r"[a-z0-9.]+", " ".join(_text(right.get(field)).lower() for field in fields)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _interval_distance(value: float, lower: float | None, upper: float | None) -> float:
    if lower is None and upper is None:
        return math.inf
    if lower is None:
        return max(0.0, value - float(upper))
    if upper is None:
        return max(0.0, float(lower) - value)
    if lower <= value <= upper:
        return 0.0
    return min(abs(value - lower), abs(value - upper))


def _claim_threshold_label(row: Mapping[str, Any]) -> int | None:
    """Return a 20% label only when the complete claim interval is one-sided."""
    value = _float_or_none(row.get("value_percent"))
    lower = _float_or_none(row.get("value_lower_percent"))
    upper = _float_or_none(row.get("value_upper_percent"))
    if lower is None and upper is None:
        if value is None:
            return None
        lower = upper = value
    if lower is not None and lower >= 20.0:
        return 1
    if upper is not None and upper < 20.0:
        return 0
    return None


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%]+", " ", _text(value).lower())).strip()


def _dedup_columns() -> list[str]:
    return [
        "parent_identity_key",
        "pmid",
        "hf_source_record_id",
        "local_source_record_id",
        "hf_value_percent",
        "local_value_percent",
        "value_distance_percent",
        "context_token_jaccard",
        "match_method",
    ]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _raw_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset", default=HF_SOURCE_DATASET)
    parser.add_argument("--hf-revision", default=HF_SOURCE_REVISION)
    parser.add_argument("--hf-source", default=str(RAW_HF_SOURCE_PATH))
    parser.add_argument("--local-source", default=str(RAW_LOCAL_SOURCE_PATH))
    parser.add_argument("--prior-direct-claims", default=str(PRIOR_DIRECT_CLAIMS_PATH))
    parser.add_argument("--canonical-dir", default=str(CANONICAL_SOURCE_DIR))
    parser.add_argument("--residual-dir", default=str(RESIDUAL_SOURCE_DIR))
    parser.add_argument(
        "--match-value-tolerance-percent",
        type=float,
        default=MATCH_VALUE_TOLERANCE_PERCENT,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
