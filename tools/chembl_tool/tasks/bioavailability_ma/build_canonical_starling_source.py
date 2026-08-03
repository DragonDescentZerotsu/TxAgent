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
from tools.chembl_tool.common.starling.benchmark_dataset import parse_numeric_interval
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    CANONICAL_SOURCE_DIR,
    CANONICAL_VERSION,
    DEDUP_AUDIT_PATH,
    DIRECT_CLAIMS_PATH,
    DIRECT_REJECTED_ROWS_PATH,
    DIRECT_REPORT_TYPES,
    DIRECT_SOURCE_ROWS_PATH,
    HF_SNAPSHOT_PATH,
    HF_SOURCE_DATASET,
    HF_SOURCE_REVISION,
    LOCAL_PARTITION_AUDIT_PATH,
    LOCAL_PARTITION_DIRECT,
    MANIFEST_PATH,
    RAW_LOCAL_SOURCE_PATH,
    RESIDUAL_MANIFEST_PATH,
    RESIDUAL_RECORDS_PATH,
    RESIDUAL_SOURCE_DIR,
    classify_local_record,
    local_classification_signals,
    local_value_percent,
)


MATCH_VALUE_TOLERANCE_PERCENT = 1.0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import HfApi

    hf_dataset = load_dataset(args.hf_dataset, split="train", revision=args.hf_revision)
    resolved_revision = HfApi().dataset_info(args.hf_dataset, revision=args.hf_revision).sha
    hf_frame = hf_dataset.to_pandas()
    hf_frame.insert(0, "source_index", range(len(hf_frame)))
    local_path = Path(args.local_source)
    local_frame = pd.read_parquet(local_path)
    local_frame.insert(0, "source_index", range(len(local_frame)))

    outputs = build_canonical_frames(
        hf_frame,
        local_frame,
        hf_dataset=args.hf_dataset,
        hf_revision=resolved_revision,
        local_source_path=local_path,
        value_tolerance_percent=args.match_value_tolerance_percent,
    )

    canonical_dir = Path(args.canonical_dir)
    residual_dir = Path(args.residual_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hf_snapshot": canonical_dir / HF_SNAPSHOT_PATH.name,
        "direct_source_rows": canonical_dir / DIRECT_SOURCE_ROWS_PATH.name,
        "direct_claims": canonical_dir / DIRECT_CLAIMS_PATH.name,
        "direct_rejected_rows": canonical_dir / DIRECT_REJECTED_ROWS_PATH.name,
        "dedup_audit": canonical_dir / DEDUP_AUDIT_PATH.name,
        "local_partition_audit": canonical_dir / LOCAL_PARTITION_AUDIT_PATH.name,
        "residual_records": residual_dir / RESIDUAL_RECORDS_PATH.name,
    }
    hf_frame.to_parquet(paths["hf_snapshot"], index=False)
    outputs["direct_source_rows"].to_parquet(paths["direct_source_rows"], index=False)
    outputs["direct_claims"].to_parquet(paths["direct_claims"], index=False)
    outputs["direct_rejected_rows"].to_parquet(paths["direct_rejected_rows"], index=False)
    outputs["dedup_audit"].to_parquet(paths["dedup_audit"], index=False)
    outputs["local_partition_audit"].to_parquet(paths["local_partition_audit"], index=False)
    outputs["residual_records"].to_parquet(paths["residual_records"], index=False)

    stats = outputs["stats"]
    manifest = {
        "contract_version": CANONICAL_VERSION,
        "hf_source": {
            "dataset": args.hf_dataset,
            "requested_revision": args.hf_revision,
            "resolved_revision": resolved_revision,
            "split": "train",
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


def build_canonical_frames(
    hf_frame: Any,
    local_frame: Any,
    *,
    hf_dataset: str = HF_SOURCE_DATASET,
    hf_revision: str = HF_SOURCE_REVISION,
    local_source_path: str | Path = RAW_LOCAL_SOURCE_PATH,
    value_tolerance_percent: float = MATCH_VALUE_TOLERANCE_PERCENT,
) -> dict[str, Any]:
    """Return inspectable derived frames without writing or mutating inputs."""
    import pandas as pd

    hf_rows, hf_rejections = _normalize_hf_rows(hf_frame, hf_dataset, hf_revision)
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
        "n_hf_direct_rejected_invalid_structure": hf_rejections,
        "n_local_input_rows": n_local,
        "n_local_absolute_rows_removed_from_residual": n_direct_partition,
        "n_local_absolute_rows_with_valid_structure": n_direct_local,
        "n_local_absolute_rows_rejected_structure": n_direct_rejected,
        "n_local_residual_rows": n_residual,
        "local_partition_counts": dict(sorted(partition_counts.items())),
        "n_direct_source_rows_before_dedup": len(direct_rows),
        "n_cross_source_matches": len(dedup_rows),
        "n_canonical_direct_claims": len(claims),
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
        "direct_source_rows": direct_frame,
        "direct_claims": claims_frame,
        "direct_rejected_rows": direct_rejected_frame,
        "dedup_audit": dedup_frame,
        "local_partition_audit": partition_frame,
        "residual_records": residual_frame,
        "stats": stats,
    }


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
    claims: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        if index in matched:
            continue
        merged_indices = [index]
        if index in reverse:
            merged_indices.append(reverse[index])
        merged_rows = [source_rows[item] for item in merged_indices]
        representative = dict(merged_rows[0])
        source_record_ids = sorted(item["source_record_id"] for item in merged_rows)
        claim_id = "BIOAVAIL_CLAIM_" + hashlib.sha256(
            (CANONICAL_VERSION + "|" + "|".join(source_record_ids)).encode("utf-8")
        ).hexdigest()[:20].upper()
        representative.update(
            {
                "canonical_claim_id": claim_id,
                "source_origins": sorted({item["source_origin"] for item in merged_rows}),
                "source_record_ids": source_record_ids,
                "source_datasets": sorted({item["source_dataset"] for item in merged_rows}),
                "n_source_records": len(merged_rows),
                "cross_source_deduplicated": len(merged_rows) > 1,
            }
        )
        claims.append(representative)
    claims.sort(key=lambda row: (row["parent_identity_key"], row["pmid"], row["canonical_claim_id"]))
    return claims, audits


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
    parser.add_argument("--local-source", default=str(RAW_LOCAL_SOURCE_PATH))
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
