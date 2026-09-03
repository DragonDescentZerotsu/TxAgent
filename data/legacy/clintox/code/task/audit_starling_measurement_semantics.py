"""Build the full-corpus ClinTox measurement census and 360-row QA sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.tasks.clintox.starling_source import DIRECT_SOURCE_ID


DEFAULT_ROOT = Path(
    "outputs/chembl_tool/tasks/clintox/evidence_library/starling_normalized_v7"
)
AUDIT_VERSION = "clintox_measurement_unit_manual_review.v2"
CONTINUOUS_SOURCES = (
    "nonclinical_in_vivo_toxicity",
    "general_cytotoxicity",
    "off_target_ddi_exposure",
)
CATEGORICAL_SOURCES = (
    DIRECT_SOURCE_ID,
    "organ_specific_toxicity",
    "genotoxicity_carcinogenicity",
    "cellular_stress",
)
_CENSOR_QUALIFIER_PATTERNS = (
    ("symbol", r"[<>≤≥]"),
    ("at_least_most", r"\bat (?:least|most)\b"),
    ("no_more_less", r"\bno (?:more|less) than\b"),
    ("greater_less", r"\b(?:greater|less|more) than\b"),
    ("excess", r"\bin excess of\b"),
)
_SOURCE_INTERVAL_RE = re.compile(
    r"^\s*(?P<center>[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?:[±]\s*[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][+-]?\d+)?)?"
    r"\s*[\(\[]\s*(?P<lower>[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)"
    r"(?:[eE][+-]?\d+)?)\s*[–—-]\s*(?P<upper>[+-]?(?:\d+(?:,\d{3})*|"
    r"\d*\.\d+)(?:[eE][+-]?\d+)?)\s*[\)\]]\s*$"
)


def _stable_key(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _counts(frame: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    available = [field for field in fields if field in frame]
    if not available:
        return []
    grouped = frame.fillna("__missing__").groupby(available, dropna=False).size()
    return [
        {**{field: value for field, value in zip(available, key if isinstance(key, tuple) else (key,))}, "records": int(count)}
        for key, count in grouped.sort_values(ascending=False).items()
    ]


def _balanced_sample(
    frame: pd.DataFrame,
    n: int,
    *,
    strata: tuple[str, ...],
    forced: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(f"requested {n} audit rows from only {len(frame)} candidates")
    chosen: list[int] = []
    if forced is not None and not forced.empty:
        chosen.extend(forced.index.tolist()[:n])
    remaining = frame.drop(index=chosen, errors="ignore").copy()
    remaining["_stable_order"] = remaining["canonical_record_id"].map(_stable_key)
    fields = [field for field in strata if field in remaining]
    if fields:
        groups = [
            group.sort_values("_stable_order").index.tolist()
            for _, group in remaining.groupby(fields, dropna=False, sort=True)
        ]
        while len(chosen) < n and groups:
            next_groups: list[list[int]] = []
            for group in groups:
                if len(chosen) >= n:
                    break
                chosen.append(group.pop(0))
                if group:
                    next_groups.append(group)
            groups = next_groups
    if len(chosen) < n:
        fallback = remaining.drop(index=chosen, errors="ignore").sort_values(
            "_stable_order"
        )
        chosen.extend(fallback.index.tolist()[: n - len(chosen)])
    return frame.loc[chosen[:n]].copy()


def _censor_qualifier(value: Any) -> str:
    text = str(value or "").casefold()
    for name, pattern in _CENSOR_QUALIFIER_PATTERNS:
        if re.search(pattern, text):
            return name
    return "other"


def _censored_support_sample(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[
        frame["assay_transfer_ineligibility_reason"]
        == "support_reports_censored_value"
    ].copy()
    candidates["censor_qualifier"] = candidates["support_text"].map(
        _censor_qualifier
    )
    candidates["_stable_order"] = candidates["canonical_record_id"].map(
        _stable_key
    )
    sample = (
        candidates.sort_values("_stable_order")
        .groupby(
            ["canonical_semantics_rule_id", "censor_qualifier"],
            dropna=False,
            sort=True,
        )
        .head(5)
        .sort_values(
            ["canonical_semantics_rule_id", "censor_qualifier", "_stable_order"]
        )
        .copy()
    )
    columns = [
        "source_id",
        "source_row_number",
        "canonical_record_id",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "canonical_semantics_rule_id",
        "censor_qualifier",
        "canonicalization_status",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
        "support_text",
    ]
    sample = sample[columns]
    sample.insert(0, "manual_review_status", "pending")
    sample.insert(1, "manual_review_notes", "")
    sample.insert(2, "manual_review_version", AUDIT_VERSION)
    return sample


def _source_interval_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Find retrieval rows whose point estimate contradicts its own interval."""
    text = frame["measurement_text"].fillna("")
    possible = text.str.contains(r"[\(\[]", regex=True) & text.str.contains(
        r"[–—-]", regex=True
    )
    candidates = frame.loc[possible]
    parsed = candidates["measurement_text"].fillna("").str.extract(
        _SOURCE_INTERVAL_RE
    )
    for column in parsed:
        parsed[column] = pd.to_numeric(
            parsed[column].str.replace(",", "", regex=False), errors="coerce"
        )
    complete = parsed.notna().all(axis=1)
    lower = parsed[["lower", "upper"]].min(axis=1)
    upper = parsed[["lower", "upper"]].max(axis=1)
    outside = complete & ~parsed["center"].between(lower, upper)
    retrieval = candidates["retrieval_eligible"].fillna(False).astype(bool)
    sample = candidates.loc[outside & retrieval].copy()
    sample["parsed_point_estimate"] = parsed.loc[sample.index, "center"]
    sample["parsed_interval_lower"] = lower.loc[sample.index]
    sample["parsed_interval_upper"] = upper.loc[sample.index]
    sample["reviewed_ineligibility_reason"] = (
        "source_point_outside_reported_interval"
    )
    columns = [
        "source_id",
        "source_row_number",
        "canonical_record_id",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "parsed_point_estimate",
        "parsed_interval_lower",
        "parsed_interval_upper",
        "reviewed_ineligibility_reason",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
        "support_text",
    ]
    sample = sample[columns].sort_values(
        ["source_id", "source_row_number", "canonical_record_id"]
    )
    sample.insert(0, "manual_review_status", "pending")
    sample.insert(1, "manual_review_notes", "")
    sample.insert(2, "manual_review_version", AUDIT_VERSION)
    return sample


def build_audit(
    records_path: str | Path, pair_buckets_path: str | Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_parquet(records_path)
    buckets = pd.read_parquet(pair_buckets_path)
    if not records["canonical_record_id"].is_unique:
        raise ValueError("Stage-03 canonical record IDs are not unique")
    if not buckets["canonical_record_id"].is_unique or len(records) != len(buckets):
        raise ValueError("Stage-04 sidecar is not one-to-one with Stage 03")
    frame = records.merge(
        buckets[
            [
                "canonical_record_id",
                "assay_transfer_eligible",
                "assay_transfer_ineligibility_reason",
                "pair_bucket_key",
            ]
        ],
        on="canonical_record_id",
        validate="one_to_one",
    )

    source_census: dict[str, Any] = {}
    for source_id, source in frame.groupby("source_id", sort=True):
        source_census[str(source_id)] = {
            "records": int(len(source)),
            "finite_scalars": int(source["finite_scalar_value"].notna().sum()),
            "assay_transfer_eligible": int(
                source["assay_transfer_eligible"].fillna(False).sum()
            ),
            "canonicalization_status": _counts(source, ["canonicalization_status"]),
            "semantic_rules": _counts(
                source,
                ["canonical_semantics_status", "canonical_semantics_rule_id"],
            ),
            "reference_semantics": _counts(
                source,
                ["canonical_reference_scope", "canonical_reference_basis"],
            ),
            "assay_transfer_exclusions": _counts(
                source, ["assay_transfer_ineligibility_reason"]
            ),
            "top_canonical_units": _counts(source, ["canonical_unit_text"])[:100],
        }

    sample_parts: list[pd.DataFrame] = []
    for source_id in CONTINUOUS_SOURCES:
        source = frame[frame["source_id"] == source_id]
        eligible = source[source["assay_transfer_eligible"].fillna(False)]
        excluded = source[~source["assay_transfer_eligible"].fillna(False)]
        sample_parts.append(
            _balanced_sample(
                eligible,
                40,
                strata=(
                    "canonical_semantics_rule_id",
                    "canonical_reference_scope",
                    "canonical_unit_text",
                ),
            )
        )
        forced = None
        if source_id == "off_target_ddi_exposure":
            forced = excluded[
                excluded["unit_text"].fillna("").str.contains(
                    "DA-8159", case=False, regex=False
                )
            ]
        sample_parts.append(
            _balanced_sample(
                excluded,
                40,
                strata=(
                    "canonicalization_status",
                    "canonical_semantics_rule_id",
                    "assay_transfer_ineligibility_reason",
                ),
                forced=forced,
            )
        )
    for source_id in CATEGORICAL_SOURCES:
        source = frame[frame["source_id"] == source_id]
        sample_parts.append(
            _balanced_sample(
                source,
                30,
                strata=(
                    "canonicalization_status",
                    "canonical_measurement_scale_id",
                    "canonical_category_id",
                ),
            )
        )

    sample = pd.concat(sample_parts, ignore_index=True)
    if len(sample) != 360 or sample["canonical_record_id"].duplicated().any():
        raise ValueError("manual audit selection is not 360 unique records")
    review_columns = [
        "source_id",
        "source_row_number",
        "canonical_record_id",
        "endpoint_name",
        "result_metric",
        "measurement_text",
        "unit_text",
        "canonical_endpoint_name",
        "canonical_measurement_text",
        "canonical_unit_text",
        "finite_scalar_value",
        "canonical_semantics_status",
        "canonical_quantity_kind",
        "canonical_semantics_rule_id",
        "canonical_reference_scope",
        "canonical_reference_basis",
        "canonicalization_status",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
        "support_text",
    ]
    sample = sample[[column for column in review_columns if column in sample]].copy()
    sample.insert(0, "manual_review_status", "pending")
    sample.insert(1, "manual_review_notes", "")
    sample.insert(2, "manual_review_version", AUDIT_VERSION)
    censored_sample = _censored_support_sample(frame)
    interval_sample = _source_interval_sample(frame)
    audit = {
        "audit_version": AUDIT_VERSION,
        "records_path": str(records_path),
        "pair_buckets_path": str(pair_buckets_path),
        "records": int(len(frame)),
        "sample_records": int(len(sample)),
        "sample_design": {
            "continuous": "40 eligible plus 40 excluded for each of three sources",
            "categorical": "30 rows for each of four sources",
            "forced": "all DA-8159 unit rows within the off-target excluded stratum",
            "order": "sha256(canonical_record_id) within round-robin strata",
        },
        "manual_review_status": "pending",
        "supplemental_censored_support_review": {
            "design": (
                "up to five SHA-256-selected rows per semantic-rule and "
                "censor-qualifier stratum"
            ),
            "records": int(len(censored_sample)),
            "status": "pending",
        },
        "supplemental_source_interval_consistency_review": {
            "design": (
                "all retrieval-eligible scalar source measurements matching "
                "point (lower-upper) where the point lies outside the interval"
            ),
            "records": int(len(interval_sample)),
            "status": "pending",
        },
        "source_census": source_census,
    }
    return audit, sample, censored_sample, interval_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", default=str(DEFAULT_ROOT / "03_records/records.parquet")
    )
    parser.add_argument(
        "--pair-buckets",
        default=str(DEFAULT_ROOT / "04_pair_buckets/pair_bucket_records.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            Path(__file__).resolve().parent
            / "data_processing/measurement_unit_audit_v2"
        ),
    )
    args = parser.parse_args(argv)
    audit, sample, censored_sample, interval_sample = build_audit(
        args.records, args.pair_buckets
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    sample.to_csv(out / "manual_audit.tsv", sep="\t", index=False)
    censored_sample.to_csv(
        out / "censored_support_manual_audit.tsv", sep="\t", index=False
    )
    interval_sample.to_csv(
        out / "source_interval_consistency_manual_audit.tsv", sep="\t", index=False
    )
    print(
        json.dumps(
            {
                "records": audit["records"],
                "sample_records": len(sample),
                "supplemental_censored_records": len(censored_sample),
                "supplemental_source_interval_records": len(interval_sample),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
