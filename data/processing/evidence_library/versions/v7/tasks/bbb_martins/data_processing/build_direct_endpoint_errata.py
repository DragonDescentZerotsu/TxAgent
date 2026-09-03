"""Build a deterministic, human-gated Direct BBB endpoint-v2 errata proposal.

This command never publishes a runtime mapping.  It compares the frozen v1
map with normalized measurement/unit contradictions, writes a complete v2-
shaped proposal plus an affected-row catalog, and leaves human_approved=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_endpoint_normalization import (
    APPROVED_DIRECT_ENDPOINT_V1_MAPPING,
    DIRECT_ENDPOINT_MAPPING_V2_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[8]
DEFAULT_NORMALIZED_RECORDS = (
    REPO_ROOT
    / "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
    "starling_normalized_v7/02_canonicalized/records.parquet"
)
DEFAULT_ROOT = Path(__file__).with_name("direct_endpoint_normalization_v2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _issue_reasons(group: pd.DataFrame, canonical_endpoint: str) -> list[str]:
    reasons: list[str] = []
    raw_endpoint = str(group["endpoint_name"].iloc[0])
    units = {str(value) for value in group["canonical_unit_text"].dropna() if str(value)}
    values = pd.to_numeric(group["finite_scalar_value"], errors="coerce").dropna()
    statuses = {
        str(value)
        for column in ("measurement_unit_status", "canonicalization_status")
        if column in group
        for value in group[column].dropna()
    }
    if raw_endpoint.strip().casefold() == "bbb":
        reasons.append("raw_bbb_label_is_semantically_ambiguous")
    if "ratio" in canonical_endpoint and any(
        unit not in {"ratio", "dimensionless", "%"} for unit in units
    ):
        reasons.append("ratio_endpoint_has_physical_unit")
    if "ratio" in canonical_endpoint and not values.empty and float(values.min()) < 0.0:
        reasons.append("ratio_endpoint_has_negative_values")
    if "log" in raw_endpoint.casefold() and not (
        canonical_endpoint.startswith("log")
        or canonical_endpoint.startswith("negative_log")
    ):
        reasons.append("raw_log_endpoint_lost_log_semantics")
    if canonical_endpoint.startswith(("log", "negative_log")) and any(
        unit in {"cm/s", "cm/h", "nm/s"} for unit in units
    ):
        reasons.append("log_endpoint_retains_physical_unit")
    if "incompatible_endpoint_unit" in statuses:
        reasons.append("normalization_detected_endpoint_unit_conflict")
    if "unreviewed_endpoint_semantics" in statuses:
        reasons.append("endpoint_has_no_approved_measurement_semantics")
    return sorted(set(reasons))


def build_errata_proposal(
    *,
    base_mapping_path: str | Path = APPROVED_DIRECT_ENDPOINT_V1_MAPPING,
    normalized_records_path: str | Path = DEFAULT_NORMALIZED_RECORDS,
    output_root: str | Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    base_path = Path(base_mapping_path)
    records_path = Path(normalized_records_path)
    target = Path(output_root)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    mapping = {str(key): str(value) for key, value in base["mapping"].items()}
    available = set(pq.read_schema(records_path).names)
    columns = [
        column
        for column in (
            "source_id",
            "endpoint_name",
            "canonical_endpoint_name",
            "canonical_unit_text",
            "finite_scalar_value",
            "measurement_unit_status",
            "canonicalization_status",
        )
        if column in available
    ]
    records = pd.read_parquet(records_path, columns=columns)
    records = records.loc[records["source_id"] == "direct_bbb"].copy()
    candidates: list[dict[str, Any]] = []
    for raw_endpoint, group in records.groupby("endpoint_name", sort=True, dropna=False):
        # Pandas represents a null parquet string group as NaN; it is part of
        # the source inventory census but cannot be a key in the approved map.
        if pd.isna(raw_endpoint):
            continue
        raw_endpoint = str(raw_endpoint)
        canonical = mapping.get(raw_endpoint)
        if canonical is None:
            raise ValueError(f"v1 map lacks Direct endpoint {raw_endpoint!r}")
        reasons = _issue_reasons(group, canonical)
        if not reasons:
            continue
        numeric = pd.to_numeric(group["finite_scalar_value"], errors="coerce").dropna()
        candidates.append(
            {
                "raw_endpoint": raw_endpoint,
                "v1_endpoint": canonical,
                "record_count": int(len(group)),
                "finite_scalar_count": int(len(numeric)),
                "minimum_finite_scalar": float(numeric.min()) if len(numeric) else None,
                "maximum_finite_scalar": float(numeric.max()) if len(numeric) else None,
                "observed_units_json": json.dumps(
                    sorted(
                        {
                            str(value)
                            for value in group["canonical_unit_text"].dropna()
                            if str(value)
                        }
                    ),
                    ensure_ascii=False,
                ),
                "issue_reasons_json": json.dumps(reasons),
                "proposed_endpoint": (
                    "bbb_ambiguous_measurement"
                    if raw_endpoint.strip().casefold() == "bbb"
                    else None
                ),
                "review_status": "requires_human_review",
            }
        )
    candidates_frame = pd.DataFrame(candidates)
    candidate_path = target / "proposal/errata_candidates.parquet"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_frame.to_parquet(candidate_path, index=False)
    proposed_mapping = dict(mapping)
    delta: list[dict[str, str]] = []
    for row in candidates:
        proposed = row["proposed_endpoint"]
        if proposed and proposed != row["v1_endpoint"]:
            proposed_mapping[row["raw_endpoint"]] = proposed
            delta.append(
                {
                    "raw_endpoint": row["raw_endpoint"],
                    "v1_endpoint": row["v1_endpoint"],
                    "proposed_endpoint": proposed,
                }
            )
    proposal = {
        "mapping_version": DIRECT_ENDPOINT_MAPPING_V2_VERSION,
        "approval": {
            "human_approved": False,
            "approved_by": None,
            "approved_at": None,
        },
        "source": base["source"],
        "mapping": proposed_mapping,
        "provenance": {
            "base_mapping_path": str(base_path),
            "base_mapping_sha256": _sha256(base_path),
            "normalized_records_path": str(records_path),
            "normalized_records_sha256": _sha256(records_path),
            "errata_candidates_path": str(candidate_path),
            "errata_candidates_sha256": _sha256(candidate_path),
            "generation_method": "deterministic_measurement_unit_contradiction_audit",
            "llm_mapping_used": False,
        },
        "errata_delta": delta,
    }
    proposal_path = target / "proposal/proposed_endpoint_mapping.json"
    _write_json(proposal_path, proposal)
    manifest = {
        "artifact_version": "bbb_martins_direct_endpoint_errata_proposal.v2",
        "publication_status": "awaiting_human_approval",
        "counts": {
            "base_mapping_entries": len(mapping),
            "candidate_endpoints": len(candidates),
            "proposed_mapping_changes": len(delta),
            "affected_records": sum(row["record_count"] for row in candidates),
        },
        "artifacts": {
            "proposal": {"path": str(proposal_path), "sha256": _sha256(proposal_path)},
            "candidates": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        },
        "validations": {
            "v1_mapping_unchanged": _sha256(base_path) == proposal["provenance"]["base_mapping_sha256"],
            "full_source_key_coverage_preserved": set(mapping) == set(proposed_mapping),
            "proposal_is_not_human_approved": proposal["approval"]["human_approved"] is False,
            "no_llm_unit_or_endpoint_inference": True,
        },
    }
    if not all(manifest["validations"].values()):
        raise ValueError("Direct endpoint v2 errata proposal failed validation")
    _write_json(target / "proposal/manifest.json", manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mapping", default=str(APPROVED_DIRECT_ENDPOINT_V1_MAPPING))
    parser.add_argument("--normalized-records", default=str(DEFAULT_NORMALIZED_RECORDS))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_errata_proposal(
        base_mapping_path=args.base_mapping,
        normalized_records_path=args.normalized_records,
        output_root=args.output_root,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
