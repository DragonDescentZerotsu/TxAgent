"""Migrate exact Oral condition reviews onto retained Stage-1 physical rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import pandas as pd

from data.processing.gold_labels.benchmark_dataset import sha256_file
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.canonical_source import (
    DIRECT_CLAIMS_PATH,
    RAW_HF_SOURCE_PATH,
    RAW_LOCAL_SOURCE_PATH,
    REVIEWED_SOURCE_REPAIRS_PATH,
)
from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.tasks.bioavailability_ma.reviewed_context_conditioned_benchmark import (
    CORE_EXTERNAL_CONDITION_GROUPS,
    REVIEW_ROOT,
    propose_source_row,
)
from tools.chembl_tool.tasks.bioavailability_ma import (
    build_canonical_starling_source as canonical_builder,
)


VERSION = "oral_stage1_condition_review_migration.v1"
SEMANTIC_FIELDS = (
    "pmid",
    "condition_text",
    "support_text",
    "raw_value",
    "bioavailability_report_type",
    "comparator",
    "species",
    "extra_details",
    "proposed_condition_group",
    "proposed_condition_atoms",
    "Y",
    "label_method",
)


def migrate_reviews(
    *,
    stage1_path: Path,
    output_dir: Path,
    legacy_claims_path: Path = DIRECT_CLAIMS_PATH,
    legacy_queue_path: Path = REVIEW_ROOT / "review_queue.jsonl",
    legacy_verdicts_path: Path = REVIEW_ROOT / "review_verdicts.jsonl",
    repairs_path: Path = REVIEWED_SOURCE_REPAIRS_PATH,
) -> dict[str, Any]:
    legacy_claims = pd.read_parquet(legacy_claims_path).astype(object)
    legacy_claims = legacy_claims.where(pd.notna(legacy_claims), None)
    claim_by_source: dict[str, str] = {}
    representative_by_claim: dict[str, str] = {}
    for row in legacy_claims.to_dict(orient="records"):
        claim_id = str(row["canonical_claim_id"])
        representative_by_claim[claim_id] = str(row["source_record_id"])
        for source_record_id in row["source_record_ids"]:
            key = str(source_record_id)
            if key in claim_by_source:
                raise ValueError(f"legacy source row maps to multiple claims: {key}")
            claim_by_source[key] = claim_id

    legacy_queue = _read_jsonl(legacy_queue_path)
    legacy_verdicts = _read_jsonl(legacy_verdicts_path)
    queue_by_id = _unique_by_id(legacy_queue, "legacy review queue")
    verdict_by_id = _unique_by_id(legacy_verdicts, "legacy verdicts")
    if set(queue_by_id) != set(verdict_by_id):
        raise ValueError("legacy queue and verdict coverage differ")
    for claim_id, candidate in queue_by_id.items():
        if verdict_by_id[claim_id].get("source_payload_sha256") != candidate.get(
            "source_payload_sha256"
        ):
            raise ValueError(f"stale legacy verdict for {claim_id}")
    repaired_uids = {
        str(row.get("evidence", {}).get("source_row_uid") or "")
        for row in _read_jsonl(repairs_path)
    } - {""}

    retained_uids = set(
        pq.read_table(stage1_path, columns=["source_row_uid"])[
            "source_row_uid"
        ].to_pylist()
    )
    hf = pd.read_parquet(RAW_HF_SOURCE_PATH)
    local = pd.read_parquet(RAW_LOCAL_SOURCE_PATH)
    for frame in (hf, local):
        if "source_index" not in frame:
            frame.insert(0, "source_index", range(len(frame)))
    repaired_hf, repaired_local, _ = canonical_builder._apply_reviewed_source_edits(
        hf,
        local,
        hf_source_path=RAW_HF_SOURCE_PATH,
        local_source_path=RAW_LOCAL_SOURCE_PATH,
    )
    direct_rows = canonical_builder.build_canonical_frames(
        repaired_hf,
        repaired_local,
        local_source_path=RAW_LOCAL_SOURCE_PATH,
        prior_claim_ids={},
    )["direct_source_rows"]
    direct_rows = direct_rows.astype(object).where(pd.notna(direct_rows), None)
    migrated: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    represented_legacy: set[str] = set()
    counts: Counter[str] = Counter()
    for source_row in direct_rows.to_dict(orient="records"):
        source_uid = str(source_row["source_row_uid"])
        if source_uid not in retained_uids or not source_row.get("qualifying_conditions"):
            continue
        source_id = (
            "hf_bioavailability"
            if source_row["source_origin"] == "hf"
            else "oral_exposure"
        )
        normalized = {
            **source_row,
            "canonical_claim_id": f"{source_id}:{source_uid}",
        }
        _, candidate = propose_source_row(
            int(normalized["source_index"]), normalized
        )
        if candidate is None:
            continue
        candidate.update(
            {
                "source_row_uid": source_uid,
                "source_id": source_id,
            }
        )
        counts["physical_candidates"] += 1
        if candidate["proposed_condition_group"] in CORE_EXTERNAL_CONDITION_GROUPS:
            counts["allowlist_candidates"] += 1

        canonical_source_id = str(source_row["source_record_id"])
        legacy_claim_id = claim_by_source.get(canonical_source_id)
        reason = ""
        if legacy_claim_id is None:
            reason = "no_legacy_claim"
        elif legacy_claim_id not in queue_by_id:
            reason = "legacy_claim_not_reviewed"
        elif representative_by_claim[legacy_claim_id] != canonical_source_id:
            reason = "nonrepresentative_legacy_member"
        else:
            legacy = queue_by_id[legacy_claim_id]
            mismatches = [
                field
                for field in SEMANTIC_FIELDS
                if candidate.get(field) != legacy.get(field)
            ]
            parent_changed = (
                candidate.get("molecule_identity_key")
                != legacy.get("molecule_identity_key")
            )
            reviewed_repair = candidate["source_row_uid"] in repaired_uids
            if not mismatches and (not parent_changed or reviewed_repair):
                verdict = verdict_by_id[legacy_claim_id]
                migrated.append(
                    {
                        **candidate,
                        "review_status": str(verdict["review_status"]),
                        "reviewer": f"{VERSION}:{verdict.get('reviewer', '')}",
                        "review_reason": str(verdict.get("review_reason") or ""),
                        "review_notes": str(verdict.get("review_notes") or ""),
                        "review_contract": VERSION,
                        "legacy_review_source_record_id": legacy_claim_id,
                        **(
                            {
                                "condition_group": str(
                                    verdict.get("condition_group")
                                    or candidate["proposed_condition_group"]
                                ),
                                "condition_atoms": list(
                                    verdict.get("condition_atoms")
                                    or candidate["proposed_condition_atoms"]
                                ),
                                "Y": int(verdict.get("Y", candidate["Y"])),
                                "label_method": str(
                                    verdict.get("label_method")
                                    or candidate["label_method"]
                                ),
                            }
                            if verdict["review_status"] == "accepted"
                            else {}
                        ),
                    }
                )
                represented_legacy.add(legacy_claim_id)
                counts["migrated"] += 1
                if parent_changed:
                    counts["migrated_reviewed_parent_repairs"] += 1
                continue
            reason = "semantic_mismatch:" + ",".join(mismatches)
            if parent_changed and not reviewed_repair:
                reason += ",unreviewed_parent_change"

        gap = {
            **candidate,
            "migration_gap_reason": reason,
            "legacy_review_source_record_id": legacy_claim_id,
            "canonical_source_record_id": canonical_source_id,
        }
        gaps.append(gap)
        counts["gaps"] += 1
        if candidate["proposed_condition_group"] in CORE_EXTERNAL_CONDITION_GROUPS:
            counts["allowlist_gaps"] += 1

    counts["legacy_queue_rows"] = len(queue_by_id)
    counts["legacy_queue_rows_not_represented"] = len(
        set(queue_by_id) - represented_legacy
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    migrated_path = output_dir / "migrated_reviews.jsonl"
    gaps_path = output_dir / "review_gaps.jsonl"
    write_jsonl_atomic(migrated_path, migrated)
    write_jsonl_atomic(gaps_path, gaps)
    manifest = {
        "version": VERSION,
        "counts": dict(sorted(counts.items())),
        "complete": not gaps and counts["legacy_queue_rows_not_represented"] == 0,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "stage1": stage1_path,
                "legacy_claims": legacy_claims_path,
                "legacy_queue": legacy_queue_path,
                "legacy_verdicts": legacy_verdicts_path,
                "reviewed_repairs": repairs_path,
                "hf_source": RAW_HF_SOURCE_PATH,
                "local_source": RAW_LOCAL_SOURCE_PATH,
                "canonical_source_adapter": Path(canonical_builder.__file__),
            }.items()
        },
        "outputs": {
            "migrated_reviews": {
                "path": str(migrated_path),
                "sha256": sha256_file(migrated_path),
            },
            "review_gaps": {
                "path": str(gaps_path),
                "sha256": sha256_file(gaps_path),
            },
        },
        "reuse_policy": (
            "Reuse only the legacy representative's verdict when condition, label, "
            "label method, and source-visible scientific fields are unchanged. A "
            "parent change is allowed only for an explicit reviewed SMILES repair."
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def _unique_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output = {str(row["source_record_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} has duplicate source_record_id")
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = migrate_reviews(stage1_path=args.stage1, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
