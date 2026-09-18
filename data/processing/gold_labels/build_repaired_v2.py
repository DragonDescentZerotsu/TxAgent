"""Rebuild BBB and Oral conditioned gold from deduplicated physical voters.

This entry point deliberately fails before writing a v2 task when either the
authoritative Stage-1 deduplication receipt or its payload-bound condition
ledger is incomplete.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from data.processing.gold_labels.benchmark_dataset import sha256_file
from data.processing.gold_labels.reviewed_conditioned_benchmark import (
    ConditionedBenchmarkConfig,
    NO_REPORTED_CONDITION,
    SPLITS,
    aggregate_reviewed_votes,
    allocate_groups,
    candidate_group_support,
    summarize_groups,
    validate_split_integrity,
)
from data.processing.gold_labels.stage1_repaired_sources import (
    label_bbb_stage1_row,
    _label_oral_stage1_row,
    scientific_payload_sha256,
)
from data.processing.gold_labels.voter_membership import (
    materialize_gold_label_record_index,
    materialize_voter_membership,
    write_gold_label_record_index,
    write_voter_membership,
)
from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)


VERSION = "conditioned_benchmark_physical_voters.v2"
LEDGER_VERSION = "physical_condition_review_ledger.v2"
DEDUP_VERSION = "stage1_exact_deduplication.v2"
EXCLUDED_EXTERNAL_CONDITION = "excluded_external_condition"
V1_MIGRATION_AGREEMENT_THRESHOLD = 2 / 3
MAPPING_ORIGINS = {
    "condition_allowlist_exclusion",
    "legacy_condition_review_replay",
    "physical_review",
    "stage1_no_external_condition",
    "unchanged_payload_v1_reuse",
    "smiles_only_reparenting_v1_reuse",
    "v1_voter_lineage_replay",
}
REQUIRED_LEDGER_FIELDS = {
    "source_row_uid",
    "source_id",
    "source_record_id",
    "source_payload_sha256",
    "vote_label",
    "label_method",
    "decision",
    "condition_group",
    "condition_atoms",
    "reviewer",
    "rationale",
    "mapping_origin",
    "reviewed_source_id",
    "reviewed_source_payload_sha256",
}
TASKS = {
    "bbb_martins": {
        "directory": "BBB_Martins",
        "sources": ("direct_bbb",),
        "external_id_prefix": "BBBCTX",
    },
    "bioavailability_ma": {
        "directory": "Bioavailability_Ma",
        "sources": ("hf_bioavailability", "oral_exposure"),
        "external_id_prefix": "BIOCTX2",
    },
}


class CoverageError(ValueError):
    """A publication-blocking Stage-1 or condition-ledger coverage failure."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(str(self.report.get("message") or "v2 coverage failure"))


def validate_stage1_authority(stage1_root: Path) -> dict[str, Any]:
    """Verify the Stage-1 records against its existing dedup/audit manifests."""
    records_path = stage1_root / "01_cleaned/records.parquet"
    stage_manifest_path = stage1_root / "01_cleaned/manifest.json"
    cleaning_manifest_path = (
        stage1_root / "01_cleaned/source_value_cleaning_manifest.json"
    )
    required = (records_path, stage_manifest_path, cleaning_manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CoverageError(
            {
                "check": "authoritative_stage1",
                "message": "required Stage-1 artifacts are missing",
                "missing_paths": missing,
            }
        )
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    cleaning_manifest = json.loads(
        cleaning_manifest_path.read_text(encoding="utf-8")
    )
    dedup = cleaning_manifest.get("canonical_deduplication") or {}
    failures = []
    if dedup.get("version") != DEDUP_VERSION:
        failures.append("canonical_deduplication.version")
    if dedup.get("authoritative_stage1_deduplication") is not True:
        failures.append("authoritative_stage1_deduplication")
    if dedup.get("deduplication_complete") is not True:
        failures.append("deduplication_complete")
    validations = dedup.get("validations") or {}
    expected_validations = {
        "input_equals_retained_plus_duplicates",
        "retained_source_row_uids_unique",
        "representatives_follow_gold_priority_then_uid",
    }
    if not expected_validations.issubset(validations) or not all(
        validations[name] is True for name in expected_validations & set(validations)
    ):
        failures.append("canonical_deduplication.validations")
    recorded_output = str((stage_manifest.get("output") or {}).get("sha256") or "")
    actual_output = sha256_file(records_path)
    if recorded_output != actual_output:
        failures.append("records.parquet sha256")

    uid_table = ds.dataset(records_path).to_table(columns=["source_row_uid"])
    retained_uids = sorted(str(value or "") for value in uid_table.column(0).to_pylist())
    if any(not value for value in retained_uids) or len(retained_uids) != len(
        set(retained_uids)
    ):
        failures.append("retained source_row_uid uniqueness")
    if dedup.get("output_records") != len(retained_uids):
        failures.append("canonical_deduplication.output_records")
    if dedup.get("retained_source_row_uid_count") != len(retained_uids):
        failures.append("retained_source_row_uid_count")
    if dedup.get("retained_source_row_uid_sha256") != _rows_digest(
        (value,) for value in retained_uids
    ):
        failures.append("retained_source_row_uid_sha256")

    sidecar = (stage_manifest.get("sidecars") or {}).get(
        "source_value_cleaning_audit"
    ) or {}
    cleaning_sidecar = (stage_manifest.get("sidecars") or {}).get(
        "source_value_cleaning_manifest"
    ) or {}
    if cleaning_sidecar.get("sha256") != sha256_file(cleaning_manifest_path):
        failures.append("source_value_cleaning_manifest sha256")
    audit_path = Path(str(sidecar.get("path") or ""))
    if not audit_path.is_file() or sha256_file(audit_path) != sidecar.get("sha256"):
        failures.append("duplicate audit path/hash")
    elif dedup:
        audit = ds.dataset(audit_path).to_table().to_pylist()
        duplicates = sorted(
            (
                str(row.get("source_row_uid") or ""),
                str(row.get("retained_source_row_uid") or ""),
                str(row.get("duplicate_scope") or ""),
            )
            for row in audit
            if row.get("field") == "record"
            and row.get("after") == "dropped"
            and row.get("rule_id") == DEDUP_VERSION
        )
        expected = dedup.get("duplicate_audit") or {}
        if len(duplicates) != expected.get("count"):
            failures.append("duplicate_audit.count")
        if _rows_digest((row[0],) for row in duplicates) != expected.get(
            "discarded_source_row_uid_sha256"
        ):
            failures.append("duplicate_audit.discarded_source_row_uid_sha256")
        if _rows_digest(duplicates) != expected.get("lineage_sha256"):
            failures.append("duplicate_audit.lineage_sha256")
    if failures:
        raise CoverageError(
            {
                "check": "authoritative_stage1",
                "message": "Stage-1 is not an authoritative, hash-consistent deduplicated survivor set",
                "failures": failures,
                "records_path": str(records_path),
            }
        )
    return {
        "records_path": str(records_path),
        "records_sha256": actual_output,
        "records": len(retained_uids),
        "duplicates_removed": int(dedup.get("duplicates_removed") or 0),
        "deduplication_manifest": dedup,
        "stage_manifest_sha256": sha256_file(stage_manifest_path),
        "source_value_cleaning_manifest_sha256": sha256_file(
            cleaning_manifest_path
        ),
        "duplicate_audit_path": str(audit_path),
        "duplicate_audit_sha256": sidecar.get("sha256"),
    }


def validate_condition_ledger(
    *,
    task: str,
    accepted_voters: Sequence[tuple[Mapping[str, Any], Any]],
    ledger_path: Path,
    allowed_condition_groups: set[str],
) -> list[dict[str, Any]]:
    """Require exactly one payload-bound condition decision per physical voter."""
    ledger_rows = _read_jsonl(ledger_path) if ledger_path.is_file() else []
    accepted = {
        str(row.get("source_row_uid") or ""): (row, decision.record)
        for row, decision in accepted_voters
    }
    rows_by_uid: dict[str, dict[str, Any]] = {}
    problems: list[dict[str, Any]] = []
    for raw in ledger_rows:
        row = dict(raw)
        uid = str(row.get("source_row_uid") or "")
        if uid in rows_by_uid:
            problems.append({"source_row_uid": uid, "reason": "duplicate_ledger_uid"})
        rows_by_uid[uid] = row
    missing = sorted(set(accepted) - set(rows_by_uid))
    extra = sorted(set(rows_by_uid) - set(accepted))
    for uid in sorted(set(accepted) & set(rows_by_uid)):
        source, record = accepted[uid]
        review = rows_by_uid[uid]
        absent = sorted(
            field
            for field in REQUIRED_LEDGER_FIELDS
            if field not in review or review[field] is None
        )
        if absent:
            problems.append(
                {"source_row_uid": uid, "reason": "missing_fields", "fields": absent}
            )
            continue
        expected = {
            "source_id": str(source.get("source_id") or ""),
            "source_record_id": str(source.get("source_record_id") or ""),
            "source_payload_sha256": scientific_payload_sha256(task, source),
            "vote_label": int(record.label),
            "label_method": str(record.label_method),
        }
        mismatched = sorted(
            field for field, value in expected.items() if review.get(field) != value
        )
        if mismatched:
            problems.append(
                {
                    "source_row_uid": uid,
                    "reason": "source_or_payload_mismatch",
                    "fields": mismatched,
                }
            )
        origin = str(review.get("mapping_origin") or "")
        if origin not in MAPPING_ORIGINS:
            problems.append({"source_row_uid": uid, "reason": "invalid_mapping_origin"})
        reviewed_source = str(review.get("reviewed_source_id") or "")
        reviewed_payload = str(review.get("reviewed_source_payload_sha256") or "")
        if (
            reviewed_source != expected["source_id"]
            or reviewed_payload != expected["source_payload_sha256"]
        ):
            problems.append(
                {
                    "source_row_uid": uid,
                    "reason": "cross_source_or_changed_payload_mapping_reuse",
                }
            )
        if (
            task == "bioavailability_ma"
            and expected["source_id"] == "oral_exposure"
            and reviewed_source != "oral_exposure"
        ):
            problems.append(
                {"source_row_uid": uid, "reason": "oral_local_inherited_hf_mapping"}
            )
        decision = str(review.get("decision") or "")
        group = str(review.get("condition_group") or "")
        atoms = [str(value) for value in (review.get("condition_atoms") or [])]
        if decision == "active_external":
            if group not in allowed_condition_groups or atoms != [group]:
                problems.append(
                    {"source_row_uid": uid, "reason": "invalid_active_condition"}
                )
        elif decision == EXCLUDED_EXTERNAL_CONDITION:
            canonical_group = "+".join(sorted(set(atoms)))
            if not group or not atoms or group != canonical_group:
                problems.append(
                    {"source_row_uid": uid, "reason": "invalid_excluded_condition"}
                )
        elif decision == NO_REPORTED_CONDITION:
            if group or atoms:
                problems.append(
                    {"source_row_uid": uid, "reason": "invalid_null_condition"}
                )
        else:
            problems.append(
                {"source_row_uid": uid, "reason": "nonterminal_condition_decision"}
            )
        if not str(review.get("reviewer") or "").strip() or not str(
            review.get("rationale") or ""
        ).strip():
            problems.append(
                {"source_row_uid": uid, "reason": "missing_review_provenance"}
            )
    report = {
        "check": "physical_condition_ledger_coverage",
        "ledger_version": LEDGER_VERSION,
        "task": task,
        "ledger_path": str(ledger_path),
        "accepted_physical_voters": len(accepted),
        "ledger_rows": len(ledger_rows),
        "missing_voters": len(missing),
        "extra_nonvoters_or_deleted_duplicates": len(extra),
        "invalid_rows": len(problems),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "problem_examples": problems[:20],
    }
    if missing or extra or problems:
        report["message"] = (
            "condition ledger must exactly cover retained, label-eligible physical voters"
        )
        raise CoverageError(report)
    return [rows_by_uid[uid] for uid in sorted(rows_by_uid)]


def prepare_task(
    *,
    task: str,
    stage1_root: Path,
    ledger_path: Path,
    lineage_path: Path,
    v1_root: Path,
) -> dict[str, Any]:
    authority = validate_stage1_authority(stage1_root)
    config = TASKS[task]
    records_path = Path(authority["records_path"])
    table = ds.dataset(records_path).to_table(
        filter=ds.field("source_id").isin(list(config["sources"]))
    )
    all_rows = table.to_pylist()
    lineage_rows = _read_jsonl(lineage_path)
    lineage_by_uid = {
        str(row["source_row_uid"]): row for row in lineage_rows
    }
    if len(lineage_by_uid) != len(lineage_rows):
        raise CoverageError(
            {
                "check": "v1_voter_lineage",
                "message": "v1 voter lineage contains duplicate source_row_uid values",
                "lineage_path": str(lineage_path),
            }
        )
    stage1_by_uid = {str(row["source_row_uid"]): row for row in all_rows}
    missing = set(lineage_by_uid) - set(stage1_by_uid)
    duplicate_audit = ds.dataset(authority["duplicate_audit_path"]).to_table().to_pylist()
    replacement = {
        str(row.get("source_row_uid") or ""): str(
            row.get("retained_source_row_uid") or ""
        )
        for row in duplicate_audit
        if row.get("field") == "record"
        and row.get("after") == "dropped"
        and row.get("rule_id") == DEDUP_VERSION
    }
    invalid_missing = sorted(
        uid
        for uid in missing
        if not replacement.get(uid) or replacement[uid] not in lineage_by_uid
    )
    if invalid_missing:
        raise CoverageError(
            {
                "check": "v1_voter_lineage",
                "message": "a v1 voter is missing without an inherited-voter duplicate survivor",
                "lineage_path": str(lineage_path),
                "invalid_missing_count": len(invalid_missing),
                "examples": invalid_missing[:20],
            }
        )
    rows = [row for row in all_rows if str(row["source_row_uid"]) in lineage_by_uid]
    if task == "bbb_martins":
        decisions = [
            label_bbb_stage1_row(row, allow_conditioned_context=True) for row in rows
        ]
    else:
        decisions = [
            _label_oral_stage1_row(row, allow_conditioned_context=True) for row in rows
        ]
    accepted_voters = [
        (row, decision)
        for row, decision in zip(rows, decisions)
        if decision.record is not None
    ]
    lineage_label_rejections = [
        {
            "source_row_uid": str(row["source_row_uid"]),
            "source_id": str(row["source_id"]),
            "source_record_id": str(row["source_record_id"]),
            "reason": decision.reason or "unspecified_rejection",
            "details": dict(decision.example),
            "v1_lineage": lineage_by_uid[str(row["source_row_uid"])],
        }
        for row, decision in zip(rows, decisions)
        if decision.record is None
    ]
    task_v1 = v1_root / config["directory"] / "v1/scaffold"
    v1_rows = _read_v1_rows(task_v1)
    allowed_groups = {
        str(row["condition_group"])
        for row in v1_rows
        if row["condition_group"] != NO_REPORTED_CONDITION
    }
    ledger = validate_condition_ledger(
        task=task,
        accepted_voters=accepted_voters,
        ledger_path=ledger_path,
        allowed_condition_groups=allowed_groups,
    )
    return {
        "task": task,
        "authority": authority,
        "stage1_rows": rows,
        "decisions": decisions,
        "accepted_voters": accepted_voters,
        "ledger": ledger,
        "ledger_path": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "lineage_path": str(lineage_path),
        "lineage_sha256": sha256_file(lineage_path),
        "lineage_rows": len(lineage_rows),
        "lineage_duplicate_rows_removed": len(missing),
        "lineage_label_rejections": lineage_label_rejections,
        "v1_root": task_v1,
        "v1_rows": v1_rows,
    }


def build_task(prepared: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    task = str(prepared["task"])
    config = TASKS[task]
    ledger_by_uid = {row["source_row_uid"]: row for row in prepared["ledger"]}
    votes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    replay = []
    excluded_external = []
    for source, decision in prepared["accepted_voters"]:
        record = decision.record
        uid = str(source["source_row_uid"])
        review = ledger_by_uid[uid]
        if review["decision"] == EXCLUDED_EXTERNAL_CONDITION:
            excluded_external.append({**dict(review), "source_row_uid": uid})
            replay.append({**dict(review), "source_row_uid": uid})
            continue
        identity = normalize_molecule_identity(record.smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            raise CoverageError(
                {
                    "check": "voter_parent_normalization",
                    "message": "an accepted physical voter has no normalized parent",
                    "task": task,
                    "source_row_uid": uid,
                }
            )
        parent = identity.parent_inchi_key or identity.parent_smiles
        condition = (
            str(review["condition_group"])
            if review["decision"] == "active_external"
            else NO_REPORTED_CONDITION
        )
        atoms = [condition] if condition != NO_REPORTED_CONDITION else []
        vote = {
            "source_row_uid": uid,
            "source_id": record.source_id,
            "source_record_id": record.source_record_id,
            "source_payload_sha256": review["source_payload_sha256"],
            "drug": identity.parent_smiles,
            "molecule_identity_key": parent,
            "molecule_identity": identity.to_dict(),
            "bemis_murcko_scaffold": bemis_murcko_scaffold(identity.parent_smiles),
            "condition_group": condition,
            "condition_atoms": atoms,
            "condition_text": str(source.get("qualifying_conditions") or ""),
            "Y": int(record.label),
            "label_method": record.label_method,
            "pmid": record.pmid,
            "raw_value": record.raw_value,
            "reviewer": review["reviewer"],
        }
        votes[(parent, condition)].append(vote)
        replay.append({**dict(review), **vote})

    v1_rows = list(prepared["v1_rows"])
    v1_by_context = {
        (row["molecule_identity_key"], row["condition_group"]): row for row in v1_rows
    }
    null_votes = {key: value for key, value in votes.items() if key[1] == NO_REPORTED_CONDITION}
    external_votes = {key: value for key, value in votes.items() if key[1] != NO_REPORTED_CONDITION}
    null_accepted, null_rejected = aggregate_reviewed_votes(
        null_votes,
        agreement_threshold=0.70,
        preserved_majorities={
            key: int(row["Y"])
            for key, row in v1_by_context.items()
            if key[1] == NO_REPORTED_CONDITION
        },
        preserved_agreement_threshold=V1_MIGRATION_AGREEMENT_THRESHOLD,
    )
    external_accepted, external_rejected = aggregate_reviewed_votes(
        external_votes, agreement_threshold=0.60
    )
    accepted_units = null_accepted + external_accepted
    rejected_units = null_rejected + external_rejected
    for row in null_accepted + null_rejected:
        row["condition_scope"] = "none_reported"
        if row.get("label_decision") == "accepted_preserved_record_majority":
            row["label_decision"] = "accepted_v1_migration_majority_2of3"
        elif row.get("label_decision") == "accepted_record_majority_60":
            row["label_decision"] = "accepted_record_majority_70"

    parent_splits, scaffold_splits = _split_assignments(v1_rows)
    allowed_groups = {
        str(row["condition_group"])
        for row in v1_rows
        if row["condition_group"] != NO_REPORTED_CONDITION
    }
    candidate_groups, group_support = candidate_group_support(
        external_accepted,
        minimum_parents=3,
        minimum_scaffolds=3,
        allowed_groups=allowed_groups,
    )
    allocation, selected_groups, allocation_audit = allocate_groups(
        external_accepted,
        candidate_groups=candidate_groups,
        parent_assignment=parent_splits,
        scaffold_assignment=scaffold_splits,
        config=ConditionedBenchmarkConfig(
            task_name=str(config["directory"]),
            lineage=VERSION,
            protocol_version=VERSION,
            frozen_root=Path(prepared["v1_root"]),
            output_root=output_root,
            source_artifacts=(),
            row_id_prefix=str(config["external_id_prefix"]),
            frozen_lineage=VERSION,
            allowed_condition_groups=tuple(sorted(allowed_groups)),
        ),
    )
    group_rejected = []
    for row in external_accepted:
        group = str(row["condition_group"])
        if group not in selected_groups:
            reasons = group_support.get(group, {}).get(
                "preallocation_exclusion_reasons", []
            )
            group_rejected.append(
                {
                    **row,
                    "drop_reason": (
                        ",".join(reasons)
                        or allocation_audit["group_exclusion_reasons"].get(
                            group, "group_not_selected_by_split_allocation"
                        )
                    ),
                }
            )
    publishable = [
        *null_accepted,
        *(row for row in external_accepted if row["condition_group"] in selected_groups),
    ]
    split_by_context = _preserved_split_assignments(
        publishable,
        v1_by_context=v1_by_context,
        parent_splits=parent_splits,
        scaffold_splits=scaffold_splits,
        external_scaffold_allocation=allocation,
        target_counts=Counter(str(row["split"]) for row in v1_rows),
    )
    published = []
    split_reallocations = []
    for row in sorted(
        publishable,
        key=lambda value: (value["molecule_identity_key"], value["condition_group"]),
    ):
        key = (row["molecule_identity_key"], row["condition_group"])
        prior = v1_by_context.get(key)
        split = split_by_context[key]
        if prior and prior["split"] != split:
            split_reallocations.append(
                {
                    "molecule_identity_key": key[0],
                    "condition_group": key[1],
                    "bemis_murcko_scaffold": row.get("bemis_murcko_scaffold", ""),
                    "v1_split": prior["split"],
                    "v2_split": split,
                    "reason": "repaired_parent_now_matches_frozen_scaffold_in_another_split",
                }
            )
        prefix = (
            "NULL"
            if row["condition_group"] == NO_REPORTED_CONDITION
            else str(config["external_id_prefix"])
        )
        published.append(
            {
                **row,
                "split": split,
                "split_policy": "v1_survivor_freeze_then_parent_scaffold_train",
                "benchmark_row_id": (
                    prior["benchmark_row_id"]
                    if prior
                    else _row_id(prefix, key[0], key[1])
                ),
            }
        )
    combined = {
        split: [row for row in published if row["split"] == split]
        for split in SPLITS
    }
    validate_split_integrity(combined)

    root = output_root / config["directory"] / "v2/scaffold"
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in combined.items():
        write_jsonl_atomic(root / f"{split}.jsonl", [_minimal_row(row) for row in rows])
        write_jsonl_atomic(
            root / f"{split}_molecule_condition_labels.jsonl", rows
        )
    write_jsonl_atomic(
        root / "heldout_molecule_condition_labels.jsonl",
        combined["valid"] + combined["test"],
    )
    write_jsonl_atomic(root / "source_condition_review.jsonl", replay)
    write_jsonl_atomic(
        root / "excluded_external_condition_reviews.jsonl", excluded_external
    )
    write_jsonl_atomic(
        root / "accepted_parent_conditions_before_group_gate.jsonl", accepted_units
    )
    write_jsonl_atomic(root / "rejected_parent_conditions.jsonl", rejected_units)
    write_jsonl_atomic(
        root / "group_gate_rejected_parent_conditions.jsonl", group_rejected
    )
    _write_csv(root / "group_distribution.csv", summarize_groups(combined))

    membership = materialize_voter_membership(
        task_name=str(config["directory"]),
        vote_groups=votes,
        published_aggregates=published,
        rejected_parent_aggregates=rejected_units,
        group_gate_rejected_aggregates=group_rejected,
    )
    membership_manifest = write_voter_membership(
        root / "voter_membership.parquet",
        membership,
        stage1_sha256=prepared["authority"]["records_sha256"],
        provenance={
            "builder_version": VERSION,
            "condition_ledger_version": LEDGER_VERSION,
            "deduplication": prepared["authority"],
        },
    )
    record_index = materialize_gold_label_record_index(membership, published)
    record_index_manifest = write_gold_label_record_index(
        root / "gold_label_record_index.parquet",
        record_index,
        membership_sha256=membership_manifest["sha256"],
    )
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(replay), provenance / "physical_voter_replay.parquet")
    diff = _v1_v2_diff(v1_rows, published)
    write_jsonl_atomic(provenance / "v1_v2_card_diff.jsonl", diff)
    write_jsonl_atomic(
        provenance / "v1_split_reallocations.jsonl", split_reallocations
    )
    write_jsonl_atomic(
        provenance / "v1_lineage_label_rejections.jsonl",
        prepared["lineage_label_rejections"],
    )
    rejection_audit = _source_rejection_audit(
        prepared["stage1_rows"], prepared["decisions"]
    )
    write_json_atomic(provenance / "source_rejection_audit.json", rejection_audit)
    summary = {
        "task": config["directory"],
        "version": VERSION,
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "stage1": prepared["authority"],
        "condition_ledger": {
            "version": LEDGER_VERSION,
            "rows": len(prepared["ledger"]),
            "path": prepared["ledger_path"],
            "sha256": prepared["ledger_sha256"],
        },
        "v1_voter_lineage": {
            "path": prepared["lineage_path"],
            "sha256": prepared["lineage_sha256"],
            "physical_members_before_dedup": prepared["lineage_rows"],
            "exact_duplicates_removed": prepared["lineage_duplicate_rows_removed"],
            "surviving_physical_members": len(prepared["accepted_voters"]),
            "label_ineligible_members_dropped": len(
                prepared["lineage_label_rejections"]
            ),
            "label_rejection_reason_counts": dict(
                sorted(
                    Counter(
                        row["reason"]
                        for row in prepared["lineage_label_rejections"]
                    ).items()
                )
            ),
        },
        "accepted_physical_source_rows": len(prepared["accepted_voters"]),
        "excluded_external_condition_rows": len(excluded_external),
        "physical_voters": len(membership),
        "agreement_policy": {
            "null_condition_threshold": 0.70,
            "v1_migration_preservation_threshold": V1_MIGRATION_AGREEMENT_THRESHOLD,
            "v1_migration_preservation_rule": (
                "existing v1 null-condition context with unchanged majority label"
            ),
            "external_condition_threshold": 0.60,
            "exact_ties_rejected": True,
        },
        "published_cards": len(published),
        "rejected_parent_conditions": len(rejected_units),
        "group_gate_rejected_parent_conditions": len(group_rejected),
        "group_gate": {
            "minimum_accepted_parents": 3,
            "minimum_distinct_scaffolds": 3,
            "allowed_condition_groups": sorted(allowed_groups),
            "support": group_support,
            "allocation": allocation_audit,
        },
        "split_counts": {split: len(rows) for split, rows in combined.items()},
        "diff_counts": dict(Counter(row["status"] for row in diff)),
        "v1_split_reallocations": len(split_reallocations),
        "voter_membership": membership_manifest,
        "gold_label_record_index": record_index_manifest,
    }
    write_json_atomic(root / "summary.json", summary)
    receipt_paths = [
        root / f"{split}.jsonl" for split in SPLITS
    ] + [
        root / f"{split}_molecule_condition_labels.jsonl" for split in SPLITS
    ] + [
        root / "voter_membership.parquet",
        root / "gold_label_record_index.parquet",
        root / "gold_label_record_index.manifest.json",
        root / "excluded_external_condition_reviews.jsonl",
        provenance / "physical_voter_replay.parquet",
        provenance / "v1_v2_card_diff.jsonl",
        provenance / "v1_split_reallocations.jsonl",
        provenance / "v1_lineage_label_rejections.jsonl",
        provenance / "source_rejection_audit.json",
        root / "summary.json",
    ]
    write_json_atomic(
        root / "migration_receipt.json",
        {
            "version": VERSION,
            "v1_root": str(prepared["v1_root"]),
            "v1_hashes": {
                path.name: sha256_file(path)
                for path in sorted(Path(prepared["v1_root"]).glob("*.jsonl"))
            },
            "artifacts": {
                str(path.relative_to(root)): sha256_file(path) for path in receipt_paths
            },
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--condition-ledger-root", type=Path, required=True)
    parser.add_argument("--voter-lineage-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, default=Path("data/gold_labels"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    prepared, blockers = {}, []
    for task in args.tasks:
        try:
            prepared[task] = prepare_task(
                task=task,
                stage1_root=args.stage1_root / task / "v10",
                ledger_path=args.condition_ledger_root
                / f"{task}.physical_condition_reviews.jsonl",
                lineage_path=args.voter_lineage_root
                / f"{task}.voter_lineage.jsonl",
                v1_root=args.v1_root,
            )
        except CoverageError as exc:
            blockers.append(exc.report)
    if blockers:
        report = {
            "version": VERSION,
            "status": "blocked",
            "message": "no v2 task was written because preflight failed",
            "required_condition_ledger_fields": sorted(REQUIRED_LEDGER_FIELDS),
            "mapping_origins": sorted(MAPPING_ORIGINS),
            "blockers": blockers,
        }
        write_json_atomic(args.output_root / "v2_build_blockers.json", report)
        print(json.dumps(report, indent=2))
        return 2
    summaries = {
        task: build_task(prepared[task], args.output_root) for task in args.tasks
    }
    write_json_atomic(args.output_root / "v2_build_summary.json", summaries)
    print(json.dumps(summaries, indent=2, default=str))
    return 0


def _read_v1_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {**row, "split": split}
        for split in SPLITS
        for row in _read_jsonl(root / f"{split}_molecule_condition_labels.jsonl")
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_assignments(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    parents, scaffolds = {}, {}
    for row in rows:
        parent = str(row["molecule_identity_key"])
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        split = str(row["split"])
        if parent in parents and parents[parent] != split:
            raise ValueError(f"v1 parent crosses splits: {parent}")
        if scaffold and scaffold in scaffolds and scaffolds[scaffold] != split:
            raise ValueError(f"v1 scaffold crosses splits: {scaffold}")
        parents[parent] = split
        if scaffold:
            scaffolds[scaffold] = split
    return parents, scaffolds


def _preserved_split_assignments(
    rows: Sequence[Mapping[str, Any]],
    *,
    v1_by_context: Mapping[tuple[str, str], Mapping[str, Any]],
    parent_splits: Mapping[str, str],
    scaffold_splits: Mapping[str, str],
    external_scaffold_allocation: Mapping[str, str],
    target_counts: Mapping[str, int],
) -> dict[tuple[str, str], str]:
    """Freeze surviving v1 contexts, then place only unconstrained scaffolds."""
    scaffold_assignment = dict(scaffold_splits)
    parent_assignment = dict(parent_splits)
    output: dict[tuple[str, str], str] = {}
    ordered = sorted(
        rows,
        key=lambda row: (str(row["molecule_identity_key"]), str(row["condition_group"])),
    )
    for row in ordered:
        parent = str(row["molecule_identity_key"])
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        context = (parent, str(row["condition_group"]))
        prior = v1_by_context.get(context)
        frozen_scaffold_split = scaffold_assignment.get(scaffold, "") if scaffold else ""
        split = (
            frozen_scaffold_split
            or parent_assignment.get(parent, "")
            or (str(prior["split"]) if prior else "")
            or (external_scaffold_allocation.get(scaffold, "") if scaffold else "")
        )
        if split:
            output[context] = split
            parent_assignment[parent] = split
            if scaffold:
                scaffold_assignment[scaffold] = split

    grouped_unassigned: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in ordered:
        parent = str(row["molecule_identity_key"])
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        context = (parent, str(row["condition_group"]))
        if context not in output:
            grouped_unassigned[scaffold or f"parent:{parent}"].append(context)
    counts = Counter(output.values())
    for identity, contexts in sorted(grouped_unassigned.items()):
        deficits = {
            split: int(target_counts.get(split, 0)) - counts[split]
            for split in SPLITS
        }
        positive = [split for split in SPLITS if deficits[split] > 0]
        split = (
            max(positive, key=lambda value: (deficits[value], -SPLITS.index(value)))
            if positive
            else "train"
        )
        for context in contexts:
            output[context] = split
        counts[split] += len(contexts)
        if not identity.startswith("parent:"):
            scaffold_assignment[identity] = split

    if len(output) != len(rows):
        raise ValueError("split assignment did not cover every publishable context")
    return output


def _v1_v2_diff(
    v1_rows: Sequence[Mapping[str, Any]], v2_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    old = {(row["molecule_identity_key"], row["condition_group"]): row for row in v1_rows}
    new = {(row["molecule_identity_key"], row["condition_group"]): row for row in v2_rows}
    output = []
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key), new.get(key)
        if before is None:
            status, changes = "added", ["card"]
        elif after is None:
            status, changes = "removed", ["card"]
        else:
            fields = ("Y", "drug", "source_record_count", "split", "benchmark_row_id")
            changes = [field for field in fields if before.get(field) != after.get(field)]
            status = "updated" if changes else "unchanged"
        output.append(
            {
                "molecule_identity_key": key[0],
                "condition_group": key[1],
                "status": status,
                "changes": changes,
                "v1": before,
                "v2": after,
            }
        )
    return output


def _source_rejection_audit(
    rows: Sequence[Mapping[str, Any]], decisions: Sequence[Any]
) -> dict[str, Any]:
    rejected = [
        {
            "source_row_uid": row.get("source_row_uid"),
            "source_id": row.get("source_id"),
            "source_record_id": row.get("source_record_id"),
            "reason": decision.reason,
        }
        for row, decision in zip(rows, decisions)
        if decision.record is None
    ]
    return {
        "considered": len(rows),
        "accepted_physical_voters": len(rows) - len(rejected),
        "rejected": len(rejected),
        "reason_counts": dict(Counter(row["reason"] for row in rejected)),
        "examples": rejected[:100],
    }


def _minimal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in (
            "drug",
            "Y",
            "condition_group",
            "condition_scope",
            "molecule_identity_key",
            "bemis_murcko_scaffold",
            "benchmark_row_id",
        )
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def _row_id(prefix: str, parent: str, condition_group: str) -> str:
    digest = hashlib.sha256(f"{parent}\0{condition_group}".encode()).hexdigest()[:20]
    return f"{prefix}_{digest.upper()}"


def _rows_digest(rows: Any) -> str:
    payload = "".join("\t".join(str(value) for value in row) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
