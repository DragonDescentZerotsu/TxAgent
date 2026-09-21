"""Build release-owned complete levels or a Gold-owned mapped subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from tools.chembl_tool.common.json_utils import write_json_atomic


LEVELS = {
    "bbb_martins": {
        "Tier 1.starling_direct_bbb_evidence": ("direct_brain_exposure", 1),
        "Proxy.central_functional_access": ("central_functional_access_proxy", 2),
        "Mechanism.passive_permeability": ("passive_permeability", 3),
        "Mechanism.efflux_transport": ("efflux_transport", 4),
        "Mechanism.influx_transport": ("influx_transport", 5),
    },
    "bioavailability_ma": {
        "Observed.direct_oral_bioavailability": ("direct_oral_bioavailability", 1),
        "Observed.nondirect_oral_bioavailability": ("nondirect_oral_bioavailability", 2),
        "Observed.oral_auc_cmax_exposure": ("oral_auc_cmax_exposure", 3),
        "Fa.absorption_solubility_permeability": ("absorption_solubility_permeability", 4),
        "Fg.gut_wall_efflux_intestinal_metabolism": ("gut_wall_efflux_intestinal_metabolism", 5),
        "Fh.hepatic_clearance_metabolic_stability": ("hepatic_clearance_metabolic_stability", 6),
    },
}


def input_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    parts = sorted(item for item in path.rglob("*") if item.is_file())
    if not parts:
        raise ValueError(f"input dataset has no files: {path}")
    for part in parts:
        digest.update(str(part.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(part).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write(
    records: pd.DataFrame,
    output_root: Path,
    manifest: dict,
    eligibility: pd.DataFrame | None = None,
    published_output_root: Path | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "records.parquet"
    records.to_parquet(path, index=False)
    manifest["output"] = {
        "path": str((published_output_root or output_root) / path.name),
        "sha256": file_sha256(path),
        "rows": len(records),
        "unique_source_row_uids": int(records.source_row_uid.nunique()),
        "rows_by_level": {
            str(level): int(count)
            for level, count in records.groupby("level").size().items()
        },
    }
    if eligibility is not None:
        eligibility_path = output_root / "assay_transfer_record_eligibility.parquet"
        eligibility.to_parquet(eligibility_path, index=False)
        manifest["assay_transfer_record_eligibility"] = {
            "version": "assay_transfer_record_eligibility.v1",
            "path": str(
                (published_output_root or output_root) / eligibility_path.name
            ),
            "sha256": file_sha256(eligibility_path),
            "rows": len(eligibility),
            "scope": "record-level scientific suitability; no bucket support or variance gates",
            "rows_by_level_and_eligibility": {
                str(level): {
                    str(bool(eligible)).lower(): int(count)
                    for eligible, count in level_rows.groupby(
                        "assay_transfer_eligible"
                    ).size().items()
                }
                for level, level_rows in eligibility.groupby("level")
            },
            "ineligibility_reason_counts": {
                str(reason): int(count)
                for reason, count in eligibility.loc[
                    ~eligibility.assay_transfer_eligible,
                    "assay_transfer_ineligibility_reason",
                ].value_counts().items()
            },
        }
    write_json_atomic(output_root / "manifest.json", manifest)


def build_complete(
    task: str,
    records_path: Path,
    prior_path: Path,
    voter_path: Path,
    output_root: Path,
    release_version: str,
    published_records_path: Path | None,
    published_output_root: Path | None = None,
    published_prior_mapping_path: Path | None = None,
) -> None:
    records = pd.read_parquet(records_path)
    prior = pd.read_parquet(prior_path)
    voters = set(pd.read_parquet(voter_path).source_row_uid.astype(str))
    record_uids = set(records.source_row_uid.astype(str))
    if not voters <= record_uids:
        raise ValueError(f"release lacks {len(voters - record_uids)} protected voters")
    prior = prior.drop(columns="canonical_record_id")
    mapped = records.merge(prior, on="source_row_uid", how="left", validate="one_to_one")
    missing = mapped.level.isna()
    if missing.any():
        routed = _route_unmapped(task, mapped.loc[missing], voters, records)
        for column in ("source_group_id", "family_key", "level"):
            mapped[column] = mapped[column].astype(object)
            mapped.loc[missing, column] = routed[column].tolist()
    mapped.loc[mapped.source_row_uid.astype(str).isin(voters), ["source_group_id", "family_key", "level"]] = [
        next(group for group, (_, level) in LEVELS[task].items() if level == 1),
        next(family for family, level in LEVELS[task].values() if level == 1),
        1,
    ]
    nonvoter_l1 = mapped.loc[mapped.level.eq(1) & ~mapped.source_row_uid.astype(str).isin(voters)]
    if len(nonvoter_l1) or mapped.level.isna().any():
        raise ValueError("release level routing violated complete exact-L1 coverage")
    eligibility_columns = [
        "source_row_uid",
        "canonical_record_id",
        "level",
        "pair_bucket_key",
        "measurement_kind",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
    ]
    if missing_columns := set(eligibility_columns) - set(mapped.columns):
        raise ValueError(
            "Stage 3 lacks assay-transfer eligibility fields: "
            f"{sorted(missing_columns)}"
        )
    output_columns = ["source_row_uid", "canonical_record_id", "source_group_id", "family_key", "level"]
    mapped["level"] = mapped.level.astype("int64")
    eligibility = mapped[eligibility_columns].copy()
    eligibility["assay_transfer_eligible"] = eligibility[
        "assay_transfer_eligible"
    ].astype(bool)
    missing_reason = eligibility.assay_transfer_ineligibility_reason.fillna("").eq("")
    if not missing_reason.eq(eligibility.assay_transfer_eligible).all():
        raise ValueError("assay-transfer eligibility and reason fields disagree")
    if eligibility.pair_bucket_key.fillna("").eq("").any():
        raise ValueError("assay-transfer eligibility row lacks pair_bucket_key")
    eligibility = eligibility.sort_values("source_row_uid").reset_index(drop=True)
    mapped = mapped[output_columns].sort_values("source_row_uid").reset_index(drop=True)
    _write(
        mapped,
        output_root,
        {
            "version": "evidence_library_level_mapping.v1",
            "status": "complete",
            "task": task,
            "evidence_library_version": release_version,
            "gold_release": "v1",
            "scope": "Complete retained Stage-3 UID universe with exact Gold-v1 physical voter membership at L1",
            "inputs": {
                "stage3_records": {"path": str(published_records_path or records_path), "sha256": file_sha256(records_path)},
                "prior_mapping": {
                    "path": str(published_prior_mapping_path or prior_path),
                    "sha256": input_sha256(prior_path),
                },
                "voter_membership": {"path": str(voter_path), "sha256": file_sha256(voter_path), "physical_voters": len(voters)},
            },
            "validation": {"stage3_uid_coverage": "exact", "l1_equals_gold_physical_voters": True},
            "newly_routed_rows": int(missing.sum()),
        },
        eligibility,
        published_output_root,
    )


def _route_unmapped(
    task: str, rows: pd.DataFrame, voters: set[str], all_records: pd.DataFrame
) -> pd.DataFrame:
    """Apply the existing reviewed purity classifier only to new physical UIDs."""
    if task == "bbb_martins":
        from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
            BBBSourceFamilyClassifier, load_near_direct_reviews,
        )
        voter_indices = {
            int(value) for value in all_records.loc[
                all_records.source_row_uid.astype(str).isin(voters), "source_index"
            ].dropna()
        }
        classifier = BBBSourceFamilyClassifier(load_near_direct_reviews(), voter_indices)
        groups = [
            classifier.classify_target(row).new_group or row["group_id"]
            for row in rows.to_dict("records")
        ]
    elif task == "bioavailability_ma":
        from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
            upstream_record_key, vote_pure_family_move,
        )
        voter_rows = all_records.loc[all_records.source_row_uid.astype(str).isin(voters)]
        voter_keys = {upstream_record_key(row) for row in voter_rows.to_dict("records")}
        groups = []
        for row in rows.to_dict("records"):
            move = vote_pure_family_move(row, voter_keys)
            groups.append(move.new_group or row["group_id"])
    else:
        raise ValueError(f"complete routing is unsupported for {task}")
    result = pd.DataFrame({"source_group_id": groups}, index=rows.index)
    result[["family_key", "level"]] = [LEVELS[task][group] for group in groups]
    return result


def build_subset(
    task: str,
    records_path: Path,
    prior_path: Path,
    output_root: Path,
    source_value_cleaning_audit: Path | None = None,
    published_records_path: Path | None = None,
    published_output_root: Path | None = None,
    published_prior_mapping_path: Path | None = None,
    published_cleaning_audit_path: Path | None = None,
    stage3_duplicate_lineage: Path | None = None,
    published_duplicate_lineage_path: Path | None = None,
    current_voter_membership: Path | None = None,
    published_voter_membership_path: Path | None = None,
) -> None:
    eligibility_columns = [
        "source_row_uid",
        "canonical_record_id",
        "pair_bucket_key",
        "measurement_kind",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
    ]
    records = pd.read_parquet(records_path)[eligibility_columns]
    prior = pd.read_parquet(prior_path)[["source_row_uid", "level"]]
    records["source_row_uid"] = records.source_row_uid.astype(str)
    prior["source_row_uid"] = prior.source_row_uid.astype(str)
    record_uids = set(records.source_row_uid)
    missing = set(prior.source_row_uid) - record_uids
    resolved = prior.loc[prior.source_row_uid.isin(record_uids)].copy()
    duplicate_remapped = pd.DataFrame(columns=["source_row_uid", "level"])
    stage1_remapped = pd.DataFrame(columns=["source_row_uid", "level"])
    stage3_duplicate_count = 0
    reviewed_drop_count = 0
    stage1_duplicate_count = 0
    if missing and stage3_duplicate_lineage is not None:
        duplicate_columns = ["source_row_uid", "retained_source_row_uid"]
        duplicates = pd.read_parquet(
            stage3_duplicate_lineage, columns=duplicate_columns
        )
        duplicates["source_row_uid"] = duplicates.source_row_uid.astype(str)
        duplicates["retained_source_row_uid"] = (
            duplicates.retained_source_row_uid.astype(str)
        )
        duplicates = duplicates.loc[duplicates.source_row_uid.isin(missing)]
        if duplicates.source_row_uid.duplicated().any():
            raise ValueError("Stage-3 duplicate lineage is not one-to-one")
        if not set(duplicates.retained_source_row_uid) <= record_uids:
            raise ValueError("Stage-3 duplicate survivor is absent from Stage 3")
        duplicate_remapped = duplicates.merge(
            prior, on="source_row_uid", validate="one_to_one"
        )[["retained_source_row_uid", "level"]].rename(
            columns={"retained_source_row_uid": "source_row_uid"}
        )
        stage3_duplicate_count = len(duplicates)
        missing -= set(duplicates.source_row_uid)
    if missing:
        if source_value_cleaning_audit is None:
            raise ValueError(f"Stage 3 lacks {len(missing)} Gold-owned mapped UIDs")
        audit = pd.read_parquet(source_value_cleaning_audit)
        required_audit_fields = {"source_row_uid", "field", "after"}
        if not required_audit_fields <= set(audit.columns):
            raise ValueError(
                "Stage-1 cleaning audit cannot resolve missing Gold mapping UIDs"
            )
        audit["source_row_uid"] = audit.source_row_uid.astype(str)
        audit = audit.loc[
            audit.source_row_uid.isin(missing)
            & audit.field.eq("record")
            & audit["after"].eq("dropped")
        ].copy()
        if (
            set(audit.source_row_uid) != missing
            or audit.source_row_uid.duplicated().any()
        ):
            raise ValueError(
                "Stage-1 lineage does not resolve every missing Gold mapping UID exactly once"
            )
        cleaning_version = audit.get(
            "cleaning_version", pd.Series("", index=audit.index)
        ).astype(str)
        rule_id = audit.get("rule_id", pd.Series("", index=audit.index)).astype(str)
        audit_type = audit.get(
            "audit_type", pd.Series("", index=audit.index)
        ).astype(str)
        action = audit.get("action", pd.Series("", index=audit.index)).astype(str)
        duplicate = cleaning_version.eq(
            "skin_reaction_stage1_exact_deduplication.v1"
        )
        reviewed_drop = rule_id.str.startswith("reviewed_drop:") | (
            action.eq("dropped") & audit_type.eq("endpoint_excluded")
        )
        if not (duplicate | reviewed_drop).all():
            raise ValueError("missing Gold mapping UID has unsupported Stage-1 lineage")
        if duplicate.any() and "retained_source_row_uid" not in audit:
            raise ValueError("Stage-1 duplicate audit lacks retained UID lineage")
        if "retained_source_row_uid" not in audit:
            audit["retained_source_row_uid"] = None
        stage1_rows = audit.loc[
            duplicate,
            ["source_row_uid", "retained_source_row_uid"],
        ]
        stage1_rows["retained_source_row_uid"] = (
            stage1_rows.retained_source_row_uid.astype(str)
        )
        if not set(stage1_rows.retained_source_row_uid) <= record_uids:
            raise ValueError("Stage-1 duplicate survivor is absent from Stage 3")
        stage1_remapped = stage1_rows.merge(
            prior, on="source_row_uid", validate="one_to_one"
        )[["retained_source_row_uid", "level"]].rename(
            columns={"retained_source_row_uid": "source_row_uid"}
        )
        stage1_duplicate_count = int(duplicate.sum())
        reviewed_drop_count = int(reviewed_drop.sum())
    resolved = pd.concat(
        [resolved, duplicate_remapped, stage1_remapped], ignore_index=True
    )
    if resolved.groupby("source_row_uid")["level"].nunique().gt(1).any():
        raise ValueError("duplicate lineage joins conflicting Gold levels")
    resolved = resolved.drop_duplicates("source_row_uid", keep="first")
    mapped = records.merge(
        resolved, on="source_row_uid", how="left", validate="one_to_one"
    )
    level_zero_rows = int(mapped.level.isna().sum())
    mapped["level"] = mapped.level.fillna(0)
    mapped["level"] = mapped.level.astype("int64")
    l1_validation = None
    if current_voter_membership is not None:
        voter_rows = pd.read_parquet(
            current_voter_membership, columns=["source_row_uid"]
        )
        voter_rows["source_row_uid"] = voter_rows.source_row_uid.astype(str)
        if voter_rows.source_row_uid.duplicated().any():
            raise ValueError("current voter membership contains duplicate UIDs")
        voters = set(voter_rows.source_row_uid)
        missing_voters = voters - record_uids
        if missing_voters:
            raise ValueError(
                f"Stage 3 lacks {len(missing_voters)} current physical voters"
            )
        output_l1 = set(mapped.loc[mapped.level.eq(1), "source_row_uid"])
        if output_l1 != voters:
            raise ValueError(
                "reviewed level mapping L1 differs from current voter membership"
            )
        l1_validation = {
            "policy": "reviewed_levels_preserved; voter_membership_is_validation_only",
            "physical_voters": len(voters),
            "l1_equals_current_physical_voters": True,
        }
    mapped = mapped.sort_values("source_row_uid").reset_index(drop=True)
    eligibility = mapped[[*eligibility_columns, "level"]].copy()
    eligibility["assay_transfer_eligible"] = eligibility[
        "assay_transfer_eligible"
    ].astype(bool)
    missing_reason = eligibility.assay_transfer_ineligibility_reason.fillna("").eq("")
    if not missing_reason.eq(eligibility.assay_transfer_eligible).all():
        raise ValueError("assay-transfer eligibility and reason fields disagree")
    if eligibility.pair_bucket_key.fillna("").eq("").any():
        raise ValueError("assay-transfer eligibility row lacks pair_bucket_key")
    eligibility = eligibility.sort_values("source_row_uid").reset_index(drop=True)
    mapped = mapped[["source_row_uid", "canonical_record_id", "level"]]
    inputs = {
        "stage3_records": {
            "path": str(published_records_path or records_path),
            "sha256": file_sha256(records_path),
        },
        "gold_level_mapping": {
            "path": str(published_prior_mapping_path or prior_path),
            "sha256": input_sha256(prior_path),
        },
    }
    if source_value_cleaning_audit is not None:
        inputs["stage1_source_value_cleaning_audit"] = {
            "path": str(published_cleaning_audit_path or source_value_cleaning_audit),
            "sha256": file_sha256(source_value_cleaning_audit),
        }
    if stage3_duplicate_lineage is not None:
        inputs["stage3_duplicate_lineage"] = {
            "path": str(
                published_duplicate_lineage_path or stage3_duplicate_lineage
            ),
            "sha256": file_sha256(stage3_duplicate_lineage),
        }
    if current_voter_membership is not None:
        inputs["current_voter_membership"] = {
            "path": str(
                published_voter_membership_path or current_voter_membership
            ),
            "sha256": file_sha256(current_voter_membership),
        }
    lineage = {
        "policy": (
            "stage3_or_stage1_exact_duplicate_to_retained_uid_or_reviewed_source_drop"
        ),
        "missing_gold_mapping_uids": int(
            stage3_duplicate_count + stage1_duplicate_count + reviewed_drop_count
        ),
        "stage3_exact_duplicate_uids_remapped": stage3_duplicate_count,
        "stage1_exact_duplicate_uids_remapped": stage1_duplicate_count,
        "unique_retained_uid_targets": int(
            pd.concat([duplicate_remapped, stage1_remapped], ignore_index=True)
            .source_row_uid.nunique()
        ),
        "reviewed_source_row_drops": reviewed_drop_count,
    }
    _write(
        mapped,
        output_root,
        {
            "version": "gold_owned_level_mapping.complete_stage3_v4",
            "task": task,
            "mode": "complete_stage3_with_unreviewed_level_zero",
            "inputs": inputs,
            "stage1_lineage_reconciliation": lineage,
            "unmapped_stage3_rows": 0,
            "level_zero_stage3_rows": level_zero_rows,
            "current_l1_validation": l1_validation,
            "validations": {
                "all_gold_mapping_uids_resolved": True,
                "all_output_uids_present_in_stage3": True,
                "duplicate_lineage_preserves_level": True,
                "exact_stage3_uid_coverage": len(mapped) == len(records),
                "unreviewed_rows_use_level_zero": True,
                "l1_equals_current_physical_voters": (
                    l1_validation is not None
                    and l1_validation["l1_equals_current_physical_voters"]
                ),
            },
        },
        eligibility=eligibility,
        published_output_root=published_output_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stage3-records", type=Path, required=True)
    parser.add_argument("--prior-mapping", type=Path, required=True)
    parser.add_argument("--voter-membership", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-version", default="v10_main_universe_v1")
    parser.add_argument("--published-stage3-path", type=Path)
    parser.add_argument("--published-output-root", type=Path)
    parser.add_argument("--published-prior-mapping-path", type=Path)
    parser.add_argument("--source-value-cleaning-audit", type=Path)
    parser.add_argument("--published-cleaning-audit-path", type=Path)
    parser.add_argument("--stage3-duplicate-lineage", type=Path)
    parser.add_argument("--published-duplicate-lineage-path", type=Path)
    parser.add_argument("--current-voter-membership", type=Path)
    parser.add_argument("--published-voter-membership-path", type=Path)
    args = parser.parse_args()
    if args.voter_membership:
        build_complete(
            args.task, args.stage3_records, args.prior_mapping,
            args.voter_membership, args.output_root, args.release_version,
            args.published_stage3_path, args.published_output_root,
            args.published_prior_mapping_path,
        )
    else:
        build_subset(
            args.task,
            args.stage3_records,
            args.prior_mapping,
            args.output_root,
            args.source_value_cleaning_audit,
            args.published_stage3_path,
            args.published_output_root,
            args.published_prior_mapping_path,
            args.published_cleaning_audit_path,
            args.stage3_duplicate_lineage,
            args.published_duplicate_lineage_path,
            args.current_voter_membership,
            args.published_voter_membership_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
