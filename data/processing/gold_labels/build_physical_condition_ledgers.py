"""Build exhaustive payload-bound BBB and Oral condition ledgers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.dataset as ds

from data.processing.gold_labels.benchmark_dataset import sha256_file
from data.processing.gold_labels.build_repaired_v2 import (
    EXCLUDED_EXTERNAL_CONDITION,
    LEDGER_VERSION,
    NO_REPORTED_CONDITION,
)
from data.processing.gold_labels.stage1_repaired_sources import (
    _label_oral_stage1_row,
    label_bbb_stage1_row,
    scientific_payload_sha256,
)
from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic


BBB_REVIEW_ROOT = Path(
    "data/artifacts/starling/bbb_martins/source_reviews/context_conditioned_review_v1"
)
ORAL_MANUAL_REVIEWS = Path(
    "data/artifacts/starling/bioavailability_ma/source_reviews/"
    "context_conditioned_review_v3/manual_physical_reviews.v1.jsonl"
)


def build_ledgers(
    *,
    stage1_root: Path,
    oral_migration_root: Path,
    voter_lineage_root: Path,
    v1_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    bbb_lineage_path = voter_lineage_root / "bbb_martins.voter_lineage.jsonl"
    oral_lineage_path = voter_lineage_root / "bioavailability_ma.voter_lineage.jsonl"
    bbb = _build_bbb(
        stage1_root / "bbb_martins/v10/01_cleaned/records.parquet",
        _allowed_groups(v1_root / "BBB_Martins/v1/scaffold"),
        _unique(_read_jsonl(bbb_lineage_path), "source_row_uid"),
    )
    oral = _build_oral(
        stage1_root / "bioavailability_ma/v10/01_cleaned/records.parquet",
        _allowed_groups(v1_root / "Bioavailability_Ma/v1/scaffold"),
        oral_migration_root,
        _unique(_read_jsonl(oral_lineage_path), "source_row_uid"),
    )
    outputs = {}
    for task, rows in (("bbb_martins", bbb), ("bioavailability_ma", oral)):
        path = output_root / f"{task}.physical_condition_reviews.jsonl"
        write_jsonl_atomic(path, sorted(rows, key=lambda row: row["source_row_uid"]))
        outputs[task] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
            "mapping_origin_counts": dict(
                sorted(Counter(row["mapping_origin"] for row in rows).items())
            ),
        }
    manifest = {
        "version": LEDGER_VERSION,
        "status": "complete",
        "outputs": outputs,
        "inputs": {
            "bbb_stage1": _file(stage1_root / "bbb_martins/v10/01_cleaned/records.parquet"),
            "oral_stage1": _file(stage1_root / "bioavailability_ma/v10/01_cleaned/records.parquet"),
            "bbb_proposal_audit": _file(BBB_REVIEW_ROOT / "proposal_audit.jsonl"),
            "bbb_review_verdicts": _file(BBB_REVIEW_ROOT / "review_verdicts.jsonl"),
            "oral_migrated_reviews": _file(oral_migration_root / "migrated_reviews.jsonl"),
            "oral_review_gaps": _file(oral_migration_root / "review_gaps.jsonl"),
            "oral_manual_reviews": _file(ORAL_MANUAL_REVIEWS),
            "bbb_voter_lineage": _file(bbb_lineage_path),
            "oral_voter_lineage": _file(oral_lineage_path),
        },
        "policy": {
            "active_external": "terminal accepted legacy or manual review in the frozen v1 condition allowlist",
            EXCLUDED_EXTERNAL_CONDITION: "reported external condition that must not vote in current gold",
            NO_REPORTED_CONDITION: "no canonical external condition proposed by the task's frozen proposal process",
        },
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def _build_bbb(
    stage1_path: Path,
    allowed: set[str],
    lineage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proposals = {
        int(row["source_index"]): row
        for row in _read_jsonl(BBB_REVIEW_ROOT / "proposal_audit.jsonl")
    }
    verdicts = _unique(_read_jsonl(BBB_REVIEW_ROOT / "review_verdicts.jsonl"), "source_record_id")
    output = []
    table = ds.dataset(stage1_path).to_table(filter=ds.field("source_id") == "direct_bbb")
    for source in table.to_pylist():
        uid = str(source["source_row_uid"])
        if uid not in lineage:
            continue
        labeled = label_bbb_stage1_row(source, allow_conditioned_context=True)
        if labeled.record is None:
            continue
        index = int(source.get("source_index") if source.get("source_index") is not None else int(source["source_row_number"]) - 1)
        proposal = proposals.get(index)
        group = str((proposal or {}).get("proposed_condition_group") or "")
        verdict = verdicts.get(f"starling-labs/BBB:row:{index}")
        if verdict and verdict.get("review_status") == "accepted" and group in allowed:
            decision, atoms = "active_external", [group]
        elif group:
            decision, atoms = EXCLUDED_EXTERNAL_CONDITION, _atoms(group)
        else:
            decision, atoms = NO_REPORTED_CONDITION, []
        if verdict and verdict.get("review_status") == "accepted" and int(verdict["Y"]) != labeled.record.label:
            raise ValueError(f"BBB label changed for reviewed row {index}")
        origin = "legacy_condition_review_replay" if verdict else (
            "condition_allowlist_exclusion" if group else "stage1_no_external_condition"
        )
        output.append(
            _ledger_row(
                "bbb_martins",
                source,
                labeled.record,
                decision=decision,
                group=group if decision != NO_REPORTED_CONDITION else "",
                atoms=atoms,
                origin=origin,
                rationale=(
                    f"Replayed terminal BBB condition review: {verdict.get('review_status')}; "
                    f"{verdict.get('review_reason')}"
                    if verdict
                    else (
                        f"Condition proposal was excluded from current gold: {(proposal or {}).get('proposal_reason')}"
                        if group
                        else "Frozen BBB proposal process found no supported external condition atom."
                    )
                ),
                extra={
                    "legacy_condition_review_id": f"starling-labs/BBB:row:{index}",
                    "legacy_review_status": (verdict or {}).get("review_status"),
                    "legacy_source_payload_sha256": (verdict or {}).get("source_payload_sha256"),
                    "proposal_reason": (proposal or {}).get("proposal_reason"),
                },
            )
        )
    return output


def _build_oral(
    stage1_path: Path,
    allowed: set[str],
    migration_root: Path,
    lineage: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    migrated = _unique(_read_jsonl(migration_root / "migrated_reviews.jsonl"), "source_row_uid")
    gaps = _unique(_read_jsonl(migration_root / "review_gaps.jsonl"), "source_row_uid")
    manual = _unique(_read_jsonl(ORAL_MANUAL_REVIEWS), "source_row_uid")
    unresolved_allowed = {
        uid
        for uid, row in gaps.items()
        if uid in lineage
        and row.get("proposed_condition_group") in allowed
        and uid not in manual
        and lineage[uid].get("v1_condition_group")
        != row.get("proposed_condition_group")
    }
    if unresolved_allowed:
        raise ValueError(f"Oral allowlisted condition gaps lack manual decisions: {sorted(unresolved_allowed)}")

    output = []
    eligible_uids: set[str] = set()
    table = ds.dataset(stage1_path).to_table(
        filter=ds.field("source_id").isin(["hf_bioavailability", "oral_exposure"])
    )
    for source in table.to_pylist():
        uid = str(source["source_row_uid"])
        if uid not in lineage:
            continue
        labeled = _label_oral_stage1_row(source, allow_conditioned_context=True)
        if labeled.record is None:
            continue
        eligible_uids.add(uid)
        inherited_group = str(lineage[uid]["v1_condition_group"])
        base = _ledger_row(
            "bioavailability_ma",
            source,
            labeled.record,
            decision=NO_REPORTED_CONDITION,
            group="",
            atoms=[],
            origin="stage1_no_external_condition",
            rationale="Frozen Oral proposal process found no canonical external condition for this physical row.",
        )
        if uid in manual:
            row = dict(manual[uid])
            for field in ("source_id", "source_record_id", "source_payload_sha256", "vote_label", "label_method"):
                if row.get(field) != base[field]:
                    raise ValueError(f"stale manual Oral review for {uid}: {field}")
            output.append(row)
            continue
        prior = migrated.get(uid)
        gap = gaps.get(uid)
        if prior:
            group = str(prior.get("condition_group") or prior.get("proposed_condition_group") or "")
            accepted = prior.get("review_status") == "accepted" and group in allowed
            if prior.get("review_status") == "accepted" and int(prior["Y"]) != labeled.record.label:
                raise ValueError(f"Oral label changed for reviewed row {uid}")
            output.append(
                {
                    **base,
                    "decision": "active_external" if accepted else EXCLUDED_EXTERNAL_CONDITION,
                    "condition_group": group,
                    "condition_atoms": [group] if accepted else _atoms(group),
                    "mapping_origin": "legacy_condition_review_replay",
                    "rationale": (
                        f"Replayed terminal Oral condition review: {prior.get('review_status')}; "
                        f"{prior.get('review_reason')}"
                    ),
                    "legacy_condition_review_id": prior.get("legacy_review_source_record_id"),
                    "legacy_review_status": prior.get("review_status"),
                }
            )
        elif gap:
            group = str(gap.get("proposed_condition_group") or "")
            if inherited_group != NO_REPORTED_CONDITION:
                if group != inherited_group:
                    raise ValueError(
                        f"Oral v1 condition lineage disagrees with physical proposal for {uid}: "
                        f"{inherited_group} != {group}"
                    )
                output.append(
                    {
                        **base,
                        "decision": "active_external",
                        "condition_group": inherited_group,
                        "condition_atoms": [inherited_group],
                        "mapping_origin": "v1_voter_lineage_replay",
                        "rationale": (
                            "Inherited the published v1 condition for this physical member "
                            "of the same canonical claim after claim decoupling."
                        ),
                        "legacy_condition_review_id": gap.get(
                            "legacy_review_source_record_id"
                        ),
                    }
                )
                continue
            output.append(
                {
                    **base,
                    "decision": EXCLUDED_EXTERNAL_CONDITION,
                    "condition_group": group,
                    "condition_atoms": _atoms(group),
                    "mapping_origin": "condition_allowlist_exclusion",
                    "rationale": (
                        "Condition is outside the current gold allowlist and cannot inherit "
                        f"a legacy representative verdict: {gap.get('migration_gap_reason')}"
                    ),
                    "legacy_condition_review_id": gap.get("legacy_review_source_record_id"),
                }
            )
        else:
            if inherited_group == NO_REPORTED_CONDITION:
                output.append(base)
            elif inherited_group in allowed:
                output.append(
                    {
                        **base,
                        "decision": "active_external",
                        "condition_group": inherited_group,
                        "condition_atoms": [inherited_group],
                        "mapping_origin": "v1_voter_lineage_replay",
                        "rationale": (
                            "Inherited the published v1 condition for this physical member "
                            "of the same canonical claim after claim decoupling."
                        ),
                        "legacy_condition_review_id": lineage[uid].get(
                            "v1_canonical_claim_id"
                        ),
                    }
                )
            else:
                raise ValueError(f"Oral v1 voter has an unsupported condition: {uid}")
    if (set(manual) & eligible_uids) - {row["source_row_uid"] for row in output}:
        raise ValueError("manual Oral reviews include a deleted or nonvoter row")
    return output


def _ledger_row(
    task: str,
    source: Mapping[str, Any],
    record: Any,
    *,
    decision: str,
    group: str,
    atoms: list[str],
    origin: str,
    rationale: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = scientific_payload_sha256(task, source)
    return {
        "source_row_uid": str(source["source_row_uid"]),
        "source_id": str(source["source_id"]),
        "source_record_id": str(source["source_record_id"]),
        "source_payload_sha256": payload,
        "vote_label": int(record.label),
        "label_method": str(record.label_method),
        "decision": decision,
        "condition_group": group,
        "condition_atoms": atoms,
        "reviewer": "physical_condition_ledger_builder.v1",
        "rationale": rationale,
        "mapping_origin": origin,
        "reviewed_source_id": str(source["source_id"]),
        "reviewed_source_payload_sha256": payload,
        **dict(extra or {}),
    }


def _allowed_groups(root: Path) -> set[str]:
    groups = {
        str(row["condition_group"])
        for split in ("train", "valid", "test")
        for row in _read_jsonl(root / f"{split}_molecule_condition_labels.jsonl")
    }
    groups.discard(NO_REPORTED_CONDITION)
    return groups


def _atoms(group: str) -> list[str]:
    return sorted(set(filter(None, group.split("+"))))


def _unique(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    output = {str(row[field]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate {field}")
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--oral-migration-root", type=Path, required=True)
    parser.add_argument("--voter-lineage-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, default=Path("data/gold_labels"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_ledgers(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
