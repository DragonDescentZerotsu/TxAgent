"""Recover the physical source rows that are allowed to vote in gold v2.

The v2 rebuild is deliberately closed over the published v1 voter universe.
BBB already stored physical row IDs.  Oral stored canonical claim IDs, so this
module replays the pinned v1 claim artifact and expands each selected claim to
its HF/local physical members before Stage-1 deduplication.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from data.processing.gold_labels.benchmark_dataset import sha256_file
from data.processing.gold_labels.reviewed_conditioned_benchmark import (
    NO_REPORTED_CONDITION,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_benchmark import (
    load_label_decisions,
)
from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity


VERSION = "gold_v1_physical_voter_lineage.v1"
EXPECTED_ORAL_CLAIMS_SHA256 = (
    "045261cbda785092143eeadd636f78399f7b02f951b23480b16fb8dde22661c5"
)
EXPECTED_ORAL_MANIFEST_SHA256 = (
    "b986a214491f3b04c6f3ae6fc71cbd4eacad08d390b9b62de53f953966b0a257"
)
HISTORICAL_ORAL_GIT_COMMIT = "1bdcc453"
HISTORICAL_ORAL_GIT_ROOT = (
    "data/starling_data/bioavailability_ma/canonical_direct_v2"
)
EXPECTED_COUNTS = {
    "bbb_martins": {"v1_claim_votes": 7_634, "physical_members": 7_634},
    "bioavailability_ma": {
        "v1_null_claim_votes": 17_929,
        "v1_external_claim_votes": 924,
        "v1_claim_votes": 18_853,
        "physical_members": 19_479,
    },
}


def build_lineage(
    *,
    v1_root: Path,
    frozen_oral_root: Path,
    historical_oral_claims: Path,
    historical_oral_manifest: Path,
    oral_physical_source_rows: Path,
    prior_stage1_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write voter-lineage rows and compact preferred-UID inputs for Stage 1."""
    _require_hash(historical_oral_claims, EXPECTED_ORAL_CLAIMS_SHA256)
    _require_hash(historical_oral_manifest, EXPECTED_ORAL_MANIFEST_SHA256)
    output_root.mkdir(parents=True, exist_ok=True)

    rows_by_task = {
        "bbb_martins": _bbb_lineage(v1_root, prior_stage1_root),
        "bioavailability_ma": _oral_lineage(
            v1_root=v1_root,
            frozen_root=frozen_oral_root,
            claims_path=historical_oral_claims,
            manifest_path=historical_oral_manifest,
            physical_source_path=oral_physical_source_rows,
        ),
    }
    outputs: dict[str, Any] = {}
    for task, rows in rows_by_task.items():
        uids = sorted(str(row["source_row_uid"]) for row in rows)
        if len(uids) != len(set(uids)):
            raise ValueError(f"{task} voter lineage contains duplicate physical UIDs")
        lineage_path = output_root / f"{task}.voter_lineage.jsonl"
        preferred_path = output_root / f"{task}.preferred_source_row_uids.json"
        write_jsonl_atomic(
            lineage_path,
            sorted(rows, key=lambda row: str(row["source_row_uid"])),
        )
        write_json_atomic(
            preferred_path,
            {
                "version": VERSION,
                "task": task,
                "source_row_uids": uids,
                "source_row_uid_sha256": _rows_digest((uid,) for uid in uids),
            },
        )
        outputs[task] = {
            "lineage_path": str(lineage_path),
            "lineage_sha256": sha256_file(lineage_path),
            "preferred_source_row_uids_path": str(preferred_path),
            "preferred_source_row_uids_sha256": sha256_file(preferred_path),
            "physical_members": len(rows),
            "condition_counts": dict(
                sorted(Counter(str(row["v1_condition_group"]) for row in rows).items())
            ),
        }

    manifest = {
        "version": VERSION,
        "status": "complete",
        "policy": {
            "universe": "physical members of published gold-v1 voter claims only",
            "oral_claim_expansion": "each distinct HF/local physical member votes",
            "dedup_preference": "retain a v1 physical voter over a non-gold duplicate; otherwise minimum UID",
        },
        "inputs": {
            "v1_root": str(v1_root),
            "frozen_oral_root": str(frozen_oral_root),
            "historical_oral_claims": _file(historical_oral_claims),
            "historical_oral_manifest": _file(historical_oral_manifest),
            "historical_oral_git_source": {
                "commit": HISTORICAL_ORAL_GIT_COMMIT,
                "claims_path": f"{HISTORICAL_ORAL_GIT_ROOT}/direct_claims.parquet",
                "manifest_path": f"{HISTORICAL_ORAL_GIT_ROOT}/merge_manifest.json",
            },
            "oral_physical_source_rows": _file(oral_physical_source_rows),
            "prior_stage1_root": str(prior_stage1_root),
        },
        "outputs": outputs,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def _bbb_lineage(v1_root: Path, prior_stage1_root: Path) -> list[dict[str, Any]]:
    clean_root = prior_stage1_root / "bbb_martins/v10/01_cleaned"
    source_rows = pq.read_table(
        clean_root / "records.parquet",
        columns=["source_record_id", "source_row_uid"],
    ).to_pylist()
    uid_by_record = {
        str(row["source_record_id"]): str(row["source_row_uid"])
        for row in source_rows
    }
    for row in pq.read_table(clean_root / "source_value_cleaning_audit.parquet").to_pylist():
        if row.get("field") == "record" and row.get("after") == "dropped":
            uid_by_record[str(row["source_record_id"])] = str(row["source_row_uid"])

    output = []
    for card in _v1_cards(v1_root / "BBB_Martins/v1/scaffold"):
        for source_id in card["source_record_ids"]:
            record_id = str(source_id).rsplit(":row:", 1)[-1].removeprefix("row:")
            uid = uid_by_record.get(record_id)
            if not uid:
                raise ValueError(f"BBB v1 voter is absent from pre-dedup Stage 1: {source_id}")
            output.append(_lineage_row(card, uid, record_id, str(source_id)))
    _require_count("bbb_martins", "v1_claim_votes", len(output))
    _require_count("bbb_martins", "physical_members", len(output))
    return output


def _oral_lineage(
    *,
    v1_root: Path,
    frozen_root: Path,
    claims_path: Path,
    manifest_path: Path,
    physical_source_path: Path,
) -> list[dict[str, Any]]:
    active_null_cards = {
        str(row["molecule_identity_key"]): row
        for row in _v1_cards(v1_root / "Bioavailability_Ma/v1/scaffold")
        if row["condition_group"] == NO_REPORTED_CONDITION
    }
    frozen = {
        str(row["molecule_identity_key"]): row
        for row in _molecule_rows(frozen_root)
    }
    accepted_by_parent: dict[str, list[Any]] = defaultdict(list)
    for decision in load_label_decisions(
        source_path=claims_path, manifest_path=manifest_path
    )[0]:
        if decision.record is None:
            continue
        identity = normalize_molecule_identity(decision.record.smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(
                f"historical Oral accepted claim has invalid identity: {decision.record.source_record_id}"
            )
        parent = identity.parent_inchi_key or identity.parent_smiles
        accepted_by_parent[parent].append(decision.record)

    null_claim_cards: dict[str, dict[str, Any]] = {}
    for parent, card in frozen.items():
        active_card = active_null_cards.get(parent)
        if active_card is None:
            raise ValueError(f"frozen Oral null parent is absent from active v1: {parent}")
        votes = accepted_by_parent.get(parent, [])
        found = Counter(record.label for record in votes)
        expected = Counter({int(key): value for key, value in card["label_counts"].items()})
        if len(votes) != int(card["source_record_count"]) or found != expected:
            raise ValueError(f"historical Oral null-vote replay differs for parent {parent}")
        for record in votes:
            null_claim_cards[record.source_record_id] = {
                **active_card,
                "condition_group": NO_REPORTED_CONDITION,
            }
    _require_count(
        "bioavailability_ma", "v1_null_claim_votes", len(null_claim_cards)
    )

    external_claim_cards: dict[str, dict[str, Any]] = {}
    for card in _v1_cards(v1_root / "Bioavailability_Ma/v1/scaffold"):
        if card["condition_group"] == NO_REPORTED_CONDITION:
            continue
        for claim_id in card["source_record_ids"]:
            claim_id = str(claim_id)
            if claim_id in external_claim_cards:
                raise ValueError(f"Oral v1 claim appears in multiple cards: {claim_id}")
            external_claim_cards[claim_id] = card
    _require_count(
        "bioavailability_ma", "v1_external_claim_votes", len(external_claim_cards)
    )
    if set(null_claim_cards) & set(external_claim_cards):
        raise ValueError("Oral v1 null and external voter claims overlap")

    claims = {
        str(row["canonical_claim_id"]): row
        for row in pq.read_table(claims_path).to_pylist()
    }
    uid_by_source = {
        str(row["source_record_id"]): str(row["source_row_uid"])
        for row in pq.read_table(
            physical_source_path,
            columns=["source_record_id", "source_row_uid"],
        ).to_pylist()
    }
    claim_cards = {**null_claim_cards, **external_claim_cards}
    output = []
    for claim_id, card in sorted(claim_cards.items()):
        claim = claims.get(claim_id)
        if claim is None:
            raise ValueError(f"Oral v1 claim is absent from pinned artifact: {claim_id}")
        for source_record_id in claim["source_record_ids"]:
            source_record_id = str(source_record_id)
            uid = uid_by_source.get(source_record_id)
            if not uid:
                raise ValueError(
                    f"Oral v1 physical member lacks a current source UID: {source_record_id}"
                )
            output.append(
                {
                    **_lineage_row(card, uid, source_record_id, claim_id),
                    "v1_canonical_claim_id": claim_id,
                    "v1_claim_physical_member_count": len(claim["source_record_ids"]),
                }
            )
    _require_count("bioavailability_ma", "v1_claim_votes", len(claim_cards))
    _require_count("bioavailability_ma", "physical_members", len(output))
    return output


def _lineage_row(
    card: dict[str, Any], uid: str, physical_id: str, v1_vote_id: str
) -> dict[str, Any]:
    return {
        "lineage_version": VERSION,
        "source_row_uid": uid,
        "physical_source_record_id": physical_id,
        "v1_vote_id": v1_vote_id,
        "v1_molecule_identity_key": str(card["molecule_identity_key"]),
        "v1_condition_group": str(card["condition_group"]),
        "v1_benchmark_row_id": str(card["benchmark_row_id"]),
        "v1_split": str(card["split"]),
    }


def _v1_cards(root: Path) -> Iterable[dict[str, Any]]:
    for split in ("train", "valid", "test"):
        yield from _read_jsonl(root / f"{split}_molecule_condition_labels.jsonl")


def _molecule_rows(root: Path) -> Iterable[dict[str, Any]]:
    for split in ("train", "valid", "test"):
        yield from _read_jsonl(root / f"{split}_molecule_labels.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"historical artifact hash mismatch for {path}: {actual}")


def _require_count(task: str, field: str, actual: int) -> None:
    expected = EXPECTED_COUNTS[task][field]
    if actual != expected:
        raise ValueError(f"{task} {field}: expected {expected}, found {actual}")


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _rows_digest(rows: Iterable[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\0".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=Path("data/gold_labels"))
    parser.add_argument("--frozen-oral-root", type=Path, required=True)
    parser.add_argument("--historical-oral-claims", type=Path, required=True)
    parser.add_argument("--historical-oral-manifest", type=Path, required=True)
    parser.add_argument("--oral-physical-source-rows", type=Path, required=True)
    parser.add_argument("--prior-stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_lineage(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
