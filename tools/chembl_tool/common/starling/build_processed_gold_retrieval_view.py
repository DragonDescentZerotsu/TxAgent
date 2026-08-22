#!/usr/bin/env python3
"""Build a train-only assay-transfer view from processed Starling gold voters."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.compact_artifacts import (
    build_relational_evidence_catalog,
    write_compact_neighbor_index,
)
from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id
from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


SCHEMA_VERSION = "processed_starling_gold_retrieval_view.v1"
TASKS = {
    "bbb_martins": {
        "task": "BBB_Martins",
        "source_id": "direct_bbb",
        "group_id": "Tier 1.starling_direct_bbb_evidence",
    },
    "bioavailability_ma": {
        "task": "Bioavailability_Ma",
        "group_id": "Observed.direct_oral_bioavailability",
    },
    "skin_reaction": {
        "task": "Skin_Reaction",
        "source_id": "direct_skin_reaction",
        "group_id": "Direct.skin_reaction",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt_fields(task_id: str, voter: dict[str, Any]) -> dict[str, Any]:
    source = dict(voter["source_record"])
    if task_id == "bbb_martins":
        source.update(
            endpoint_name=source.get("quant_metric"),
            measurement_text=source.get("quant_value"),
            unit_text=source.get("quant_units"),
        )
    elif task_id == "skin_reaction":
        source.update(
            endpoint_name=source.get("reaction_type"),
            measurement_text=source.get("effect_metric"),
            unit_text=None,
        )
    else:
        source.update(
            endpoint_name="oral_bioavailability",
            measurement_text=source.get("oral_bioavailability_value"),
            unit_text=source.get("value_units"),
            statistic_type=source.get("value_parse_method"),
            oral_dose=source.get("dose"),
            study_context=source.get("species_or_population"),
            comparator_exposure=source.get("comparator"),
        )
    source["smiles"] = voter["drug"]
    return source


def _record(task_id: str, voter: dict[str, Any]) -> dict[str, Any]:
    config = TASKS[task_id]
    fields = _prompt_fields(task_id, voter)
    source_id = config.get("source_id")
    if task_id == "bioavailability_ma":
        source_id = (
            "hf_bioavailability"
            if fields.get("source_origin") == "hf"
            else "oral_exposure"
        )
    value_type = str(voter["voting_value_type"])
    return {
        **fields,
        "source_id": source_id,
        "source_name": str(voter["source_id"]),
        "source_row_number": int(voter["source_row_index"]),
        "source_record_id": str(voter["source_record_id"]),
        "canonical_record_id": str(voter["voting_record_key"]),
        "processed_gold_voting_record_key": str(voter["voting_record_key"]),
        "processed_gold_lineage": str(voter["source_lineage"]),
        "processed_gold_split": str(voter["split"]),
        "record_vote": int(voter["record_vote"]),
        "molecule_Y": int(voter["molecule_Y"]),
        "group_id": str(config["group_id"]),
        "canonical_smiles": str(voter["drug"]),
        "molecule_id": starling_molecule_id(str(voter["drug"])),
        "measurement_kind": "continuous" if value_type == "continuous" else "binary",
        "finite_scalar_value": voter.get("continuous_value_contribution"),
        "retrieval_eligible": True,
    }


def build_view(
    task_id: str, processed_root: Path, output: Path, workers: int = 1
) -> dict[str, Any]:
    config = TASKS[task_id]
    task_root = processed_root / config["task"] / "scaffold"
    summary_path = task_root / "summary.json"
    voters_path = task_root / "voting_records.jsonl"
    labels_path = task_root / "train_molecule_labels.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "processed_starling_gold.v1":
        raise ValueError("processed gold schema is not processed_starling_gold.v1")
    expected_hash = summary["artifact_sha256"]["voting_records"]
    if file_sha256(voters_path) != expected_hash:
        raise ValueError("processed gold voting-record hash differs from summary")

    labels = {str(row["molecule_identity_key"]): row for row in _read_jsonl(labels_path)}
    voters = [row for row in _read_jsonl(voters_path) if row["split"] == "train"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for voter in voters:
        key = str(voter["voting_record_key"])
        if key in seen:
            raise ValueError(f"duplicate processed gold voting record: {key}")
        seen.add(key)
        molecule = str(voter["molecule_identity_key"])
        if molecule not in labels:
            raise ValueError(f"train voter has no train molecule label: {key}")
        grouped[molecule].append(voter)
    if set(grouped) != set(labels):
        raise ValueError("processed train voters and molecule labels cover different parents")
    for molecule, rows in grouped.items():
        label = labels[molecule]
        counts = Counter(str(row["record_vote"]) for row in rows)
        if counts != Counter(label["label_counts"]) or len(rows) != label["voting_record_count"]:
            raise ValueError(f"processed voter aggregation differs for {molecule}")
        if any(int(row["molecule_Y"]) != int(label["Y"]) for row in rows):
            raise ValueError(f"processed voter parent label differs for {molecule}")

    records = [_record(task_id, voter) for voter in voters]
    families, bridge = build_relational_evidence_catalog(records)
    candidate = output.with_name(f".{output.name}.candidate")
    if candidate.exists():
        shutil.rmtree(candidate)
    records_dir = candidate / "06_records"
    evidence_dir = candidate / "07_molecule_evidence"
    index_dir = candidate / "08_neighbor_index"
    records_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    records_file = records_dir / "records.parquet"
    families_file = evidence_dir / "molecule_families.parquet"
    bridge_file = evidence_dir / "molecule_family_records.parquet"
    write_parquet(records_file, records)
    write_parquet(families_file, families)
    write_parquet(bridge_file, bridge)

    module = __import__(
        f"tools.chembl_tool.tasks.{task_id}.starling_policy",
        fromlist=["POLICY"],
    )
    profile = module.POLICY.compact_profile_for_contract("starling_record_contract.v7")
    index_manifest = write_compact_neighbor_index(
        profile=profile, families=families, output_dir=index_dir, workers=workers
    )
    files = {
        "records": {"path": "06_records/records.parquet", "sha256": file_sha256(records_file)},
        "families": {
            "path": "07_molecule_evidence/molecule_families.parquet",
            "sha256": file_sha256(families_file),
        },
        "bridge": {
            "path": "07_molecule_evidence/molecule_family_records.parquet",
            "sha256": file_sha256(bridge_file),
        },
    }
    records_manifest = {"schema_version": SCHEMA_VERSION, **files["records"]}
    evidence_manifest = {"schema_version": SCHEMA_VERSION, **files["bridge"]}
    (records_dir / "manifest.json").write_text(
        json.dumps(records_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence_dir / "manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "source_lineage": summary["source_lineage"],
        "reference_split": "train",
        "n_molecules": len(labels),
        "n_voting_records": len(voters),
        "processed_gold": {
            "summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
            "voting_records": {"path": str(voters_path), "sha256": expected_hash},
            "train_molecule_labels": {
                "path": str(labels_path),
                "sha256": file_sha256(labels_path),
            },
        },
        "files": files,
        "index": index_manifest,
    }
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output.exists():
        shutil.rmtree(output)
    candidate.replace(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = build_view(args.task, args.processed_root, args.output, args.workers)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
