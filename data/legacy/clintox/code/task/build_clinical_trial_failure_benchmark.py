"""Build the strict parent-level ClinTox clinical-trial-failure benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    allocate_scaffold_groups,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import BUILD_ROOT
from tools.chembl_tool.tasks.clintox.clinical_trial_failure_benchmark import (
    AACT_SOURCE_PATH,
    CANONICAL_ROOT,
    COMPARATOR_SOURCE_PATH,
    CONTRACT_VERSION,
    LINEAGE,
    REFERENCE_SOURCE_PATH,
    SOURCE_SHA256,
    SOURCE_ROOT,
    SEND_V2_HUMAN_PATH,
    SEND_V2_HUMAN_SHA256,
    TASK_NAME,
    build_canonical_frames,
    build_starling_coverage,
)
from tools.chembl_tool.tasks.clintox.starling_retrieval import (
    DIRECT_GATE_VERSION,
    trial_failure_gate_reason,
)
from tools.chembl_tool.tasks.clintox.starling_source import (
    DEFAULT_DATA_ROOT as SEND_V2_ROOT,
    SOURCE_RELEASE as SEND_V2_RELEASE,
    SOURCE_SPECS as SEND_V2_SPECS,
)


SOURCE_REVISION = "61620011ac2d376606219e2f3fa539139f80d385"
PROTOCOL_VERSION = "clintox_clinical_trial_failure_benchmark.v1"
DEFAULT_OUTPUT_ROOT = BUILD_ROOT
SEED = 20260815
EVAL_FRACTION = 0.10
HISTORICAL_BASE_PATH = Path(
    "data/starling_data/clintox/clintox_base_v1/extractions.parquet"
)
HISTORICAL_BASE_SHA256 = (
    "472b5239e38c59f91d1a60eece7caf7c37a83f3b00b785f9ead59a460b83c285"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build(
        canonical_root=Path(args.canonical_root),
        output_root=Path(args.output_root),
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build(
    *,
    canonical_root: Path = CANONICAL_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Build canonical source audits and one scaffold-disjoint benchmark."""
    _verify_sources()
    aacttox = pd.read_csv(AACT_SOURCE_PATH)
    comparator = pd.read_csv(COMPARATOR_SOURCE_PATH)
    reference = pd.read_csv(REFERENCE_SOURCE_PATH)
    canonical = build_canonical_frames(aacttox, comparator, reference)

    starling_table = pq.read_table(SEND_V2_HUMAN_PATH, columns=["SMILES"])
    coverage = build_starling_coverage(
        canonical["parent_labels"], starling_table.column("SMILES").to_pylist()
    )
    residual_candidates = _direct_residual_candidates()
    canonical_root.mkdir(parents=True, exist_ok=True)
    canonical_paths = {
        "source_identity_audit": canonical_root / "source_identity_audit.parquet",
        "parent_labels": canonical_root / "parent_labels.parquet",
        "reference_reconciliation": canonical_root / "reference_reconciliation.parquet",
        "human_clinical_toxicity_coverage": (
            canonical_root / "human_clinical_toxicity_coverage.parquet"
        ),
        "direct_residual_candidates": (
            canonical_root / "direct_residual_candidates.parquet"
        ),
        "qa_sample": canonical_root / "qa_sample.parquet",
    }
    _write_parquet(
        canonical_paths["source_identity_audit"], canonical["source_identity_audit"]
    )
    _write_parquet(canonical_paths["parent_labels"], canonical["parent_labels"])
    _write_parquet(
        canonical_paths["reference_reconciliation"],
        canonical["reference_reconciliation"],
    )
    _write_parquet(
        canonical_paths["human_clinical_toxicity_coverage"], coverage["coverage"]
    )
    _write_parquet(
        canonical_paths["direct_residual_candidates"], residual_candidates
    )
    qa_sample = _qa_sample(canonical["parent_labels"], seed=seed)
    _write_parquet(canonical_paths["qa_sample"], qa_sample)
    source_inventory = _source_inventory(canonical, residual_candidates)
    source_inventory_path = canonical_root / "source_inventory.json"
    write_json_atomic(source_inventory_path, source_inventory)
    canonical_paths["source_inventory"] = source_inventory_path

    canonical_manifest = _canonical_manifest(
        canonical,
        coverage,
        canonical_paths,
        source_inventory=source_inventory,
        qa_sample_size=len(qa_sample),
    )
    write_json_atomic(canonical_root / "manifest.json", canonical_manifest)
    with atomic_output_path(canonical_root / "DATA_QUALITY_REPORT.md") as temporary:
        temporary.write_text(
            _render_data_quality_report(canonical_manifest), encoding="utf-8"
        )

    benchmark_summary = _write_benchmark(
        canonical["parent_labels"],
        canonical["source_identity_audit"],
        output_root=output_root,
        canonical_root=canonical_root,
        canonical_manifest=canonical_manifest,
        seed=seed,
    )
    root_summary = {
        "lineage": LINEAGE,
        "protocol_version": PROTOCOL_VERSION,
        "canonical_manifest": str(canonical_root / "manifest.json"),
        "tasks": {"clintox": benchmark_summary},
    }
    write_json_atomic(output_root / "summary.json", root_summary)
    return root_summary


def _write_benchmark(
    frame: pd.DataFrame,
    source_audit: pd.DataFrame,
    *,
    output_root: Path,
    canonical_root: Path,
    canonical_manifest: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rows = frame.to_dict(orient="records")
    target = int(EVAL_FRACTION * len(rows))
    assignment, optimization = allocate_scaffold_groups(
        rows,
        target_size=target,
        seed=seed,
        optimize_record_support=False,
        exclude_empty_scaffold_from_heldout=True,
    )
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for row in rows:
        split = assignment[str(row.get("bemis_murcko_scaffold") or "")]
        clean = _json_safe_row(row)
        clean.update(
            {
                "split": split,
                "split_policy": LINEAGE,
            }
        )
        split_rows[split].append(clean)
    for values in split_rows.values():
        values.sort(key=lambda row: str(row["molecule_identity_key"]))

    task_root = output_root / TASK_NAME
    split_root = task_root / "scaffold"
    all_rows = sorted(
        [row for values in split_rows.values() for row in values],
        key=lambda row: str(row["molecule_identity_key"]),
    )
    parents = {str(row["molecule_identity_key"]): row for row in all_rows}
    voting_records = []
    for source in source_audit.to_dict(orient="records"):
        if not bool(source["label_eligible"]):
            continue
        parent = parents[str(source["molecule_identity_key"])]
        vote = int(source["declared_Y"])
        voting_records.append(
            {
                "voting_record_key": str(source["source_record_id"]),
                "task": TASK_NAME,
                "source_lineage": LINEAGE,
                "split": parent["split"],
                "drug": parent["drug"],
                "molecule_identity_key": parent["molecule_identity_key"],
                "molecule_Y": int(parent["Y"]),
                "record_vote": vote,
                "vote_matches_final_parent_label": vote == int(parent["Y"]),
                "voting_value_type": "binary",
                "source_id": source["source_name"],
                "source_record_id": source["source_record_id"],
                "source_row_index": int(source["source_row_number"]),
                "label_method": source["source_role"],
                "source_record": {
                    "smiles": source["source_smiles"],
                    "declared_Y": vote,
                },
            }
        )
    voting_records.sort(
        key=lambda row: (
            ("train", "valid", "test").index(row["split"]),
            row["molecule_identity_key"],
            row["source_id"],
            row["source_row_index"],
        )
    )
    write_jsonl_atomic(task_root / "molecule_labels.jsonl", all_rows)
    write_jsonl_atomic(split_root / "voting_records.jsonl", voting_records)
    for split, values in split_rows.items():
        write_jsonl_atomic(
            split_root / f"{split}.jsonl",
            [{"drug": row["drug"], "Y": int(row["Y"])} for row in values],
        )
        write_jsonl_atomic(
            split_root / f"{split}_molecule_labels.jsonl", values
        )
    heldout = split_rows["valid"] + split_rows["test"]
    write_jsonl_atomic(split_root / "heldout_molecule_labels.jsonl", heldout)

    split_summary = _split_summary(
        split_rows, voting_records, target, optimization, split_root
    )
    write_json_atomic(split_root / "summary.json", split_summary)
    task_summary = {
        "task": TASK_NAME,
        "status": "benchmark_ready",
        "lineage": LINEAGE,
        "protocol_version": PROTOCOL_VERSION,
        "gold_contract_version": CONTRACT_VERSION,
        "seed": seed,
        "task_definition": canonical_manifest["task_definition"],
        "parent_label_policy": canonical_manifest["label_policy"],
        "n_binary_molecules": len(all_rows),
        "n_voting_records": len(voting_records),
        "n_votes_differing_from_final_parent_label": sum(
            not row["vote_matches_final_parent_label"] for row in voting_records
        ),
        "all_label_counts": _label_counts(all_rows),
        "split_size_policy": {
            "formula": "floor(0.10 * n_binary_parent_molecules) for valid and test",
            "target_valid_size": target,
            "target_test_size": target,
        },
        "splits": {"scaffold": split_summary},
        "canonical_source_stats": canonical_manifest["stats"],
        "starling_coverage_stats": canonical_manifest["starling_evidence_coverage"],
        "paths": {
            "canonical_manifest": str(canonical_root / "manifest.json"),
            "molecule_labels": str(task_root / "molecule_labels.jsonl"),
            "scaffold_root": str(split_root),
            "voting_records": str(split_root / "voting_records.jsonl"),
        },
    }
    task_summary["artifact_sha256"] = _benchmark_hashes(task_root)
    task_summary["build_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "lineage": LINEAGE,
                "contract": CONTRACT_VERSION,
                "artifacts": task_summary["artifact_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json_atomic(task_root / "summary.json", task_summary)
    with atomic_output_path(task_root / "REPORT.md") as temporary:
        temporary.write_text(_render_benchmark_report(task_summary), encoding="utf-8")
    return task_summary


def _split_summary(
    split_rows: dict[str, list[dict[str, Any]]],
    voting_records: list[dict[str, Any]],
    target: int,
    optimization: dict[str, Any],
    split_root: Path,
) -> dict[str, Any]:
    identities = {
        split: {str(row["molecule_identity_key"]) for row in rows}
        for split, rows in split_rows.items()
    }
    scaffolds = {
        split: {str(row.get("bemis_murcko_scaffold") or "") for row in rows}
        for split, rows in split_rows.items()
    }
    overlap_identity = _pairwise_overlaps(identities)
    overlap_scaffold = _pairwise_overlaps(scaffolds)
    if any(overlap_identity.values()) or any(overlap_scaffold.values()):
        raise AssertionError("generated split has identity or scaffold leakage")
    summary: dict[str, Any] = {
        "method": "label_balanced_lexicographic_milp_scaffold_groups",
        "lineage": LINEAGE,
        "target_valid_size": target,
        "target_test_size": target,
        "actual_valid_size": len(split_rows["valid"]),
        "actual_test_size": len(split_rows["test"]),
        "optimization": optimization,
        "empty_bemis_murcko_policy": (
            "parents without a Bemis-Murcko scaffold remain train-only"
        ),
        "pairwise_identity_overlap": overlap_identity,
        "pairwise_scaffold_overlap": overlap_scaffold,
    }
    for split, rows in split_rows.items():
        summary[f"n_{split}"] = len(rows)
        summary[f"{split}_label_counts"] = _label_counts(rows)
        summary[f"n_{split}_scaffolds"] = len(scaffolds[split])
        summary[f"n_{split}_voting_records"] = sum(
            record["split"] == split for record in voting_records
        )
    summary["paths"] = {
        split: str(split_root / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    summary["paths"].update(
        {
            f"{split}_molecule_labels": str(
                split_root / f"{split}_molecule_labels.jsonl"
            )
            for split in ("train", "valid", "test")
        }
    )
    summary["paths"]["heldout_molecule_labels"] = str(
        split_root / "heldout_molecule_labels.jsonl"
    )
    summary["paths"]["voting_records"] = str(split_root / "voting_records.jsonl")
    return summary


def _canonical_manifest(
    canonical: dict[str, Any],
    coverage: dict[str, Any],
    paths: dict[str, Path],
    *,
    source_inventory: dict[str, Any],
    qa_sample_size: int,
) -> dict[str, Any]:
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "lineage": LINEAGE,
        "status": "benchmark_ready",
        "source_revision": {
            "repository": "deepchem/deepchem",
            "commit": SOURCE_REVISION,
        },
        "source_inputs": {
            "positive_aacttox": _source_entry(AACT_SOURCE_PATH),
            "negative_comparator_sweetfda": _source_entry(COMPARATOR_SOURCE_PATH),
            "joined_reference_only": _source_entry(REFERENCE_SOURCE_PATH),
            "human_clinical_toxicity_coverage_only": {
                "path": str(SEND_V2_HUMAN_PATH),
                "sha256": sha256_file(SEND_V2_HUMAN_PATH),
                "used_for_labels": False,
            },
        },
        "task_definition": {
            "grain": "molecular parent",
            "positive": (
                "at least one source record in the AACT-derived list of drugs from "
                "clinical trials that failed for toxicity reasons"
            ),
            "negative": (
                "FDA-approved SWEETLEAD comparator with no AACT toxicity-failure "
                "record after parent normalization"
            ),
            "out_of_scope_for_labels": [
                "ordinary adverse events without trial failure",
                "preclinical or in-vitro toxicity",
                "organ injury without development failure",
                "market withdrawal or black-box warning alone",
                "mechanistic toxicity liabilities",
                "Starling extraction categories or free-text inference",
            ],
        },
        "label_policy": {
            "method": "source-role existential positive",
            "positive_rule": "any AACT toxicity-failure source record => Y=1",
            "negative_rule": "SWEETLEAD FDA-approved source and no AACT record => Y=0",
            "source_role_overlap": "Y=1; approval and a toxicity-failed trial can coexist",
            "record_majority_vote": False,
            "starling_rows_create_labels": False,
            "identity_normalization": canonical["stats"]["identity_normalizer_version"],
        },
        "stats": canonical["stats"],
        "stage1_source_inventory": source_inventory,
        "starling_evidence_coverage": coverage["stats"],
        "qa_sample": {
            "method": "stable-hash sample, up to 30 parents per label",
            "n_rows": qa_sample_size,
            "purpose": "inspectable lineage QA, not a source for labels",
        },
        "known_limitations": [
            "The negative class is an approved-drug comparator, not proof that a molecule can never be toxic.",
            "The source lists are historical and do not represent clinical developments after the frozen source snapshot.",
            "The upstream joined clintox.csv used a lossy string join; seven valid comparator parents are retained by this source-first reconstruction but absent from that joined reference.",
            "Individual dataset licensing is not specified by TDC/DeepChem.",
        ],
        "paths": {},
    }
    for name, path in paths.items():
        manifest["paths"][name] = str(path)
        manifest["paths"][f"{name}_sha256"] = sha256_file(path)
    return manifest


def _verify_sources() -> None:
    source_manifest_path = SOURCE_ROOT / "SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_revision") != SOURCE_REVISION:
        raise ValueError("unexpected ClinTox source revision")
    for name, expected in SOURCE_SHA256.items():
        path = SOURCE_ROOT / name
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"unexpected source hash for {path}: {observed}")
    observed_starling = sha256_file(SEND_V2_HUMAN_PATH)
    if observed_starling != SEND_V2_HUMAN_SHA256:
        raise ValueError(
            f"unexpected ClinTox human source hash: {observed_starling}"
        )
    for spec in SEND_V2_SPECS:
        path = SEND_V2_ROOT / spec.source_id / "extractions.parquet"
        if pq.ParquetFile(path).metadata.num_rows != spec.expected_rows:
            raise ValueError(f"unexpected row count for {path}")
        if sha256_file(path) != spec.parquet_sha256:
            raise ValueError(f"unexpected source hash for {path}")


def _direct_residual_candidates() -> pd.DataFrame:
    """Map the frozen 338 strict old-base rows onto their send_v2 records."""
    if sha256_file(HISTORICAL_BASE_PATH) != HISTORICAL_BASE_SHA256:
        raise ValueError("unexpected historical ClinTox base source hash")
    columns = list(pq.ParquetFile(HISTORICAL_BASE_PATH).schema_arrow.names)
    candidates: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    old_row_number = 0
    for batch in pq.ParquetFile(HISTORICAL_BASE_PATH).iter_batches(
        columns=columns, batch_size=32_768
    ):
        for row in batch.to_pylist():
            if trial_failure_gate_reason(row).startswith("accept_"):
                key = (
                    str(row.get("pmid") or ""),
                    str(row.get("extraction_id") or ""),
                )
                if key in candidates:
                    raise ValueError(
                        f"duplicate historical direct candidate key: {key}"
                    )
                candidates[key] = (old_row_number, row)
            old_row_number += 1
    if len(candidates) != 338:
        raise ValueError(
            f"expected 338 historical direct candidates, found {len(candidates)}"
        )

    matched: list[dict[str, Any]] = []
    send_row_number = 0
    for batch in pq.ParquetFile(SEND_V2_HUMAN_PATH).iter_batches(
        columns=columns, batch_size=32_768
    ):
        for row in batch.to_pylist():
            key = (
                str(row.get("pmid") or ""),
                str(row.get("extraction_id") or ""),
            )
            historical = candidates.get(key)
            if historical is not None:
                old_index, old_row = historical
                if not _rows_equal(old_row, row, columns):
                    raise ValueError(
                        f"send_v2 row differs from historical candidate: {key}"
                    )
                identity = normalize_molecule_identity(str(row.get("SMILES") or ""))
                matched.append(
                    {
                        "source_id": "human_clinical_toxicity",
                        "source_release": SEND_V2_RELEASE,
                        "source_row_number": send_row_number,
                        "source_record_id": (
                            f"{SEND_V2_RELEASE}:human_clinical_toxicity:"
                            f"{send_row_number:09d}"
                        ),
                        "pmid": key[0],
                        "extraction_id": key[1],
                        "source_smiles": row.get("SMILES"),
                        "structure_status": identity.status,
                        "canonical_smiles": identity.canonical_smiles,
                        "parent_smiles": identity.parent_smiles,
                        "parent_inchi_key": identity.parent_inchi_key,
                        "source_role": "indirect",
                        "direct_residual_candidate": True,
                        "direct_gate_version": DIRECT_GATE_VERSION,
                        "direct_gate_reason": "accept_literal_toxicity_trial_failure",
                        "review_status": "pending_manual_review",
                        "retrieval_eligible": False,
                        "historical_base_source_row_number": old_index,
                    }
                )
            send_row_number += 1
    if len(matched) != len(candidates):
        raise ValueError(
            f"only {len(matched)} of {len(candidates)} direct candidates matched send_v2"
        )
    matched.sort(key=lambda row: int(row["source_row_number"]))
    return pd.DataFrame(matched)


def _rows_equal(
    left: dict[str, Any], right: dict[str, Any], columns: list[str]
) -> bool:
    def comparable(value: Any) -> Any:
        return None if isinstance(value, float) and math.isnan(value) else value

    return all(
        comparable(left.get(name)) == comparable(right.get(name)) for name in columns
    )


def _source_inventory(
    canonical: dict[str, Any], residual_candidates: pd.DataFrame
) -> dict[str, Any]:
    audit = canonical["source_identity_audit"]
    sources = [
        {
            "source_id": "aacttox",
            "path": str(AACT_SOURCE_PATH),
            "sha256": SOURCE_SHA256[AACT_SOURCE_PATH.name],
            "rows": int((audit["source_name"] == "aacttox").sum()),
            "label_eligible_rows": int(
                ((audit["source_name"] == "aacttox") & audit["label_eligible"]).sum()
            ),
            "task_relation": "direct_vote",
        },
        {
            "source_id": "sweetfda",
            "path": str(COMPARATOR_SOURCE_PATH),
            "sha256": SOURCE_SHA256[COMPARATOR_SOURCE_PATH.name],
            "rows": int((audit["source_name"] == "sweetfda").sum()),
            "label_eligible_rows": int(
                ((audit["source_name"] == "sweetfda") & audit["label_eligible"]).sum()
            ),
            "task_relation": "direct_vote",
        },
    ]
    for spec in SEND_V2_SPECS:
        entry = {
            "source_id": spec.source_id,
            "path": str(SEND_V2_ROOT / spec.source_id / "extractions.parquet"),
            "sha256": spec.parquet_sha256,
            "rows": spec.expected_rows,
            "task_relation": "indirect",
        }
        if spec.source_id == "human_clinical_toxicity":
            entry.update(
                direct_residual_candidate_rows=len(residual_candidates),
                direct_residual_review_status="pending_manual_review",
            )
        sources.append(entry)
    return {
        "schema_version": "clintox_stage1_source_inventory.v1",
        "benchmark_lineage": LINEAGE,
        "source_separation": "records remain in their provenance source",
        "cross_source_deduplication": False,
        "sources": sources,
        "counts": {
            "source_datasets": len(sources),
            "source_rows": sum(int(source["rows"]) for source in sources),
            "direct_vote_rows": int(audit["label_eligible"].sum()),
            "excluded_direct_source_rows": int((~audit["label_eligible"]).sum()),
            "indirect_rows": sum(spec.expected_rows for spec in SEND_V2_SPECS),
            "direct_residual_candidate_rows": len(residual_candidates),
        },
        "direct_residual_candidates": {
            "membership": (
                "frozen historical strict-gate rows exactly reconciled to send_v2"
            ),
            "source_role": "indirect",
            "retrieval_eligible": False,
            "review_status": "pending_manual_review",
            "gate_version": DIRECT_GATE_VERSION,
        },
        "historical_clintox_base_v1": {
            "ingested_as_source": False,
            "purpose": "freeze and reconcile the 338 residual candidates only",
            "path": str(HISTORICAL_BASE_PATH),
            "sha256": HISTORICAL_BASE_SHA256,
        },
    }


def _source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    with atomic_output_path(path) as temporary:
        frame.to_parquet(temporary, index=False)


def _qa_sample(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    selected = []
    for label in (0, 1):
        rows = frame[frame["Y"] == label].to_dict(orient="records")
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0qa\0{row['molecule_identity_key']}".encode("utf-8")
            ).hexdigest()
        )
        selected.extend(rows[:30])
    return pd.DataFrame(selected)


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        output[key] = value
    return output


def _pairwise_overlaps(values: dict[str, set[str]]) -> dict[str, int]:
    return {
        "train_valid": len(values["train"] & values["valid"]),
        "train_test": len(values["train"] & values["test"]),
        "valid_test": len(values["valid"] & values["test"]),
    }


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["Y"]) for row in rows)
    return {"0": counts.get(0, 0), "1": counts.get(1, 0)}


def _benchmark_hashes(task_root: Path) -> dict[str, str]:
    relative = [Path("molecule_labels.jsonl")]
    relative.extend(
        Path("scaffold") / name
        for name in (
            "train.jsonl",
            "valid.jsonl",
            "test.jsonl",
            "train_molecule_labels.jsonl",
            "valid_molecule_labels.jsonl",
            "test_molecule_labels.jsonl",
            "heldout_molecule_labels.jsonl",
            "voting_records.jsonl",
            "summary.json",
        )
    )
    hashes = {
        str(path): sha256_file(task_root / path)
        for path in relative
    }
    hashes["voting_records"] = hashes["scaffold/voting_records.jsonl"]
    return hashes


def _render_data_quality_report(manifest: dict[str, Any]) -> str:
    stats = manifest["stats"]
    coverage = manifest["starling_evidence_coverage"]
    reconciliation = stats["reference_reconciliation_counts"]
    return "\n".join(
        [
            "# ClinTox clinical-trial-failure canonical source QA",
            "",
            f"- contract: `{CONTRACT_VERSION}`",
            f"- status: `{manifest['status']}`",
            f"- source rows: {stats['n_source_identity_audit_rows']:,}",
            f"- labeled molecular parents: {stats['n_binary_parent_molecules']:,}",
            f"- Y=0/Y=1: {stats['parent_label_counts']['0']:,}/{stats['parent_label_counts']['1']:,}",
            f"- AACT/FDA source-role overlap parents: {stats['n_parent_source_role_overlaps']:,}",
            "",
            "## Definition",
            "",
            "Y=1 requires membership in the frozen AACT-derived toxicity-failed-trial list. "
            "Y=0 requires membership in the FDA-approved comparator list and absence from "
            "the positive source after molecular-parent normalization. Starling prose and "
            "mechanistic toxicity records do not vote.",
            "",
            "## Completeness and validity",
            "",
            f"- identity audit reconciliation: {stats['n_source_identity_audit_rows']:,} rows",
            f"- invalid/unresolved source structures: "
            f"{stats['source_identity_status_counts'].get('invalid_or_unresolved_smiles', 0):,}",
            f"- joined-reference common parent label matches/mismatches: "
            f"{stats['n_common_reference_label_matches']:,}/"
            f"{stats['n_common_reference_label_mismatches']:,}",
            f"- source-only/reference-only parents: "
            f"{reconciliation.get('source_reconstruction_only', 0):,}/"
            f"{reconciliation.get('reference_only', 0):,}",
            "",
            "## Starling evidence coverage (not labels)",
            "",
            f"- benchmark parents with send_v2 human clinical evidence: "
            f"{coverage['n_benchmark_parents_with_human_clinical_toxicity_evidence']:,}",
            "- Starling rows used to create labels: 0",
            "",
            "## Leakage boundary",
            "",
            "The canonical coverage table is descriptive only. Before retrieval-based "
            "evaluation, valid/test parents must be removed from every ClinTox evidence "
            "library and query-time retrieval must remain parent-disjoint.",
            "",
        ]
    )


def _render_benchmark_report(summary: dict[str, Any]) -> str:
    split = summary["splits"]["scaffold"]
    return "\n".join(
        [
            "# ClinTox clinical-trial-failure benchmark v1",
            "",
            f"- status: `{summary['status']}`",
            f"- lineage: `{LINEAGE}`",
            f"- parent molecules: {summary['n_binary_molecules']:,}",
            f"- Y=0/Y=1: {summary['all_label_counts']['0']:,}/{summary['all_label_counts']['1']:,}",
            f"- scaffold train/valid/test: {split['n_train']:,}/{split['n_valid']:,}/{split['n_test']:,}",
            f"- valid Y=0/Y=1: {split['valid_label_counts']['0']:,}/{split['valid_label_counts']['1']:,}",
            f"- test Y=0/Y=1: {split['test_label_counts']['0']:,}/{split['test_label_counts']['1']:,}",
            "- parent identity overlap: 0",
            "- Bemis-Murcko scaffold overlap: 0",
            "",
            "Y=1 is an existential source event: an AACT-derived record says the drug was "
            "associated with a clinical trial that failed for toxicity. FDA approval does "
            "not erase such an event. Y=0 is the FDA-approved comparator class with no "
            "positive source record after parent normalization.",
            "",
            "General toxicity evidence remains retrieval context only and cannot create or "
            "override this benchmark label.",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
