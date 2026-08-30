"""Build the strict parent-level ClinTox clinical-trial-failure benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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
    STARLING_BASE_PATH,
    STARLING_BASE_SHA256,
    TASK_NAME,
    build_canonical_frames,
    build_starling_coverage,
)


SOURCE_REVISION = "61620011ac2d376606219e2f3fa539139f80d385"
PROTOCOL_VERSION = "clintox_clinical_trial_failure_benchmark.v1"
DEFAULT_OUTPUT_ROOT = BUILD_ROOT
SEED = 20260815
EVAL_FRACTION = 0.10


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

    starling_table = pq.read_table(STARLING_BASE_PATH, columns=["SMILES"])
    coverage = build_starling_coverage(
        canonical["parent_labels"], starling_table.column("SMILES").to_pylist()
    )
    canonical_root.mkdir(parents=True, exist_ok=True)
    canonical_paths = {
        "source_identity_audit": canonical_root / "source_identity_audit.parquet",
        "parent_labels": canonical_root / "parent_labels.parquet",
        "reference_reconciliation": canonical_root / "reference_reconciliation.parquet",
        "starling_base_coverage": canonical_root / "starling_base_coverage.parquet",
        "qa_sample": canonical_root / "qa_sample.parquet",
    }
    _write_parquet(canonical_paths["source_identity_audit"], canonical["source_identity_audit"])
    _write_parquet(canonical_paths["parent_labels"], canonical["parent_labels"])
    _write_parquet(
        canonical_paths["reference_reconciliation"],
        canonical["reference_reconciliation"],
    )
    _write_parquet(canonical_paths["starling_base_coverage"], coverage["coverage"])
    qa_sample = _qa_sample(canonical["parent_labels"], seed=seed)
    _write_parquet(canonical_paths["qa_sample"], qa_sample)

    canonical_manifest = _canonical_manifest(
        canonical,
        coverage,
        canonical_paths,
        qa_sample_size=len(qa_sample),
    )
    write_json_atomic(canonical_root / "manifest.json", canonical_manifest)
    with atomic_output_path(canonical_root / "DATA_QUALITY_REPORT.md") as temporary:
        temporary.write_text(
            _render_data_quality_report(canonical_manifest), encoding="utf-8"
        )

    benchmark_summary = _write_benchmark(
        canonical["parent_labels"],
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
    write_jsonl_atomic(task_root / "molecule_labels.jsonl", all_rows)
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

    split_summary = _split_summary(split_rows, target, optimization, split_root)
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
    with atomic_output_path(task_root / "report_zh.md") as temporary:
        temporary.write_text(_render_benchmark_report(task_summary), encoding="utf-8")
    return task_summary


def _split_summary(
    split_rows: dict[str, list[dict[str, Any]]],
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
    return summary


def _canonical_manifest(
    canonical: dict[str, Any],
    coverage: dict[str, Any],
    paths: dict[str, Path],
    *,
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
            "starling_coverage_only": {
                "path": str(STARLING_BASE_PATH),
                "sha256": sha256_file(STARLING_BASE_PATH),
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
    for entry in source_manifest["files"]:
        path = SOURCE_ROOT / str(entry["name"])
        if path.stat().st_size != int(entry["size"]):
            raise ValueError(f"unexpected source size for {path}")
        observed = sha256_file(path)
        if observed != str(entry["sha256"]):
            raise ValueError(f"unexpected source hash for {path}: {observed}")
    for name, expected in SOURCE_SHA256.items():
        path = SOURCE_ROOT / name
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"unexpected source hash for {path}: {observed}")
    observed_starling = sha256_file(STARLING_BASE_PATH)
    if observed_starling != STARLING_BASE_SHA256:
        raise ValueError(
            f"unexpected ClinTox base source hash: {observed_starling}"
        )


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
            "summary.json",
        )
    )
    return {
        str(path): sha256_file(task_root / path)
        for path in relative
    }


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
            f"- benchmark parents with clintox_base_v1 evidence: "
            f"{coverage['n_benchmark_parents_with_clintox_base_evidence']:,}",
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
