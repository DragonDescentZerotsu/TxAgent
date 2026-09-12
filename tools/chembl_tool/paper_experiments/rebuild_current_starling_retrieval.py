"""Rebuild and verify the sole current Starling progressive retrieval lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
from tools.chembl_tool.common.json_utils import atomic_output_path
from tools.chembl_tool.common.build_runtime import sha256_file as _sha256
from tools.chembl_tool.common.starling.current_retrieval_artifacts import (
    DEFAULT_LOCAL_ROOT,
    TASKS,
    current_records_path,
    require_current_records,
    restore,
    restore_published_files,
    verify_local,
    verify_packaged,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = Path(__file__).with_name("current_starling_retrieval.json")
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT / "outputs/paper/starling_conditioned_assay_family_curve_v1"
)

def _contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "current_starling_retrieval.v1":
        raise ValueError(f"unsupported current retrieval contract: {path}")
    if set(contract.get("tasks") or {}) != set(TASKS):
        raise ValueError(f"current retrieval contract task set is incomplete: {path}")
    return contract


def _run(arguments: list[str]) -> None:
    command = [sys.executable, "-m", *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _overlay_paths(
    root: Path, contract: dict[str, Any], *, records_root: Path = DEFAULT_LOCAL_ROOT,
) -> dict[str, Path]:
    return {
        task: (
            current_records_path(task, local_root=records_root).parent
            if contract["tasks"][task].get("overlay_is_canonical")
            else root / str(contract["tasks"][task]["overlay"])
        )
        for task in TASKS
    }


def build_overlays(
    root: Path,
    *,
    records_root: Path = DEFAULT_LOCAL_ROOT,
    tasks: tuple[str, ...] = TASKS,
) -> None:
    contract = _contract()
    for task in tasks:
        require_current_records(task, local_root=records_root)
    overlays = _overlay_paths(root, contract, records_root=records_root)
    # Reviewed canonical snapshots (Ames/DILI/Carcinogens) need no overlay replay.
    if "bbb_martins" in tasks:
        _run([
            "tools.chembl_tool.paper_experiments.build_bbb_source_family_purity",
            "--records", str(current_records_path("bbb_martins", local_root=records_root)),
            "--output-dir", str(overlays["bbb_martins"]),
        ])
    if "bioavailability_ma" in tasks:
        intermediate = root / contract["tasks"]["bioavailability_ma"]["intermediate_overlay"]
        _run([
            "tools.chembl_tool.tasks.bioavailability_ma.build_nondirect_assay_context",
            "--input-records", str(current_records_path("bioavailability_ma", local_root=records_root)),
            "--output-dir", str(intermediate),
        ])
        _run([
            "tools.chembl_tool.paper_experiments.build_bioavailability_vote_pure_source",
            "--input-records", str(intermediate / "records.parquet"),
            "--output-dir", str(overlays["bioavailability_ma"]),
        ])
    if "skin_reaction" in tasks:
        _run([
            "tools.chembl_tool.paper_experiments.build_conditioned_source_family_purity",
            "--task", "skin_reaction",
            "--input-records", str(current_records_path("skin_reaction", local_root=records_root)),
            "--output-root", str(overlays["skin_reaction"].parent),
            "--batch-size", "20000",
        ])


def build_catalogs(root: Path, *, records_root: Path = DEFAULT_LOCAL_ROOT, tasks: tuple[str, ...] = TASKS) -> None:
    contract = _contract()
    overlays = _overlay_paths(root, contract, records_root=records_root)
    for task in tasks:
        catalog = root / str(contract["tasks"][task]["catalog"])
        _run(
            [
                "tools.chembl_tool.paper_experiments.build_assay_family_catalog",
                "--tasks",
                task,
                "--records",
                str(overlays[task] / "records.parquet"),
                "--output-root",
                str(catalog.parent),
                "--output-name",
                catalog.name,
                "--config-name",
                str(contract["tasks"][task]["catalog_config"]),
            ]
        )


def build_indices(
    root: Path, *, workers: int, records_root: Path = DEFAULT_LOCAL_ROOT,
    tasks: tuple[str, ...] = TASKS,
) -> None:
    contract = _contract()
    settings = contract["contract"]
    overlays = _overlay_paths(root, contract, records_root=records_root)
    for split_scheme in ("scaffold", "random"):
        identity_policy = settings[f"{split_scheme}_identity_policy"]
        for task in tasks:
            task_contract = contract["tasks"][task]
            catalog = root / str(task_contract["catalog"])
            index_dir = root / str(task_contract["indices"][split_scheme]["path"])
            heldout = task_root(task, split_scheme) / "heldout_molecule_condition_labels.jsonl"
            _run(
                [
                    "tools.chembl_tool.common.assay_retrieval",
                    "build-index",
                    "--task",
                    task,
                    "--records",
                    str(overlays[task] / "records.parquet"),
                    "--ranked-assays",
                    str(catalog / "family_assays.jsonl"),
                    "--output-dir",
                    str(index_dir),
                    "--workers",
                    str(workers),
                    "--neighbor-identity-policy-default",
                    identity_policy,
                    "--max-record-examples",
                    str(settings["max_record_examples"]),
                    "--max-support-text-chars",
                    str(settings["max_support_text_chars"]),
                    "--heldout-molecules-jsonl",
                    str(heldout),
                    "--heldout-smiles-field",
                    "drug",
                    "--filter-scope-field",
                    str(task_contract.get("filter_scope_field", "group_id")),
                    "--filter-scope-value",
                    str(task_contract.get("filter_scope_value", task_contract["direct_group"])),
                    "--evidence-prompt-profile",
                    str(settings["evidence_prompt_profile"]),
                ] + ([
                    "--identity-contract", task_contract["identity_contract"],
                    "--identity-cache", str(overlays[task] / "retrieval_identity_cache.jsonl"),
                ] if task_contract.get("identity_contract") else [])
            )


def verify(root: Path, *, records_root: Path = DEFAULT_LOCAL_ROOT, tasks: tuple[str, ...] = TASKS) -> dict[str, Any]:
    contract = _contract()
    overlays = _overlay_paths(root, contract, records_root=records_root)
    report: dict[str, Any] = {"schema_version": contract["schema_version"], "tasks": {}}
    failures: list[str] = []
    for task in tasks:
        task_contract = contract["tasks"][task]
        task_report: dict[str, Any] = {}
        for check_name, check in (
            ("packaged_stage03", lambda: verify_packaged(task)),
            (
                "local_stage03_inventory",
                lambda: verify_local(task, local_root=records_root),
            ),
        ):
            try:
                check()
                task_report[check_name] = {"ok": True}
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                task_report[check_name] = {"ok": False, "error": str(error)}
                failures.append(f"{task}.{check_name}")
        records = current_records_path(task, local_root=records_root)
        artifacts = {
            "canonical_records": (records, task_contract["canonical_records_sha256"]),
            "overlay_records": (
                overlays[task] / "records.parquet",
                task_contract["overlay_records_sha256"],
            ),
            "catalog": (
                root / str(task_contract["catalog"]) / "family_assays.jsonl",
                task_contract["catalog_sha256"],
            ),
        }
        if task_contract.get("source_records"):
            artifacts["published_source_records"] = (
                PROJECT_ROOT / task_contract["source_records"],
                task_contract["canonical_records_sha256"],
            )
        if task == "bioavailability_ma":
            artifacts["intermediate_overlay_records"] = (
                root / str(task_contract["intermediate_overlay"]) / "records.parquet",
                task_contract["intermediate_overlay_records_sha256"],
            )
        for split_scheme, index_contract in task_contract["indices"].items():
            index_root = root / str(index_contract["path"])
            artifacts[f"{split_scheme}_index"] = (
                index_root / "assay_neighbor_index.pkl",
                index_contract["index_sha256"],
            )
            artifacts[f"{split_scheme}_evidence"] = (
                index_root / "assay_molecule_evidence.jsonl",
                index_contract["evidence_sha256"],
            )
        for name, (path, expected) in artifacts.items():
            actual = _sha256(path) if path.is_file() else ""
            ok = actual == expected
            task_report[name] = {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "ok": ok,
            }
            if not ok:
                failures.append(f"{task}.{name}")
        report["tasks"][task] = task_report
    report["ok"] = not failures
    report["failures"] = failures
    if failures:
        raise ValueError("current Starling retrieval hash mismatch: " + ", ".join(failures))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "restore-records",
            "build",
            "build-overlays",
            "build-catalogs",
            "build-indices",
            "verify",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a nonmatching restored Stage-03 directory (restore-records only).",
    )
    args = parser.parse_args(argv)
    tasks = tuple(args.tasks)
    if args.force and args.action != "restore-records":
        parser.error("--force is only valid with restore-records")
    if args.action == "restore-records":
        for task in tasks:
            restored = restore(task, local_root=args.records_root, force=args.force)
            print(restored)
            if args.records_root.resolve() == DEFAULT_LOCAL_ROOT.resolve():
                restore_published_files(task, restored)
            source_records = _contract()["tasks"][task].get("source_records")
            # Restore missing source-adapter output, preserving existing reviews.
            if source_records and args.records_root.resolve() == DEFAULT_LOCAL_ROOT.resolve():
                published = PROJECT_ROOT / source_records
                if not published.exists():
                    with atomic_output_path(published) as temporary:
                        shutil.copyfile(current_records_path(task), temporary)
    elif args.action == "build":
        build_overlays(args.output_root, records_root=args.records_root, tasks=tasks)
        build_catalogs(args.output_root, records_root=args.records_root, tasks=tasks)
        build_indices(args.output_root, workers=args.workers, records_root=args.records_root, tasks=tasks)
        print(json.dumps(verify(args.output_root, records_root=args.records_root, tasks=tasks), indent=2))
    elif args.action == "build-overlays":
        build_overlays(args.output_root, records_root=args.records_root, tasks=tasks)
    elif args.action == "build-catalogs":
        build_catalogs(args.output_root, records_root=args.records_root, tasks=tasks)
    elif args.action == "build-indices":
        build_indices(args.output_root, workers=args.workers, records_root=args.records_root, tasks=tasks)
    else:
        print(json.dumps(verify(args.output_root, records_root=args.records_root, tasks=tasks), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
