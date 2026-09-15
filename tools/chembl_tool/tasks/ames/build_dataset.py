"""Build Ames source, conditioned splits, and retrieval without inference.

Run with --phase source|benchmark|retrieval|all or refresh-retrieval to preserve gold.
Identity responses must already be frozen; this adapter makes no name queries.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import json
import importlib
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    read_jsonl,
    write_json_atomic,
)
from tools.chembl_tool.common.build_runtime import sha256_file
from tools.chembl_tool.common.starling.build_conditioned_random_split import build_all
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    BUILD_ROOT,
    TASK_DIRECTORIES,
    task_root,
)
from tools.chembl_tool.common.starling.fresh_conditioned_benchmark import (
    build_fresh_conditioned_benchmark,
)
from tools.chembl_tool.common.starling.publish_conditioned_benchmark import publish
from tools.chembl_tool.paper_experiments.build_assay_family_catalog import TASKS
from tools.chembl_tool.tasks.ames import build_conditioned_source as source
from tools.chembl_tool.tasks.ames.identity_review import RESPONSES
from tools.chembl_tool.tasks.ames.reviewed_source import (
    DECISIONS,
    REVIEW_METHOD,
    PLACEMENTS,
)
from tools.chembl_tool.tasks.ames.source_contract import RETRIEVAL_VERSION, VERSION


MANIFEST = source.ROOT / "dataset_manifest.json"
CATALOG_ROOT = Path(TASKS["ames"]["output_root"]) / TASKS["ames"]["output_name"]
INDEX_ROOT = CATALOG_ROOT.parent.parent / "indices" / "ames"
POLICY = Path(__file__).with_name("source_contract.py")
SOURCE_BUILDER = Path(source.__file__)
IDENTITY_POLICY = Path(__file__).with_name("identity_review.py")
VOTES = source.CANONICAL_ROOT / "source_votes.jsonl"
RECORDS = source.CANONICAL_ROOT / "records.parquet"
SPLITS = ("train", "valid", "test")
IDENTITY_POLICIES = {"scaffold": "scaffold_disjoint", "random": "parent_disjoint"}
REVIEW_POLICY = Path(__file__).with_name("reviewed_source.py")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hashes(paths) -> list[dict]:
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def _verify(files: list[dict]) -> None:
    for item in files:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Artifact changed; rebuild its phase: {item['path']}")


def _expect(actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Ames lineage mismatch for {key}: {actual.get(key)!r}")


def _cli(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _source_dependencies():
    from rdkit import rdBase

    modules = (
        "tools.chembl_tool.common.molecule_identity",
        "tools.chembl_tool.common.json_utils",
        "tools.chembl_tool.common.build_runtime",
    )
    return {
        "files": {
            name: sha256_file(Path(importlib.import_module(name).__file__))
            for name in modules
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": source.pd.__version__,
            "pyarrow": importlib.import_module("pyarrow").__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }


def _source_files() -> list[dict]:
    manifest = _read(source.CANONICAL_ROOT / "manifest.json")
    _expect(
        manifest,
        {
            "contract": VERSION,
            "retrieval_contract": RETRIEVAL_VERSION,
            "build_dependencies": _source_dependencies(),
            "vote_unit": "one_unambiguous_pmid_parent_exact_condition",
            "review_method": REVIEW_METHOD,
            "source_manifest_sha256": sha256_file(source.RAW_ROOT / "manifest.json"),
            "pubchem_name_responses_sha256": sha256_file(RESPONSES),
            "policy_source_sha256": sha256_file(POLICY),
            "builder_sha256": sha256_file(SOURCE_BUILDER),
            "identity_policy_sha256": sha256_file(IDENTITY_POLICY),
            "semantic_reviews_sha256": sha256_file(DECISIONS),
            "placement_reviews_sha256": sha256_file(PLACEMENTS),
            "semantic_review_policy_sha256": sha256_file(REVIEW_POLICY),
        },
    )
    # Require receipts for the actual inputs, not merely whichever files an old
    # or partial manifest happens to list. Never follow draft artifact paths.
    required = {
        "records.parquet",
        "record_audit.parquet",
        "source_votes.jsonl",
        "study_vote_audit.jsonl",
        "pending_identity.jsonl",
    }
    if required - manifest["files"].keys():
        raise ValueError("Incomplete Ames canonical manifest; rebuild source phase")
    for name in required:
        if (
            Path(manifest["files"][name]["path"]).resolve()
            != (source.CANONICAL_ROOT / name).resolve()
        ):
            raise ValueError(f"Noncanonical Ames artifact path: {name}")
    files = list(manifest["files"].values())
    _verify(files)
    raw = _read(source.RAW_ROOT / "manifest.json")
    raw_files = [
        {"path": str(source.RAW_ROOT / name), "sha256": item["sha256"]}
        for name, item in raw["files"].items()
    ]
    _verify(raw_files)
    return (
        files
        + raw_files
        + _hashes(
            [
                source.RAW_ROOT / "manifest.json",
                source.CANONICAL_ROOT / "manifest.json",
                RESPONSES,
                POLICY,
                SOURCE_BUILDER,
                IDENTITY_POLICY,
                DECISIONS,
                PLACEMENTS,
                REVIEW_POLICY,
            ]
        )
    )


def _build_benchmark(source_files: list[dict]) -> list[dict]:
    build_fresh_conditioned_benchmark(
        task="ames",
        swap_evaluation_splits=True,
        record_votes=read_jsonl(VOTES),
        source_artifacts=(
            VOTES,
            source.CANONICAL_ROOT / "manifest.json",
            POLICY,
            RESPONSES,
            SOURCE_BUILDER,
            IDENTITY_POLICY,
        ),
    )
    # publish bootstraps a missing random split, but does not refresh an existing one.
    had_random = all(
        (task_root("ames", "random") / f"{split}.jsonl").is_file() for split in SPLITS
    )
    _verify(source_files)
    publish(tasks=("ames",))
    if had_random:
        build_all(tasks=("ames",))
    # The publisher replaces summary.json with publication counts. Preserve the
    # scientific group gates, label policy, and optimizer receipt in public data.
    staged_summary = BUILD_ROOT / TASK_DIRECTORIES["ames"] / "scaffold" / "summary.json"
    public_summary = (
        task_root("ames").parent / "provenance" / "scaffold_build_summary.json"
    )
    summary_hash = sha256_file(staged_summary)
    with atomic_output_path(public_summary) as temporary:
        shutil.copyfile(staged_summary, temporary)
        if sha256_file(temporary) != summary_hash:
            raise ValueError(
                "Ames staged scientific summary changed during publication"
            )
    paths = [public_summary]
    for scheme in IDENTITY_POLICIES:
        root = task_root("ames", scheme)
        paths.extend(
            root / name
            for name in (
                "summary.json",
                "heldout_molecule_condition_labels.jsonl",
                *(f"{split}.jsonl" for split in SPLITS),
                *(f"{split}_molecule_condition_labels.jsonl" for split in SPLITS),
            )
        )
    return _hashes(paths)


def _build_retrieval(workers: int) -> dict:
    _cli(
        "tools.chembl_tool.paper_experiments.build_assay_family_catalog",
        "--tasks",
        "ames",
        "--workers",
        str(workers),
    )
    catalog = CATALOG_ROOT / "family_assays.jsonl"
    _expect(
        _read(CATALOG_ROOT / "manifest.json"),
        {
            "records_sha256": sha256_file(RECORDS),
            "catalog_sha256": sha256_file(catalog),
            "family_assignment_unit": "source_record",
        },
    )
    split_count = min(workers, len(IDENTITY_POLICIES))

    def build_split(item):
        position, (scheme, policy) = item
        split_workers = workers // split_count + (position < workers % split_count)
        heldout = task_root("ames", scheme) / "heldout_molecule_condition_labels.jsonl"
        root = INDEX_ROOT / scheme
        _cli(
            "tools.chembl_tool.common.assay_retrieval",
            "build-index",
            "--task",
            "ames",
            "--records",
            str(RECORDS),
            "--ranked-assays",
            str(catalog),
            "--output-dir",
            str(root),
            "--heldout-molecules-jsonl",
            str(heldout),
            "--heldout-smiles-field",
            "drug",
            "--filter-scope-field",
            "heldout_filter_scope",
            "--filter-scope-value",
            "bacterial_outcome",
            "--neighbor-identity-policy-default",
            policy,
            "--evidence-prompt-profile",
            "assay_compact.mechanism_tagged_v4",
            "--max-record-examples",
            "3",
            "--max-support-text-chars",
            "0",
            "--workers",
            str(split_workers),
        )
        metadata = _read(root / "manifest.json")
        _expect(
            metadata,
            {
                "records_sha256": sha256_file(RECORDS),
                "ranked_assays_sha256": sha256_file(catalog),
                "heldout_molecules_jsonl_sha256": sha256_file(heldout),
                "filter_source_id": "",
                "filter_scope_field": "heldout_filter_scope",
                "filter_scope_value": "bacterial_outcome",
                "n_direct_heldout_records_after_filter": 0,
                "neighbor_identity_policy_default": policy,
                "index_sha256": sha256_file(root / "assay_neighbor_index.pkl"),
                "evidence_sha256": sha256_file(root / "assay_molecule_evidence.jsonl"),
            },
        )
        return scheme, _hashes(
            root / name
            for name in (
                "manifest.json",
                "assay_neighbor_index.pkl",
                "assay_molecule_evidence.jsonl",
            )
        )

    items = list(enumerate(IDENTITY_POLICIES.items()))
    if split_count == 1:
        indices = dict(map(build_split, items))
    else:
        with ThreadPoolExecutor(max_workers=split_count) as pool:
            indices = dict(pool.map(build_split, items))
    return {
        "catalog": _hashes(
            [
                CATALOG_ROOT / "manifest.json",
                catalog,
                Path(__file__).with_name("experiment_config.py"),
            ]
        ),
        "indices": indices,
    }


def build(*, phase: str = "all", workers: int = 128) -> dict:
    if (
        phase not in {"source", "benchmark", "retrieval", "all", "refresh-retrieval"}
        or workers < 1
    ):
        raise ValueError(
            "Expected phase source/benchmark/retrieval/all and positive workers"
        )
    frozen = None
    if phase == "refresh-retrieval":
        frozen = _read(MANIFEST)
        _verify(frozen["benchmark"])
        frozen_votes = sha256_file(VOTES)
        recorded_votes = next(
            item for item in frozen["source"] if Path(item["path"]) == VOTES
        )
        if frozen_votes != recorded_votes["sha256"]:
            raise ValueError(
                "Existing gold votes differ from the frozen dataset manifest"
            )
        frozen_manifest = sha256_file(MANIFEST)
    reuse_source = False
    if phase == "refresh-retrieval":
        try:
            _source_files()
        except (FileNotFoundError, ValueError, KeyError):
            pass
        else:
            reuse_source = True
            print(
                "Reusing canonical source: all input and output hashes match",
                flush=True,
            )
    if phase in {"source", "all", "refresh-retrieval"} and not reuse_source:
        # The source builder owns gold-candidate response coverage and the
        # identity loader rejects duplicate names; raw noncandidates need no query.
        dependencies = _source_dependencies()
        source.build(workers=workers)
        if _source_dependencies() != dependencies:
            raise RuntimeError("Source build dependencies changed during construction")
        source_manifest = source.CANONICAL_ROOT / "manifest.json"
        metadata = _read(source_manifest)
        metadata["build_dependencies"] = dependencies
        write_json_atomic(source_manifest, metadata)
    if frozen is not None and sha256_file(VOTES) != frozen_votes:
        raise ValueError(
            "Retrieval refresh changed gold votes; refusing downstream publication"
        )
    source_files = _source_files()
    result = {
        "schema_version": "ames_dataset.v1",
        "task": "ames",
        "phase": phase,
        "source_contract": VERSION,
        "retrieval_contract": RETRIEVAL_VERSION,
        "vote_unit": "one_unambiguous_pmid_parent_exact_condition",
        "review_method": REVIEW_METHOD,
        "model_evaluation_performed": False,
        "benchmark_scope": "identity_verified_source_subset_with_explicit_pending_identity_exclusions",
        "source": source_files,
        "verification_method": (
            "SHA256 of frozen sources, identity responses, policy, source votes and outputs; "
            "shared builders validate split integrity and condition coverage; index manifests "
            "must match records/catalog/heldout hashes and contain zero heldout bacterial "
            "outcome records across L1 and L2. No inference validation."
        ),
    }
    if frozen is not None:
        result["benchmark"] = frozen["benchmark"]
        result["gold_preservation"] = {
            "source_votes_sha256": frozen_votes,
            "previous_dataset_manifest_sha256": frozen_manifest,
            "benchmark_files_unchanged": True,
            "split_allocation_repeated": False,
        }
    elif phase == "retrieval":
        previous = _read(MANIFEST)
        _expect(previous, {"source": source_files, "source_contract": VERSION})
        if "benchmark" not in previous:
            raise ValueError("Run --phase benchmark before retrieval")
        _verify(previous["benchmark"])
        result["benchmark"] = previous["benchmark"]
        if "gold_preservation" in previous:
            result["gold_preservation"] = previous["gold_preservation"]
    elif phase in {"benchmark", "all"}:
        result["benchmark"] = _build_benchmark(source_files)
    if phase in {"retrieval", "all", "refresh-retrieval"}:
        _verify(source_files)
        result["retrieval"] = _build_retrieval(workers)
    _verify(source_files)
    if "benchmark" in result:
        _verify(result["benchmark"])
    if frozen is not None and {
        key: value for key, value in result.items() if key != "gold_preservation"
    } == {key: value for key, value in frozen.items() if key != "gold_preservation"}:
        print(
            "All build artifacts unchanged; preserving the dataset manifest", flush=True
        )
        return frozen
    # A successful upstream-only phase deliberately drops old downstream hashes.
    write_json_atomic(MANIFEST, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("source", "benchmark", "retrieval", "all", "refresh-retrieval"),
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(
            1,
            min(
                128,
                (
                    len(os.sched_getaffinity(0)) // 2
                    if hasattr(os, "sched_getaffinity")
                    else (os.cpu_count() or 2) // 2
                ),
            ),
        ),
    )
    args = parser.parse_args(argv)
    build(phase=args.phase, workers=args.workers)
    print(json.dumps({"manifest": str(MANIFEST), "phase": args.phase}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
