"""Build one complete current Starling task release under a single lock."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    load_task_policy,
    parse_args as parse_record_args,
    run_with_args,
)
from tools.chembl_tool.common.starling.build_runtime import starling_build_session
from tools.chembl_tool.common.starling.final_endpoint_pruning import (
    ARTIFACT_DIR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MANIFEST_FILENAME as REVIEW_MANIFEST_FILENAME,
    PruningBudgetExhausted,
    generate_final_endpoint_pruning,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.v7_benchmark_view import (
    ALL_PARENT_FILTER,
    DIRECT_SOURCE_ONLY_FILTER,
    FULL_VIEW,
    HELDOUT_FILTER_MODES,
    build_v7_benchmark_view,
)


TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")


def build_current_release(
    *,
    task_id: str,
    normalized_root: str | Path | None = None,
    workers: int = 1,
    validation_level: str = "strict",
    cache_mode: str = "auto",
    review_api_key_env: str = "OPENAI_API_KEY",
    review_base_url: str = DEFAULT_BASE_URL,
    review_model: str = DEFAULT_MODEL,
    review_workers: int | None = None,
    reuse_valid_cache_across_models: bool = False,
    heldout_labels: str | Path | None = None,
    view_out: str | Path | None = None,
    benchmark_split: str = "scaffold",
    heldout_filter_mode: str = ALL_PARENT_FILTER,
    record_argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Run Stage 2, review discovery, final Stage 3, and an optional view."""
    if task_id not in TASKS:
        raise ValueError(f"unsupported current Starling task: {task_id}")
    if (heldout_labels is None) != (view_out is None):
        raise ValueError("--heldout-labels and --view-out must be supplied together")

    policy = load_task_policy(task_id)
    root = Path(normalized_root or policy.default_out_dir)
    downstream = importlib.import_module(
        f"tools.chembl_tool.tasks.{task_id}.build_starling_downstream_artifacts"
    )
    stage2_args = parse_record_args(
        policy,
        [
            *record_argv,
            "--out-dir",
            str(root),
            "--from-stage",
            "source",
            "--through-stage",
            "normalize",
            "--workers",
            str(workers),
            "--validation-level",
            validation_level,
            "--cache-mode",
            cache_mode,
        ],
        "normalize",
        True,
    )

    with starling_build_session(root, complete=True):
        if run_with_args(policy, stage2_args):
            raise RuntimeError("Stage 2 build failed")
        downstream.build_canonical_artifacts(
            normalized_root=root,
            workers=workers,
            validation_level=validation_level,
            cache_mode="off",
            apply_outlier_review=False,
        )
        generate_final_endpoint_pruning(
            task_id=task_id,
            normalized_root=root,
            api_key_env=review_api_key_env,
            base_url=review_base_url,
            model=review_model,
            workers=review_workers or workers,
            reuse_valid_cache_across_models=reuse_valid_cache_across_models,
        )
        downstream.build_canonical_artifacts(
            normalized_root=root,
            workers=workers,
            validation_level=validation_level,
            cache_mode="off",
            apply_outlier_review=True,
        )
        view_manifest = None
        if heldout_labels is not None and view_out is not None:
            view_manifest = build_v7_benchmark_view(
                policy=policy,
                normalized_root=root,
                heldout_labels_jsonl=heldout_labels,
                out_dir=view_out,
                benchmark_split=benchmark_split,
                view=FULL_VIEW,
                heldout_filter_mode=heldout_filter_mode,
                downstream_spec=(
                    downstream.get_spec()
                    if heldout_filter_mode == DIRECT_SOURCE_ONLY_FILTER
                    else None
                ),
                workers=workers,
            )
        return _validate_release(task_id, root, view_manifest)


def _validate_release(
    task_id: str, root: Path, view_manifest: dict[str, Any] | None
) -> dict[str, Any]:
    canonical_records = root / "02_canonicalized/records.parquet"
    canonical_manifest_path = root / "02_canonicalized/manifest.json"
    review_manifest_path = root / ARTIFACT_DIR / REVIEW_MANIFEST_FILENAME
    stage3 = root / "03_pair_buckets"
    stage3_records = stage3 / "records.parquet"
    stage3_manifest_path = stage3 / "manifest.json"
    canonical_manifest = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    stage3_manifest = json.loads(stage3_manifest_path.read_text(encoding="utf-8"))

    canonical_sha = file_sha256(canonical_records)
    review_sha = file_sha256(review_manifest_path)
    stage3_sha = file_sha256(stage3_records)
    if canonical_manifest.get("output", {}).get("sha256") != canonical_sha:
        raise ValueError("Stage 2 records differ from their manifest")
    if review_manifest.get("task_id") != task_id or review_manifest.get(
        "inputs", {}
    ).get("canonical_records", {}).get("sha256") != canonical_sha:
        raise ValueError("pruning review is stale for Stage 2")
    for filename, expected in review_manifest.get("files", {}).items():
        if file_sha256(review_manifest_path.parent / filename) != expected:
            raise ValueError(f"pruning review output hash mismatch: {filename}")
    outputs = stage3_manifest.get("outputs", {})
    if stage3_manifest.get("task_id") != task_id or not outputs:
        raise ValueError("final Stage 3 manifest is incomplete or for another task")
    if stage3_manifest.get("input_hashes") != {
        "canonical_records": canonical_sha,
        "auxiliary_mapping_manifest": file_sha256(
            root / "02_canonicalized/auxiliary_mapping_manifest.json"
        ),
        "assay_transfer_outlier_review": review_sha,
    }:
        raise ValueError("final Stage 3 input hashes are incomplete or stale")
    for filename, expected in outputs.items():
        if file_sha256(stage3 / filename) != expected:
            raise ValueError(f"final Stage 3 output hash mismatch: {filename}")

    if view_manifest is not None:
        source = view_manifest.get("source_v7_records", {})
        if source.get("sha256") != stage3_sha or source.get(
            "manifest_sha256"
        ) != file_sha256(stage3_manifest_path):
            raise ValueError("held-out view is stale for final Stage 3")
        if not view_manifest.get("heldout_filter", {}).get(
            "zero_filter_scope_parent_overlap"
        ):
            raise ValueError("held-out view contains direct-scope parent overlap")

    result = {
        "task_id": task_id,
        "normalized_root": str(root),
        "canonical_records_sha256": canonical_sha,
        "review_manifest_sha256": review_sha,
        "stage3_records_sha256": stage3_sha,
        "stage3_manifest_sha256": file_sha256(stage3_manifest_path),
    }
    if view_manifest is not None:
        result["view_records_sha256"] = view_manifest["files"]["records"]
        result["view_index_manifest_sha256"] = view_manifest["files"][
            "index_manifest"
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--normalized-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--validation-level", choices=("strict", "full"), default="strict"
    )
    parser.add_argument("--cache-mode", choices=("auto", "off"), default="auto")
    parser.add_argument("--review-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--review-base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--review-model", default=DEFAULT_MODEL)
    parser.add_argument("--review-workers", type=int)
    parser.add_argument("--reuse-valid-cache-across-models", action="store_true")
    parser.add_argument("--heldout-labels", type=Path)
    parser.add_argument("--view-out", type=Path)
    parser.add_argument("--benchmark-split", default="scaffold")
    parser.add_argument(
        "--heldout-filter-mode", choices=HELDOUT_FILTER_MODES, default=ALL_PARENT_FILTER
    )
    args, record_argv = parser.parse_known_args(argv)
    try:
        result = build_current_release(
            task_id=args.task,
            normalized_root=args.normalized_root,
            workers=args.workers,
            validation_level=args.validation_level,
            cache_mode=args.cache_mode,
            review_api_key_env=args.review_api_key_env,
            review_base_url=args.review_base_url,
            review_model=args.review_model,
            review_workers=args.review_workers,
            reuse_valid_cache_across_models=args.reuse_valid_cache_across_models,
            heldout_labels=args.heldout_labels,
            view_out=args.view_out,
            benchmark_split=args.benchmark_split,
            heldout_filter_mode=args.heldout_filter_mode,
            record_argv=record_argv,
        )
    except PruningBudgetExhausted as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
