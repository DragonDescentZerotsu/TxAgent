"""Build one complete current Starling task release under a single lock."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v1.build_runtime import (
    assert_unpublished_build_root,
    starling_build_session,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v8.assay_transfer_record_pruning import (
    ARTIFACT_DIR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RecordPruningBudgetExhausted,
    generate_assay_transfer_record_pruning,
)
from data.processing.evidence_library.versions.v8.assay_transfer_record_pruning import (
    MANIFEST_FILENAME as PRUNING_MANIFEST_FILENAME,
)
from data.processing.evidence_library.versions.v8.build_normalized_evidence_library import (
    load_task_policy,
    run_with_args,
)
from data.processing.evidence_library.versions.v8.build_normalized_evidence_library import (
    parse_args as parse_record_args,
)
from data.processing.evidence_library.versions.v8.task_registry import (
    import_task_module,
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
    review_budget_epoch: str = "",
    start_new_review_budget_epoch: bool = False,
    review_budget_max_tokens: int = 10_000_000,
    reuse_valid_cache_across_models: bool = False,
    record_argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Run Stage 2, record pruning, and final Stage 3."""
    if task_id not in TASKS:
        raise ValueError(f"unsupported current Starling task: {task_id}")
    policy = load_task_policy(task_id)
    root = Path(normalized_root or policy.default_out_dir)
    assert_unpublished_build_root(root)
    downstream = import_task_module(task_id, "build_starling_downstream_artifacts")
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
            apply_record_pruning=False,
        )
        generate_assay_transfer_record_pruning(
            task_id=task_id,
            normalized_root=root,
            api_key_env=review_api_key_env,
            base_url=review_base_url,
            model=review_model,
            workers=review_workers or workers,
            budget_epoch=review_budget_epoch,
            start_new_budget_epoch=start_new_review_budget_epoch,
            budget_max_tokens=review_budget_max_tokens,
            reuse_valid_cache_across_models=reuse_valid_cache_across_models,
        )
        downstream.build_canonical_artifacts(
            normalized_root=root,
            workers=workers,
            validation_level=validation_level,
            cache_mode="off",
            apply_record_pruning=True,
        )
        return _validate_release(task_id, root)


def _validate_release(task_id: str, root: Path) -> dict[str, Any]:
    canonical_records = root / "02_canonicalized/records.parquet"
    canonical_manifest_path = root / "02_canonicalized/manifest.json"
    pruning_manifest_path = root / ARTIFACT_DIR / PRUNING_MANIFEST_FILENAME
    stage3 = root / "03_pair_buckets"
    stage3_records = stage3 / "records.parquet"
    stage3_manifest_path = stage3 / "manifest.json"
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    pruning_manifest = json.loads(pruning_manifest_path.read_text(encoding="utf-8"))
    stage3_manifest = json.loads(stage3_manifest_path.read_text(encoding="utf-8"))

    canonical_sha = file_sha256(canonical_records)
    pruning_sha = file_sha256(pruning_manifest_path)
    stage3_sha = file_sha256(stage3_records)
    if canonical_manifest.get("output", {}).get("sha256") != canonical_sha:
        raise ValueError("Stage 2 records differ from their manifest")
    if (
        pruning_manifest.get("task_id") != task_id
        or pruning_manifest.get("inputs", {}).get("canonical_records", {}).get("sha256")
        != canonical_sha
    ):
        raise ValueError("assay-transfer record pruning is stale for Stage 2")
    for filename, expected in pruning_manifest.get("files", {}).items():
        if file_sha256(pruning_manifest_path.parent / filename) != expected:
            raise ValueError(
                f"assay-transfer record-pruning output hash mismatch: {filename}"
            )
    outputs = stage3_manifest.get("outputs", {})
    if stage3_manifest.get("task_id") != task_id or not outputs:
        raise ValueError("final Stage 3 manifest is incomplete or for another task")
    if stage3_manifest.get("input_hashes") != {
        "canonical_records": canonical_sha,
        "auxiliary_mapping_manifest": file_sha256(
            root / "02_canonicalized/auxiliary_mapping_manifest.json"
        ),
        "assay_transfer_record_pruning": pruning_sha,
    }:
        raise ValueError("final Stage 3 input hashes are incomplete or stale")
    for filename, expected in outputs.items():
        if file_sha256(stage3 / filename) != expected:
            raise ValueError(f"final Stage 3 output hash mismatch: {filename}")

    return {
        "task_id": task_id,
        "normalized_root": str(root),
        "canonical_records_sha256": canonical_sha,
        "record_pruning_manifest_sha256": pruning_sha,
        "stage3_records_sha256": stage3_sha,
        "stage3_manifest_sha256": file_sha256(stage3_manifest_path),
    }


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
    parser.add_argument("--review-budget-epoch", default="")
    parser.add_argument("--start-new-review-budget-epoch", action="store_true")
    parser.add_argument("--review-budget-max-tokens", type=int, default=10_000_000)
    parser.add_argument("--reuse-valid-cache-across-models", action="store_true")
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
            review_budget_epoch=args.review_budget_epoch,
            start_new_review_budget_epoch=args.start_new_review_budget_epoch,
            review_budget_max_tokens=args.review_budget_max_tokens,
            reuse_valid_cache_across_models=args.reuse_valid_cache_across_models,
            record_argv=record_argv,
        )
    except RecordPruningBudgetExhausted as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
