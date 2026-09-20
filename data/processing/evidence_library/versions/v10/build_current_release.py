"""Build one complete current Starling task release under a single lock."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.build_runtime import (
    assert_unpublished_build_root,
    starling_build_session,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.assay_transfer_record_pruning import (
    ARTIFACT_DIR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RecordPruningBudgetExhausted,
    generate_assay_transfer_record_pruning,
)
from data.processing.evidence_library.versions.v10.assay_transfer_record_pruning import (
    MANIFEST_FILENAME as PRUNING_MANIFEST_FILENAME,
)
from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    load_task_policy,
    run_with_args,
)
from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    parse_args as parse_record_args,
)
from data.processing.evidence_library.versions.v10.task_registry import (
    import_task_module,
)

TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")


def _validate_existing_mappings(task_id: str) -> dict[str, Any]:
    module = import_task_module(task_id, "mapping_registry")
    module.validate_mapping_hashes()
    registry = module.mapping_registry()
    return {
        "mode": "existing",
        "registry_version": registry["version"],
        "registry_path": str(module.REGISTRY_PATH),
        "registry_sha256": file_sha256(module.REGISTRY_PATH),
        "mappings": {
            mapping_id: {
                "path": entry.get("path"),
                "sha256": entry.get("sha256"),
            }
            for mapping_id, entry in registry["mappings"].items()
            if entry.get("path")
        },
    }


def _require_selected_mapping_args(task_id: str, args: argparse.Namespace) -> None:
    module = import_task_module(task_id, "mapping_registry")
    for mapping_id, argument in {
        "measurement_resolution": "measurement_resolution_mapping",
        "exact_measurement_units": "exact_unit_mapping",
        "auxiliary_context": "auxiliary_mapping",
    }.items():
        if mapping_id not in module.mapping_registry()["mappings"]:
            continue
        selected = module.mapping_path(mapping_id)
        supplied = Path(str(getattr(args, argument, "") or ""))
        if supplied.resolve() != selected.resolve():
            raise ValueError(
                f"--use_existing requires the registry-selected {mapping_id}: "
                f"{selected}"
            )


def build_current_release(
    *,
    task_id: str,
    normalized_root: str | Path | None = None,
    workers: int = 1,
    validation_level: str = "full",
    cache_mode: str = "auto",
    review_api_key_env: str = "OPENAI_API_KEY_ONE",
    review_base_url: str = DEFAULT_BASE_URL,
    review_model: str = DEFAULT_MODEL,
    review_workers: int | None = None,
    review_budget_epoch: str = "",
    start_new_review_budget_epoch: bool = False,
    review_budget_max_tokens: int = 10_000_000,
    reuse_valid_cache_across_models: bool = False,
    frozen_prior_pruning_manifest: str | Path | None = None,
    published_pruning_input_root: str | Path | None = None,
    from_stage: str = "source",
    use_existing: bool = True,
    record_argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Run Stage 2, record pruning, and final Stage 3."""
    if task_id not in TASKS:
        raise ValueError(f"unsupported current Starling task: {task_id}")
    if validation_level != "full":
        raise ValueError("completed releases require validation_level='full'")
    if from_stage not in {"source", "normalize"}:
        raise ValueError("current release must start from source or a prepared clean stage")
    if not use_existing:
        raise ValueError(
            "release builds consume reviewed maps; regenerate maps first with "
            "canonical_reconciliation.py"
        )
    canonicalization = _validate_existing_mappings(task_id)
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
            from_stage,
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
    _require_selected_mapping_args(task_id, stage2_args)

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
            frozen_prior_manifest=frozen_prior_pruning_manifest,
            published_stage3_input_root=published_pruning_input_root,
        )
        downstream.build_canonical_artifacts(
            normalized_root=root,
            workers=workers,
            validation_level=validation_level,
            cache_mode="off",
            apply_record_pruning=True,
        )
        result = _validate_release(task_id, root)
        result["canonicalization"] = canonicalization
        return result


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
    validations = canonical_manifest.get("validations") or {}
    if validations.get("measurement_unit_pair_validation") != "passed":
        raise ValueError("Stage 2 measurement/unit pair validation did not pass")
    if validations.get("final_assay_transfer_measurement_validation") not in {
        "passed",
        "not_applicable",
    }:
        raise ValueError("Stage 2 final assay-transfer measurement validation did not pass")
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
    expected_stage3_inputs = {
        "canonical_records": canonical_sha,
        "auxiliary_mapping_manifest": file_sha256(
            root / "02_canonicalized/auxiliary_mapping_manifest.json"
        ),
        "assay_transfer_record_pruning": pruning_sha,
    }
    if stage3_manifest.get("contract", {}).get("authoritative_stage1_deduplication"):
        expected_stage3_inputs["stage1_source_value_cleaning_manifest"] = file_sha256(
            root / "01_cleaned/source_value_cleaning_manifest.json"
        )
    if stage3_manifest.get("input_hashes") != expected_stage3_inputs:
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
        "--validation-level", choices=("full",), default="full"
    )
    parser.add_argument("--cache-mode", choices=("auto", "off"), default="auto")
    parser.add_argument("--review-api-key-env", default="OPENAI_API_KEY_ONE")
    parser.add_argument(
        "--review-base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--review-model", default=DEFAULT_MODEL)
    parser.add_argument("--review-workers", type=int)
    parser.add_argument("--review-budget-epoch", default="")
    parser.add_argument("--start-new-review-budget-epoch", action="store_true")
    parser.add_argument("--review-budget-max-tokens", type=int, default=10_000_000)
    parser.add_argument("--reuse-valid-cache-across-models", action="store_true")
    parser.add_argument("--frozen-prior-pruning-manifest", type=Path)
    parser.add_argument("--published-pruning-input-root", type=Path)
    parser.add_argument("--from-stage", choices=("source", "normalize"), default="source")
    parser.add_argument(
        "--use_existing",
        "--use-existing",
        action="store_true",
        default=True,
        help="Validate and apply the task registry's existing reviewed maps (default).",
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
            review_budget_epoch=args.review_budget_epoch,
            start_new_review_budget_epoch=args.start_new_review_budget_epoch,
            review_budget_max_tokens=args.review_budget_max_tokens,
            reuse_valid_cache_across_models=args.reuse_valid_cache_across_models,
            frozen_prior_pruning_manifest=args.frozen_prior_pruning_manifest,
            published_pruning_input_root=args.published_pruning_input_root,
            from_stage=args.from_stage,
            use_existing=args.use_existing,
            record_argv=record_argv,
        )
    except RecordPruningBudgetExhausted as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
