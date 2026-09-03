"""CLI for the isolated train-only KNN-agent router experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic
from .agent import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    run_agent_oof,
)
from .contract import (
    DEFAULT_N_FOLDS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    FOLD_SCHEMA_VERSION,
    TASK_SPECS,
    selected_task_specs,
    task_root,
)
from .features import FEATURE_PROFILES, build_task_features
from .folds import prepare_task_folds
from .indices import build_fold_index, load_frozen_index
from .knn import run_task_oof_knn
from .progress import summarize_agent_progress
from .post_features import POST_FEATURE_PROFILES, build_post_selector_features
from .post_train import POST_MODEL_FAMILIES, train_post_selectors
from .post_valid import evaluate_post_selector_valid
from .post_v31_train import V31_FEATURE_PROFILES, train_post_selector_v31
from .post_v31_learning_curve import (
    DEFAULT_SEEDS as POST_V31_LEARNING_CURVE_SEEDS,
    run_post_v31_learning_curve,
)
from .post_v31_transfer import run_post_v31_transfer_diagnosis
from .post_v31_valid import evaluate_post_selector_v31_valid
from .train import MODEL_FAMILIES, train_task_routers
from .valid import (
    DEFAULT_VALID_AGENT_ROOT,
    DEFAULT_VALID_DATA_ROOT,
    DEFAULT_VALID_KNN_ROOT,
    evaluate_task_valid,
)


DEFAULT_DATA_ROOT = Path("data/gold_labels/legacy/processed_starling")
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = selected_task_specs(args.tasks)
    output_root = Path(args.output_root)

    if args.command == "prepare":
        manifests = [
            prepare_task_folds(
                spec,
                data_root=args.data_root,
                output_root=output_root,
                split=args.split,
                n_folds=args.n_folds,
                seed=args.seed,
            )
            for spec in specs
        ]
        root_manifest = {
            "schema_version": "router_oof_experiment.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fold_schema_version": FOLD_SCHEMA_VERSION,
            "split": args.split,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "task_training_policy": "task_local",
            "tasks": [manifest["task"] for manifest in manifests],
            "model_candidate": DEFAULT_MODEL,
            "visibility_mode": "identity_blind",
            "neighbor_identity_policy": "parent_disjoint",
            "task_manifests": [
                str(task_root(output_root, manifest["task"]) / "fold_manifest.json")
                for manifest in manifests
            ],
        }
        write_json_atomic(output_root / "manifest.json", root_manifest)
        _print_json(root_manifest)
        return 0

    if args.command == "evaluate-valid":
        results = [
            evaluate_task_valid(
                spec,
                output_root=output_root,
                data_root=args.data_root,
                knn_root=args.knn_root,
                agent_root=args.agent_root,
            )
            for spec in specs
        ]
        result = {
            "schema_version": "task_local_knn_agent_router.valid_matrix.v2",
            "tasks": [spec.task for spec in specs],
            "valid": results,
        }
        write_json_atomic(output_root / "valid_evaluation_result_v2.json", result)
        _print_json(result)
        return 0

    if args.command == "build-post-features":
        results = [
            build_post_selector_features(spec, output_root=output_root)
            for spec in specs
        ]
        _print_json({"post_selector_features": results})
        return 0

    if args.command == "train-post-selector":
        results = [
            train_post_selectors(
                spec,
                output_root=output_root,
                model_families=args.model_families,
                feature_profiles=args.feature_profiles,
            )
            for spec in specs
        ]
        _print_json({"post_selectors": results})
        return 0

    if args.command == "evaluate-post-valid":
        results = [
            evaluate_post_selector_valid(spec, output_root=output_root)
            for spec in specs
        ]
        result = {
            "schema_version": "task_local_knn_agent_post_selector.valid_matrix.v3",
            "tasks": [spec.task for spec in specs],
            "valid": results,
        }
        write_json_atomic(output_root / "post_selector_valid_result_v3.json", result)
        _print_json(result)
        return 0

    if args.command == "train-post-selector-v31":
        results = [
            train_post_selector_v31(
                spec,
                output_root=output_root,
                model_families=args.model_families,
                feature_profiles=args.feature_profiles,
            )
            for spec in specs
        ]
        _print_json({"post_selectors_v31": results})
        return 0

    if args.command == "evaluate-post-valid-v31":
        results = [
            evaluate_post_selector_v31_valid(spec, output_root=output_root)
            for spec in specs
        ]
        result = {
            "schema_version": "task_local_knn_agent_post_selector.valid_matrix.v3.1",
            "tasks": [spec.task for spec in specs],
            "valid": results,
        }
        write_json_atomic(output_root / "post_selector_valid_result_v31.json", result)
        _print_json(result)
        return 0

    if args.command == "diagnose-post-v31-learning-curve":
        results = [
            run_post_v31_learning_curve(
                spec,
                output_root=output_root,
                budgets=args.budgets,
                seeds=args.seeds,
                workers=args.workers,
                bootstrap_repetitions=args.bootstrap_repetitions,
            )
            for spec in specs
        ]
        result = {
            "schema_version": "task_local_knn_agent_post_selector.v3.1.learning_curve_matrix",
            "tasks": [spec.task for spec in specs],
            "results": results,
        }
        write_json_atomic(output_root / "post_selector_v31_learning_curve_result.json", result)
        _print_json(result)
        return 0

    if args.command == "diagnose-post-v31-transfer":
        if set(args.tasks or TASK_SPECS) != set(TASK_SPECS):
            raise ValueError(
                "diagnose-post-v31-transfer requires all three tasks because the base representation is shared"
            )
        result = run_post_v31_transfer_diagnosis(
            output_root=output_root,
            workers=args.workers,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
        _print_json(result)
        return 0

    folds = _resolve_folds(specs, output_root, args.folds)
    if args.command == "build-indices":
        results = []
        for spec in specs:
            source_payload = load_frozen_index(spec.source_index)
            for fold in folds[spec.task]:
                results.append(
                    {
                        "task": spec.task,
                        "fold": fold,
                        "meta": build_fold_index(
                            spec,
                            output_root=output_root,
                            fold=fold,
                            workers=args.workers,
                            progress_every=args.progress_every,
                            source_payload=source_payload,
                        ),
                    }
                )
            del source_payload
        _print_json({"built": results})
        return 0

    if args.command == "run-knn":
        _require_all_folds(specs, output_root, folds)
        results = [run_task_oof_knn(spec, output_root=output_root, k=args.k) for spec in specs]
        _print_json({"knn": results})
        return 0

    if args.command == "run-agent":
        common_folds = sorted(set.intersection(*(set(folds[spec.task]) for spec in specs)))
        if any(set(folds[spec.task]) != set(common_folds) for spec in specs):
            raise SystemExit("run-agent currently requires the same selected fold IDs for every task")
        result = run_agent_oof(
            specs,
            output_root=output_root,
            folds=common_folds,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            reasoning_effort=args.reasoning_effort,
            parallelism=args.parallelism,
            timeout_s=args.timeout_s,
            max_stage_requeues=args.max_stage_requeues,
            limit=args.limit,
            manifest_only=args.manifest_only,
        )
        _print_json(result)
        return 1 if result["failed"] else 0

    if args.command == "build-features":
        results = [
            build_task_features(spec, output_root=output_root, folds=folds[spec.task])
            for spec in specs
        ]
        _print_json({"features": results})
        return 0

    if args.command == "train":
        results = [
            train_task_routers(
                spec,
                output_root=output_root,
                model_families=args.model_families,
                feature_profiles=args.feature_profiles,
            )
            for spec in specs
        ]
        _print_json({"routers": results})
        return 0

    if args.command == "finalize":
        progress = summarize_agent_progress(specs, output_root=output_root, folds=folds)
        if progress["paired_complete"] != progress["expected_samples"]:
            raise SystemExit(
                "Agent OOF is incomplete: "
                f"{progress['paired_complete']}/{progress['expected_samples']} paired samples"
            )
        feature_results = [
            build_task_features(spec, output_root=output_root, folds=folds[spec.task])
            for spec in specs
        ]
        router_results = [
            train_task_routers(
                spec,
                output_root=output_root,
                model_families=args.model_families,
                feature_profiles=args.feature_profiles,
            )
            for spec in specs
        ]
        result = {
            "schema_version": "router_oof_finalization.v1",
            "progress": progress,
            "features": feature_results,
            "routers": router_results,
        }
        write_json_atomic(output_root / "finalization_result.json", result)
        _print_json(result)
        return 0

    if args.command == "audit":
        audit = audit_artifacts(specs, output_root=output_root, folds=folds)
        write_json_atomic(output_root / "audit.json", audit)
        _print_json(audit)
        return 0 if audit["passed"] else 1

    if args.command == "progress":
        _print_json(
            summarize_agent_progress(
                specs,
                output_root=output_root,
                folds=folds,
                recent_window_minutes=args.recent_window_minutes,
            )
        )
        return 0

    raise AssertionError(args.command)


def audit_artifacts(
    specs,
    *,
    output_root: str | Path,
    folds: dict[str, list[int]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for spec in specs:
        task_manifest_path = task_root(output_root, spec.task) / "fold_manifest.json"
        checks.append(_exists_check(spec.task, None, "task_manifest", task_manifest_path))
        for fold in folds[spec.task]:
            current_root = Path(output_root) / spec.task / f"fold_{fold:02d}"
            fold_manifest_path = current_root / "fold_manifest.json"
            fold_check = _exists_check(spec.task, fold, "fold_manifest", fold_manifest_path)
            if fold_check["passed"]:
                manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
                fold_check["passed"] = bool(
                    manifest.get("zero_group_overlap")
                    and manifest.get("zero_benchmark_parent_overlap")
                )
            checks.append(fold_check)
            meta_path = (
                current_root
                / "evidence"
                / spec.direct_index_name
                / spec.meta_filename
            )
            meta_check = _exists_check(spec.task, fold, "index_meta", meta_path)
            if meta_check["passed"]:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                filter_stats = ((meta.get("source_stats") or {}).get("heldout_filter") or {})
                meta_check["passed"] = bool(filter_stats.get("zero_parent_overlap"))
                meta_check["n_residual_heldout_parent_identities"] = filter_stats.get(
                    "n_residual_heldout_parent_identities"
                )
            checks.append(meta_check)
            checks.append(
                _exists_check(
                    spec.task,
                    fold,
                    "knn_predictions",
                    current_root / "knn_predictions.jsonl",
                    required=False,
                )
            )
            checks.append(
                _exists_check(
                    spec.task,
                    fold,
                    "agent_predictions",
                    (
                        current_root
                        / "agent"
                        / "runs_identity_blind_parent_disjoint"
                        / spec.task
                        / spec.direct_condition
                        / "predictions.jsonl"
                    ),
                    required=False,
                )
            )
    required_checks = [check for check in checks if check["required"]]
    return {
        "schema_version": "router_oof_audit.v1",
        "passed": all(check["passed"] for check in required_checks),
        "n_required_checks": len(required_checks),
        "n_required_failures": sum(not check["passed"] for check in required_checks),
        "checks": checks,
    }


def _exists_check(
    task: str,
    fold: int | None,
    name: str,
    path: Path,
    *,
    required: bool = True,
) -> dict[str, Any]:
    exists = path.exists()
    return {
        "task": task,
        "fold": fold,
        "check": name,
        "path": str(path),
        "required": required,
        "passed": exists if required else True,
        "present": exists,
    }


def _resolve_folds(specs, output_root: Path, requested: list[int] | None) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for spec in specs:
        manifest_path = task_root(output_root, spec.task) / "fold_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run prepare first: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        available = list(range(int(manifest["n_folds"])))
        selected = available if requested is None else sorted(set(requested))
        invalid = sorted(set(selected) - set(available))
        if invalid:
            raise ValueError(f"Invalid folds for {spec.task}: {invalid}")
        output[spec.task] = selected
    return output


def _require_all_folds(specs, output_root: Path, folds: dict[str, list[int]]) -> None:
    for spec in specs:
        manifest = json.loads(
            (task_root(output_root, spec.task) / "fold_manifest.json").read_text(encoding="utf-8")
        )
        expected = set(range(int(manifest["n_folds"])))
        if set(folds[spec.task]) != expected:
            raise ValueError(f"run-knn requires all folds for {spec.task}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_SPECS), default=list(TASK_SPECS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_common(prepare)
    prepare.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    prepare.add_argument("--split", default=DEFAULT_SPLIT, choices=("random", "scaffold"))
    prepare.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)

    indices = subparsers.add_parser("build-indices")
    _add_common(indices)
    indices.add_argument("--folds", nargs="+", type=int)
    indices.add_argument("--workers", type=int, default=1)
    indices.add_argument("--progress-every", type=int, default=10000)

    knn = subparsers.add_parser("run-knn")
    _add_common(knn)
    knn.add_argument("--folds", nargs="+", type=int)
    knn.add_argument("-k", type=int, default=3)

    agent = subparsers.add_parser("run-agent")
    _add_common(agent)
    agent.add_argument("--folds", nargs="+", type=int)
    agent.add_argument("--base-url", default=DEFAULT_BASE_URL)
    agent.add_argument("--model", default=DEFAULT_MODEL)
    agent.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    agent.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    agent.add_argument("--parallelism", type=int, default=128)
    agent.add_argument("--timeout-s", type=int, default=300)
    agent.add_argument("--max-stage-requeues", type=int, default=1)
    agent.add_argument("--limit", type=int, default=0)
    agent.add_argument("--manifest-only", action="store_true")

    features = subparsers.add_parser("build-features")
    _add_common(features)
    features.add_argument("--folds", nargs="+", type=int)

    train = subparsers.add_parser("train")
    _add_common(train)
    train.add_argument("--folds", nargs="+", type=int)
    train.add_argument("--model-families", nargs="+", choices=MODEL_FAMILIES, default=list(MODEL_FAMILIES))
    train.add_argument(
        "--feature-profiles",
        nargs="+",
        choices=sorted(FEATURE_PROFILES),
        default=list(FEATURE_PROFILES),
    )

    valid = subparsers.add_parser("evaluate-valid")
    _add_common(valid)
    valid.add_argument("--data-root", default=str(DEFAULT_VALID_DATA_ROOT))
    valid.add_argument("--knn-root", default=str(DEFAULT_VALID_KNN_ROOT))
    valid.add_argument("--agent-root", default=str(DEFAULT_VALID_AGENT_ROOT))

    post_features = subparsers.add_parser("build-post-features")
    _add_common(post_features)

    post_train = subparsers.add_parser("train-post-selector")
    _add_common(post_train)
    post_train.add_argument(
        "--model-families",
        nargs="+",
        choices=POST_MODEL_FAMILIES,
        default=list(POST_MODEL_FAMILIES),
    )
    post_train.add_argument(
        "--feature-profiles",
        nargs="+",
        choices=sorted(POST_FEATURE_PROFILES),
        default=list(POST_FEATURE_PROFILES),
    )

    post_valid = subparsers.add_parser("evaluate-post-valid")
    _add_common(post_valid)

    post_v31_train = subparsers.add_parser("train-post-selector-v31")
    _add_common(post_v31_train)
    post_v31_train.add_argument(
        "--model-families",
        nargs="+",
        choices=POST_MODEL_FAMILIES,
        default=list(POST_MODEL_FAMILIES),
    )
    post_v31_train.add_argument(
        "--feature-profiles",
        nargs="+",
        choices=sorted(V31_FEATURE_PROFILES),
        default=list(V31_FEATURE_PROFILES),
    )

    post_v31_valid = subparsers.add_parser("evaluate-post-valid-v31")
    _add_common(post_v31_valid)

    post_v31_curve = subparsers.add_parser(
        "diagnose-post-v31-learning-curve",
        help="Run the train-only matched-size v3.1 termination diagnosis",
    )
    _add_common(post_v31_curve)
    post_v31_curve.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        help="Optional full-equivalent disagreement budgets; defaults are task-specific",
    )
    post_v31_curve.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(POST_V31_LEARNING_CURVE_SEEDS),
    )
    post_v31_curve.add_argument("--workers", type=int, default=1)
    post_v31_curve.add_argument("--bootstrap-repetitions", type=int, default=2000)

    post_v31_transfer = subparsers.add_parser(
        "diagnose-post-v31-transfer",
        help="Run shared-representation/task-specific-calibration train-only diagnosis",
    )
    _add_common(post_v31_transfer)
    post_v31_transfer.add_argument("--workers", type=int, default=1)
    post_v31_transfer.add_argument("--bootstrap-repetitions", type=int, default=2000)

    finalize = subparsers.add_parser("finalize")
    _add_common(finalize)
    finalize.add_argument("--folds", nargs="+", type=int)
    finalize.add_argument(
        "--model-families",
        nargs="+",
        choices=MODEL_FAMILIES,
        default=list(MODEL_FAMILIES),
    )
    finalize.add_argument(
        "--feature-profiles",
        nargs="+",
        choices=sorted(FEATURE_PROFILES),
        default=list(FEATURE_PROFILES),
    )

    audit = subparsers.add_parser("audit")
    _add_common(audit)
    audit.add_argument("--folds", nargs="+", type=int)

    progress = subparsers.add_parser("progress")
    _add_common(progress)
    progress.add_argument("--folds", nargs="+", type=int)
    progress.add_argument("--recent-window-minutes", type=float, default=10.0)
    return parser.parse_args(argv)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
