"""Build leakage-safe paper indices directly from canonical Starling v7 rows."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling import build_heldout_starling_index
from tools.chembl_tool.common.starling.v7_benchmark_view import (
    ALL_PARENT_FILTER,
    ALL_SCAFFOLD_FILTER,
    DIRECT_SOURCE_ONLY_FILTER,
    FULL_VIEW,
    HELDOUT_FILTER_MODES,
    build_v7_benchmark_view,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/paper")
BENCHMARK_SPLITS = ("random", "scaffold")
BENCHMARK_LINEAGE = "record_agreement70_split811_v1"
HELDOUT_SUBSETS = ("valid", "test")

INDEX_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "bbb_starling_v7",
        "task": "BBB_Martins",
        "task_id": "bbb_martins",
        "normalized_root": (
            "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
            "starling_normalized_v7"
        ),
        "view": FULL_VIEW,
    },
    {
        "name": "bioavailability_starling_v7",
        "task": "Bioavailability_Ma",
        "task_id": "bioavailability_ma",
        "normalized_root": (
            "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
            "starling_normalized_v7"
        ),
        "view": FULL_VIEW,
    },
    {
        "name": "skin_reaction_starling_v7",
        "task": "Skin_Reaction",
        "task_id": "skin_reaction",
        "normalized_root": (
            "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
            "starling_normalized_v7"
        ),
        "view": FULL_VIEW,
    },
    {
        "name": "clintox_starling_full",
        "task": "ClinTox",
        "source_evidence": (
            "outputs/chembl_tool/tasks/clintox/evidence_library/"
            "starling_clinical_trial_failure_v1/starling_clintox_evidence.jsonl"
        ),
        "evidence_filename": "starling_clintox_evidence.jsonl",
        "index_filename": "starling_clintox_neighbor_index.pkl",
        "meta_filename": "starling_clintox_neighbor_index.meta.json",
    },
)


def paper_root_for_benchmark_split(
    split: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    lineage: str = BENCHMARK_LINEAGE,
) -> Path:
    if split not in BENCHMARK_SPLITS:
        raise ValueError(f"Unknown Starling benchmark split: {split}")
    return Path(output_root) / f"molecular_evidence_agent_starling_{split}_{lineage}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    heldout_subsets = normalize_heldout_subsets(args.heldout_subsets)
    splits = args.splits or list(BENCHMARK_SPLITS)
    specs = _apply_source_evidence_overrides(
        _select_specs(args.indices),
        args.source_evidence,
    )
    summary_name = (
        "starling_benchmark_index_summary.json"
        if args.benchmark_lineage == BENCHMARK_LINEAGE
        else f"starling_benchmark_index_summary_{args.benchmark_lineage}.json"
    )
    summary_path = Path(args.output_root) / summary_name
    if args.summarize_existing:
        results = _collect_existing_index_meta(
            splits=splits,
            specs=specs,
            output_root=args.output_root,
            lineage=args.benchmark_lineage,
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"summary": str(summary_path)}, indent=2), flush=True)
        return 0
    results = _load_existing_summary(summary_path)
    for split in splits:
        paper_root = paper_root_for_benchmark_split(
            split,
            output_root=args.output_root,
            lineage=args.benchmark_lineage,
        )
        split_results = dict(results.get(split, {}))
        for spec in specs:
            if split not in spec.get("benchmark_splits", BENCHMARK_SPLITS):
                continue
            heldout_path = heldout_labels_path(
                args.benchmark_data_root,
                spec["task"],
                split,
                heldout_subsets,
            )
            out_dir = paper_root / "evidence" / spec["name"]
            print(f"[starling_benchmark_index] split={split} index={spec['name']}", flush=True)
            if "normalized_root" in spec:
                policy_module = importlib.import_module(
                    f"tools.chembl_tool.tasks.{spec['task_id']}.starling_policy"
                )
                downstream_spec = None
                if args.heldout_filter_mode == DIRECT_SOURCE_ONLY_FILTER:
                    downstream_module = importlib.import_module(
                        f"tools.chembl_tool.tasks.{spec['task_id']}."
                        "build_starling_downstream_artifacts"
                    )
                    downstream_spec = downstream_module.get_spec()
                meta = build_v7_benchmark_view(
                    policy=policy_module.POLICY,
                    normalized_root=spec["normalized_root"],
                    heldout_labels_jsonl=heldout_path,
                    out_dir=out_dir,
                    benchmark_split=split,
                    view=spec["view"],
                    heldout_filter_mode=args.heldout_filter_mode,
                    downstream_spec=downstream_spec,
                    workers=args.workers,
                    progress_every=args.progress_every,
                )
            else:
                meta = build_heldout_starling_index(
                    source_evidence_jsonl=spec["source_evidence"],
                    heldout_labels_jsonl=heldout_path,
                    out_dir=out_dir,
                    index_version=(
                        f"{spec['name']}.heldout_{'_'.join(heldout_subsets)}_{split}."
                        f"{args.benchmark_lineage}.v1"
                    ),
                    evidence_filename=spec["evidence_filename"],
                    index_filename=spec["index_filename"],
                    meta_filename=spec["meta_filename"],
                    workers=args.workers,
                    progress_every=args.progress_every,
                )
            split_results[spec["name"]] = meta
        results[split] = split_results

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2), flush=True)
    return 0


def _load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in existing index summary: {path}")
    return payload


def _collect_existing_index_meta(
    *,
    splits: list[str],
    specs: list[dict[str, Any]],
    output_root: str | Path,
    lineage: str = BENCHMARK_LINEAGE,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    missing: list[str] = []
    for split in splits:
        paper_root = paper_root_for_benchmark_split(
            split,
            output_root=output_root,
            lineage=lineage,
        )
        split_results: dict[str, Any] = {}
        for spec in specs:
            if split not in spec.get("benchmark_splits", BENCHMARK_SPLITS):
                continue
            meta_path = (
                paper_root
                / "evidence"
                / spec["name"]
                / spec.get("meta_filename", "manifest.json")
            )
            if not meta_path.exists():
                missing.append(str(meta_path))
                continue
            split_results[spec["name"]] = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
        results[split] = split_results
    if missing:
        raise SystemExit("Missing held-out index metadata:\n" + "\n".join(missing))
    return results


def _select_specs(names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return [dict(spec) for spec in INDEX_SPECS if spec.get("default", True)]
    by_name = {spec["name"]: spec for spec in INDEX_SPECS}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown indices: {', '.join(missing)}")
    return [dict(by_name[name]) for name in names]


def _apply_source_evidence_overrides(
    specs: list[dict[str, Any]],
    overrides: list[str],
) -> list[dict[str, Any]]:
    """Return copied specs with explicit NAME=JSONL source overrides."""
    parsed: dict[str, str] = {}
    for item in overrides:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise SystemExit(
                "--source-evidence entries must use INDEX_NAME=EVIDENCE_JSONL"
            )
        if name in parsed:
            raise SystemExit(f"Duplicate --source-evidence override: {name}")
        parsed[name] = path
    selected = {spec["name"] for spec in specs}
    unknown = sorted(set(parsed) - selected)
    if unknown:
        raise SystemExit(
            "Source override does not match a selected index: " + ", ".join(unknown)
        )
    updated: list[dict[str, Any]] = []
    for spec in specs:
        replacement = parsed.get(spec["name"])
        if replacement is None:
            updated.append(dict(spec))
            continue
        field = "source_evidence" if "source_evidence" in spec else "normalized_root"
        updated.append({**spec, field: replacement})
    return updated


def heldout_labels_path(
    benchmark_data_root: str | Path,
    task: str,
    split: str,
    heldout_subsets: list[str] | tuple[str, ...],
) -> Path:
    """Resolve the benchmark identity file excluded from one reference pool."""
    subsets = normalize_heldout_subsets(heldout_subsets)
    split_dir = Path(benchmark_data_root) / task / split
    if set(subsets) == set(HELDOUT_SUBSETS):
        return split_dir / "heldout_molecule_labels.jsonl"
    if len(subsets) == 1:
        return split_dir / f"{subsets[0]}_molecule_labels.jsonl"
    raise ValueError(f"Unsupported held-out subset combination: {subsets}")


def normalize_heldout_subsets(
    heldout_subsets: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Validate and canonicalize held-out subset names for stable receipts."""
    requested = tuple(heldout_subsets)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(subset not in HELDOUT_SUBSETS for subset in requested)
    ):
        raise ValueError(f"Unsupported held-out subsets: {requested}")
    normalized = tuple(subset for subset in HELDOUT_SUBSETS if subset in requested)
    if normalized not in (HELDOUT_SUBSETS, ("test",)):
        raise ValueError(
            "Held-out scope must represent a train pool (valid test) or a "
            "train+valid pool (test)"
        )
    return normalized


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", choices=BENCHMARK_SPLITS, default=[])
    parser.add_argument("--indices", nargs="*", default=[])
    parser.add_argument(
        "--source-evidence",
        action="append",
        default=[],
        metavar="INDEX_NAME=SOURCE_PATH",
        help=(
            "Override one selected index source without changing the frozen default spec; "
            "v7 specs expect a normalized-artifact root."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--benchmark-data-root",
        default="data/processed_starling",
        help="Root containing <Task>/<split>/heldout_molecule_labels.jsonl.",
    )
    parser.add_argument("--benchmark-lineage", default=BENCHMARK_LINEAGE)
    parser.add_argument(
        "--heldout-filter-mode",
        choices=HELDOUT_FILTER_MODES,
        default=ALL_PARENT_FILTER,
        help=(
            "all_parents preserves the historical paper-index contract; "
            "direct_source_only removes held-out parents only from the task's "
            "declared direct label source and retains mechanism evidence; "
            f"{ALL_SCAFFOLD_FILTER} removes every held-out scaffold from all sources."
        ),
    )
    parser.add_argument(
        "--heldout-subsets",
        nargs="+",
        choices=HELDOUT_SUBSETS,
        default=list(HELDOUT_SUBSETS),
        help=(
            "Evaluation subsets excluded from the retrieval reference pool. "
            "Use 'test' for a post-selection train+valid test reference pool."
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild only the root summary from existing held-out index metadata.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
