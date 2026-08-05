"""Build leakage-safe paper indices directly from canonical Starling v7 rows."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.v7_benchmark_view import (
    DIRECT_NUMERIC_VIEW,
    FULL_VIEW,
    build_v7_benchmark_view,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/paper")
BENCHMARK_SPLITS = ("random", "scaffold")
BENCHMARK_LINEAGE = "record_agreement70_split811_v1"

INDEX_SPECS: tuple[dict[str, str], ...] = (
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
        "name": "bioavailability_starling_v7_direct_numeric",
        "task": "Bioavailability_Ma",
        "task_id": "bioavailability_ma",
        "normalized_root": (
            "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
            "starling_normalized_v7"
        ),
        "view": DIRECT_NUMERIC_VIEW,
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
)


def paper_root_for_benchmark_split(split: str, *, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    if split not in BENCHMARK_SPLITS:
        raise ValueError(f"Unknown Starling benchmark split: {split}")
    return Path(output_root) / f"molecular_evidence_agent_starling_{split}_{BENCHMARK_LINEAGE}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = args.splits or list(BENCHMARK_SPLITS)
    specs = _select_specs(args.indices)
    summary_path = Path(args.output_root) / "starling_benchmark_index_summary.json"
    if args.summarize_existing:
        results = _collect_existing_index_meta(
            splits=splits,
            specs=specs,
            output_root=args.output_root,
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
        paper_root = paper_root_for_benchmark_split(split, output_root=args.output_root)
        split_results = dict(results.get(split, {}))
        for spec in specs:
            heldout_path = (
                Path("data/processed_starling")
                / spec["task"]
                / split
                / "heldout_molecule_labels.jsonl"
            )
            out_dir = paper_root / "evidence" / spec["name"]
            print(f"[starling_benchmark_index] split={split} index={spec['name']}", flush=True)
            policy_module = importlib.import_module(
                f"tools.chembl_tool.tasks.{spec['task_id']}.starling_policy"
            )
            meta = build_v7_benchmark_view(
                policy=policy_module.POLICY,
                normalized_root=spec["normalized_root"],
                heldout_labels_jsonl=heldout_path,
                out_dir=out_dir,
                benchmark_split=split,
                view=spec["view"],
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
    specs: list[dict[str, str]],
    output_root: str | Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    missing: list[str] = []
    for split in splits:
        paper_root = paper_root_for_benchmark_split(split, output_root=output_root)
        split_results: dict[str, Any] = {}
        for spec in specs:
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


def _select_specs(names: list[str]) -> list[dict[str, str]]:
    if not names:
        return [dict(spec) for spec in INDEX_SPECS]
    by_name = {spec["name"]: spec for spec in INDEX_SPECS}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown indices: {', '.join(missing)}")
    return [dict(by_name[name]) for name in names]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", choices=BENCHMARK_SPLITS, default=[])
    parser.add_argument("--indices", nargs="*", default=[])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
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
