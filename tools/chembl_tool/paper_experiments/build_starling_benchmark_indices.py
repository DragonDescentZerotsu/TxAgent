"""Build random/scaffold Starling indices with all valid/test parents excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling import build_heldout_starling_index


DEFAULT_OUTPUT_ROOT = Path("outputs/paper")
BENCHMARK_SPLITS = ("random", "scaffold")
BENCHMARK_LINEAGE = "record_agreement70_split811_v1"

INDEX_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "bbb_starling_direct",
        "task": "BBB_Martins",
        "source_evidence": (
            "outputs/paper/molecular_evidence_agent/evidence/bbb_starling/all/"
            "starling_bbb_evidence.jsonl"
        ),
        "evidence_filename": "starling_bbb_evidence.jsonl",
        "index_filename": "starling_bbb_neighbor_index.pkl",
        "meta_filename": "starling_bbb_neighbor_index.meta.json",
    },
    {
        "name": "bbb_starling_full",
        "task": "BBB_Martins",
        "source_evidence": (
            "outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full/"
            "starling_bbb_evidence.jsonl"
        ),
        "evidence_filename": "starling_bbb_evidence.jsonl",
        "index_filename": "starling_bbb_neighbor_index.pkl",
        "meta_filename": "starling_bbb_neighbor_index.meta.json",
    },
    {
        "name": "bioavailability_starling_direct_numeric",
        "task": "Bioavailability_Ma",
        "source_evidence": (
            "outputs/paper/molecular_evidence_agent/evidence/"
            "bioavailability_starling_direct_numeric_v2/starling_factor_evidence.jsonl"
        ),
        "evidence_filename": "starling_factor_evidence.jsonl",
        "index_filename": "starling_factor_neighbor_index.pkl",
        "meta_filename": "starling_factor_neighbor_index.meta.json",
    },
    {
        "name": "bioavailability_starling_full",
        "task": "Bioavailability_Ma",
        "source_evidence": (
            "outputs/paper/molecular_evidence_agent/evidence/"
            "bioavailability_starling_full_v2/starling_factor_evidence.jsonl"
        ),
        "evidence_filename": "starling_factor_evidence.jsonl",
        "index_filename": "starling_factor_neighbor_index.pkl",
        "meta_filename": "starling_factor_neighbor_index.meta.json",
    },
    {
        "name": "skin_reaction_starling_full",
        "task": "Skin_Reaction",
        "source_evidence": (
            "outputs/paper/molecular_evidence_agent/evidence/"
            "skin_reaction_starling_full/starling_skin_reaction_evidence.jsonl"
        ),
        "evidence_filename": "starling_skin_reaction_evidence.jsonl",
        "index_filename": "starling_skin_reaction_neighbor_index.pkl",
        "meta_filename": "starling_skin_reaction_neighbor_index.meta.json",
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
            heldout_path = (
                Path(args.benchmark_data_root)
                / spec["task"]
                / split
                / "heldout_molecule_labels.jsonl"
            )
            out_dir = paper_root / "evidence" / spec["name"]
            print(f"[starling_benchmark_index] split={split} index={spec['name']}", flush=True)
            meta = build_heldout_starling_index(
                source_evidence_jsonl=spec["source_evidence"],
                heldout_labels_jsonl=heldout_path,
                out_dir=out_dir,
                index_version=(
                    f"{spec['name']}.heldout_valid_test_{split}."
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
    specs: list[dict[str, str]],
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
            meta_path = paper_root / "evidence" / spec["name"] / spec["meta_filename"]
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


def _apply_source_evidence_overrides(
    specs: list[dict[str, str]],
    overrides: list[str],
) -> list[dict[str, str]]:
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
    return [
        {**spec, "source_evidence": parsed.get(spec["name"], spec["source_evidence"])}
        for spec in specs
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", choices=BENCHMARK_SPLITS, default=[])
    parser.add_argument("--indices", nargs="*", default=[])
    parser.add_argument(
        "--source-evidence",
        action="append",
        default=[],
        metavar="INDEX_NAME=EVIDENCE_JSONL",
        help="Override one selected index source without changing the frozen default spec.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--benchmark-data-root",
        default="data/processed_starling",
        help="Root containing <Task>/<split>/heldout_molecule_labels.jsonl.",
    )
    parser.add_argument("--benchmark-lineage", default=BENCHMARK_LINEAGE)
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
