"""Build random/scaffold Starling indices with all test parents excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling import build_heldout_starling_index


DEFAULT_OUTPUT_ROOT = Path("outputs/paper")
BENCHMARK_SPLITS = ("random", "scaffold")

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
            "bioavailability_starling_direct_numeric/starling_factor_evidence.jsonl"
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
            "bioavailability_starling_full/starling_factor_evidence.jsonl"
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


def paper_root_for_benchmark_split(split: str, *, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    if split not in BENCHMARK_SPLITS:
        raise ValueError(f"Unknown Starling benchmark split: {split}")
    return Path(output_root) / f"molecular_evidence_agent_starling_{split}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = args.splits or list(BENCHMARK_SPLITS)
    specs = _select_specs(args.indices)
    results: dict[str, Any] = {}
    for split in splits:
        paper_root = paper_root_for_benchmark_split(split, output_root=args.output_root)
        split_results: dict[str, Any] = {}
        for spec in specs:
            heldout_path = (
                Path("data/processed_starling")
                / spec["task"]
                / split
                / "test_molecule_labels.jsonl"
            )
            out_dir = paper_root / "evidence" / spec["name"]
            print(f"[starling_benchmark_index] split={split} index={spec['name']}", flush=True)
            meta = build_heldout_starling_index(
                source_evidence_jsonl=spec["source_evidence"],
                heldout_labels_jsonl=heldout_path,
                out_dir=out_dir,
                index_version=f"{spec['name']}.heldout_{split}.v1",
                evidence_filename=spec["evidence_filename"],
                index_filename=spec["index_filename"],
                meta_filename=spec["meta_filename"],
                workers=args.workers,
                progress_every=args.progress_every,
            )
            split_results[spec["name"]] = meta
        results[split] = split_results

    summary_path = Path(args.output_root) / "starling_benchmark_index_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2), flush=True)
    return 0


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
