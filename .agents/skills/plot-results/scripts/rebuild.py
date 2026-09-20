#!/usr/bin/env python3
"""Rebuild the frozen Joseph BBB/Oral comparison figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PLOTTER_COMMIT = "8972834103833c18a8d269a92f898bad07f7e5ab"
PLOTTER_PATH = "tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py"
PLOTTER_SHA256 = "60b89e54fcde807cc2e7be58a6fabea7ec0d863753f3e2dea5ce0b661a8a3bdb"
PATCHED_SHA256 = "aa240b3a2ec797997b71778bb9e946b677bc238aad10d8f6346896c530018ecd"
DEFAULT_STUDY = Path(
    "outputs/analysis/assay_retrieval_curve/"
    "joseph_full_flat_optimized_provisional_v2"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("Pinned renderer no longer matches the layout patch")
    return source.replace(old, new)


def patched_plotter(repo: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{PLOTTER_COMMIT}:{PLOTTER_PATH}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    if sha256(result.stdout) != PLOTTER_SHA256:
        raise RuntimeError("Pinned renderer hash mismatch")
    source = result.stdout.decode()
    source = replace_once(
        source,
        "grid_rows, columns, figsize=(6.2 * columns, 3.6 * grid_rows + 1.6),",
        "grid_rows, columns, figsize=(6.2 * columns, 3.6 * grid_rows + (3.6 if grid_rows == 1 else 1.6)),",
    )
    source = replace_once(source, 'x=0.055, y=0.985,', 'x=0.055, y=0.975,')
    source = replace_once(
        source,
        '''    fig.text(0.055, 0.943,
             f"{receipt['split_scheme'].title()} {subset_label} · {similarity_label} · {visibility_label} · "
             f"direct budget {receipt['direct_budget']} + indirect budget {receipt['indirect_budget']} records · "
             "shared Macro-F1 scale", fontsize=10, color="#555555")''',
        '''    subtitle = receipt.get('figure_subtitle') or (
        f"{receipt['split_scheme'].title()} {subset_label} · {similarity_label} · {visibility_label} · "
        f"direct budget {receipt['direct_budget']} + indirect budget {receipt['indirect_budget']} records · "
        "shared Macro-F1 scale"
    )
    fig.text(0.055, 0.91 if grid_rows == 1 else 0.943,
             subtitle, fontsize=10, color="#555555")''',
    )
    source = replace_once(
        source,
        '''    fig.text(0.055, 0.12,
             "Each point is an independent full-flat setting. Curves use the None/Direct references specified in their metadata.\\n"
             f"Right-side ML baselines use the same {subset_label} rows. One run per setting; no uncertainty intervals. "
             f"Recorded tool failures: {receipt['tool_failures']}.",
             fontsize=9, color="#555555", linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.23,''',
        '''    figure_note = receipt.get('figure_note') or (
        "Each point is an independent full-flat setting. Curves use the None/Direct references specified in their metadata.\\n"
        f"Right-side ML baselines use the same {subset_label} rows. One run per setting; no uncertainty intervals. "
        f"Recorded tool failures: {receipt['tool_failures']}."
    )
    fig.text(0.055, 0.17 if grid_rows == 1 else 0.12,
             figure_note,
             fontsize=9, color="#555555", linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.79 if grid_rows == 1 else 0.86, bottom=0.31 if grid_rows == 1 else 0.23,''',
    )
    if sha256(source.encode()) != PATCHED_SHA256:
        raise RuntimeError("Patched renderer hash mismatch")
    return source


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/plot-results-rebuild"))
    parser.add_argument("--output-stem", default="ours_with_published_llm_provisional_valid")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    study = args.study_dir if args.study_dir.is_absolute() else repo / args.study_dir
    output = args.output_dir.resolve()
    points, metadata = study / "points.csv", study / "metadata.json"
    if not points.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Missing points.csv or metadata.json under {study}")

    with tempfile.TemporaryDirectory(prefix="plot-results-") as temporary:
        plotter = Path(temporary) / "plot_assay_retrieval_curve.py"
        plotter.write_text(patched_plotter(repo), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(repo), environment.get("PYTHONPATH")) if value
        )
        subprocess.run(
            [
                sys.executable,
                str(plotter),
                "--baseline-curves-csv",
                str(points),
                "--baseline-curves-metadata",
                str(metadata),
                "--analysis-dir",
                str(output),
                "--output-stem",
                args.output_stem,
            ],
            cwd=repo,
            env=environment,
            check=True,
        )

    generated_csv = output / f"{args.output_stem}.csv"
    if rows(points) != rows(generated_csv):
        raise RuntimeError("Rendered CSV does not match points.csv")
    receipt = json.loads((output / f"{args.output_stem}_receipt.json").read_text())
    if len(rows(points)) != 26 or len(receipt["curve_settings"]) != 4:
        raise RuntimeError("Expected 26 task/method rows and four LLM curves")
    print(json.dumps(receipt["outputs"], indent=2))


if __name__ == "__main__":
    main()
