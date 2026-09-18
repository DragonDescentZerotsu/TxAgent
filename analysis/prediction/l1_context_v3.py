"""Consolidate the completed L1-context-v3 Morgan-width studies.

The command reads complete organized run leaves for widths 10, 15, and 25,
joins their L1 metrics and mandatory diagnostics, and writes compact TSVs plus
an input/output hash manifest. It never changes raw inference runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from predict.utils.json import sha256_file


WIDTHS = (10, 15, 25)
PRIORS = ("with_query_prior", "no_query_prior")
DEFAULT_RESULTS = Path("outputs/paper/assay_transfer_harness/joseph")
DEFAULT_OUTPUT = Path(
    "outputs/analysis/prediction/l1_context_v3_morgan10_15_25_k10_m0_m2_v1"
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build(results: Path = DEFAULT_RESULTS, output: Path = DEFAULT_OUTPUT) -> Path:
    results, output = results.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {output}")
    metrics, neighborhoods, coverage = [], [], []
    inputs: set[Path] = set()
    for width in WIDTHS:
        for prior_dir in PRIORS:
            parent = results / f"contrastive_morgan{width}/assay_transfer/{prior_dir}"
            for minimum in (0, 1, 2):
                leaves = sorted(parent.glob(f"k10_m{minimum}_*/run.json"))
                if len(leaves) != 1:
                    raise ValueError(
                        f"Expected one completed width={width}, prior={prior_dir}, M={minimum} run"
                    )
                run_path = leaves[0]
                root = run_path.parent
                run = json.loads(run_path.read_text())
                if run.get("status") != "complete" or run.get("metric_status") != "complete":
                    raise ValueError(f"Run is not complete: {root}")
                paths = {
                    "metrics": root / "macro_f1.tsv",
                    "neighborhood": root / "neighborhood_label_mix.summary.tsv",
                    "coverage": root / "reasoning_reference_coverage.summary.tsv",
                    "diagnostics": root / "diagnostics_manifest.json",
                    "experiment": root / "experiment_manifest.json",
                }
                if not all(path.is_file() for path in paths.values()):
                    raise ValueError(f"Completed run lacks required artifacts: {root}")
                inputs.update({run_path, *paths.values()})
                cache_bundle = Path(run["retrieval_cache_bundle"])
                batch_manifest = Path(run["batch_root"]) / "matrix.json"
                if not cache_bundle.is_file() or not batch_manifest.is_file():
                    raise ValueError(f"Run provenance inputs are missing: {root}")
                inputs.update({cache_bundle, batch_manifest})
                common = {
                    "morgan_candidate_width": width,
                    "k": 10,
                    "m": minimum,
                    "query_prior": run["query_prior"],
                    "prompt_version": run["prompt_version"],
                }
                for row in _read_tsv(paths["metrics"]):
                    metrics.append({**common, "task": row["task"], "macro_f1": row["L1"]})
                for row in _read_tsv(paths["neighborhood"]):
                    neighborhoods.append({**common, **row})
                for row in _read_tsv(paths["coverage"]):
                    if row["level"] == "1":
                        coverage.append({**common, **row})
    output.mkdir(parents=True)
    tables = {
        "macro_f1.tsv": metrics,
        "neighborhood_label_mix.tsv": neighborhoods,
        "reasoning_reference_coverage.tsv": coverage,
    }
    for name, rows in tables.items():
        _write_tsv(output / name, rows)
    manifest = {
        "schema_version": "l1_context_v3_morgan_width_analysis.v1",
        "status": "complete",
        "selection_unit": "parent_condition_context",
        "widths": list(WIDTHS),
        "k": 10,
        "minimum_contrasts": [0, 1, 2],
        "query_prior_modes": ["cached", "none"],
        "inputs": {str(path): sha256_file(path) for path in sorted(inputs)},
        "outputs": {name: sha256_file(output / name) for name in tables},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(build(args.results_root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
