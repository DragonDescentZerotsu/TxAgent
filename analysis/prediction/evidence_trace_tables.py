"""Add measured final-call trace references to the selected Evidence score tables."""

from __future__ import annotations

from predict.harnesses.branches.manifests import read_record

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analysis.prediction.flat_reasoning_diagnostics import _measure_run


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs/paper/assay_transfer_harness/evidence/final"
STUDY = ROOT / "outputs/analysis/prediction/evidence_final_trace_metrics_20260925_v1"
ABLATIONS = FINAL / "ablations"
ANALYSIS = ROOT / "outputs/analysis/record_selection"
METRICS = (
    ("reasoning_molecules_mean", "mentioned_molecule_count"),
    ("reasoning_records_mean", "mentioned_record_count"),
    ("reasoning_tokens_mean", "reasoning_tokens"),
    ("final_claim_molecules_mean", "final_mentioned_molecule_count"),
    ("final_claim_records_mean", "final_mentioned_record_count"),
)
EXTRA = ("reasoning_text_n", "token_estimated_n")
ABLATION_LABELS = {
    "Control": "control", "Assay off": "assay_off",
    "Informativeness off": "informativeness_off",
    "Molecular diversity off": "molecular_off",
    "Evidence + semantic diversity off": "semantic_level_off",
}


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rows(path: Path, inputs: dict[str, str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    inputs[str(path.relative_to(ROOT))] = _hash(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _source_rows(
    source: Path, task: str, condition: str, budget: str, expected: int,
    inputs: dict[str, str], target_score: float,
) -> list[tuple[int, Path, str]]:
    """Follow a result row to either a FINAL arm or its query ledger."""
    if source.name == "manifest.json":
        manifest = read_record(source)
        inputs[str(source.relative_to(ROOT))] = _hash(source)
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete FINAL arm: {source}")
        return [(int(run.name), run, "") for run in sorted((source.parent / "runs").iterdir())
                if run.is_dir()]
    result_rows = _rows(source, inputs)
    selected_results = [row for row in result_rows
                        if (not row.get("task") or row["task"] == task)
                        and (not condition or "condition" not in row or
                             row.get("condition") in (condition, condition + "_reused"))
                        and (not budget or not ("budget" in row or "arm" in row) or
                             row.get("budget", row.get("arm", "")) in
                             (budget, "budget" + budget, "control_budget" + budget))]
    if len(selected_results) != 1:
        selected_results = [row for row in result_rows
                            if (not row.get("task") or row["task"] == task)
                            and row.get("macro_f1")
                            and abs(float(row["macro_f1"]) - target_score) < 0.000001]
    selected_result = selected_results[0] if len(selected_results) == 1 else {}
    profile = selected_result.get("profile", "")
    ledger = source.parent / "query_provenance.tsv"
    if ledger.is_file():
        selected = [row for row in _rows(ledger, inputs)
                    if (not row.get("task") or row["task"] == task)
                    and (not row.get("condition") or row["condition"] ==
                         ("full" if budget == "50" and not condition else condition))
                    and (not row.get("budget") or row["budget"] == budget)
                    and (not profile or not row.get("profile") or row["profile"] == profile)
                    and (not budget or row.get("budget") or
                         f"/budget{budget}/" in row["run_dir"] or
                         source.parent.name.startswith("gold_v1_objective_terms"))]
        if len(selected) == expected:
            return [(int(row["query_index"]), ROOT / row["run_dir"], row.get("final_sha256", ""))
                    for row in selected]
    if selected_result.get("source_results"):
        return _source_rows(ROOT / selected_result["source_results"], task, condition,
                            budget, expected, inputs, target_score)
    reused = source.parent / "results.json"
    if reused.is_file():
        data = json.loads(reused.read_text())
        inputs[str(reused.relative_to(ROOT))] = _hash(reused)
        candidates = [row for group in ("reused_informativeness_results", "results")
                      for row in data.get(group, []) if isinstance(row, dict)
                      and row.get("task") == task
                      and row.get("condition") == condition + "_reused"
                      and row.get("source_results")]
        if len(candidates) == 1:
            return _source_rows(ROOT / candidates[0]["source_results"], task,
                                condition, budget, expected, inputs, target_score)
    raise ValueError(f"cannot resolve one run source: {source} {task} {condition} {budget}")


def _measure(
    source: Path, task: str, condition: str, budget: str, expected: int,
    target_score: float,
    inputs: dict[str, str], cache: dict[tuple[str, str, str, str], list[dict]],
) -> list[dict]:
    key = (str(source), task, condition, budget)
    if key in cache:
        return cache[key]
    runs = _source_rows(source, task, condition, budget, expected, inputs, target_score)
    indices = [index for index, _, _ in runs]
    if len(indices) != expected or len(set(indices)) != expected:
        raise ValueError(f"run coverage mismatch: {source} {task} {condition} {budget}: {len(indices)}/{expected}")
    measured = []
    for index, run, expected_hash in runs:
        gaps: list[dict] = []
        row = _measure_run(matrix_path=source, task=task, subset="test", metadata={},
                           run_dir=run, expected_index=index, inputs={}, gaps=gaps)
        if gaps or row is None:
            raise ValueError(f"unmeasurable run {run}: {gaps}")
        if expected_hash and row["output_sha256"] != expected_hash:
            raise ValueError(f"source final-output hash mismatch: {run}")
        measured.append({"query_index": index, "run_dir": str(run.relative_to(ROOT)), **row})
    cache[key] = measured
    return measured


def _summary(rows: list[dict]) -> dict[str, str | int]:
    def mean(field: str) -> str:
        values = [float(row[field]) for row in rows if row[field] != ""]
        return f"{sum(values) / len(values):.6f}" if values else ""
    return {
        **{name: mean(field) for name, field in METRICS},
        "reasoning_text_n": sum(int(row["reasoning_text_available"]) for row in rows),
        "token_estimated_n": sum(row["reasoning_token_source"] == "deepseek_tokenizer_estimate"
                                 for row in rows),
    }


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(output: Path = STUDY) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"immutable analysis already exists: {output}")
    inputs: dict[str, str] = {}
    cache: dict[tuple[str, str, str, str], list[dict]] = {}
    summaries: list[dict] = []
    per_query: list[dict] = []
    tables: dict[Path, tuple[list[dict], list[str]]] = {}

    def add(table: str, task: str, cell: str, source: Path, expected: int,
            score: float, condition: str = "", budget: str = "") -> dict:
        rows = _measure(source, task, condition, budget, expected, score, inputs, cache)
        summary = _summary(rows)
        source_name = str(source.relative_to(ROOT))
        summaries.append({"table": table, "task": task, "cell": cell, "source": source_name,
                          "n_queries": expected, **summary})
        per_query.extend({"table": table, "task": task, "cell": cell, "source": source_name,
                          **row} for row in rows)
        return summary

    for model in ("deepseek", "luna"):
        source = FINAL / f"test_macro_f1_12_task_benchmarks_{model}_selected_20260925_v1.tsv"
        original = _rows(source, inputs)
        augmented = []
        fields = list(original[0])
        for prefix in ("direct", "direct_plus_indirect"):
            fields.extend(f"{prefix}_{name}" for name, _ in METRICS)
            fields.extend(f"{prefix}_{name}" for name in EXTRA)
        for row in original:
            task = row["task"]
            for prefix in ("direct", "direct_plus_indirect"):
                arm = FINAL / row[f"{prefix}_final_arm"] / "manifest.json"
                result = add(model, task, prefix, arm, int(row["n_queries"]),
                             float(row[f"{prefix}_macro_f1"]))
                row.update({f"{prefix}_{key}": value for key, value in result.items()})
            augmented.append(row)
        tables[FINAL / f"test_macro_f1_12_task_benchmarks_{model}_selected_trace_metrics_20260925_v1.tsv"] = (augmented, fields)

    ablation_long = {(row["task"], row["condition"]): row for row in
                     _rows(ABLATIONS / "gold_v1_ablation_test_20260925_v1.tsv", inputs)}
    ablation = _rows(ABLATIONS / "gold_v1_ablation_test_20260925_v3.tsv", inputs)
    fields = list(ablation[0])
    for label in ABLATION_LABELS:
        fields.extend(f"{label} {name}" for name, _ in METRICS)
        fields.extend(f"{label} {name}" for name in EXTRA)
    for row in ablation:
        for label, condition in ABLATION_LABELS.items():
            source_row = ablation_long[(row["Task"], condition)]
            if abs(float(row[label]) - float(source_row["arm_macro_f1"])) > 0.000001:
                raise ValueError(f"ablation score mismatch: {row['Task']} {condition}")
            source = ROOT / source_row["arm_source"]
            if condition == "informativeness_off" and row["Task"] == "dili":
                source = ROOT / ablation_long[("dili", "control")]["arm_source"]
            elif condition == "informativeness_off" and row["Task"] == "skin_reaction":
                source = ANALYSIS / "gold_v1_objective_terms_skin_bbb_test_20260924_v1/results.tsv"
            result = add("ablation", row["Task"], condition, source,
                         int(source_row["n_queries"]), float(row[label]),
                         condition=condition)
            row.update({f"{label} {key}": value for key, value in result.items()})
    tables[ABLATIONS / "gold_v1_ablation_test_20260925_v4.tsv"] = (ablation, fields)

    scaling_long = {(row["task"], row["indirect_budget"]): row for row in
                    _rows(ABLATIONS / "gold_v1_scaling_test_20260925_v2.tsv", inputs)}
    scaling = _rows(ABLATIONS / "gold_v1_scaling_test_20260925_v3.tsv", inputs)
    fields = list(scaling[0])
    for budget in ("10", "25", "50", "75", "100"):
        fields.extend(f"{budget} {name}" for name, _ in METRICS)
        fields.extend(f"{budget} {name}" for name in EXTRA)
    for row in scaling:
        for budget in ("10", "25", "50", "75", "100"):
            source_row = scaling_long[(row["Task"], budget)]
            if abs(float(row[budget]) - float(source_row["macro_f1"])) > 0.000001:
                raise ValueError(f"scaling score mismatch: {row['Task']} {budget}")
            source = ROOT / source_row["source_results"]
            if budget == "50" and row["Task"] in ("dili", "skin_reaction"):
                source = ROOT / ablation_long[(row["Task"], "control")]["arm_source"]
            result = add("scaling", row["Task"], budget, source,
                         int(source_row["n_queries"]), float(row[budget]), budget=budget)
            row.update({f"{budget} {key}": value for key, value in result.items()})
    tables[ABLATIONS / "gold_v1_scaling_test_20260925_v4.tsv"] = (scaling, fields)

    output.mkdir(parents=True)
    for path, (rows, fields) in tables.items():
        _write(path, rows, fields)
    _write(output / "cell_summary.tsv", summaries, list(summaries[0]))
    _write(output / "per_query.tsv", per_query, list(per_query[0]))
    report = (
        "# Evidence FINAL trace reference metrics\n\n"
        "Each cell retains its published test macro-F1. Counts are means per query. "
        "Reasoning counts use distinct visible IDs in the final call's private reasoning; "
        "final-claim counts use distinct IDs in structured claim citation lists. "
        "Only readable reasoning contributes to reasoning-reference means. "
        "The `reasoning_text_n` column gives that denominator; unavailable text is never zero. "
        "Reasoning tokens use provider counts when present and a DeepSeek tokenizer estimate "
        "for visible text otherwise. Token estimates and provider counts are different measurements. "
        "The tables are descriptive; provider routes and prompt choices vary across arms.\n\n"
        "Outputs: `cell_summary.tsv`, `per_query.tsv`, both model-specific direct/mixed "
        "tables, and v4 Gold-v1 ablation and scaling tables. No inference was run.\n"
    )
    (output / "report.md").write_text(report)
    inputs[str(Path(__file__).relative_to(ROOT))] = _hash(Path(__file__))
    reader = ROOT / "analysis/prediction/flat_reasoning_diagnostics.py"
    inputs[str(reader.relative_to(ROOT))] = _hash(reader)
    matcher = ROOT / "predict/harnesses/reasoning_references.py"
    inputs[str(matcher.relative_to(ROOT))] = _hash(matcher)
    outputs = {str(path.relative_to(ROOT)): _hash(path)
               for path in [*tables, output / "cell_summary.tsv", output / "per_query.tsv", output / "report.md"]}
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "evidence_final_trace_metrics.v1", "status": "complete",
        "n_cells": len(summaries), "n_per_query_rows": len(per_query),
        "inputs": inputs, "outputs": outputs,
    }, indent=2) + "\n")
    return {"cells": len(summaries), "per_query_rows": len(per_query)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY)
    print(build(parser.parse_args().output))
