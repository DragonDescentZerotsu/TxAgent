"""Analyze the BBB direct-anchored residual adjudication valid experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.final_decision_prior import TrainRatioPrior
from tools.chembl_tool.tasks.bbb_martins.final_decision_profiles import (
    DIRECT_ANCHORED_RESIDUAL_V1,
    DIRECT_OVERRIDE_RECHECK_V1,
    bbb_final_decision_validation_errors,
)
from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)


REUSED_ARTIFACTS = (
    "retrieval.json",
    "single_molecule_reasoning_output.json",
    "group_reasoning_outputs.jsonl",
)
BBB_PRIOR = TrainRatioPrior(
    dataset_lineage="experimental_meaningful_cns_access_v2",
    split="scaffold/train",
    positive_count=2162,
    negative_count=773,
    positive_label="pass",
    negative_label="fail",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_rows(batch: Path) -> dict[int, dict[str, Any]]:
    metrics = _read_json(batch / "metrics.json")
    if int(metrics.get("n_failed_runs") or 0):
        raise ValueError(f"Batch has failed runs: {batch}")
    rows = [
        json.loads(line)
        for line in (batch / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {int(row["query_index"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate query indices: {batch}")
    return indexed


def _final_content(row: dict[str, Any]) -> dict[str, Any]:
    final_path = Path(row["final_reasoning_output"])
    content = (_read_json(final_path).get("llm") or {}).get("content") or {}
    if not isinstance(content, dict):
        raise ValueError(f"Invalid final content: {final_path}")
    return content


def proposed_override_indices(residual_batch: Path) -> list[int]:
    rows = _prediction_rows(residual_batch)
    return sorted(
        index
        for index, row in rows.items()
        if _final_content(row).get("mechanism_override") != "no_change"
    )


def _audit_final_only_reuse(
    source_batch: Path,
    candidate_batch: Path,
    candidate_rows: dict[int, dict[str, Any]],
    *,
    profile: str,
) -> dict[str, Any]:
    source_rows = _prediction_rows(source_batch)
    checked = 0
    for index, candidate in sorted(candidate_rows.items()):
        source = source_rows.get(index)
        if source is None:
            raise ValueError(f"Missing source row: index={index}")
        candidate_run = Path(candidate["run_dir"])
        manifest = _read_json(candidate_run / "manifest.json")
        source_value = Path(str(manifest.get("final_only_source_batch") or ""))
        if source_value.resolve() != source_batch.resolve():
            raise ValueError(f"Final-only source mismatch: index={index}")
        if manifest.get("final_decision_profile") != profile:
            raise ValueError(f"Final profile mismatch: index={index}")
        source_run = Path(source["run_dir"])
        for name in REUSED_ARTIFACTS:
            if sha256_file(source_run / name) != sha256_file(candidate_run / name):
                raise ValueError(f"Reused artifact differs: index={index} name={name}")
            checked += 1
    return {
        "status": "pass",
        "candidate_batch": str(candidate_batch),
        "n_rows": len(candidate_rows),
        "n_artifact_pairs_checked": checked,
        "n_mismatches": 0,
    }


def _validate_audit_content(content: dict[str, Any], profile: str, index: int) -> None:
    errors = bbb_final_decision_validation_errors(
        content,
        profile=profile,
        prior=BBB_PRIOR,
        prediction_field="bbb_prediction",
    )
    if errors:
        raise ValueError(f"Invalid decision audit: index={index} errors={errors}")


def _condition_summary(
    labels: list[int],
    direct: list[int],
    candidate: list[int],
) -> dict[str, Any]:
    paired = paired_binary_summary(
        labels,
        direct,
        candidate,
        bootstrap_replicates=20_000,
        seed=20260813,
    )
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, candidate, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, candidate, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, candidate, strict=True))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, candidate, strict=True))
    return {**paired, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def analyze(
    direct_batch: Path,
    standard_batch: Path,
    residual_batch: Path,
    recheck_batch: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direct_rows = _prediction_rows(direct_batch)
    standard_rows = _prediction_rows(standard_batch)
    residual_rows = _prediction_rows(residual_batch)
    indices = sorted(direct_rows)
    if len(indices) != 366 or set(standard_rows) != set(indices) or set(residual_rows) != set(indices):
        raise ValueError("Direct, standard, and residual batches must align on all 366 rows")

    recheck_rows = _prediction_rows(recheck_batch) if recheck_batch else {}
    override_indices = proposed_override_indices(residual_batch)
    if set(recheck_rows) != set(override_indices):
        raise ValueError("Recheck rows must exactly equal proposed residual overrides")

    labels: list[int] = []
    direct_predictions: list[int] = []
    standard_predictions: list[int] = []
    residual_predictions: list[int] = []
    hybrid_predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    anchor_states: Counter[str] = Counter()
    mechanism_weights: Counter[str] = Counter()
    override_directions: Counter[str] = Counter()
    recheck_verdicts: Counter[str] = Counter()

    for index in indices:
        direct = direct_rows[index]
        standard = standard_rows[index]
        residual = residual_rows[index]
        label = int(direct["label"])
        if int(standard["label"]) != label or int(residual["label"]) != label:
            raise ValueError(f"Label mismatch: index={index}")
        residual_content = _final_content(residual)
        _validate_audit_content(residual_content, DIRECT_ANCHORED_RESIDUAL_V1, index)
        anchor_states[str(residual_content["direct_anchor_state"])] += 1
        mechanism_weights[str(residual_content["mechanism_evidence_weight"])] += 1
        override = str(residual_content["mechanism_override"])
        override_directions[override] += 1

        hybrid = int(residual["pred_label"])
        recheck_content: dict[str, Any] | None = None
        if index in recheck_rows:
            recheck = recheck_rows[index]
            if int(recheck["label"]) != label:
                raise ValueError(f"Recheck label mismatch: index={index}")
            recheck_content = _final_content(recheck)
            _validate_audit_content(recheck_content, DIRECT_OVERRIDE_RECHECK_V1, index)
            recheck_verdicts[str(recheck_content["override_verdict"])] += 1
            hybrid = int(recheck["pred_label"])

        labels.append(label)
        direct_predictions.append(int(direct["pred_label"]))
        standard_predictions.append(int(standard["pred_label"]))
        residual_predictions.append(int(residual["pred_label"]))
        hybrid_predictions.append(hybrid)
        rows.append(
            {
                "query_index": index,
                "label": label,
                "direct_prediction": int(direct["pred_label"]),
                "standard_full_mechanism_prediction": int(standard["pred_label"]),
                "residual_prediction": int(residual["pred_label"]),
                "hybrid_prediction": hybrid,
                "direct_anchor_state": residual_content["direct_anchor_state"],
                "direct_anchor_prediction": residual_content["direct_anchor_prediction"],
                "mechanism_evidence_weight": residual_content["mechanism_evidence_weight"],
                "mechanism_override": override,
                "recheck_verdict": (recheck_content or {}).get("override_verdict"),
            }
        )

    conditions = {
        "direct_anchor": _condition_summary(labels, direct_predictions, direct_predictions),
        "standard_full_mechanism": _condition_summary(labels, direct_predictions, standard_predictions),
        "direct_anchored_residual": _condition_summary(labels, direct_predictions, residual_predictions),
        "residual_with_override_recheck": _condition_summary(labels, direct_predictions, hybrid_predictions),
    }
    hybrid = conditions["residual_with_override_recheck"]
    ci_low = hybrid["delta_macro_f1_bootstrap_95ci"][0]
    promotion = bool(
        hybrid["delta_macro_f1"] > 0
        and hybrid["right_accuracy"] >= hybrid["left_accuracy"]
        and ci_low >= 0
    )
    reuse = {
        "residual": _audit_final_only_reuse(
            standard_batch,
            residual_batch,
            residual_rows,
            profile=DIRECT_ANCHORED_RESIDUAL_V1,
        ),
        "recheck": (
            _audit_final_only_reuse(
                standard_batch,
                recheck_batch,
                recheck_rows,
                profile=DIRECT_OVERRIDE_RECHECK_V1,
            )
            if recheck_batch
            else {"status": "not_needed", "n_rows": 0}
        ),
    }
    return {
        "schema_version": "bbb.residual_adjudication.valid.v1",
        "dataset_lineage": "experimental_meaningful_cns_access_v2",
        "split": "scaffold/valid",
        "n": len(labels),
        "conditions": conditions,
        "audit_counts": {
            "direct_anchor_state": dict(sorted(anchor_states.items())),
            "mechanism_evidence_weight": dict(sorted(mechanism_weights.items())),
            "mechanism_override": dict(sorted(override_directions.items())),
            "recheck_verdict": dict(sorted(recheck_verdicts.items())),
        },
        "proposed_override_indices": override_indices,
        "reuse_audit": reuse,
        "promotion_gate_pass": promotion,
        "formal_test_started": False,
    }, rows


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# BBB direct-anchored residual adjudication — scaffold-valid",
        "",
        "| condition | macro-F1 | delta vs direct | 95% CI | accuracy | flips | corrected / broken |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, result in summary["conditions"].items():
        ci = result["delta_macro_f1_bootstrap_95ci"]
        lines.append(
            f"| {name} | {result['right_macro_f1']:.4f} | {result['delta_macro_f1']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {result['right_accuracy']:.4f} | "
            f"{result['prediction_flips']} | {result['right_only_correct']} / {result['left_only_correct']} |"
        )
    lines.extend(
        [
            "",
            f"- Audit counts: `{json.dumps(summary['audit_counts'], sort_keys=True)}`",
            f"- Reused-artifact audit: `{json.dumps(summary['reuse_audit'], sort_keys=True)}`",
            f"- Promotion gate: **{'PASS' if summary['promotion_gate_pass'] else 'FAIL'}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-batch", type=Path, required=True)
    parser.add_argument("--emit-recheck-indices", type=Path)
    parser.add_argument("--direct-batch", type=Path)
    parser.add_argument("--standard-batch", type=Path)
    parser.add_argument("--recheck-batch", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.emit_recheck_indices:
        indices = proposed_override_indices(args.residual_batch)
        args.emit_recheck_indices.parent.mkdir(parents=True, exist_ok=True)
        args.emit_recheck_indices.write_text(json.dumps(indices, indent=2) + "\n", encoding="utf-8")
        print(" ".join(map(str, indices)))
        if not args.output_dir:
            return 0

    if not all((args.direct_batch, args.standard_batch, args.output_dir)):
        parser.error("full analysis requires --direct-batch, --standard-batch, and --output-dir")
    summary, rows = analyze(
        args.direct_batch,
        args.standard_batch,
        args.residual_batch,
        args.recheck_batch,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
