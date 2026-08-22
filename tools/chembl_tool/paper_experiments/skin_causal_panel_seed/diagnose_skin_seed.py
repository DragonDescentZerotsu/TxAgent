"""Trace-backed failure diagnosis for the no-go Skin causal-panel seed."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import re
from typing import Any

from tools.chembl_tool.common.json_utils import read_jsonl, write_jsonl_atomic
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.build_skin_seed import (
    DEFAULT_OUTPUT_ROOT,
)
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.analyze_skin_seed import (
    DEFAULT_DIRECT_BATCH,
    binary_metrics,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    diagnosis = diagnose_skin_seed(args.output_root)
    print(json.dumps(diagnosis, indent=2), flush=True)
    return 0


def diagnose_skin_seed(output_root: Path) -> dict[str, Any]:
    selection = json.loads((output_root / "selected_indices.json").read_text(encoding="utf-8"))
    selected = [int(value) for value in selection["selected_indices"]]
    retrieval = {
        int(row["query_index"]): row
        for row in read_jsonl(output_root / "retrieval_audit.jsonl")
    }
    flips = {
        int(row["query_index"]): row
        for row in read_jsonl(output_root / "analysis" / "paired_flips.jsonl")
    }
    predictions = {
        int(row["query_index"]): row
        for row in read_jsonl(
            output_root
            / "runs_identity_blind_parent_disjoint/skin_reaction/"
            "skin_reaction__starling_causal_panel_seed_v1/predictions.jsonl"
        )
    }
    direct_predictions = {
        int(row["query_index"]): row
        for row in read_jsonl(DEFAULT_DIRECT_BATCH / "predictions.jsonl")
    }
    source_build = json.loads((output_root / "source_build_audit.json").read_text(encoding="utf-8"))
    with Path(source_build["paths"]["index_pkl"]).open("rb") as handle:
        source_index = pickle.load(handle)
    materialized_direction_counts: Counter[str] = Counter()
    for groups in source_index["evidence_by_molecule_group"].values():
        for row in groups.get("Mechanism.sensitization_causal_panel", []):
            materialized_direction_counts[
                str((row.get("causal_panel") or {}).get("observed_direction") or "missing")
            ] += 1
    run_root = output_root / (
        "runs_identity_blind_parent_disjoint/skin_reaction/"
        "skin_reaction__starling_causal_panel_seed_v1/runs"
    )

    direction_counts: Counter[str] = Counter()
    transferability_counts: Counter[str] = Counter()
    usefulness_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()
    top3_patterns: Counter[tuple[str, ...]] = Counter()
    gold_by_top_direction: Counter[tuple[str, int]] = Counter()
    max_morgan_by_query: dict[int, float | None] = {}
    branch_by_query: dict[int, dict[str, Any]] = {}
    for query_index in selected:
        run_dir = run_root / f"skin_reaction__starling_causal_panel_seed_v1_idx{query_index:05d}"
        groups = read_jsonl(run_dir / "group_reasoning_outputs.jsonl")
        causal = next(row for row in groups if row.get("group_id") == "Mechanism.causal_panel")
        content = dict((causal.get("llm") or {}).get("content") or {})
        branch_by_query[query_index] = content
        direction_counts[str(content.get("sensitization_evidence_direction") or "missing")] += 1
        transferability_counts[str(content.get("transferability") or "missing")] += 1
        usefulness_counts[str(bool(content.get("useful_for_skin_sensitization_reasoning")))] += 1
        endpoint_counts[str(content.get("endpoint_scope") or "missing")] += 1
        top3_patterns[
            tuple(
                str(neighbor.get("observed_direction") or "")
                for neighbor in retrieval[query_index]["causal_neighbors"]
            )
        ] += 1
        gold_by_top_direction[
            (str(retrieval[query_index]["top_causal_direction"]), int(predictions[query_index]["label"]))
        ] += 1
        max_morgan_by_query[query_index] = _max_reported_morgan(content)

    flip_diagnoses = []
    for query_index, flip in sorted(flips.items()):
        content = branch_by_query[query_index]
        transferability = str(content.get("transferability") or "")
        direction = str(content.get("sensitization_evidence_direction") or "")
        causal_signal = transferability in {"moderate", "high"} and direction in {
            "supports_sensitizer",
            "argues_against_sensitizer",
        }
        flip_diagnoses.append(
            {
                "query_index": query_index,
                "flip_effect": flip["flip_effect"],
                "branch_transferability": transferability,
                "branch_direction": direction,
                "branch_useful": bool(content.get("useful_for_skin_sensitization_reasoning")),
                "max_reported_morgan": max_morgan_by_query[query_index],
                "top3_source_directions": [
                    neighbor["observed_direction"]
                    for neighbor in retrieval[query_index]["causal_neighbors"]
                ],
                "classification": (
                    "directional_transfer" if causal_signal else "neutral_low_transfer_final_instability"
                ),
                "branch_summary": str(content.get("reasoning_summary") or ""),
                "direct_summary": flip["direct_summary"],
                "seed_summary": flip["seed_summary"],
            }
        )

    parsed_morgan = [value for value in max_morgan_by_query.values() if value is not None]
    neutral_flip_count = sum(
        row["classification"] == "neutral_low_transfer_final_instability"
        for row in flip_diagnoses
    )
    neutral_harmful_count = sum(
        row["classification"] == "neutral_low_transfer_final_instability"
        and row["flip_effect"] == "harmful"
        for row in flip_diagnoses
    )
    directional_transfer_indices = sorted(
        query_index
        for query_index, content in branch_by_query.items()
        if str(content.get("transferability") or "") in {"moderate", "high"}
        and str(content.get("sensitization_evidence_direction") or "")
        in {"supports_sensitizer", "argues_against_sensitizer"}
    )
    posthoc_rows = [
        predictions[query_index]
        if query_index in directional_transfer_indices
        else direct_predictions[query_index]
        for query_index in sorted(direct_predictions)
    ]
    direct_metrics = binary_metrics(list(direct_predictions.values()))
    posthoc_metrics = binary_metrics(posthoc_rows)
    posthoc_flips = [
        query_index
        for query_index in directional_transfer_indices
        if int(predictions[query_index]["pred_label"])
        != int(direct_predictions[query_index]["pred_label"])
    ]
    diagnosis = {
        "type": "skin_causal_panel_seed_trace_diagnosis.v1",
        "verdict": "no_go_do_not_start_bbb_seed",
        "n_selected": len(selected),
        "source_panel": {
            "full_cards": source_build["compiler"]["n_compiled_cards"],
            "full_direction_counts": source_build["compiler"]["direction_counts"],
            "materialized_cards": source_build["materialization"]["n_materialized_cards"],
            "materialized_direction_counts": dict(sorted(materialized_direction_counts.items())),
            "diagnosis": "strict outcome calibration retained quality but left a severe positive-source imbalance",
        },
        "retrieval": {
            "top3_direction_patterns": {
                "|".join(pattern): count for pattern, count in sorted(top3_patterns.items())
            },
            "gold_by_top_causal_direction": {
                f"{direction}|Y={label}": count
                for (direction, label), count in sorted(gold_by_top_direction.items())
            },
            "n_queries_with_parsed_morgan": len(parsed_morgan),
            "n_max_morgan_below_0_30": sum(value < 0.30 for value in parsed_morgan),
            "n_max_morgan_below_0_40": sum(value < 0.40 for value in parsed_morgan),
            "max_of_query_max_morgan": max(parsed_morgan) if parsed_morgan else None,
            "diagnosis": (
                "MiniMol found embedding-near cards, but most lacked structural/reactive-route compatibility; "
                "high cosine was not a mechanism-transfer gate"
            ),
        },
        "causal_branch": {
            "transferability_counts": dict(sorted(transferability_counts.items())),
            "direction_counts": dict(sorted(direction_counts.items())),
            "useful_counts": dict(sorted(usefulness_counts.items())),
            "endpoint_scope_counts": dict(sorted(endpoint_counts.items())),
            "diagnosis": (
                "the model usually recognized non-transferability, so the new branch contributed uncertainty "
                "rather than usable causal evidence"
            ),
        },
        "flips": {
            "n_total": len(flip_diagnoses),
            "n_neutral_low_transfer": neutral_flip_count,
            "n_neutral_low_transfer_harmful": neutral_harmful_count,
            "rows_path": str(output_root / "analysis" / "trace_flip_diagnosis.jsonl"),
            "diagnosis": (
                "most flips were final-synthesis/default-policy instability after a neutral branch, not "
                "successful transfer of a causal signal"
            ),
        },
        "exploratory_directional_transfer_sensitivity": {
            "status": "posthoc_diagnostic_not_a_promotion_result",
            "eligibility_rule": "branch transferability moderate/high and direction non-neutral",
            "n_eligible": len(directional_transfer_indices),
            "eligible_indices": directional_transfer_indices,
            "n_prediction_flips": len(posthoc_flips),
            "flip_indices": posthoc_flips,
            "macro_f1": posthoc_metrics["macro_f1"],
            "macro_f1_delta_vs_direct": posthoc_metrics["macro_f1"] - direct_metrics["macro_f1"],
            "accuracy": posthoc_metrics["accuracy"],
            "accuracy_delta_vs_direct": posthoc_metrics["accuracy"] - direct_metrics["accuracy"],
            "interpretation": (
                "the cards may contain a small useful signal after transfer gating, but the rule was "
                "examined after this seed and changed only one prediction; it requires a fresh independent seed"
            ),
        },
        "root_causes": [
            {
                "rank": 1,
                "cause": "reference cards are causal internally, but query-to-reference transfer is not causal",
                "evidence": "55/64 branches were low-transferability and 48/64 neutral_or_unclear",
            },
            {
                "rank": 2,
                "cause": "MiniMol cosine retrieves broad semantic/physicochemical neighbors, not shared reactive routes",
                "evidence": f"{sum(value < 0.30 for value in parsed_morgan)}/{len(parsed_morgan)} query panels had max reported Morgan <0.30",
            },
            {
                "rank": 3,
                "cause": "negative causal reference coverage is too sparse",
                "evidence": "heldout-filtered pool contains 89 positive versus 10 negative cards",
            },
            {
                "rank": 4,
                "cause": "an extra neutral branch destabilizes the final binary default",
                "evidence": f"{neutral_flip_count}/10 flips followed neutral/low-transfer evidence; {neutral_harmful_count} were harmful",
            },
        ],
        "recommended_starling_redesign": [
            "Extract molecule-level reactive mechanism family and activation route together with direct outcome and AOP events.",
            "Expand experimentally supported non-sensitizer panels within each reactive mechanism family instead of adding more generic negatives.",
            "Retrieve by shared reactive motif/activation route first, then rank within that compatible family; do not use MiniMol cosine alone as the transfer gate.",
            "Expose a causal branch to final synthesis only when its independent transfer assessment is moderate/high and directional; otherwise retain the frozen direct result.",
            "Run a new label-blind Skin seed only after zero-cost gates show balanced mechanism-family coverage and nontrivial structurally compatible query coverage.",
        ],
    }
    analysis_dir = output_root / "analysis"
    write_jsonl_atomic(analysis_dir / "trace_flip_diagnosis.jsonl", flip_diagnoses)
    (analysis_dir / "trace_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8"
    )
    (analysis_dir / "trace_diagnosis.md").write_text(
        _render_markdown(diagnosis), encoding="utf-8"
    )
    return diagnosis


def _max_reported_morgan(content: dict[str, Any]) -> float | None:
    values = []
    pattern = re.compile(
        r"(?:Morgan(?: fingerprint)?(?: Tanimoto)?|Tanimoto)(?: similarities?)?"
        r"\s*(?:=|≤|only|of|:)?\s*(0\.\d+)",
        flags=re.IGNORECASE,
    )
    for evidence in content.get("key_evidence") or []:
        values.extend(float(value) for value in pattern.findall(str(evidence.get("tool_summary") or "")))
    return max(values) if values else None


def _render_markdown(diagnosis: dict[str, Any]) -> str:
    source = diagnosis["source_panel"]
    retrieval = diagnosis["retrieval"]
    branch = diagnosis["causal_branch"]
    flips = diagnosis["flips"]
    lines = [
        "# Skin causal-panel seed trace diagnosis",
        "",
        "Verdict: **NO-GO**. Do not start the BBB seed from this design.",
        "",
        f"- Reference pool: {source['materialized_cards']} cards "
        f"({source['materialized_direction_counts']['positive']} positive / "
        f"{source['materialized_direction_counts']['negative']} negative).",
        f"- Transferability: {branch['transferability_counts']}.",
        f"- Branch directions: {branch['direction_counts']}.",
        f"- Structural mismatch: {retrieval['n_max_morgan_below_0_30']}/"
        f"{retrieval['n_queries_with_parsed_morgan']} query panels had max reported Morgan <0.30.",
        f"- Final instability: {flips['n_neutral_low_transfer']}/10 flips followed a neutral/low-transfer branch; "
        f"{flips['n_neutral_low_transfer_harmful']} were harmful.",
        f"- Post-hoc transfer gate: {diagnosis['exploratory_directional_transfer_sensitivity']['n_eligible']} eligible, "
        f"{diagnosis['exploratory_directional_transfer_sensitivity']['n_prediction_flips']} flip, macro-F1 delta "
        f"{diagnosis['exploratory_directional_transfer_sensitivity']['macro_f1_delta_vs_direct']:+.6f}; "
        "diagnostic only.",
        "",
        "## Root causes",
        "",
    ]
    lines.extend(
        f"{row['rank']}. {row['cause']}: {row['evidence']}"
        for row in diagnosis["root_causes"]
    )
    lines.extend(["", "## Recommended Starling redesign", ""])
    lines.extend(f"- {item}" for item in diagnosis["recommended_starling_redesign"])
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
