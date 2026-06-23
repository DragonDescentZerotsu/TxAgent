"""Bioavailability_Ma reasoning pipeline v2 with source-agnostic final evidence compilation.

This module preserves the original retrieval and single/group reasoning flow in
run_reasoning_pipeline.py. It only replaces the final synthesis step with a
role-aware, source-agnostic evidence compiler.
"""

from __future__ import annotations

import json
from typing import Any

from tools.chembl_tool.tasks.bioavailability_ma import run_reasoning_pipeline as v1
from tools.chembl_tool.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
from tools.chembl_tool.tasks.bioavailability_ma.evidence_compiler import compile_final_evidence


PIPELINE_VARIANT = "bioavailability_ma_deepseek_v2_source_agnostic_final_compiler"


def main(argv: list[str] | None = None) -> int:
    v1._run_final_reasoning = _run_final_reasoning_v2
    return v1.main(argv)


def _run_final_reasoning_v2(
    client: v1.DeepSeekClient,
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    compiled = compile_final_evidence(retrieval, single_output, group_outputs)
    decision_view = compiled.get("llm_decision_view") or compiled
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior oral bioavailability reasoning model. "
                "Integrate evidence using the provided source-agnostic compiler summary. "
                "Be conservative about analog transfer: endpoint role and transferability gates are mandatory. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Final oral bioavailability prediction from role-aware analog evidence.",
                    "pipeline_variant": PIPELINE_VARIANT,
                    "query": v1._llm_query_payload(retrieval["query"]),
                    "compiled_evidence_summary": decision_view,
                    "instructions": [
                        "Return compact complete JSON.",
                        f"Use bioavailability_prediction='high' only when the best-supported conclusion is oral bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (Bioavailability_Ma label 1). Use 'low' for F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (label 0).",
                        "Respect the compiled decision_rules, blocked_numeric_evidence_policy, and proxy_evidence_policy.",
                        "Direct label evidence must pass the direct_label_vote_allowed gate. Do not override that gate.",
                        (
                            "For direct F% evidence, use compiler_threshold_direction and threshold_summary as an "
                            "unweighted numeric audit, not as an automatic override of the group transferability "
                            "analysis. If they disagree with llm_evidence_direction or reasoning_summary, inspect "
                            "which source molecules the group analysis considered structurally and mechanistically "
                            "transferable; discount numeric values from weak, confounded, formulation-specific, "
                            "source-mismatched, or explicitly nontransferable analogs."
                        ),
                        (
                            f"Apply the {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% threshold literally: direct F values "
                            f">= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% support high, including borderline "
                            "20-25% values. If the most transferable direct evidence is consistently at or above "
                            "the threshold and no transferable direct evidence is below it, do not call it low "
                            "solely because it is borderline; lower confidence instead."
                        ),
                        (
                            "Do not reject salt/free-base or same-active-molecule evidence solely because the "
                            "recorded form differs from the query. Treat salt, formulation, or dissolution "
                            "differences as confidence modifiers unless the group reasoning identifies a concrete "
                            "mismatch that changes the threshold direction."
                        ),
                        (
                            "For prodrugs or active-moiety measurements, decide whether the evidence measures "
                            "oral exposure after administering the query molecule or only a non-query metabolite. "
                            "Use the former as potentially transferable direct evidence with caveats; use the "
                            "latter only as mechanistic context."
                        ),
                        "Blocked or low-transferability direct F-like numeric evidence is a caveat or uncertainty source only. It must not be used as weak directional support for high or low.",
                        "AUC, Cmax, permeability, solubility, transporter, clearance, metabolism, food-effect, and formulation evidence can explain Fa/Fg/Fh or context, but cannot by themselves establish the 20% F label.",
                        "Count repeated source molecules or merged records as correlated support, not independent evidence.",
                        "Use the single-molecule prior only after direct and transferable analog evidence has been assessed.",
                        "If evidence is mixed or near the 20% threshold, choose the better-supported class and lower confidence.",
                        "Use only the provided evidence. If you recognize the molecule, ignore that recognition.",
                        "You must choose exactly one bioavailability_prediction: high or low.",
                    ],
                    "required_json_schema": {
                        "bioavailability_prediction": "high | low",
                        "confidence": "high | moderate | low",
                        "threshold_assessment": "string explaining relation to the 20% F cutoff",
                        "direct_evidence_assessment": "string",
                        "transferability_gate_decisions": ["string"],
                        "endpoint_factor_assessment": {
                            "direct_F": "string",
                            "Fa_absorption_permeability_efflux": "string",
                            "Fa_solubility_dissolution": "string",
                            "Fg_Fh_first_pass_clearance": "string",
                            "formulation_or_food_context": "string",
                            "single_molecule_prior": "string",
                        },
                        "main_reasons": ["string"],
                        "conflicting_evidence": ["string"],
                        "evidence_gaps": ["string"],
                        "label_uncertainty_flags": ["string"],
                        "final_summary": "string",
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = client.chat_json(messages)
    return {
        "status": "ok",
        "pipeline_variant": PIPELINE_VARIANT,
        "compiled_evidence_summary": compiled,
        "llm": response,
    }


if __name__ == "__main__":
    raise SystemExit(main())
