"""Bioavailability_Ma specific reasoning pipeline with Fa/Fg/Fh synthesis.

This module keeps the original service/tool infrastructure but changes the
task-specific reasoning surface: source retrieval rows are regrouped into
observed-F context plus Fa/Fg/Fh factor groups, group prompts analyze those
typed groups, and final synthesis reasons over F = Fa * Fg * Fh.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tools.chembl_tool.tasks.bioavailability_ma import run_reasoning_pipeline as v1
from tools.chembl_tool.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
from tools.chembl_tool.tasks.bioavailability_ma.specific_evidence_compiler import (
    build_specific_decision_policy,
    compile_specific_final_evidence,
)
from tools.chembl_tool.tasks.bioavailability_ma.specific_retrieval import retrieve_specific_neighbors


PIPELINE_VARIANT = "bioavailability_ma_specific_fa_fg_fh_v16_mechanism_alert_view"


def main(argv: list[str] | None = None) -> int:
    v1.retrieve_neighbors = retrieve_specific_neighbors
    v1._reason_single_molecule = _reason_single_molecule_specific
    v1._reason_one_group = _reason_one_specific_group
    v1._run_final_reasoning = _run_final_reasoning_specific
    v1._parse_json_content = _parse_json_content_specific
    return v1.main(argv)


def _reason_single_molecule_specific(
    client: v1.DeepSeekClient,
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        "Do not identify the molecule by drug name, therapeutic class, label, or known clinical PK. If you recognize it, ignore that recognition.",
        "Use only computed structure/property evidence from molecule_properties: molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, and functional groups.",
        "Assess Fa prior from dissolution/solubility/permeability/charge descriptors.",
        "Assess Fg/Fh prior only from structural metabolic liabilities such as esters, amides, oxidizable aromatics/alkyl groups, CYP-soft spots, UGT/glucuronidation-like groups, and ionization/permeability constraints.",
        "Do not cite external facts such as known first-pass metabolism, marketed formulation, exact human F, or disease/indication.",
        "Return JSON with oral_bioavailability_prior, absorption_prior, solubility_or_dissolution_prior, metabolism_or_clearance_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ]
    payload: dict[str, Any] = {
        "task": "Single-molecule Bioavailability_Ma property-only Fa/Fg/Fh plausibility analysis.",
        "query": query,
        "instructions": instructions,
        "required_json_schema": {
            "oral_bioavailability_prior": "high | low | mixed_or_unclear",
            "absorption_prior": "favorable | unfavorable | mixed_or_unclear",
            "solubility_or_dissolution_prior": "favorable | unfavorable | mixed_or_unclear",
            "metabolism_or_clearance_prior": "favorable | unfavorable | mixed_or_unclear",
            "forbidden_external_knowledge_check": "string confirming no drug-name or known-PK facts were used",
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "property_drivers": ["string"],
            "caveats": ["string"],
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry property-only oral bioavailability analyst. "
                "You may call exactly one tool: molecule_properties. "
                "You must ignore molecule identity, drug names, and known PK facts. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]
    response = client.chat_json_with_tools(
        messages,
        tools=v1.SINGLE_MOLECULE_TOOLS,
        allowed_tool_names={"molecule_properties"},
        first_tool_choice="auto",
    )
    return {
        "analysis_id": "single_molecule",
        "status": "ok",
        "llm": response,
    }


def _reason_one_specific_group(client: v1.DeepSeekClient, query: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical pharmacology oral bioavailability factor analyst. "
                "Analyze exactly one typed evidence group for Bioavailability_Ma. "
                "Do not make a final high/low prediction unless this is clean direct parent F evidence. "
                "Use tools only to judge analog transferability. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(_specific_group_prompt_payload(query, group), ensure_ascii=False),
        },
    ]
    response = client.chat_json_with_group_tools(messages)
    return {
        "group_id": group["group_id"],
        "status": "ok",
        "tier": group["tier"],
        "endpoint_group": group["endpoint_group"],
        "specific_group_role": group.get("specific_group_role"),
        "bioavailability_factor": group.get("bioavailability_factor"),
        "n_neighbors": len(group["neighbors"]),
        "llm": response,
    }


def _specific_group_prompt_payload(query: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    role = str(group.get("specific_group_role") or "")
    factor = str(group.get("bioavailability_factor") or "")
    return {
        "task": "Bioavailability_Ma typed factor-group analog analysis.",
        "pipeline_variant": PIPELINE_VARIANT,
        "query": query,
        "group": {
            "group_id": group["group_id"],
            "tier": group["tier"],
            "endpoint_group": group["endpoint_group"],
            "specific_group_role": role,
            "bioavailability_factor": factor,
            "specific_group_description": group.get("specific_group_description") or "",
            "evidence_source": v1._group_evidence_source(group),
            "source_group_ids": sorted(
                {
                    str(source_group_id)
                    for neighbor in group.get("neighbors") or []
                    for source_group_id in neighbor.get("source_group_ids", [])
                    if source_group_id
                }
            ),
        },
        "neighbors": [
            {
                "rank": neighbor["rank"],
                "molecule_chembl_id": neighbor["molecule_chembl_id"],
                "canonical_smiles": neighbor["canonical_smiles"],
                "similarity": neighbor["similarity"],
                "similarity_bucket": neighbor["similarity_bucket"],
                "source_group_ids": neighbor.get("source_group_ids") or [],
                "evidence_rows": [v1._clean_evidence_row(row) for row in neighbor["evidence_rows"]],
                "shared_assay_context": v1._clean_shared_assay_context(neighbor.get("shared_assay_context") or {}),
            }
            for neighbor in group["neighbors"]
        ],
        "instructions": _specific_group_instructions(role=role, factor=factor),
        "required_json_schema": {
            "useful_for_bioavailability_reasoning": "boolean",
            "specific_group_role": role,
            "bioavailability_factor": factor,
            "transferability": "high | moderate | low | not_applicable",
            "evidence_direction": (
                "supports_high_bioavailability | argues_against_high_bioavailability | absorption_support | "
                "permeability_support | solubility_support | solubility_risk | metabolic_stability_support | "
                "first_pass_or_clearance_risk | transporter_efflux_risk | neutral_or_unclear"
            ),
            "factor_effect": "supports_higher_F | risk_for_lower_F | mixed_or_context | neutral_or_unclear",
            "confidence": "high | moderate | low",
            "parent_scope_assessment": "string",
            "factor_transfer_assessment": "string",
            "reasoning_summary": "string",
            "key_evidence": [
                {
                    "molecule_chembl_id": "string",
                    "similarity": "number or null",
                    "similarity_bucket": "string",
                    "specific_evidence_role": "string",
                    "assay_signal": "string",
                    "activity_values": ["string"],
                    "tool_summary": "string",
                    "transferability": "high | moderate | low | not_applicable",
                    "parent_or_analyte_scope": "clean_parent | active_metabolite_or_moiety | prodrug_or_conversion | formulation_or_route_specific | unclear | not_applicable",
                    "effect_on_bioavailability_reasoning": "string",
                }
            ],
            "caveats": ["string"],
        },
    }


def _specific_group_instructions(*, role: str, factor: str) -> list[str]:
    common = [
        "Use only this typed evidence group.",
        "Assess source analog transferability to the query; low-similarity analogs are intentionally included and must be explicitly gated.",
        "Use mmp_structure_compare for scaffold/MCS/matched-pair differences when similarity bucket is not enough.",
        "Use properties_compare when pKa, logD, TPSA, charge, HBD/HBA, size, polarity, or functional groups could change oral bioavailability transfer.",
        "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
        "For Starling evidence, source_record_examples bind condition, F%, and support text. Do not detach a number from its species, dose, formulation, route, or analyte context.",
        "If starling_transfer_tool is present, treat it as auxiliary analog-transfer evidence, not an automatic label.",
        "Return key_evidence as structured cards. Include specific_evidence_role from each row when relevant.",
        "Do not use external knowledge; use only provided evidence and tool results.",
    ]
    if role == "direct_parent_f_or_systemic_exposure_context":
        return [
            *common,
            f"Only clean absolute oral parent F values can directly support the {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% threshold.",
            "Assign high transferability to a direct-F analog only when it is the same active parent/connectivity or a very close analog (roughly Tanimoto >=0.90) with matching administered molecule, measured analyte, swallowed oral route, species/population context, and formulation/salt scope. Class analogs below that bar are at most moderate, often low.",
            "De-correlate repeated records before judging direction: salt/free-base/enantiomer/literature duplicates and ChEMBL/Starling records for the same active source molecule count as one source-level signal, not independent votes.",
            "If source-level direct F values from the same scaffold family straddle the 20% threshold, do not let one high analog dominate; mark threshold sensitivity and use factor evidence only as a low-confidence tie-breaker.",
            "Oral AUC/Cmax/exposure rows are systemic exposure context, not direct F. They may support plausibility only after route, dose, formulation, clearance, and analyte scope checks.",
            "Separate administered/query parent molecule from measured analyte, active metabolite, active moiety, salt, prodrug conversion product, formulation-specific, or non-swallowed route evidence.",
            "If evidence measures active metabolite/moiety after parent or prodrug dosing, mark parent_or_analyte_scope accordingly and do not treat it as clean parent F.",
            "For direct F values around 15-25%, flag threshold sensitivity instead of overclaiming.",
            "Do not use known drug names or class reputation as evidence. If a name appears in a source row, use only the numeric/context fields provided for that source.",
        ]
    if factor == "Fa":
        return [
            *common,
            "Analyze Fa only: dissolution, solubility, permeability, intestinal absorption, absorptive uptake, and absorption-side efflux barriers.",
            "Favorable Fa cannot by itself prove high systemic F because Fg and Fh may still be poor.",
            "Unfavorable solubility/permeability/absorption can lower F, but do not convert it into a direct 20% threshold claim without direct F evidence.",
            "Distinguish passive permeability/solubility from transporter inhibition or target binding assays that are not absorption evidence.",
        ]
    if factor == "Fg":
        return [
            *common,
            "Analyze Fg only: fraction escaping gut-wall metabolism and intestinal efflux before portal circulation.",
            "Relevant mechanisms include intestinal CYP3A, enterocyte metabolism, P-gp/BCRP/MRP efflux, gut presystemic extraction, and food/formulation effects that change gut availability.",
            "Transporter inhibition/binding is not the same as being an efflux substrate unless the assay context supports substrate/transport behavior.",
            "Fg risk can explain low F, but by itself does not prove parent F <20% when clean direct F evidence is high or threshold-straddling.",
        ]
    if factor == "Fh":
        return [
            *common,
            "Analyze Fh only: fraction escaping hepatic first-pass extraction and hepatic/systemic metabolic clearance.",
            "Relevant mechanisms include hepatic clearance, extraction ratio, intrinsic clearance, liver microsome/hepatocyte/S9 stability, CYP/UGT/glucuronidation, and substrate depletion.",
            "Distinguish metabolic clearance of parent from prodrug activation or active-metabolite formation; parent F and active-moiety exposure are different endpoints.",
            "Strong metabolism or clearance wording should be translated into Fh risk, not automatically into a low label without direct F or coherent factor evidence.",
        ]
    return common


def _parse_json_content_specific(content: str) -> Any:
    """Parse JSON from model output with a small repair for invalid escapes."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as first_error:
        extracted = _extract_json_object(content)
        candidates = [extracted] if extracted != content else []
        candidates.append(_escape_invalid_json_backslashes(extracted))
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {
            "unparsed_text": content,
            "parse_error": str(first_error),
        }


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    return content


def _escape_invalid_json_backslashes(content: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", content)


def _run_final_reasoning_specific(
    client: v1.DeepSeekClient,
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    compiled = compile_specific_final_evidence(retrieval, single_output, group_outputs)
    decision_view = compiled.get("llm_decision_view") or compiled
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior clinical pharmacology and oral bioavailability reasoning model. "
                "For this Bioavailability_Ma-specific pipeline, do not reason as a flat high/low vote aggregator. "
                "Reason through parent oral F, then the Fa/Fg/Fh factor model. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Final Bioavailability_Ma prediction with parent-F and Fa/Fg/Fh factor synthesis.",
                    "pipeline_variant": PIPELINE_VARIANT,
                    "query": v1._llm_query_payload(retrieval["query"]),
                    "compiled_evidence_summary": decision_view,
                    "instructions": [
                        "Return compact complete JSON.",
                        f"Use bioavailability_prediction='high' only when the best-supported conclusion is oral parent bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (Bioavailability_Ma label 1). Use 'low' for parent F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (label 0).",
                        "Default to parent/unchanged-drug F for the query molecule. If the evidence appears to use active-metabolite, active-moiety, prodrug, formulation-specific, or route-specific scope, explicitly flag it before using it.",
                        "Read evidence_gate_summary before making the label decision. Treat final_reasoning_contract as binding unless you can cite a clean_parent_like_analog_f card that directly contradicts it.",
                        "Read deterministic_decision_policy.decision_state before choosing the class. If it is force_high or force_low, output the forced_class exactly and use the rest of the answer only to explain it.",
                        "If deterministic_decision_policy.decision_state is fallback_uncertain, do not infer a label from hidden or diagnostic class blocks. Treat this as genuine uncertainty and choose the forced binary class only from visible source-level direct-F, soft context, factor evidence, and property priors.",
                        "factor_product_guard.class_constraints are diagnostic-only in this LLM view. Do not cite or infer hidden diagnostic blocks as a reason for high or low.",
                        "Read soft_direct_f_context as context-only de-correlated direct-F evidence. It is not a hard vote, but it can counter weak metabolism priors or weak analog-only factor risks.",
                        "If soft_direct_f_context.net_soft_context_direction is soft_context_threshold_sensitive_mixed, treat it as uncertainty, not as low evidence.",
                        "Read review_required_counter_context as non-voting aggregate caution. It must not directly vote high or low, but review-required low/straddling context should prevent a fallback high call from relying only on absence of strong low evidence, and review-required high context should prevent a fallback low call from relying only on broad metabolism risk.",
                        "Read mechanism_transfer_alerts before using soft/context analog direct-F evidence. These alerts identify mechanisms such as permanent charge, beta-lactam anionic motifs, DHP diesters, prodrug/active-moiety scope, or dissolution-risk scaffolds that can make raw analog high-F transfer unsafe.",
                        "mechanism_transfer_alerts are not direct low votes. Use them only as transfer-mismatch guards for soft/context evidence or property priors; eligible clean direct-F anchors still take priority.",
                        "Use source_level_direct_f_summary as the authoritative de-correlated direct-F view. Do not count raw ChEMBL/Starling/enantiomer/salt/repeated rows as independent votes when the source-level summary has already collapsed them.",
                        "If source_level_direct_f_consensus is mixed, threshold-straddling, or has no eligible clean source-level vote, do not let one high-F analog or one low-F analog dominate the label.",
                        "Do not invent numeric bounds such as Fg > 0.5 or Fh ~= 0.6 unless the compiled evidence explicitly provides that bound. Missing Fg/Fh is uncertainty, not proof of high F.",
                        "Do not multiply several moderate analog-only risks into F < 20% unless factor_product_guard allows low; low needs strong limiting evidence, a strong absorption-limited property prior, or prodrug/parent-Fh risk when no direct F vote is eligible.",
                        "Do not use any parent_analyte_review_required direct-F value as the main reason for high or low. Those values are audit/context only unless the provided evidence itself resolves parent/analyte scope.",
                        "If no clean parent-like direct-F label vote is available, make a factor/prior-based forced choice. Do not default to low solely because direct F is absent; low requires credible limiting Fa, Fg, or Fh evidence. Do not treat missing or mixed Fg/Fh as proof that those factors are non-limiting for a high label.",
                        "First evaluate direct_f_analog_evidence. These are direct-F endpoint values for retrieved source molecules, not exact-query F, because exact query context is disabled in this benchmark.",
                        "Only clean_parent_like_analog_f cards may act as transferable direct-F analog evidence. Parent_analyte_review_required cards are context until you prove administered molecule and measured analyte match the query parent-F endpoint.",
                        "If evidence is prodrug, active metabolite, active moiety, metabolite-after-parent-dosing, salt/formulation-specific, or route-specific, do not silently use it as parent F. State the scope and downweight or move it to context.",
                        "Then evaluate Fa: solubility, dissolution, permeability, absorbed fraction, uptake, and efflux barriers.",
                        "Then evaluate Fg: gut-wall escape, intestinal metabolism, gut CYP3A, intestinal efflux, and food/formulation effects that alter gut availability.",
                        "Then evaluate Fh: hepatic first-pass extraction, metabolic stability, hepatic clearance, CYP/UGT/glucuronidation, prodrug conversion, and systemic presystemic metabolism.",
                        "High Fa, permeability, HIA, oral AUC, or Cmax does not prove high F when Fg/Fh evidence is poor.",
                        "Strong first-pass or metabolism wording does not prove low F when credible numeric parent F remains >=20%; lower confidence or mark borderline instead.",
                        "Animal low-F or first-pass evidence cannot override same-molecule human parent F unless human evidence is absent or clearly non-matching.",
                        "Dose, steady-state, food, salt, formulation, and route-specific evidence must match the benchmark context before it can drive the final label.",
                        "If clean direct-F analog evidence is mixed, weak, or threshold-sensitive and Fg/Fh evidence indicates strong first-pass or clearance, do not let favorable Fa alone drive high.",
                        "If the only low evidence is a single borderline 15-25% analog, animal/species-specific value, or context-mismatched value, do not let metabolism wording alone drive low; use the 20% threshold literally and lower confidence.",
                        "For prodrug or active-metabolite cases, default benchmark scope is unchanged parent F. Active metabolite exposure after oral parent/prodrug dosing cannot establish parent F high.",
                        "For threshold-straddling evidence around 15-25%, choose the better-supported forced class but set low confidence and add label_uncertainty_flags.",
                        "Use only the provided evidence. If you recognize the molecule, ignore that recognition.",
                        "You must choose exactly one bioavailability_prediction: high or low.",
                    ],
                    "required_json_schema": {
                        "bioavailability_prediction": "high | low",
                        "confidence": "high | moderate | low",
                        "parent_scope_assessment": "string explaining administered molecule vs measured analyte and whether direct evidence is clean parent F",
                        "direct_f_analog_assessment": "string explaining source analog direct-F values and transfer gates",
                        "factor_assessment": {
                            "Fa": "string",
                            "Fg": "string",
                            "Fh": "string",
                            "overall_factor_synthesis": "string explaining F = Fa * Fg * Fh",
                        },
                        "context_gate_assessment": {
                            "prodrug_or_active_metabolite": "string",
                            "species_translation": "string",
                            "dose_steady_state_food_formulation_route": "string",
                            "threshold_sensitivity": "string",
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
    response = _apply_specific_decision_policy(client.chat_json(messages), compiled)
    return {
        "status": "ok",
        "pipeline_variant": PIPELINE_VARIANT,
        "compiled_evidence_summary": compiled,
        "llm": response,
    }


def _apply_specific_decision_policy(response: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic final-label validation on top of the LLM explanation.

    The LLM still writes the reasoning, but Bioavailability_Ma has a few hard
    evidentiary constraints: a blocked class cannot be selected, and no-direct
    low calls need admissible low evidence rather than prior-only metabolism
    speculation.
    """
    if not isinstance(response, dict) or not isinstance(response.get("content"), dict):
        return response
    content = dict(response["content"])
    prediction = str(content.get("bioavailability_prediction") or "").strip().lower()
    override = _specific_policy_override(prediction, compiled)
    if not override:
        return response

    new_response = dict(response)
    old_prediction = prediction or content.get("bioavailability_prediction")
    new_prediction = override["prediction"]
    content["bioavailability_prediction"] = new_prediction
    content["confidence"] = _downgraded_confidence(content.get("confidence"))
    content["policy_override"] = {
        "applied": True,
        "old_prediction": old_prediction,
        "new_prediction": new_prediction,
        "reason": override["reason"],
    }
    flags = list(content.get("label_uncertainty_flags") or [])
    flags.append(f"deterministic_policy_override:{override['reason_code']}")
    content["label_uncertainty_flags"] = flags
    reasons = list(content.get("main_reasons") or [])
    reasons.append(f"Deterministic final policy set prediction to {new_prediction}: {override['reason']}")
    content["main_reasons"] = reasons
    summary = str(content.get("final_summary") or "").strip()
    suffix = f" Deterministic final policy changed the forced label from {old_prediction} to {new_prediction}: {override['reason']}"
    content["final_summary"] = (summary + suffix).strip()
    new_response["content"] = content
    return new_response


def _specific_policy_override(prediction: str, compiled: dict[str, Any]) -> dict[str, str] | None:
    policy = build_specific_decision_policy(compiled)
    forced_class = str(policy.get("forced_class") or "").strip().lower()
    decision_state = str(policy.get("decision_state") or "").strip().lower()
    if decision_state in {"force_high", "force_low"} and forced_class in {"high", "low"} and forced_class != prediction:
        return {
            "prediction": forced_class,
            "reason_code": decision_state,
            "reason": (
                f"Deterministic admissibility policy state is {decision_state} with forced_class={forced_class}: "
                f"{policy.get('reason_codes') or []}"
            ),
        }
    return None


def _downgraded_confidence(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value == "high":
        return "moderate"
    return value if value in {"moderate", "low"} else "low"


if __name__ == "__main__":
    raise SystemExit(main())
