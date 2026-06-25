"""Bioavailability-specific final compiler using an Fa/Fg/Fh factor model.

This compiler is intentionally downstream of retrieval and group reasoning: it
does not change which evidence sources are fetched. It rewrites the final-stage
view so oral bioavailability is treated as F = Fa * Fg * Fh instead of a flat
collection of high/low endpoint votes.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from tools.chembl_tool.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
from tools.chembl_tool.tasks.bioavailability_ma.evidence_compiler import compile_final_evidence


SPECIFIC_COMPILER_VERSION = "bioavailability_specific_fa_fg_fh_v16_mechanism_alert_view"

FA_ROLES = {
    "absorption_fa",
    "in_vivo_absorption_or_permeability",
    "in_vitro_permeability",
    "solubility_or_dissolution",
}
FG_HINTS = {
    "gut",
    "intestinal",
    "intestine",
    "enterocyte",
    "enteric",
    "presystemic",
    "p-gp",
    "pgp",
    "efflux",
    "bcrp",
    "cyp3a",
    "food",
    "fed",
    "fasted",
    "dissolution",
}
FH_HINTS = {
    "hepatic",
    "liver",
    "clearance",
    "extraction",
    "microsomal",
    "microsome",
    "hepatocyte",
    "cyp",
    "ugt",
    "glucuronid",
    "oxidation",
    "metabolic stability",
    "first-pass",
    "first pass",
}
PRODRUG_ANALYTE_HINTS = {
    "prodrug",
    "measured analyte",
    "conversion product",
    "metabolite exposure",
    "metabolite after",
    "diacid",
}
SCOPE_MISMATCH_HINTS = {
    "measured analyte differs",
    "differs from parent",
    "not unchanged parent",
    "not parent f",
    "not clean parent",
    "not direct parent",
    "after prodrug",
    "prodrug dosing",
}
SPECIES_HINTS = {
    "rat",
    "mouse",
    "mice",
    "dog",
    "canine",
    "monkey",
    "primate",
    "rabbit",
    "guinea pig",
}
DOSE_CONTEXT_HINTS = {
    "dose-dependent",
    "dose dependent",
    "saturable",
    "steady-state",
    "steady state",
    "single dose",
    "multiple dose",
    "autoinduction",
    "induction",
}
FORMULATION_HINTS = {
    "salt",
    "free-base",
    "free base",
    "formulation",
    "fed",
    "fasted",
    "food",
    "meal",
    "intranasal",
    "buccal",
    "transmucosal",
    "solution",
    "tablet",
    "capsule",
}
FORMULATION_REVIEW_HINTS = {
    "formulation-specific",
    "self-emulsified",
    "self emulsified",
    "sedd",
    "sedds",
    "snedd",
    "snedds",
    "solid dispersion",
    "nanoparticle",
    "nanosuspension",
    "lipid-based",
    "lipid based",
    "micellar",
    "cyclodextrin",
    "salt form",
    "free-base",
    "free base",
}
SPECIAL_FORMULATION_REVIEW_HINTS = FORMULATION_REVIEW_HINTS - {
    "salt form",
    "free-base",
    "free base",
}
NON_SWALLOWED_ROUTE_HINTS = {
    "intranasal",
    "buccal",
    "sublingual",
    "transmucosal",
    "transdermal",
    "rectal",
    "inhaled",
    "pulmonary",
    "ocular",
}
RELATIVE_OR_PROXY_SCOPE_HINTS = {
    "relative bioavailability",
    "relative_comparison",
    "exposure ratio",
    "fold higher",
    "fold lower",
    "fold increase",
    "fold decrease",
}
TOTAL_RADIOACTIVITY_HINTS = {
    "total radioactivity",
    "drug-related radioactivity",
    "drug related radioactivity",
    "radiolabeled",
    "radiolabelled",
}
DIRECT_OR_DIRECT_CONTEXT_ROLES = {
    "direct_oral_f",
    "direct_oral_f_context",
    "oral_exposure_ratio",
}
SOURCE_DIRECT_TRANSFERABLE = {"high"}


def compile_specific_final_evidence(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile final evidence into a Bioavailability_Ma-specific factor view."""
    base = compile_final_evidence(retrieval, single_output, group_outputs)
    source_scope_context = _source_scope_context_map(retrieval)
    group_cards = list(base.get("group_cards") or [])
    factor_cards = {"Fa": [], "Fg": [], "Fh": []}
    contextual_cards: list[dict[str, Any]] = []

    for card in group_cards:
        if not card.get("useful_for_bioavailability_reasoning") and not card.get("threshold_evidence"):
            continue
        factors = _assign_factors(card)
        slim = _factor_card(card)
        if factors:
            for factor in factors:
                factor_cards[factor].append(dict(slim, factor=factor))
        elif card.get("endpoint_role") not in DIRECT_OR_DIRECT_CONTEXT_ROLES:
            contextual_cards.append(slim)

    direct_cards = [_direct_parent_f_card(card) for card in base.get("direct_label_votes") or []]
    direct_evidence = _split_direct_evidence(direct_cards)
    source_level_direct_f_summary = _source_level_direct_f_summary(
        group_cards,
        source_scope_context=source_scope_context,
    )
    context_flags = _context_flags(group_cards)
    source_transfer_summary = _source_transfer_summary(source_level_direct_f_summary)
    single_content = (single_output.get("llm") or {}).get("content") or {}
    mechanism_alerts = _mechanism_transfer_alerts(single_content)
    gate_summary = _evidence_gate_summary(
        direct_evidence=direct_evidence,
        source_level_direct_f_summary=source_level_direct_f_summary,
        factor_cards=factor_cards,
        single_output=single_output,
        context_flags=context_flags,
        mechanism_alerts=mechanism_alerts,
    )
    gate_summary["deterministic_decision_policy"] = build_specific_decision_policy(
        {"evidence_gate_summary": gate_summary}
    )
    decision_view = _specific_decision_view(
        base=base,
        direct_evidence=direct_evidence,
        factor_cards=factor_cards,
        contextual_cards=contextual_cards,
        context_flags=context_flags,
        gate_summary=gate_summary,
        mechanism_alerts=mechanism_alerts,
    )
    return {
        "compiler_version": SPECIFIC_COMPILER_VERSION,
        "base_compiler_version": base.get("compiler_version"),
        "task_model": {
            "model": "F = Fa * Fg * Fh",
            "Fa": "fraction absorbed through dissolution, solubility, permeability, uptake, and efflux barriers",
            "Fg": "fraction escaping gut-wall metabolism and intestinal efflux before portal circulation",
            "Fh": "fraction escaping hepatic first-pass extraction, clearance, and systemic presystemic metabolism",
            "direct_F_policy": (
                "Credible same-molecule or transferable parent absolute oral F is observed product evidence. "
                "Use it before inferring from factors, but review administered molecule and measured analyte first."
            ),
        },
        "direct_f_analog_evidence": direct_evidence,
        "source_level_direct_f_summary": source_level_direct_f_summary,
        "source_transfer_summary": source_transfer_summary,
        "mechanism_transfer_alerts": mechanism_alerts,
        "source_scope_context_summary": _source_scope_context_summary(source_scope_context),
        "factor_evidence": factor_cards,
        "factor_summary": {factor: _summarize_factor(cards) for factor, cards in factor_cards.items()},
        "contextual_evidence": contextual_cards,
        "context_flags": context_flags,
        "evidence_gate_summary": gate_summary,
        "base_compiled_evidence": base,
        "llm_decision_view": decision_view,
    }


def _specific_decision_view(
    *,
    base: dict[str, Any],
    direct_evidence: dict[str, Any],
    factor_cards: dict[str, list[dict[str, Any]]],
    contextual_cards: list[dict[str, Any]],
    context_flags: dict[str, Any],
    gate_summary: dict[str, Any],
    mechanism_alerts: dict[str, Any],
) -> dict[str, Any]:
    clean_direct_cards = list(direct_evidence.get("clean_parent_like_analog_f") or [])
    review_direct_cards = list(direct_evidence.get("parent_analyte_review_required") or [])
    llm_direct_evidence = _direct_evidence_for_llm(direct_evidence)
    llm_gate_summary = _gate_summary_for_llm(gate_summary)
    return {
        "compiler_version": SPECIFIC_COMPILER_VERSION,
        "task_threshold": {
            "high_label_rule": f"parent oral bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}%",
            "low_label_rule": f"parent oral bioavailability F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}%",
            "default_endpoint_scope": (
                "Use the query/administered parent molecule unless evidence clearly states that the benchmark "
                "or source uses an active-moiety formulation-specific scope."
            ),
        },
        "factor_model": {
            "equation": "F = Fa * Fg * Fh",
            "interpretation": [
                "Fa evidence can support absorption but cannot establish systemic F if Fg/Fh are poor.",
                "Fg/Fh first-pass evidence can lower F but is not automatically below 20%; compare numeric F when available.",
                "Direct parent absolute F evidence is the observed product of all factors and has priority when clean.",
            ],
        },
        "evidence_state": {
            "n_clean_parent_like_direct_f_analog_cards": len(clean_direct_cards),
            "n_parent_analyte_review_required_direct_f_cards": len(review_direct_cards),
            "factor_card_counts": {key: len(value) for key, value in factor_cards.items()},
            "contextual_card_count": len(contextual_cards),
            "clean_direct_f_analog_threshold_direction_counts": _direction_counts(clean_direct_cards),
            "review_required_direct_f_threshold_direction_counts": "redacted_from_llm_view",
            "base_blocked_numeric_group_count": (
                base.get("llm_decision_view", {})
                .get("evidence_state", {})
                .get("n_blocked_numeric_groups", 0)
            ),
        },
        "direct_f_analog_evidence": llm_direct_evidence,
        "source_level_direct_f_summary": llm_gate_summary.get("source_level_direct_f_summary") or {},
        "factor_evidence": {key: value[:10] for key, value in factor_cards.items()},
        "factor_summary": {factor: _summarize_factor(cards) for factor, cards in factor_cards.items()},
        "contextual_evidence": contextual_cards[:12],
        "context_flags": context_flags,
        "mechanism_transfer_alerts": _mechanism_alerts_for_llm(mechanism_alerts),
        "evidence_gate_summary": llm_gate_summary,
        "factor_product_guard": llm_gate_summary.get("factor_product_guard") or {},
        "duplicate_or_correlated_source_summary": base.get("duplicate_or_correlated_source_summary") or [],
        "decision_rules": [
            "Do not use a flat high/low vote across tiers. Reason through direct parent F, then Fa, Fg, and Fh.",
            "Retrieval excludes exact query evidence in the prospective benchmark. Direct-F cards are source-analog endpoint evidence, not query direct F, unless explicitly stated otherwise.",
            "If measured_analyte or active moiety differs from administered/query parent, do not treat the value as clean parent-F evidence.",
            "Parent/analyte review-required direct-F cards are context only until you explain why the measured analyte matches the query parent endpoint.",
            "If evidence_gate_summary says no clean direct label vote is available, do not say a review-required direct-F value compels high or low.",
            "Use mechanism_transfer_alerts only to check whether soft/context analog evidence is mechanism-mismatched; do not use alerts to override an eligible clean direct-F anchor.",
            "High Fa, permeability, absorption, oral AUC, or Cmax cannot establish high F when gut/hepatic first-pass evidence is strong.",
            "Strong first-pass, clearance, or metabolism wording cannot establish low F when credible human parent F is >=20%; it lowers confidence or explains borderline F.",
            "Animal first-pass or low-F data are mechanism caveats unless human direct F is absent or concordant.",
            "Dose, steady-state, food, salt, formulation, and route-specific evidence must not be generalized to ordinary swallowed oral parent F without a matching context.",
            "If F values straddle 15-25%, mark threshold sensitivity and use low confidence unless one context clearly matches the benchmark.",
            "When clean direct-F analog evidence is mixed or borderline and Fg/Fh cards show strong first-pass or clearance risk, do not let favorable Fa alone drive high.",
            "When clean direct-F analog evidence is borderline-low only because of a single analog or species/context mismatch, do not let metabolism wording alone drive low.",
        ],
    }


def _evidence_gate_summary(
    *,
    direct_evidence: dict[str, Any],
    source_level_direct_f_summary: dict[str, Any],
    factor_cards: dict[str, list[dict[str, Any]]],
    single_output: dict[str, Any],
    context_flags: dict[str, Any],
    mechanism_alerts: dict[str, Any],
) -> dict[str, Any]:
    clean_cards = list(direct_evidence.get("clean_parent_like_analog_f") or [])
    review_cards = list(direct_evidence.get("parent_analyte_review_required") or [])
    factor_signal_counts = {
        factor: dict(sorted(Counter(str(card.get("signal") or "mixed_or_context") for card in cards).items()))
        for factor, cards in factor_cards.items()
    }
    single_content = (single_output.get("llm") or {}).get("content") or {}
    factor_product_guard = _factor_product_guard(
        source_level_direct_f_summary=source_level_direct_f_summary,
        factor_cards=factor_cards,
        single_content=single_content,
        context_flags=context_flags,
    )
    soft_direct_f_context = _soft_direct_f_context(source_level_direct_f_summary)
    no_clean_direct = not clean_cards
    return {
        "clean_direct_label_vote_available": bool(clean_cards),
        "n_clean_parent_like_direct_f_cards": len(clean_cards),
        "n_review_required_direct_f_cards": len(review_cards),
        "clean_direct_direction_counts": _direction_counts(clean_cards),
        "review_required_direction_counts_for_audit_only": _direction_counts(review_cards),
        "source_level_direct_f_summary": source_level_direct_f_summary,
        "source_level_direct_f_consensus": source_level_direct_f_summary.get("source_level_consensus"),
        "source_level_direct_f_decision_hint": source_level_direct_f_summary.get("decision_hint"),
        "soft_direct_f_context": soft_direct_f_context,
        "review_required_label_vote_policy": (
            "forbidden_as_direct_label_vote"
            if review_cards
            else "not_applicable"
        ),
        "factor_signal_counts": factor_signal_counts,
        "single_molecule_prior": {
            "oral_bioavailability_prior": single_content.get("oral_bioavailability_prior"),
            "absorption_prior": single_content.get("absorption_prior"),
            "solubility_or_dissolution_prior": single_content.get("solubility_or_dissolution_prior"),
            "metabolism_or_clearance_prior": single_content.get("metabolism_or_clearance_prior"),
            "confidence": single_content.get("confidence"),
        },
        "mechanism_transfer_alerts": mechanism_alerts,
        "context_flags_present": sorted(context_flags),
        "factor_product_guard": factor_product_guard,
        "final_reasoning_contract": _final_reasoning_contract(
            no_clean_direct=no_clean_direct,
            review_cards=review_cards,
            source_level_direct_f_summary=source_level_direct_f_summary,
        ),
        "forbidden_reasoning_patterns": [
            "Using parent_analyte_review_required values as the main reason for a high or low label.",
            "Treating active-metabolite, active-moiety, prodrug-conversion, salt/formulation, or species-mismatched F as clean parent F.",
            "Counting the same source molecule's ChEMBL and Starling records as independent votes.",
            "Treating oral AUC, Cmax, Papp, Peff, CL, CLint, recovery, stability, IC50, or transporter ratios as F% threshold evidence.",
        ],
    }


def _direct_evidence_for_llm(direct_evidence: dict[str, Any]) -> dict[str, Any]:
    result = dict(direct_evidence)
    result["clean_parent_like_analog_f"] = [
        _clean_direct_card_for_llm(card)
        for card in direct_evidence.get("clean_parent_like_analog_f") or []
    ]
    result["parent_analyte_review_required"] = [
        _redact_review_required_direct_card(card)
        for card in direct_evidence.get("parent_analyte_review_required") or []
    ]
    if result["parent_analyte_review_required"]:
        result["review_required_redaction_policy"] = (
            "Numeric values and threshold directions are withheld from the final LLM view because these cards failed "
            "parent/analyte/scope review. They remain in the full compiled audit object."
        )
    return result


def _clean_direct_card_for_llm(card: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in card.items()
        if key not in {"threshold_evidence", "threshold_summary", "reasoning_summary", "caveats"}
    } | {
        "threshold_evidence": "collapsed_to_source_level_direct_f_summary",
        "threshold_summary": {
            "collapsed": True,
            "reason": "Use source_level_direct_f_summary for de-correlated direct-F values and label-admissibility.",
        },
        "reasoning_summary": "Original group-level direct-F narrative omitted; use source-level direct-F summary.",
        "caveats": ["Raw direct-F rows may be correlated or context-mismatched; source-level summary is authoritative."],
    }


def _redact_review_required_direct_card(card: dict[str, Any]) -> dict[str, Any]:
    redacted = {
        key: value
        for key, value in card.items()
        if key
        not in {
            "threshold_evidence",
            "threshold_summary",
            "compiler_threshold_direction",
            "llm_evidence_direction",
            "reasoning_summary",
            "caveats",
        }
    }
    redacted["threshold_evidence"] = []
    redacted["threshold_summary"] = {
        "redacted": True,
        "reason": "parent/analyte/scope review required; not eligible for direct label vote",
    }
    redacted["compiler_threshold_direction"] = "redacted_review_required_context_only"
    redacted["llm_evidence_direction"] = "redacted_review_required_context_only"
    redacted["reasoning_summary"] = (
        "Original numeric/direct-F summary withheld from final LLM view because parent/analyte/scope review is required."
    )
    redacted["caveats"] = ["Direct-F values are context only until parent/analyte/scope compatibility is established."]
    return redacted


def _gate_summary_for_llm(gate_summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(gate_summary)
    source_summary = dict(result.get("source_level_direct_f_summary") or {})
    result["review_required_direction_counts_for_audit_only"] = "redacted_from_llm_view"
    result["source_level_direct_f_summary"] = _source_level_summary_for_llm(source_summary)
    if isinstance(result.get("factor_product_guard"), dict):
        guard = dict(result["factor_product_guard"])
        if guard.get("class_constraints"):
            guard["class_constraints"] = "hidden_from_llm_diagnostic_only"
            guard["class_constraints_policy"] = (
                "Diagnostic class blocks are hidden from final label synthesis unless the deterministic "
                "decision policy converts them into force_high or force_low."
            )
        result["factor_product_guard"] = guard
    if isinstance(result.get("deterministic_decision_policy"), dict):
        policy = dict(result["deterministic_decision_policy"])
        if policy.get("diagnostic_blocked_classes"):
            policy["diagnostic_blocked_classes"] = "hidden_from_llm_diagnostic_only"
        result["deterministic_decision_policy"] = policy
    return result


def _source_level_summary_for_llm(source_summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(source_summary)
    result["review_context_source_direction_counts"] = "redacted_from_llm_view"
    result["clean_context_source_direction_counts"] = source_summary.get("clean_context_source_direction_counts") or {}
    result["review_required_counter_context"] = _review_required_counter_context(source_summary)
    result["source_summaries"] = [
        _source_summary_item_for_llm(item)
        for item in source_summary.get("source_summaries") or []
    ]
    if result.get("n_review_required_context_only_sources"):
        result["review_required_redaction_policy"] = (
            "Review-required source values and high/low directions are hidden from final synthesis. They indicate "
            "uncertainty/scope mismatch only, not a label trend."
        )
    return result


def _source_summary_item_for_llm(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("label_vote_bucket") == "eligible_clean_direct_vote":
        return item
    if item.get("label_vote_bucket") == "clean_but_not_direct_vote":
        return _soft_source_item_for_llm(item)
    redaction_reason = (
        "parent/analyte/scope review required; context only"
        if item.get("label_vote_bucket") == "review_required_context_only"
        else "clean source but source-level transferability/confidence did not pass the direct label gate"
    )
    return {
        "source_molecule": item.get("source_molecule"),
        "source_merge_key": item.get("source_merge_key"),
        "source_molecule_ids": item.get("source_molecule_ids"),
        "label_vote_bucket": item.get("label_vote_bucket"),
        "source_direction": f"redacted_{item.get('label_vote_bucket')}",
        "n_values": item.get("n_values"),
        "n_above_or_equal_20_percent": "redacted",
        "n_below_20_percent": "redacted",
        "n_borderline_15_to_25_percent": "redacted",
        "min_value_percent": "redacted",
        "max_value_percent": "redacted",
        "groups": item.get("groups"),
        "evidence_sources": item.get("evidence_sources"),
        "parent_applicabilities": item.get("parent_applicabilities"),
        "parent_scope_reasons": item.get("parent_scope_reasons"),
        "transferabilities": item.get("transferabilities"),
        "confidences": item.get("confidences"),
        "redaction_reason": redaction_reason,
    }


def _source_level_direct_f_summary(
    group_cards: list[dict[str, Any]],
    *,
    source_scope_context: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Aggregate direct-F threshold evidence by source molecule for de-correlation."""
    source_records: dict[str, dict[str, Any]] = {}
    source_scope_context = source_scope_context or {}
    for card in group_cards:
        if card.get("endpoint_role") != "direct_oral_f" or not card.get("threshold_evidence"):
            continue
        direct_card = _direct_parent_f_card(card)
        card_parent_applicability = str(direct_card.get("parent_applicability") or "requires_parent_analyte_review")
        source_transferability = _source_transferability_map(card)
        group_scope_context = source_scope_context.get(str(card.get("group_id") or "")) or {}
        for evidence in card.get("threshold_evidence") or []:
            raw_source_key = str(evidence.get("source_molecule") or "").strip() or "unknown_source"
            source_key = str(evidence.get("source_merge_key") or raw_source_key).strip() or "unknown_source"
            scope_context = group_scope_context.get(source_key) or group_scope_context.get(raw_source_key) or {}
            parent_applicability = str(
                scope_context.get("parent_applicability") or card_parent_applicability
            )
            transferability = source_transferability.get(raw_source_key) or str(card.get("transferability") or "")
            transferability = transferability.strip().lower()
            source_direct_vote_allowed = (
                bool(card.get("direct_label_vote_allowed"))
                and parent_applicability == "clean_parent_f"
                and transferability in SOURCE_DIRECT_TRANSFERABLE
            )
            record = source_records.setdefault(
                source_key,
                {
                    "source_molecule": source_key,
                    "source_merge_key": source_key,
                    "source_molecule_ids": set(),
                    "source_inchi_keys": set(),
                    "canonical_smiles": set(),
                    "groups": set(),
                    "evidence_sources": set(),
                    "parent_applicabilities": set(),
                    "parent_scope_reasons": set(),
                    "source_scope_contexts": [],
                    "source_transfer_scope_classes": set(),
                    "values_percent": [],
                    "source_min_values_percent": [],
                    "source_max_values_percent": [],
                    "threshold_relations": [],
                    "borderline_value_count": 0,
                    "range_crosses_20_percent_count": 0,
                    "direct_vote_allowed_by_any_group": False,
                    "transferabilities": set(),
                    "confidences": set(),
                },
            )
            record["source_molecule_ids"].add(raw_source_key)
            if evidence.get("source_inchi_key"):
                record["source_inchi_keys"].add(str(evidence.get("source_inchi_key") or ""))
            if evidence.get("canonical_smiles"):
                record["canonical_smiles"].add(str(evidence.get("canonical_smiles") or ""))
            record["groups"].add(str(card.get("group_id") or ""))
            if evidence.get("evidence_source"):
                record["evidence_sources"].add(str(evidence.get("evidence_source") or ""))
            record["parent_applicabilities"].add(parent_applicability)
            for reason in scope_context.get("scope_review_reasons") or []:
                record["parent_scope_reasons"].add(str(reason))
            if scope_context:
                record["source_scope_contexts"].append(scope_context)
            for scope_class in scope_context.get("source_transfer_scope_classes") or []:
                record["source_transfer_scope_classes"].add(str(scope_class))
            value = evidence.get("value_percent")
            if isinstance(value, (int, float)):
                record["values_percent"].append(float(value))
            min_value = evidence.get("source_value_min_percent")
            if isinstance(min_value, (int, float)):
                record["source_min_values_percent"].append(float(min_value))
            max_value = evidence.get("source_value_max_percent")
            if isinstance(max_value, (int, float)):
                record["source_max_values_percent"].append(float(max_value))
            relation = str(evidence.get("threshold_relation") or "")
            if relation:
                record["threshold_relations"].append(relation)
            if evidence.get("borderline_to_20_percent"):
                record["borderline_value_count"] += 1
            if evidence.get("range_crosses_20_percent"):
                record["range_crosses_20_percent_count"] += 1
            record["direct_vote_allowed_by_any_group"] = (
                record["direct_vote_allowed_by_any_group"] or source_direct_vote_allowed
            )
            if transferability:
                record["transferabilities"].add(transferability)
            if card.get("confidence"):
                record["confidences"].add(str(card.get("confidence") or ""))

    source_summaries = [_summarize_direct_f_source(record) for record in source_records.values()]
    source_summaries = sorted(
        source_summaries,
        key=lambda item: (
            item["label_vote_bucket"] != "eligible_clean_direct_vote",
            item["source_direction"] == "mixed_threshold_straddling",
            item["source_molecule"],
        ),
    )
    eligible = [item for item in source_summaries if item["label_vote_bucket"] == "eligible_clean_direct_vote"]
    review = [item for item in source_summaries if item["label_vote_bucket"] == "review_required_context_only"]
    clean_context = [item for item in source_summaries if item["label_vote_bucket"] == "clean_but_not_direct_vote"]
    consensus = _source_level_consensus(eligible, clean_context, review)
    return {
        "policy": (
            "Source-level direct-F evidence is de-correlated by source molecule before final reasoning. "
            "Eligible clean sources can guide the label; review-required or clean-but-blocked sources are context only."
        ),
        "n_source_molecules_with_direct_f_values": len(source_summaries),
        "n_eligible_clean_direct_vote_sources": len(eligible),
        "n_clean_but_not_direct_vote_sources": len(clean_context),
        "n_review_required_context_only_sources": len(review),
        "eligible_clean_source_direction_counts": _count_source_directions(eligible),
        "clean_context_source_direction_counts": _count_source_directions(clean_context),
        "review_context_source_direction_counts": _count_source_directions(review),
        "n_threshold_straddling_sources": sum(
            1 for item in source_summaries if item["source_direction"] == "mixed_threshold_straddling"
        ),
        "source_level_consensus": consensus,
        "decision_hint": _source_level_decision_hint(consensus, eligible, clean_context, review),
        "source_summaries": source_summaries[:20],
    }


def _summarize_direct_f_source(record: dict[str, Any]) -> dict[str, Any]:
    relations = Counter(record["threshold_relations"])
    n_above = relations.get("above_or_equal_high_threshold", 0)
    n_below = relations.get("below_high_threshold", 0)
    if record["range_crosses_20_percent_count"] or (n_above and n_below):
        direction = "mixed_threshold_straddling"
    elif n_above:
        direction = "supports_high_bioavailability"
    elif n_below:
        direction = "argues_against_high_bioavailability"
    else:
        direction = "no_interpretable_threshold_direction"
    parent_applicabilities = sorted(record["parent_applicabilities"])
    if any(item != "clean_parent_f" for item in parent_applicabilities):
        bucket = "review_required_context_only"
    elif record["direct_vote_allowed_by_any_group"]:
        bucket = "eligible_clean_direct_vote"
    else:
        bucket = "clean_but_not_direct_vote"
    values = list(record["values_percent"])
    range_values = list(record["source_min_values_percent"]) + list(record["source_max_values_percent"])
    all_values = values + range_values
    return {
        "source_molecule": record["source_molecule"],
        "source_merge_key": record["source_merge_key"],
        "source_molecule_ids": sorted(record["source_molecule_ids"]),
        "source_inchi_keys": sorted(record["source_inchi_keys"]),
        "canonical_smiles": sorted(record["canonical_smiles"])[:3],
        "label_vote_bucket": bucket,
        "source_direction": direction,
        "n_values": len(values),
        "n_above_or_equal_20_percent": n_above,
        "n_below_20_percent": n_below,
        "n_borderline_15_to_25_percent": int(record["borderline_value_count"]),
        "n_range_crosses_20_percent": int(record["range_crosses_20_percent_count"]),
        "min_value_percent": round(min(all_values), 3) if all_values else None,
        "max_value_percent": round(max(all_values), 3) if all_values else None,
        "groups": sorted(record["groups"]),
        "evidence_sources": sorted(record["evidence_sources"]),
        "parent_applicabilities": parent_applicabilities,
        "parent_scope_reasons": sorted(record["parent_scope_reasons"]),
        "source_transfer_scope_classes": sorted(record["source_transfer_scope_classes"])
        or ["clean_parent_or_close_analog_context"],
        "n_source_scope_context_rows": len(record["source_scope_contexts"]),
        "transferabilities": sorted(record["transferabilities"]),
        "confidences": sorted(record["confidences"]),
    }


def _source_transfer_summary(source_level_direct_f_summary: dict[str, Any]) -> dict[str, Any]:
    """Audit-only source-transfer layer for future direct-F anchoring.

    This deliberately does not change v14.1 final policy. It makes the source
    semantics that subagents repeatedly found important explicit and testable
    before any later prompt or deterministic policy consumes them.
    """
    sources = list(source_level_direct_f_summary.get("source_summaries") or [])
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        for scope_class in source.get("source_transfer_scope_classes") or ["unknown_scope"]:
            by_scope.setdefault(str(scope_class), []).append(source)

    return {
        "policy": (
            "Audit-only direct-F source-transfer summary. It separates clean parent analogs, same-active-moiety "
            "salt/free-base context, prodrug/active-moiety context, special formulation/route context, and "
            "relative/proxy context before any future final policy decides whether these sources can anchor a label."
        ),
        "status": "audit_only_not_used_for_v14_1_label_policy",
        "scope_class_counts": {
            scope_class: len(scope_sources)
            for scope_class, scope_sources in sorted(by_scope.items())
        },
        "scope_direction_counts": {
            scope_class: _count_source_directions(scope_sources)
            for scope_class, scope_sources in sorted(by_scope.items())
        },
        "anchor_candidate_sources": [
            _source_transfer_item(source)
            for source in sources
            if source.get("label_vote_bucket") == "eligible_clean_direct_vote"
            or "same_active_moiety_salt_or_freebase_context" in set(source.get("source_transfer_scope_classes") or [])
        ][:12],
        "context_only_source_classes": {
            scope_class: [_source_transfer_item(source) for source in scope_sources[:8]]
            for scope_class, scope_sources in sorted(by_scope.items())
            if scope_class
            in {
                "active_metabolite_or_prodrug_scope",
                "special_formulation_or_route_context",
                "relative_or_proxy_context",
            }
        },
    }

def _source_transfer_item(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_molecule": source.get("source_molecule"),
        "source_merge_key": source.get("source_merge_key"),
        "source_molecule_ids": source.get("source_molecule_ids"),
        "label_vote_bucket": source.get("label_vote_bucket"),
        "source_direction": source.get("source_direction"),
        "min_value_percent": source.get("min_value_percent"),
        "max_value_percent": source.get("max_value_percent"),
        "transferabilities": source.get("transferabilities"),
        "confidences": source.get("confidences"),
        "parent_applicabilities": source.get("parent_applicabilities"),
        "parent_scope_reasons": source.get("parent_scope_reasons"),
        "source_transfer_scope_classes": source.get("source_transfer_scope_classes"),
    }


def _mechanism_transfer_alerts(single_content: dict[str, Any]) -> dict[str, Any]:
    text = _single_molecule_mechanism_text(single_content)
    alerts = {
        "permanent_quaternary_charge": {
            "present": _has_any(
                text,
                {
                    "quaternary ammonium",
                    "permanent positive charge",
                    "permanently charged",
                    "permanent cation",
                },
            ),
            "transfer_guard": (
                "High-F tertiary-amine analogs may not transfer to permanently charged quaternary ammonium "
                "parents unless matched quaternary/same-active evidence supports it."
            ),
        },
        "beta_lactam_anionic": {
            "present": _has_any(text, {"β-lactam", "beta-lactam"})
            and _has_any(text, {"anion", "anionic", "dianion", "carboxylic acid", "free carboxylic acid"}),
            "transfer_guard": (
                "High-F beta-lactam/cephalosporin analogs that depend on transporter motifs should not transfer "
                "to an anionic beta-lactam parent when same-active or low-source evidence conflicts."
            ),
        },
        "dihydropyridine_diester": {
            "present": _has_any(text, {"1,4-dihydropyridine", "dihydropyridine"})
            and _has_any(text, {"two carboxylic ester", "two ester", "diester"}),
            "transfer_guard": (
                "Dihydropyridine diester analogs can have strong first-pass/oxidative metabolism; do not let "
                "soft high DHP analog context override matched low or threshold-sensitive context by itself."
            ),
        },
        "flat_high_logp_low_tpsa_hbd0": {
            "present": _has_any(text, {"low tpsa", "tpsa 34", "tpsa (34"})
            and _has_any(text, {"hbd 0", "zero hbd"})
            and _has_any(text, {"logp/logd", "logd 3", "logp 3", "high logd", "high logp"})
            and _has_any(text, {"aromatic rings", "multiple aromatic", "flat", "rigid"}),
            "transfer_guard": (
                "Neutral flat high-logP low-TPSA HBD0 scaffolds may be dissolution-limited despite favorable "
                "permeability; soft high analog context needs matched solubility or direct-F support."
            ),
        },
        "ester_prodrug_or_active_moiety": {
            "present": _has_any(text, {"prodrug", "active metabolite", "active moiety"})
            or (
                _has_any(text, {"carboxylic ester", "hydroly", "ester"})
                and _has_any(text, {"active", "activation", "metabolite"})
            ),
            "transfer_guard": (
                "Prodrug or active-moiety exposure may describe active moiety F rather than unchanged parent F. "
                "Use it as scope context unless the benchmark/source scope explicitly matches active-moiety exposure."
            ),
        },
        "high_ionization_low_permeability": {
            "present": _has_any(
                text,
                {
                    "neutral fraction 0",
                    "fully ionized",
                    "dianion",
                    "net anionic",
                    "permanently charged",
                    "permanent positive charge",
                },
            )
            and _has_any(text, {"poor passive", "low permeability", "very poor passive", "permeability restriction"}),
            "transfer_guard": (
                "Strong ionization/permeability restriction can block raw high-F analog transfer unless uptake, "
                "prodrug, or same-active evidence explains absorption."
            ),
        },
    }
    present_alerts = [name for name, payload in alerts.items() if payload["present"]]
    return {
        "policy": (
            "Mechanism alerts are derived from the single-molecule structural prior. They are transfer guards for "
            "soft/context analog evidence, not direct high/low votes."
        ),
        "present_alerts": present_alerts,
        "n_present_alerts": len(present_alerts),
        "alerts": alerts,
    }


def _mechanism_alerts_for_llm(mechanism_alerts: dict[str, Any]) -> dict[str, Any]:
    alerts = mechanism_alerts.get("alerts") or {}
    return {
        "policy": mechanism_alerts.get("policy"),
        "present_alerts": mechanism_alerts.get("present_alerts") or [],
        "n_present_alerts": mechanism_alerts.get("n_present_alerts") or 0,
        "transfer_guards": {
            name: payload.get("transfer_guard")
            for name, payload in sorted(alerts.items())
            if payload.get("present")
        },
    }


def _single_molecule_mechanism_text(single_content: dict[str, Any]) -> str:
    parts = [
        single_content.get("reasoning_summary"),
    ]
    for key in ("property_drivers", "caveats"):
        values = single_content.get(key)
        if isinstance(values, list):
            parts.extend(values)
    return " ".join(str(part or "") for part in parts).lower()


def _source_transferability_map(card: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in card.get("key_evidence") or []:
        if not isinstance(item, dict):
            continue
        source_key = str(item.get("molecule_chembl_id") or item.get("source_molecule") or "").strip()
        transferability = str(item.get("transferability") or "").strip().lower()
        if not source_key or not transferability:
            continue
        old = result.get(source_key)
        if old is None or _transferability_rank(transferability) > _transferability_rank(old):
            result[source_key] = transferability
    return result


def _transferability_rank(value: str) -> int:
    return {"not_applicable": 0, "low": 1, "moderate": 2, "high": 3}.get(value, 0)


def _source_level_consensus(
    eligible_sources: list[dict[str, Any]],
    clean_context: list[dict[str, Any]] | None = None,
    review: list[dict[str, Any]] | None = None,
) -> str:
    if not eligible_sources:
        return "no_eligible_clean_source_level_direct_f_vote"
    directions = Counter(item["source_direction"] for item in eligible_sources)
    if directions.get("mixed_threshold_straddling"):
        return "eligible_clean_sources_threshold_straddling"
    if len(directions) == 1 and directions.get("supports_high_bioavailability"):
        context_directions = Counter(
            item["source_direction"] for item in (clean_context or []) + (review or [])
        )
        if any(
            context_directions.get(direction)
            for direction in {
                "argues_against_high_bioavailability",
                "mixed_threshold_straddling",
            }
        ):
            return "eligible_clean_sources_consensus_high_with_context_conflict"
        return "eligible_clean_sources_consensus_high"
    if len(directions) == 1 and directions.get("argues_against_high_bioavailability"):
        context_directions = Counter(
            item["source_direction"] for item in (clean_context or []) + (review or [])
        )
        if any(
            context_directions.get(direction)
            for direction in {
                "supports_high_bioavailability",
                "mixed_threshold_straddling",
            }
        ):
            return "eligible_clean_sources_consensus_low_with_context_conflict"
        return "eligible_clean_sources_consensus_low"
    return "eligible_clean_sources_mixed_high_low"


def _source_level_decision_hint(
    consensus: str,
    eligible: list[dict[str, Any]],
    clean_context: list[dict[str, Any]],
    review: list[dict[str, Any]],
) -> str:
    if consensus == "eligible_clean_sources_consensus_high":
        return "Eligible clean source-level direct-F evidence consistently supports high; use factors mainly as caveats or contradiction checks."
    if consensus == "eligible_clean_sources_consensus_high_with_context_conflict":
        return "Eligible clean source-level direct-F evidence supports high, but blocked/context sources conflict; treat high as supportive, not decisive."
    if consensus == "eligible_clean_sources_consensus_low":
        return "Eligible clean source-level direct-F evidence consistently supports low; use factors mainly as caveats or contradiction checks."
    if consensus == "eligible_clean_sources_consensus_low_with_context_conflict":
        return "Eligible clean source-level direct-F evidence supports low, but blocked/context sources conflict; treat low as supportive, not decisive."
    if consensus in {"eligible_clean_sources_threshold_straddling", "eligible_clean_sources_mixed_high_low"}:
        return "Eligible clean source-level direct-F evidence is mixed or threshold-straddling; do not let any single high or low analog dominate."
    if clean_context or review:
        return "No eligible clean source-level direct-F vote is available; clean-blocked and review-required direct-F sources are context only."
    return "No source-level direct-F threshold evidence is available; final decision must rely on Fa/Fg/Fh and property prior."


def _count_source_directions(sources: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["source_direction"] for item in sources).items()))


def _soft_direct_f_context(source_summary: dict[str, Any]) -> dict[str, Any]:
    """Summarize clean but non-voting direct-F sources as soft context.

    These sources failed the direct label gate, usually because analog
    transferability was not high. They must not become hard votes, but hiding
    their source-level direction entirely lets weak property priors dominate
    no-direct cases.
    """
    sources = [
        item
        for item in source_summary.get("source_summaries") or []
        if item.get("label_vote_bucket") == "clean_but_not_direct_vote"
    ]
    counts = _count_source_directions(sources)
    high = int(counts.get("supports_high_bioavailability") or 0)
    low = int(counts.get("argues_against_high_bioavailability") or 0)
    straddling = int(counts.get("mixed_threshold_straddling") or 0)
    low_or_straddling = low + straddling
    if high >= 3 and high >= 2 * max(1, low_or_straddling):
        net_direction = "soft_context_leans_high"
    elif low >= 2 and low > high and low >= straddling:
        net_direction = "soft_context_leans_low"
    elif straddling and low_or_straddling >= high:
        net_direction = "soft_context_threshold_sensitive_mixed"
    elif sources:
        net_direction = "soft_context_mixed_or_weak"
    else:
        net_direction = "no_soft_clean_direct_f_context"
    all_range_values = [
        value
        for item in sources
        for value in (item.get("min_value_percent"), item.get("max_value_percent"))
        if isinstance(value, (int, float))
    ]
    return {
        "policy": (
            "Clean but non-eligible direct-F sources are soft context only. They cannot directly vote on the "
            "label, but their de-correlated high/low/threshold-straddling direction can counter weak factor-only "
            "or single-molecule priors."
        ),
        "n_soft_clean_direct_f_sources": len(sources),
        "soft_clean_source_direction_counts": counts,
        "net_soft_context_direction": net_direction,
        "n_threshold_straddling_sources": straddling,
        "threshold_sensitive_policy": (
            "Threshold-straddling clean direct-F context is uncertainty/counterweight, not low evidence."
            if straddling
            else "not_applicable"
        ),
        "min_value_percent": round(min(all_range_values), 3) if all_range_values else None,
        "max_value_percent": round(max(all_range_values), 3) if all_range_values else None,
        "source_summaries": [_soft_source_item_for_llm(item) for item in sources[:12]],
    }


def _review_required_counter_context(source_summary: dict[str, Any]) -> dict[str, Any]:
    """Expose non-voting review-required direction as a caution without values."""
    sources = [
        item
        for item in source_summary.get("source_summaries") or []
        if item.get("label_vote_bucket") == "review_required_context_only"
    ]
    counts = _count_source_directions(sources)
    reason_counts = Counter(
        reason
        for item in sources
        for reason in item.get("parent_scope_reasons") or []
    )
    low_or_straddling = int(counts.get("argues_against_high_bioavailability") or 0) + int(
        counts.get("mixed_threshold_straddling") or 0
    )
    high = int(counts.get("supports_high_bioavailability") or 0)
    if low_or_straddling and low_or_straddling >= high:
        counterweight = "review_context_cautions_against_high"
    elif high and high > low_or_straddling:
        counterweight = "review_context_cautions_against_low"
    elif sources:
        counterweight = "review_context_mixed_or_weak"
    else:
        counterweight = "no_review_required_counter_context"
    return {
        "policy": (
            "Review-required source directions are non-voting and numeric values remain hidden, but aggregate "
            "direction and scope reasons can prevent overconfident fallback decisions."
        ),
        "n_review_required_sources": len(sources),
        "review_required_source_direction_counts": counts,
        "parent_scope_reason_counts": dict(sorted(reason_counts.items())),
        "counterweight_direction": counterweight,
    }


def _soft_source_item_for_llm(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_molecule": item.get("source_molecule"),
        "source_merge_key": item.get("source_merge_key"),
        "source_molecule_ids": item.get("source_molecule_ids"),
        "label_vote_bucket": item.get("label_vote_bucket"),
        "source_direction": item.get("source_direction"),
        "n_values": item.get("n_values"),
        "n_above_or_equal_20_percent": item.get("n_above_or_equal_20_percent"),
        "n_below_20_percent": item.get("n_below_20_percent"),
        "n_borderline_15_to_25_percent": item.get("n_borderline_15_to_25_percent"),
        "n_range_crosses_20_percent": item.get("n_range_crosses_20_percent"),
        "min_value_percent": item.get("min_value_percent"),
        "max_value_percent": item.get("max_value_percent"),
        "groups": item.get("groups"),
        "evidence_sources": item.get("evidence_sources"),
        "parent_applicabilities": item.get("parent_applicabilities"),
        "parent_scope_reasons": item.get("parent_scope_reasons"),
        "transferabilities": item.get("transferabilities"),
        "confidences": item.get("confidences"),
        "soft_context_policy": "context_only_not_direct_label_vote",
    }


def _final_reasoning_contract(
    *,
    no_clean_direct: bool,
    review_cards: list[dict[str, Any]],
    source_level_direct_f_summary: dict[str, Any],
) -> str:
    source_consensus = source_level_direct_f_summary.get("source_level_consensus")
    if source_consensus == "eligible_clean_sources_consensus_high":
        return (
            "De-correlated source-level clean direct-F evidence consistently supports high. Use this as the main "
            "direct-F signal unless Fa/Fg/Fh evidence gives a strong mechanistic contradiction."
        )
    if source_consensus == "eligible_clean_sources_consensus_high_with_context_conflict":
        return (
            "De-correlated eligible direct-F sources support high, but blocked/context sources conflict or straddle "
            "the threshold. Treat direct-F as supportive but not decisive; use Fa/Fg/Fh and scope checks as tie-breakers."
        )
    if source_consensus == "eligible_clean_sources_consensus_low":
        return (
            "De-correlated source-level clean direct-F evidence consistently supports low. Use this as the main "
            "direct-F signal unless factor evidence shows the source context is not transferable."
        )
    if source_consensus == "eligible_clean_sources_consensus_low_with_context_conflict":
        return (
            "De-correlated eligible direct-F sources support low, but blocked/context sources conflict or straddle "
            "the threshold. Treat direct-F as supportive but not decisive; use Fa/Fg/Fh and scope checks as tie-breakers."
        )
    if source_consensus in {"eligible_clean_sources_threshold_straddling", "eligible_clean_sources_mixed_high_low"}:
        return (
            "Clean direct-F source-level evidence is mixed or threshold-straddling. Do not let one high or low analog "
            "dominate; use Fa/Fg/Fh factors as tie-breakers and keep confidence low."
        )
    if no_clean_direct and review_cards:
        return (
            "No clean parent-like direct F label vote is available. Review-required direct-F values are audit/context "
            "only and must not be the main reason for the forced class. Use single-molecule prior and Fa/Fg/Fh factor "
            "evidence. Do not default to low solely because direct F is absent; choose low only when property/factor "
            "evidence shows absorption, gut escape, or hepatic escape is strongly limiting. Missing or mixed Fg/Fh is "
            "uncertainty, not evidence that those factors are non-limiting. Keep confidence low unless multiple factors "
            "are coherent."
        )
    if no_clean_direct:
        return (
            "No clean parent-like direct F label vote is available. Make only a factor/prior-based forced choice and "
            "do not default to low solely because direct F is absent. Choose low only when property/factor evidence "
            "shows absorption, gut escape, or hepatic escape is strongly limiting. Choose high only from coherent "
            "factor support, not from missing Fg/Fh. Keep confidence low unless Fa/Fg/Fh evidence is unusually coherent."
        )
    return (
        "Clean parent-like direct-F analog evidence exists. It may guide the label only after transferability, "
        "source de-correlation, threshold sensitivity, species, route, formulation, and analyte scope are checked."
    )


def _factor_product_guard(
    *,
    source_level_direct_f_summary: dict[str, Any],
    factor_cards: dict[str, list[dict[str, Any]]],
    single_content: dict[str, Any],
    context_flags: dict[str, Any],
) -> dict[str, Any]:
    """Structured guardrails for avoiding unsupported Fa*Fg*Fh arithmetic."""
    factor_signals = {
        factor: [str(card.get("signal") or "mixed_or_context") for card in cards]
        for factor, cards in factor_cards.items()
    }
    factor_strength = {
        factor: [_limiting_strength(card) for card in cards]
        for factor, cards in factor_cards.items()
    }
    class_constraints: dict[str, str] = {}
    audit_warnings: dict[str, str] = {}
    source_consensus = str(source_level_direct_f_summary.get("source_level_consensus") or "")
    source_counts = source_level_direct_f_summary.get("eligible_clean_source_direction_counts") or {}
    clean_context_counts = source_level_direct_f_summary.get("clean_context_source_direction_counts") or {}
    review_context_counts = source_level_direct_f_summary.get("review_context_source_direction_counts") or {}
    n_eligible = int(source_level_direct_f_summary.get("n_eligible_clean_direct_vote_sources") or 0)
    mixed_direct = source_consensus in {
        "eligible_clean_sources_mixed_high_low",
        "eligible_clean_sources_threshold_straddling",
        "eligible_clean_sources_consensus_high_with_context_conflict",
        "eligible_clean_sources_consensus_low_with_context_conflict",
    }
    high_sources = int(source_counts.get("supports_high_bioavailability") or 0)
    low_sources = int(source_counts.get("argues_against_high_bioavailability") or 0)
    soft_high_sources = int(clean_context_counts.get("supports_high_bioavailability") or 0)
    soft_low_sources = int(clean_context_counts.get("argues_against_high_bioavailability") or 0)
    soft_straddling_sources = int(clean_context_counts.get("mixed_threshold_straddling") or 0)
    review_low_sources = int(review_context_counts.get("argues_against_high_bioavailability") or 0)
    review_straddling_sources = int(review_context_counts.get("mixed_threshold_straddling") or 0)
    soft_low_or_straddling = soft_low_sources + soft_straddling_sources
    review_low_or_straddling = review_low_sources + review_straddling_sources
    soft_high_counterweight = soft_high_sources >= 3 and soft_high_sources >= 2 * max(1, soft_low_or_straddling)
    soft_low_context_present = soft_low_or_straddling > 0
    low_or_threshold_context_present = soft_low_or_straddling > 0 or review_low_or_straddling > 0

    has_fa_support = "supports_higher_F" in factor_signals.get("Fa", [])
    has_fg_support = "supports_higher_F" in factor_signals.get("Fg", [])
    has_fh_support = "supports_higher_F" in factor_signals.get("Fh", [])
    has_fa_risk = "risk_for_lower_F" in factor_signals.get("Fa", [])
    has_fh_risk = "risk_for_lower_F" in factor_signals.get("Fh", [])

    if mixed_direct and (low_sources >= high_sources) and has_fa_support and has_fh_risk:
        class_constraints["high"] = (
            "blocked_mixed_direct_without_explicit_fg_fh_bounds: mixed source-level direct-F evidence cannot be "
            "upgraded to high from favorable Fa when Fh is a risk and no explicit Fg/Fh lower-bound supports high."
        )

    low_property_prior = (
        single_content.get("oral_bioavailability_prior") == "low"
        and (
            single_content.get("absorption_prior") == "unfavorable"
            or single_content.get("solubility_or_dissolution_prior") == "unfavorable"
            or single_content.get("metabolism_or_clearance_prior") == "unfavorable"
        )
    )
    if (
        n_eligible == 0
        and low_property_prior
        and has_fa_support
        and has_fg_support
        and not has_fh_support
        and not soft_high_counterweight
        and low_or_threshold_context_present
    ):
        class_constraints["high"] = (
            "blocked_no_direct_low_property_prior_without_fh_support: when no label-admissible direct-F source "
            "exists, favorable Fa/Fg cannot force high if the property prior is low, Fh is unsupported, and "
            "soft/review direct-F context is low or threshold-sensitive."
        )
    if (
        n_eligible == 0
        and low_property_prior
        and has_fa_support
        and not has_fg_support
        and not has_fh_support
        and not soft_high_counterweight
        and (low_or_threshold_context_present or has_fh_risk)
    ):
        class_constraints["high"] = (
            "blocked_no_direct_fa_only_high_with_low_prior: when no label-admissible direct-F source exists, "
            "favorable Fa alone cannot force high if both Fg and Fh are missing or mixed, the property prior is low, "
            "and soft direct-F context or Fh risk does not counter that uncertainty."
        )
    if n_eligible == 0 and (has_fa_risk or single_content.get("absorption_prior") == "unfavorable") and not has_fa_support:
        audit_warnings["no_direct_absorption_risk_without_fa_support"] = (
            "Without label-admissible direct F, Fa or absorption risk weakens high. This is an audit warning, not "
            "a hard class block, because full-set validation showed this rule was too broad."
        )
    if (
        n_eligible == 0
        and not has_fa_support
        and not has_fg_support
        and has_fh_support
        and single_content.get("absorption_prior") == "unfavorable"
        and not soft_high_counterweight
        and low_or_threshold_context_present
    ):
        class_constraints["high"] = (
            "blocked_no_direct_escape_only_high: favorable hepatic escape alone cannot establish high oral F when "
            "Fa and Fg support are absent, absorption prior is unfavorable, and direct-F context is low or "
            "threshold-sensitive."
        )

    all_risks = [
        (factor, card, _limiting_strength(card))
        for factor, cards in factor_cards.items()
        for card in cards
        if card.get("signal") == "risk_for_lower_F"
    ]
    strong_risk_exists = any(strength == "strong_limiting" for _, _, strength in all_risks)
    absorption_limited_prior = (
        single_content.get("absorption_prior") == "unfavorable"
        or (
            single_content.get("oral_bioavailability_prior") == "low"
            and single_content.get("solubility_or_dissolution_prior") == "unfavorable"
        )
    )
    prodrug_fh_risk = "prodrug_or_active_metabolite" in context_flags and has_fh_risk
    prior_supports_low_metabolism = (
        single_content.get("oral_bioavailability_prior") == "low"
        or single_content.get("metabolism_or_clearance_prior") == "unfavorable"
    )
    if (
        n_eligible == 0
        and all_risks
        and not strong_risk_exists
        and not absorption_limited_prior
        and not prodrug_fh_risk
        and not prior_supports_low_metabolism
        and not soft_low_context_present
        and not review_low_or_straddling
    ):
        class_constraints["low"] = (
            "blocked_factor_only_moderate_analog_risks: without label-admissible direct F, low requires strong "
            "limiting evidence, a strong absorption-limited property prior, or prodrug/parent-Fh risk; moderate "
            "analog-only risks cannot be multiplied into F < 20%."
        )
    if (
        n_eligible == 0
        and not all_risks
        and not absorption_limited_prior
        and not prodrug_fh_risk
        and not soft_low_context_present
        and not review_low_or_straddling
    ):
        class_constraints["low"] = (
            "blocked_prior_only_metabolism_low: single-molecule metabolism/clearance prior is hypothesis-level "
            "evidence. Without label-admissible direct F or factor evidence showing a limiting Fa/Fg/Fh bottleneck, "
            "it cannot by itself force F < 20%."
        )

    return {
        "policy": (
            "Do not compute Fa*Fg*Fh with invented numeric bounds. Missing or mixed Fg/Fh is uncertainty, not "
            "evidence for high; single-molecule metabolism prior is not direct Fh evidence; moderate analog-only "
            "risk is uncertainty, not automatically evidence for low."
        ),
        "source_level_consensus": source_consensus,
        "eligible_direct_source_direction_counts": source_counts,
        "clean_context_source_direction_counts": clean_context_counts,
        "review_context_source_direction_counts_for_policy_only": review_context_counts,
        "soft_high_context_counterweight": soft_high_counterweight,
        "factor_signals": factor_signals,
        "factor_limiting_strength": factor_strength,
        "single_molecule_prior": {
            "oral_bioavailability_prior": single_content.get("oral_bioavailability_prior"),
            "absorption_prior": single_content.get("absorption_prior"),
            "solubility_or_dissolution_prior": single_content.get("solubility_or_dissolution_prior"),
            "metabolism_or_clearance_prior": single_content.get("metabolism_or_clearance_prior"),
        },
        "class_constraints": class_constraints,
        "audit_warnings": audit_warnings,
    }


def build_specific_decision_policy(compiled_or_gate: dict[str, Any]) -> dict[str, Any]:
    """Build a narrow deterministic policy state for final-stage use.

    This is deliberately not a full rule-based classifier. It identifies
    admissible strong/soft source states and narrow class vetoes so the final
    LLM can explain a bounded decision instead of freely converting weak
    metabolism priors into labels.
    """
    gate = compiled_or_gate.get("evidence_gate_summary") or compiled_or_gate
    source_summary = gate.get("source_level_direct_f_summary") or {}
    guard = gate.get("factor_product_guard") or {}
    constraints = guard.get("class_constraints") or {}
    soft_context = gate.get("soft_direct_f_context") or _soft_direct_f_context(source_summary)
    allowed_classes = ["high", "low"]
    reason_codes: list[str] = []
    uncertainty_flags: list[str] = []

    source_consensus = str(source_summary.get("source_level_consensus") or gate.get("source_level_direct_f_consensus") or "")
    eligible_counts = source_summary.get("eligible_clean_source_direction_counts") or {}
    eligible_high = int(eligible_counts.get("supports_high_bioavailability") or 0)
    eligible_low = int(eligible_counts.get("argues_against_high_bioavailability") or 0)
    eligible_straddling = int(eligible_counts.get("mixed_threshold_straddling") or 0)
    soft_counts = soft_context.get("soft_clean_source_direction_counts") or {}
    soft_high = int(soft_counts.get("supports_high_bioavailability") or 0)
    soft_low = int(soft_counts.get("argues_against_high_bioavailability") or 0)
    soft_straddling = int(soft_counts.get("mixed_threshold_straddling") or 0)
    net_soft = str(soft_context.get("net_soft_context_direction") or "")
    strengths = guard.get("factor_limiting_strength") or {}
    strong_limiting = any("strong_limiting" in list(values or []) for values in strengths.values())
    prior = guard.get("single_molecule_prior") or {}
    strong_absorption_prior = (
        prior.get("absorption_prior") == "unfavorable"
        and prior.get("solubility_or_dissolution_prior") == "unfavorable"
    )

    recommendation: str | None = None
    strength = "none"

    if source_consensus == "eligible_clean_sources_consensus_high" and "high" in allowed_classes:
        recommendation = "high"
        strength = "strong"
        reason_codes.append("eligible_clean_direct_f_consensus_high")
    elif source_consensus == "eligible_clean_sources_consensus_low" and "low" in allowed_classes:
        recommendation = "low"
        strength = "strong"
        reason_codes.append("eligible_clean_direct_f_consensus_low")
    elif eligible_high > eligible_low and eligible_high >= 2 and "high" in allowed_classes:
        recommendation = "high"
        strength = "moderate" if eligible_straddling else "strong"
        reason_codes.append("eligible_direct_f_sources_lean_high")
        if eligible_straddling:
            uncertainty_flags.append("eligible_direct_f_threshold_straddling")
    elif eligible_low > eligible_high and eligible_low >= 2 and "low" in allowed_classes:
        recommendation = "low"
        strength = "moderate" if eligible_straddling else "strong"
        reason_codes.append("eligible_direct_f_sources_lean_low")
        if eligible_straddling:
            uncertainty_flags.append("eligible_direct_f_threshold_straddling")
    elif net_soft == "soft_context_leans_high" and "high" in allowed_classes and not strong_limiting and not strong_absorption_prior:
        recommendation = "high"
        strength = "moderate"
        reason_codes.append("soft_clean_direct_f_context_leans_high_without_strong_limiting_evidence")
    elif (
        net_soft == "soft_context_leans_low"
        and "low" in allowed_classes
        and (strong_limiting or strong_absorption_prior)
    ):
        recommendation = "low"
        strength = "moderate"
        reason_codes.append("soft_clean_direct_f_context_leans_low_with_limiting_evidence")
    elif net_soft == "soft_context_threshold_sensitive_mixed":
        reason_codes.append("soft_clean_direct_f_context_threshold_sensitive_mixed")
        uncertainty_flags.append("soft_clean_direct_f_threshold_sensitive_mixed")

    if recommendation is None:
        if constraints:
            reason_codes.append("class_constraints_only_no_positive_recommendation")
        if soft_high or soft_low or soft_straddling:
            reason_codes.append("soft_direct_f_context_is_context_only")
        if source_consensus in {
            "no_eligible_clean_source_level_direct_f_vote",
            "eligible_clean_sources_mixed_high_low",
            "eligible_clean_sources_threshold_straddling",
        }:
            uncertainty_flags.append(source_consensus)

    if recommendation == "high" and strength == "strong":
        decision_state = "force_high"
        forced_class = "high"
    elif recommendation == "low" and strength == "strong":
        decision_state = "force_low"
        forced_class = "low"
    else:
        decision_state = "fallback_uncertain"
        forced_class = None

    return {
        "policy_version": "bioavailability_specific_v14_1_soft_threshold_counterweight_policy",
        "policy_role": (
            "Three-state admissibility policy for final synthesis. force_high/force_low are bounded deterministic "
            "states. fallback_uncertain means evidence is insufficient for a deterministic label, so diagnostic "
            "blocks must not be treated as label evidence."
        ),
        "decision_state": decision_state,
        "forced_class": forced_class,
        "fallback_candidate_class": recommendation if decision_state == "fallback_uncertain" else None,
        "allowed_classes": allowed_classes,
        "diagnostic_blocked_classes": constraints,
        "recommended_class": recommendation,
        "recommendation_strength": strength,
        "reason_codes": reason_codes,
        "uncertainty_flags": uncertainty_flags,
        "soft_direct_f_context": soft_context,
        "source_level_consensus": source_consensus,
        "eligible_direct_source_direction_counts": eligible_counts,
    }


def _limiting_strength(card: dict[str, Any]) -> str:
    if card.get("signal") != "risk_for_lower_F":
        return "not_limiting"
    if card.get("transferability") == "high" and card.get("confidence") in {"high", "moderate"}:
        return "strong_limiting"
    if card.get("transferability") == "moderate" and card.get("confidence") in {"high", "moderate"}:
        return "moderate_analog_risk"
    return "weak_context"


def _assign_factors(card: dict[str, Any]) -> list[str]:
    group_id = str(card.get("group_id") or "")
    if group_id.startswith("Fa."):
        return ["Fa"]
    if group_id.startswith("Fg."):
        return ["Fg"]
    if group_id.startswith("Fh."):
        return ["Fh"]
    role = str(card.get("endpoint_role") or "")
    text = _card_text(card)
    factors: list[str] = []
    if role in FA_ROLES:
        factors.append("Fa")
    if role == "transporter_or_efflux":
        factors.extend(["Fa", "Fg"])
    if role in {"clearance_or_first_pass", "metabolic_stability"}:
        if _has_any(text, FG_HINTS):
            factors.append("Fg")
        if _has_any(text, FH_HINTS):
            factors.append("Fh")
        if not factors:
            factors.append("Fh")
    if role == "food_or_formulation_context":
        if _has_any(text, {"dissolution", "solubility", "food", "fed", "fasted"}):
            factors.append("Fa")
        if _has_any(text, {"gut", "intestinal", "cyp3a", "p-gp", "efflux"}):
            factors.append("Fg")
    return _dedupe_keep_order(factors)


def _source_scope_context_map(retrieval: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Build source-level parent/analyte/scope assessments from raw retrieval rows."""
    group_map: dict[str, dict[str, dict[str, Any]]] = {}
    for group in retrieval.get("groups") or []:
        group_id = str(group.get("group_id") or "")
        if not group_id:
            continue
        group_context = group_map.setdefault(group_id, {})
        for neighbor in group.get("neighbors") or []:
            for row in neighbor.get("evidence_rows") or []:
                source_key = _source_merge_key_from_row(row, neighbor)
                raw_source_key = _source_molecule_key_from_row(row, neighbor)
                if not source_key and not raw_source_key:
                    continue
                assessment = _row_parent_scope_assessment(row, group, neighbor)
                for key in {source_key, raw_source_key}:
                    if not key:
                        continue
                    context = group_context.setdefault(
                        key,
                        {
                            "source_molecule": key,
                            "source_merge_key": source_key or key,
                            "parent_applicability": "clean_parent_f",
                            "scope_review_reasons": set(),
                            "source_transfer_scope_classes": set(),
                            "source_report_types": set(),
                            "source_species_or_populations": set(),
                            "source_oral_exposure_modes": set(),
                            "source_condition_contexts": set(),
                            "n_scope_context_rows": 0,
                        },
                    )
                    context["n_scope_context_rows"] += 1
                    context["parent_applicability"] = _merge_parent_applicability(
                        str(context["parent_applicability"]),
                        assessment["parent_applicability"],
                    )
                    for field in (
                        "scope_review_reasons",
                        "source_transfer_scope_classes",
                        "source_report_types",
                        "source_species_or_populations",
                        "source_oral_exposure_modes",
                        "source_condition_contexts",
                    ):
                        context[field].update(assessment.get(field) or [])

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for group_id, group_context in group_map.items():
        normalized[group_id] = {}
        for key, context in group_context.items():
            normalized[group_id][key] = {
                **context,
                "scope_review_reasons": sorted(context["scope_review_reasons"]),
                "source_transfer_scope_classes": sorted(context["source_transfer_scope_classes"]),
                "source_report_types": sorted(context["source_report_types"])[:8],
                "source_species_or_populations": sorted(context["source_species_or_populations"])[:8],
                "source_oral_exposure_modes": sorted(context["source_oral_exposure_modes"])[:8],
                "source_condition_contexts": sorted(context["source_condition_contexts"])[:8],
            }
    return normalized


def _source_scope_context_summary(source_scope_context: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    unique_by_group_and_source: dict[tuple[str, str], dict[str, Any]] = {}
    for group_id, group_context in source_scope_context.items():
        if "direct" not in group_id.lower() and "Observed." not in group_id:
            continue
        for source_key, context in group_context.items():
            merge_key = str(context.get("source_merge_key") or source_key)
            unique_by_group_and_source.setdefault((group_id, merge_key), context)
    contexts = list(unique_by_group_and_source.values())
    return {
        "n_group_source_scope_contexts": len(unique_by_group_and_source),
        "summary_scope": "direct_f_groups_merge_key_deduplicated",
        "parent_applicability_counts": dict(
            sorted(Counter(str(context.get("parent_applicability") or "") for context in contexts).items())
        ),
        "scope_review_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for context in contexts
                    for reason in context.get("scope_review_reasons") or []
                ).items()
            )
        ),
        "source_transfer_scope_class_counts": dict(
            sorted(
                Counter(
                    scope_class
                    for context in contexts
                    for scope_class in context.get("source_transfer_scope_classes") or []
                ).items()
            )
        ),
    }


def _row_parent_scope_assessment(
    row: dict[str, Any],
    group: dict[str, Any],
    neighbor: dict[str, Any],
) -> dict[str, Any]:
    text = _row_scope_text(
        row,
        group,
        neighbor,
        include_qualitative_examples=not bool(row.get("source_record_examples")),
        include_aggregate_context=not bool(row.get("source_record_examples")),
    )
    reasons: list[str] = []
    if _requires_parent_analyte_review(text) or _has_any(text, TOTAL_RADIOACTIVITY_HINTS):
        reasons.append("active_metabolite_prodrug_or_total_radioactivity_scope")
    if _has_any(text, NON_SWALLOWED_ROUTE_HINTS):
        reasons.append("non_swallowed_or_nonstandard_route_scope")
    if _has_any(text, RELATIVE_OR_PROXY_SCOPE_HINTS) and not _has_any(text, {"absolute oral bioavailability", "absolute bioavailability"}):
        reasons.append("relative_or_exposure_proxy_scope")
    if _has_any(text, FORMULATION_REVIEW_HINTS):
        reasons.append("formulation_or_salt_specific_scope")

    if any(reason.startswith("active_metabolite") for reason in reasons):
        parent_applicability = "requires_parent_analyte_review"
    elif reasons:
        parent_applicability = "requires_context_scope_review"
    else:
        parent_applicability = "clean_parent_f"

    return {
        "parent_applicability": parent_applicability,
        "scope_review_reasons": reasons,
        "source_transfer_scope_classes": _source_transfer_scope_classes(text, reasons),
        "source_report_types": _row_example_values(row, "bioavailability_report_type"),
        "source_species_or_populations": _row_example_values(row, "species_or_population"),
        "source_oral_exposure_modes": _row_example_values(row, "oral_exposure_mode"),
        "source_condition_contexts": _row_condition_contexts(row),
    }


def _source_transfer_scope_classes(text: str, reasons: list[str]) -> set[str]:
    classes: set[str] = set()
    if "active_metabolite_prodrug_or_total_radioactivity_scope" in reasons:
        if _has_any(text, TOTAL_RADIOACTIVITY_HINTS):
            classes.add("total_radioactivity_scope")
        else:
            classes.add("active_metabolite_or_prodrug_scope")
    if "non_swallowed_or_nonstandard_route_scope" in reasons:
        classes.add("non_swallowed_route_context")
    if "formulation_or_salt_specific_scope" in reasons:
        if _has_any(text, SPECIAL_FORMULATION_REVIEW_HINTS):
            classes.add("special_formulation_or_route_context")
        if _has_any(text, {"salt form", "free-base", "free base"}):
            classes.add("same_active_moiety_salt_or_freebase_context")
    if "relative_or_exposure_proxy_scope" in reasons:
        classes.add("relative_or_proxy_context")
    if not classes:
        classes.add("clean_parent_or_close_analog_context")
    return classes


def _row_scope_text(
    row: dict[str, Any],
    group: dict[str, Any],
    neighbor: dict[str, Any],
    *,
    include_qualitative_examples: bool = True,
    include_aggregate_context: bool = True,
) -> str:
    parts = [
        group.get("group_id"),
        group.get("endpoint_group"),
        group.get("specific_group_role"),
        row.get("specific_evidence_role"),
        row.get("standard_type"),
        row.get("standard_units"),
        row.get("activity_comment"),
        row.get("organism"),
        row.get("target_pref_name"),
        neighbor.get("molecule_chembl_id"),
    ]
    if include_aggregate_context:
        parts.extend(
            [
                row.get("assay_description"),
                row.get("source_report_types"),
                row.get("source_support_texts"),
            ]
        )
    examples = list(row.get("source_record_examples") or [])
    if include_qualitative_examples:
        examples.extend(row.get("source_qualitative_examples") or [])
    for example in examples:
        if not isinstance(example, dict):
            continue
        parts.extend(
            [
                example.get("molecule_name"),
                example.get("bioavailability_report_type"),
                example.get("oral_exposure_mode"),
                example.get("condition_text"),
                example.get("qualifying_conditions"),
                example.get("comparator"),
                example.get("extra_details"),
                example.get("support_text"),
                example.get("oral_bioavailability_value_text"),
            ]
        )
    return " ".join(str(part or "") for part in parts).lower()


def _row_example_values(row: dict[str, Any], key: str) -> set[str]:
    values: set[str] = set()
    report_types = row.get("source_report_types") or []
    if isinstance(report_types, str):
        report_types = [report_types]
    for value in report_types:
        if key == "bioavailability_report_type" and str(value or "").strip():
            values.add(str(value).strip())
    for example in (row.get("source_record_examples") or []) + (row.get("source_qualitative_examples") or []):
        if isinstance(example, dict) and str(example.get(key) or "").strip():
            values.add(str(example.get(key)).strip())
    return values


def _row_condition_contexts(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for example in (row.get("source_record_examples") or []) + (row.get("source_qualitative_examples") or []):
        if not isinstance(example, dict):
            continue
        text = " | ".join(
            str(example.get(key) or "").strip()
            for key in ("qualifying_conditions", "comparator", "extra_details")
            if str(example.get(key) or "").strip()
        )
        if text:
            values.add(text[:240])
    return values


def _source_molecule_key_from_row(row: dict[str, Any], neighbor: dict[str, Any]) -> str:
    for key in ("combined_merge_key", "source_molecule_id", "molecule_chembl_id", "standard_inchi_key", "canonical_smiles"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(neighbor.get("molecule_chembl_id") or neighbor.get("canonical_smiles") or "").strip()


def _source_merge_key_from_row(row: dict[str, Any], neighbor: dict[str, Any]) -> str:
    for key in ("standard_inchi_key", "canonical_smiles", "combined_merge_key", "source_molecule_id", "molecule_chembl_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(neighbor.get("standard_inchi_key") or neighbor.get("canonical_smiles") or neighbor.get("molecule_chembl_id") or "").strip()


def _merge_parent_applicability(left: str, right: str) -> str:
    rank = {
        "clean_parent_f": 0,
        "requires_context_scope_review": 1,
        "requires_parent_analyte_review": 2,
    }
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _direct_parent_f_card(card: dict[str, Any]) -> dict[str, Any]:
    text = _card_text(card)
    parent_applicability = "clean_parent_f"
    if _requires_parent_analyte_review(text):
        parent_applicability = "requires_parent_analyte_review"
    context_flags = _flags_for_text(text)
    return {
        "group_id": card.get("group_id"),
        "endpoint_role": card.get("endpoint_role"),
        "parent_applicability": parent_applicability,
        "transferability": card.get("transferability"),
        "confidence": card.get("confidence"),
        "llm_evidence_direction": card.get("llm_evidence_direction"),
        "factor_effect": card.get("factor_effect") or "",
        "compiler_threshold_direction": (card.get("threshold_summary") or {}).get("compiler_threshold_direction"),
        "threshold_summary": card.get("threshold_summary"),
        "threshold_evidence": (card.get("threshold_evidence") or [])[:6],
        "reasoning_summary": card.get("reasoning_summary") or "",
        "caveats": card.get("caveats") or [],
        "context_flags": context_flags,
    }


def _split_direct_evidence(direct_cards: list[dict[str, Any]]) -> dict[str, Any]:
    clean = []
    review = []
    for card in direct_cards:
        if card.get("parent_applicability") == "clean_parent_f":
            clean.append(card)
        else:
            review.append(card)
    return {
        "note": (
            "These are direct-F endpoint measurements on retrieved source molecules. Because exact-query context is "
            "disabled for the benchmark, they should be treated as analog direct-F evidence and transferred only "
            "after structure, species, route, formulation, and parent/analyte scope checks."
        ),
        "clean_parent_like_analog_f": clean,
        "parent_analyte_review_required": review,
    }


def _factor_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": card.get("group_id"),
        "endpoint_role": card.get("endpoint_role"),
        "bioavailability_factor": card.get("bioavailability_factor"),
        "transferability": card.get("transferability"),
        "confidence": card.get("confidence"),
        "signal": _signal_from_direction(card),
        "llm_evidence_direction": card.get("llm_evidence_direction"),
        "factor_effect": card.get("factor_effect") or "",
        "reasoning_summary": card.get("reasoning_summary") or "",
        "caveats": card.get("caveats") or [],
        "context_flags": _flags_for_text(_card_text(card)),
    }


def _summarize_factor(cards: list[dict[str, Any]]) -> dict[str, Any]:
    signals = Counter(str(card.get("signal") or "unclear") for card in cards)
    transferable = [
        {
            "group_id": card.get("group_id"),
            "signal": card.get("signal"),
            "transferability": card.get("transferability"),
            "confidence": card.get("confidence"),
        }
        for card in cards
        if card.get("transferability") in {"high", "moderate"} or card.get("confidence") in {"high", "moderate"}
    ]
    return {
        "n_cards": len(cards),
        "signal_counts": dict(sorted(signals.items())),
        "strongest_cards": transferable[:8],
    }


def _context_flags(group_cards: list[dict[str, Any]]) -> dict[str, Any]:
    flag_to_groups: dict[str, list[str]] = {
        "prodrug_or_active_metabolite": [],
        "species_translation": [],
        "dose_or_steady_state": [],
        "food_formulation_salt_or_route": [],
        "threshold_sensitive_15_25_percent": [],
        "first_pass_or_clearance_mechanism": [],
    }
    for card in group_cards:
        group_id = str(card.get("group_id") or "")
        text = _card_text(card)
        flags = _flags_for_text(text)
        for flag in flags:
            if flag in flag_to_groups:
                flag_to_groups[flag].append(group_id)
        summary = card.get("threshold_summary") or {}
        if summary.get("n_borderline_15_to_25_percent"):
            flag_to_groups["threshold_sensitive_15_25_percent"].append(group_id)
        if card.get("endpoint_role") in {"clearance_or_first_pass", "metabolic_stability"} or _has_any(text, FH_HINTS):
            flag_to_groups["first_pass_or_clearance_mechanism"].append(group_id)
    return {
        flag: sorted(set(groups))[:20]
        for flag, groups in flag_to_groups.items()
        if groups
    }


def _flags_for_text(text: str) -> list[str]:
    flags = []
    if _requires_parent_analyte_review(text):
        flags.append("prodrug_or_active_metabolite")
    if _has_any(text, SPECIES_HINTS):
        flags.append("species_translation")
    if _has_any(text, DOSE_CONTEXT_HINTS):
        flags.append("dose_or_steady_state")
    if _has_any(text, FORMULATION_HINTS):
        flags.append("food_formulation_salt_or_route")
    if _has_any(text, FH_HINTS | FG_HINTS):
        flags.append("first_pass_or_clearance_mechanism")
    return flags


def _signal_from_direction(card: dict[str, Any]) -> str:
    factor_effect = str(card.get("factor_effect") or "").strip().lower()
    if factor_effect in {"supports_higher_f", "supports_higher_F".lower()}:
        return "supports_higher_F"
    if factor_effect == "risk_for_lower_f":
        return "risk_for_lower_F"
    if factor_effect in {"mixed_or_context", "neutral_or_unclear"}:
        return "mixed_or_context"
    evidence_direction = str(card.get("llm_evidence_direction") or "").strip().lower()
    if evidence_direction in {
        "supports_high_bioavailability",
        "absorption_support",
        "permeability_support",
        "solubility_support",
        "metabolic_stability_support",
    }:
        return "supports_higher_F"
    if evidence_direction in {
        "argues_against_high_bioavailability",
        "solubility_risk",
        "first_pass_or_clearance_risk",
        "transporter_efflux_risk",
    }:
        return "risk_for_lower_F"
    if evidence_direction == "neutral_or_unclear":
        return "mixed_or_context"
    text = " ".join(
        [
            str(card.get("llm_evidence_direction") or ""),
            str(card.get("reasoning_summary") or ""),
            " ".join(str(item) for item in (card.get("caveats") or [])),
        ]
    ).lower()
    low_terms = ("argues_against", "against high", "low bioavailability", "poor", "decrease", "reduce", "risk")
    high_terms = ("supports_high", "support high", "high bioavailability", "favorable", "good", "increase")
    if any(term in text for term in low_terms):
        return "risk_for_lower_F"
    if any(term in text for term in high_terms):
        return "supports_higher_F"
    return "mixed_or_context"


def _requires_parent_analyte_review(text: str) -> bool:
    return (
        _has_any(text, PRODRUG_ANALYTE_HINTS)
        or _has_any(text, SCOPE_MISMATCH_HINTS)
        or _has_parent_analyte_phrase(text)
    )


def _has_parent_analyte_phrase(text: str) -> bool:
    patterns = [
        r"\bactive\s+metabolites?\b",
        r"\bactive\s+moiet(?:y|ies)\b",
        r"\bmeasured\s+metabolites?\b",
        r"\bmetabolite\s+after\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _direction_counts(cards: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(card.get("compiler_threshold_direction") or "") for card in cards).items()))


def _card_text(card: dict[str, Any]) -> str:
    parts = [
        card.get("group_id"),
        card.get("endpoint_role"),
        card.get("bioavailability_factor"),
        card.get("llm_evidence_direction"),
        card.get("factor_effect"),
        card.get("reasoning_summary"),
        " ".join(str(item) for item in (card.get("caveats") or [])),
        " ".join(str(item) for item in (card.get("key_evidence") or [])),
    ]
    return " ".join(str(part or "") for part in parts).lower()


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
