"""Source-agnostic evidence compiler for Bioavailability_Ma final reasoning.

The compiler is intentionally conservative: it does not make the final label,
but it converts heterogeneous retrieval/group outputs into role-aware evidence
cards so the final LLM sees direct-F evidence, proxy evidence, transferability
gates, and duplicate-source warnings separately.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from tools.chembl_tool.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT


BORDERLINE_LOW_PERCENT = 15.0
BORDERLINE_HIGH_PERCENT = 25.0

DIRECT_ROLES = {"direct_oral_f"}
CONTEXT_DIRECT_ROLES = {"direct_oral_f_context", "oral_exposure_ratio"}
TRANSFERABLE = {"high", "moderate"}


def compile_final_evidence(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile retrieval and group-level LLM outputs into final-stage evidence cards."""
    retrieval_groups = {str(group.get("group_id") or ""): group for group in retrieval.get("groups", [])}
    output_by_group = {str(item.get("group_id") or ""): item for item in group_outputs}
    group_ids = sorted(set(retrieval_groups) | set(output_by_group))

    group_cards = [
        _compile_group_card(group_id, retrieval_groups.get(group_id) or {}, output_by_group.get(group_id) or {})
        for group_id in group_ids
    ]
    group_cards = [card for card in group_cards if card["n_neighbors"] or card["group_status"] != "missing"]

    direct_votes = [
        card
        for card in group_cards
        if card["direct_label_vote_allowed"]
    ]
    transferability_blocked = [
        {
            "group_id": card["group_id"],
            "endpoint_role": card["endpoint_role"],
            "transferability": card["transferability"],
            "confidence": card["confidence"],
            "reason": card["direct_vote_block_reason"],
            "threshold_evidence": card["threshold_evidence"],
            "threshold_summary": card["threshold_summary"],
            "llm_evidence_direction": card["llm_evidence_direction"],
        }
        for card in group_cards
        if card["threshold_evidence"] and not card["direct_label_vote_allowed"]
    ]
    proxy_cards = [
        card
        for card in group_cards
        if card["endpoint_role"] not in DIRECT_ROLES and card["useful_for_bioavailability_reasoning"]
    ]
    duplicate_clusters = _duplicate_source_summary(group_cards)

    return {
        "compiler_version": "bioavailability_final_evidence_v2",
        "task_threshold": {
            "high_label_rule": f"oral bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}%",
            "low_label_rule": f"oral bioavailability F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}%",
            "borderline_zone_for_uncertainty": f"{BORDERLINE_LOW_PERCENT:g}-{BORDERLINE_HIGH_PERCENT:g}%",
        },
        "source_policy": (
            "Evidence is compiled by endpoint role, transferability, and unique source molecule. "
            "No rule depends on whether the retrieval provider is ChEMBL, Starling, or another source."
        ),
        "endpoint_hierarchy": [
            "1. Transferable direct oral F% evidence from same/close analogs.",
            "2. Context-qualified oral/IV ratio or direct-F-like evidence.",
            "3. Oral exposure/absorption evidence interpreted as Fa/Fg/Fh context, not absolute F.",
            "4. Permeability, solubility, dissolution, transporter, clearance, metabolism, food, and formulation proxies.",
            "5. Single-molecule physicochemical prior.",
        ],
        "single_molecule_prior": {
            "status": single_output.get("status"),
            "content": (single_output.get("llm") or {}).get("content"),
        },
        "retrieval_coverage": retrieval.get("coverage") or {},
        "group_cards": group_cards,
        "direct_label_votes": [_slim_vote_card(card) for card in direct_votes],
        "transferability_blocked_direct_or_numeric_evidence": transferability_blocked,
        "proxy_evidence_cards": [_slim_proxy_card(card) for card in proxy_cards],
        "duplicate_or_correlated_source_summary": duplicate_clusters,
        "factor_summary": _factor_summary(group_cards),
        "llm_decision_view": _llm_decision_view(
            retrieval=retrieval,
            single_output=single_output,
            group_cards=group_cards,
            direct_votes=direct_votes,
            proxy_cards=proxy_cards,
            duplicate_clusters=duplicate_clusters,
        ),
        "decision_guardrails": [
            "Only direct oral F% or clearly equivalent oral/IV exposure-ratio evidence can directly vote high/low.",
            "A numeric F-like value with low/not_applicable transferability must not be used as a direct label vote.",
            "AUC, Cmax, Papp, solubility, dissolution, transporter, and clearance evidence are mechanism modifiers unless the assay explicitly measures F%.",
            "Repeated evidence from the same source molecule, document, assay family, or merged record should be counted as correlated support, not independent votes.",
            "Prodrug, active metabolite, salt, formulation, route, species, and food-state mismatches should lower transferability unless resolved by the group analysis.",
            "For values near the 20% label threshold, express uncertainty rather than over-weighting small numeric differences.",
        ],
    }


def _llm_decision_view(
    *,
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_cards: list[dict[str, Any]],
    direct_votes: list[dict[str, Any]],
    proxy_cards: list[dict[str, Any]],
    duplicate_clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    blocked_cards = [card for card in group_cards if card["threshold_evidence"] and not card["direct_label_vote_allowed"]]
    no_direct_vote = not direct_votes
    direct_direction_counts = Counter(card["threshold_summary"]["compiler_threshold_direction"] for card in direct_votes)
    return {
        "compiler_version": "bioavailability_final_decision_view_v2",
        "task_threshold": {
            "high_label_rule": f"oral bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}%",
            "low_label_rule": f"oral bioavailability F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}%",
            "borderline_zone_for_uncertainty": f"{BORDERLINE_LOW_PERCENT:g}-{BORDERLINE_HIGH_PERCENT:g}%",
        },
        "retrieval_coverage": retrieval.get("coverage") or {},
        "single_molecule_prior": {
            "status": single_output.get("status"),
            "content": (single_output.get("llm") or {}).get("content"),
        },
        "evidence_state": {
            "n_direct_label_vote_groups": len(direct_votes),
            "direct_vote_threshold_direction_counts": dict(sorted(direct_direction_counts.items())),
            "n_blocked_numeric_groups": len(blocked_cards),
            "blocked_numeric_role_counts": dict(Counter(card["endpoint_role"] for card in blocked_cards)),
            "n_useful_proxy_groups": len(proxy_cards),
            "no_direct_label_vote": no_direct_vote,
        },
        "direct_label_votes": [_slim_vote_card(card) for card in direct_votes],
        "blocked_numeric_evidence_policy": {
            "rule": "Blocked numeric evidence failed endpoint-role, transferability, usefulness, or confidence gates. Treat it as uncertainty/context only, not directional support.",
            "blocked_group_summaries": [_blocked_card_for_llm(card) for card in blocked_cards],
        },
        "proxy_evidence_policy": {
            "rule": "Proxy endpoints can explain mechanisms for Fa/Fg/Fh but cannot by themselves prove F >= 20% or F < 20%.",
            "proxy_cards": [_slim_proxy_card(card) for card in proxy_cards],
        },
        "duplicate_or_correlated_source_summary": duplicate_clusters,
        "decision_rules": [
            "Allowed direct_label_votes are the only evidence cards permitted to vote directly on the 20% F label.",
            "If allowed direct F votes all support high, do not predict low solely from solubility, permeability, or metabolism proxies; lower confidence instead unless direct low evidence is also allowed.",
            "If allowed direct F votes all support low, do not predict high solely from Fa, AUC, Cmax, permeability, or favorable physicochemical prior; lower confidence instead unless direct high evidence is also allowed.",
            "If direct votes are mixed, balanced, or borderline, state that the evidence is threshold-sensitive and use proxies only as tie-breakers with reduced confidence.",
            "If no direct label vote exists, do not claim the 20% threshold is experimentally established. Make the forced high/low choice from the best available prior/proxy evidence with low confidence unless multiple transferable in vivo proxy groups are coherent.",
            "Blocked direct-F values must not be used as weak positive or weak negative trends in the final label. They can only explain uncertainty and evidence gaps.",
        ],
    }


def _blocked_card_for_llm(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": card["group_id"],
        "endpoint_role": card["endpoint_role"],
        "bioavailability_factor": card["bioavailability_factor"],
        "transferability": card["transferability"],
        "confidence": card["confidence"],
        "block_reason": card["direct_vote_block_reason"],
        "n_numeric_values": card["threshold_summary"]["n_numeric_values"],
        "n_source_molecules_with_numeric_values": card["threshold_summary"].get("n_source_molecules_with_numeric_values"),
        "n_borderline_15_to_25_percent": card["threshold_summary"].get("n_borderline_15_to_25_percent"),
        "reasoning_summary": card["reasoning_summary"],
        "caveats": card["caveats"],
        "redaction_note": (
            "Numeric threshold evidence and compiler threshold directions are omitted because this group failed the "
            "direct-label gate. The group-level qualitative summary is retained as context only, not as a direct "
            "high/low label vote."
        ),
    }


def _compile_group_card(
    group_id: str,
    retrieval_group: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    llm_content = (output.get("llm") or {}).get("content") if output else None
    if not isinstance(llm_content, dict):
        llm_content = {}

    tier = str(retrieval_group.get("tier") or output.get("tier") or _tier_from_group_id(group_id))
    endpoint_group = str(retrieval_group.get("endpoint_group") or output.get("endpoint_group") or _endpoint_from_group_id(group_id))
    rows = _all_rows(retrieval_group)
    role = _endpoint_role(group_id, endpoint_group, rows)
    threshold_evidence = _threshold_evidence(rows)
    threshold_summary = _threshold_summary(threshold_evidence)
    transferability = _norm_lower(llm_content.get("transferability") or "not_applicable")
    confidence = _norm_lower(llm_content.get("confidence") or "low")
    useful = bool(llm_content.get("useful_for_bioavailability_reasoning"))
    evidence_direction = _norm_lower(llm_content.get("evidence_direction") or "neutral_or_unclear")
    n_unique, top_neighbors, source_keys = _neighbor_summary(retrieval_group)
    direct_allowed, block_reason = _direct_vote_gate(
        role=role,
        threshold_evidence=threshold_evidence,
        transferability=transferability,
        confidence=confidence,
        useful=useful,
    )
    return {
        "group_id": group_id,
        "tier": tier,
        "endpoint_group": endpoint_group,
        "endpoint_role": role,
        "bioavailability_factor": _factor_for_role(role),
        "group_status": output.get("status") or ("missing" if not output else ""),
        "n_neighbors": len(retrieval_group.get("neighbors") or []),
        "n_unique_source_molecules": n_unique,
        "top_neighbors": top_neighbors,
        "source_keys": source_keys,
        "threshold_evidence": threshold_evidence,
        "threshold_summary": threshold_summary,
        "useful_for_bioavailability_reasoning": useful,
        "transferability": transferability,
        "confidence": confidence,
        "llm_evidence_direction": evidence_direction,
        "direct_label_vote_allowed": direct_allowed,
        "direct_vote_block_reason": block_reason,
        "reasoning_summary": llm_content.get("reasoning_summary") or "",
        "key_evidence": llm_content.get("key_evidence") or [],
        "caveats": llm_content.get("caveats") or [],
    }


def _direct_vote_gate(
    *,
    role: str,
    threshold_evidence: list[dict[str, Any]],
    transferability: str,
    confidence: str,
    useful: bool,
) -> tuple[bool, str]:
    if role not in DIRECT_ROLES:
        return False, "endpoint is not direct absolute oral bioavailability"
    if not threshold_evidence:
        return False, "no interpretable direct F% threshold evidence"
    if not useful:
        return False, "group LLM did not mark evidence useful"
    if transferability not in TRANSFERABLE:
        return False, "transferability gate blocks direct label vote"
    if confidence == "low":
        return False, "low confidence blocks direct label vote"
    return True, ""


def _endpoint_role(group_id: str, endpoint_group: str, rows: list[dict[str, Any]]) -> str:
    text = " ".join(
        [
            group_id,
            endpoint_group,
            " ".join(str(row.get("standard_type") or "") for row in rows[:5]),
            " ".join(str(row.get("assay_description") or "") for row in rows[:3]),
        ]
    ).lower()
    if "direct_absolute_bioavailability" in text:
        return "direct_oral_f"
    if "oral bioavailability" in text or "absolute bioavailability" in text or "bioavailability (%)" in text:
        return "direct_oral_f"
    if "direct_oral_iv_exposure_ratio" in text:
        return "oral_exposure_ratio"
    if "context_dependent" in text and "bioavailability" in text:
        return "direct_oral_f_context"
    if "oral_auc_exposure" in text or "oral_cmax_exposure" in text or "auc" in text or "cmax" in text:
        return "oral_exposure"
    if "absorption_fraction_or_hia" in text or "fraction absorbed" in text or "human intestinal absorption" in text:
        return "absorption_fa"
    if "intestinal_permeability" in text or "intestinal_uptake" in text:
        return "in_vivo_absorption_or_permeability"
    if "pampa" in text or "permeability_papp" in text or "papp" in text:
        return "in_vitro_permeability"
    if "efflux" in text or "transporter" in text:
        return "transporter_or_efflux"
    if "solubility" in text or "dissolution" in text:
        return "solubility_or_dissolution"
    if "clearance" in text or "first_pass" in text or "extraction" in text:
        return "clearance_or_first_pass"
    if "metabolic_stability" in text or "microsomal" in text or "hepatocyte" in text:
        return "metabolic_stability"
    if "food_effect" in text or "formulation" in text or "relative_bioavailability" in text:
        return "food_or_formulation_context"
    return "context_or_other"


def _factor_for_role(role: str) -> str:
    if role in DIRECT_ROLES:
        return "direct_F"
    if role in CONTEXT_DIRECT_ROLES or role == "oral_exposure":
        return "contextual_F_or_exposure"
    if role in {"absorption_fa", "in_vivo_absorption_or_permeability", "in_vitro_permeability", "transporter_or_efflux"}:
        return "Fa_absorption_permeability_efflux"
    if role == "solubility_or_dissolution":
        return "Fa_solubility_dissolution"
    if role in {"clearance_or_first_pass", "metabolic_stability"}:
        return "Fg_Fh_first_pass_clearance"
    if role == "food_or_formulation_context":
        return "formulation_or_food_context"
    return "context"


def _threshold_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in rows:
        value = _best_percent_value(row)
        if value is None:
            continue
        relation = str(row.get("standard_relation") or row.get("parse_modifier") or "=").strip() or "="
        threshold_relation = _threshold_relation(value, relation)
        cards.append(
            {
                "source_molecule": _source_molecule_key_from_row(row),
                "assay_or_source_id": row.get("assay_chembl_id") or row.get("source_group_id") or "",
                "standard_type": row.get("standard_type") or "",
                "relation": relation,
                "value_percent": round(value, 3),
                "threshold_relation": threshold_relation,
                "borderline_to_20_percent": BORDERLINE_LOW_PERCENT <= value <= BORDERLINE_HIGH_PERCENT,
                "evidence_source": row.get("evidence_source") or "",
            }
        )
    return cards[:12]


def _threshold_summary(cards: list[dict[str, Any]]) -> dict[str, Any]:
    row_counts = Counter(card["threshold_relation"] for card in cards)
    source_to_cards: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, card in enumerate(cards):
        source_key = str(card.get("source_molecule") or f"row:{i}")
        source_to_cards[source_key].append(card)
    source_directions = Counter(_source_threshold_direction(source_cards) for source_cards in source_to_cards.values())

    n_above = row_counts.get("above_or_equal_high_threshold", 0)
    n_below = row_counts.get("below_high_threshold", 0)
    if not cards:
        direction = "no_numeric_threshold_evidence"
    elif n_above and not n_below:
        direction = "supports_high_bioavailability"
    elif n_below and not n_above:
        direction = "argues_against_high_bioavailability"
    elif n_above > n_below:
        direction = "mixed_numeric_evidence_lean_high"
    elif n_below > n_above:
        direction = "mixed_numeric_evidence_lean_low"
    else:
        direction = "mixed_numeric_evidence_balanced"
    values = [card["value_percent"] for card in cards if isinstance(card.get("value_percent"), (int, float))]
    return {
        "n_numeric_values": len(cards),
        "n_above_or_equal_20_percent_rows": row_counts.get("above_or_equal_high_threshold", 0),
        "n_below_20_percent_rows": row_counts.get("below_high_threshold", 0),
        "n_source_molecules_with_numeric_values": len(source_to_cards),
        "source_molecule_direction_counts": dict(sorted(source_directions.items())),
        "n_borderline_15_to_25_percent": sum(1 for card in cards if card.get("borderline_to_20_percent")),
        "min_value_percent": min(values) if values else None,
        "max_value_percent": max(values) if values else None,
        "compiler_threshold_direction": direction,
    }


def _source_threshold_direction(cards: list[dict[str, Any]]) -> str:
    counts = Counter(card["threshold_relation"] for card in cards)
    n_above = counts.get("above_or_equal_high_threshold", 0)
    n_below = counts.get("below_high_threshold", 0)
    if n_above and not n_below:
        return "supports_high_bioavailability"
    if n_below and not n_above:
        return "argues_against_high_bioavailability"
    if n_above > n_below:
        return "mixed_numeric_evidence_lean_high"
    if n_below > n_above:
        return "mixed_numeric_evidence_lean_low"
    return "mixed_numeric_evidence_balanced"


def _threshold_relation(value: float, relation: str = "=") -> str:
    relation = relation.strip()
    cutoff = BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
    if relation in {">", ">="} and value >= cutoff:
        return "above_or_equal_high_threshold"
    if relation in {"<", "<="} and value <= cutoff:
        return "below_high_threshold"
    if value >= cutoff:
        return "above_or_equal_high_threshold"
    return "below_high_threshold"


def _best_percent_value(row: dict[str, Any]) -> float | None:
    for key in ("source_value_median_percent", "oral_bioavailability_value_percent"):
        value = _parse_float(row.get(key))
        if value is not None and 0 <= value <= 150:
            return value
    standard_value = _parse_float(row.get("standard_value"))
    if standard_value is not None:
        standard_type = str(row.get("standard_type") or "").lower()
        standard_units = str(row.get("standard_units") or "").lower()
        if "fraction" in standard_type and 0 <= standard_value <= 1:
            return standard_value * 100
        if standard_units in {"fraction", "ratio"} and 0 <= standard_value <= 1:
            return standard_value * 100
        if 0 <= standard_value <= 150:
            return standard_value
    return None


def _all_rows(group: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for neighbor in group.get("neighbors") or []:
        for row in neighbor.get("evidence_rows") or []:
            item = dict(row)
            item.setdefault("molecule_chembl_id", neighbor.get("molecule_chembl_id"))
            item.setdefault("canonical_smiles", neighbor.get("canonical_smiles"))
            rows.append(item)
    return rows


def _neighbor_summary(group: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[str]]:
    keys = []
    top_neighbors = []
    for neighbor in group.get("neighbors") or []:
        source_key = _source_molecule_key_from_neighbor(neighbor)
        keys.append(source_key)
        if len(top_neighbors) < 5:
            top_neighbors.append(
                {
                    "source_molecule": source_key,
                    "similarity": neighbor.get("similarity"),
                    "similarity_bucket": neighbor.get("similarity_bucket"),
                    "n_evidence_rows": len(neighbor.get("evidence_rows") or []),
                }
            )
    unique = sorted({key for key in keys if key})
    return len(unique), top_neighbors, unique[:12]


def _duplicate_source_summary(group_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_to_groups: dict[str, list[str]] = defaultdict(list)
    for card in group_cards:
        for key in card.get("source_keys") or []:
            source_to_groups[key].append(card["group_id"])
    duplicates = [
        {"source_molecule": key, "groups": sorted(set(groups)), "n_groups": len(set(groups))}
        for key, groups in source_to_groups.items()
        if len(set(groups)) > 1
    ]
    return sorted(duplicates, key=lambda item: (-item["n_groups"], item["source_molecule"]))[:20]


def _factor_summary(group_cards: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(card["bioavailability_factor"] for card in group_cards if card["useful_for_bioavailability_reasoning"])
    strongest = defaultdict(list)
    for card in group_cards:
        if not card["useful_for_bioavailability_reasoning"]:
            continue
        factor = card["bioavailability_factor"]
        if card["transferability"] in TRANSFERABLE or card["direct_label_vote_allowed"]:
            strongest[factor].append(
                {
                    "group_id": card["group_id"],
                    "endpoint_role": card["endpoint_role"],
                    "transferability": card["transferability"],
                    "confidence": card["confidence"],
                    "evidence_direction": card["llm_evidence_direction"],
                }
            )
    return {
        "useful_group_count_by_factor": dict(sorted(counts.items())),
        "transferable_or_direct_groups_by_factor": {key: value[:8] for key, value in sorted(strongest.items())},
    }


def _slim_vote_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": card["group_id"],
        "endpoint_role": card["endpoint_role"],
        "transferability": card["transferability"],
        "confidence": card["confidence"],
        "llm_evidence_direction": card["llm_evidence_direction"],
        "compiler_threshold_direction": card["threshold_summary"]["compiler_threshold_direction"],
        "threshold_evidence": card["threshold_evidence"][:6],
        "threshold_summary": card["threshold_summary"],
        "reasoning_summary": card["reasoning_summary"],
        "caveats": card["caveats"],
    }


def _slim_proxy_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": card["group_id"],
        "endpoint_role": card["endpoint_role"],
        "bioavailability_factor": card["bioavailability_factor"],
        "transferability": card["transferability"],
        "confidence": card["confidence"],
        "llm_evidence_direction": card["llm_evidence_direction"],
        "reasoning_summary": card["reasoning_summary"],
        "caveats": card["caveats"],
    }


def _source_molecule_key_from_neighbor(neighbor: dict[str, Any]) -> str:
    for row in neighbor.get("evidence_rows") or []:
        key = _source_molecule_key_from_row(row)
        if key:
            return key
    return str(neighbor.get("molecule_chembl_id") or neighbor.get("canonical_smiles") or "")


def _source_molecule_key_from_row(row: dict[str, Any]) -> str:
    for key in ("combined_merge_key", "source_molecule_id", "molecule_chembl_id", "standard_inchi_key", "canonical_smiles"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _tier_from_group_id(group_id: str) -> str:
    return group_id.split(".", 1)[0] if "." in group_id else ""


def _endpoint_from_group_id(group_id: str) -> str:
    return group_id.split(".", 1)[1] if "." in group_id else group_id


def _norm_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
