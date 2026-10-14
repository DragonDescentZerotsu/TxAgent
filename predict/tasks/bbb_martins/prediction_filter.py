"""BBB prediction-only evidence screen from the reviewed indirect ablation."""

import re
from typing import Any, Mapping

PREDICTION = re.compile(
    r"\bin[ -]silico\b|\bqikprop\b|swiss\s*adme|admetlab|admet[s ]?sar|"
    r"\bpredicted\b|\bprediction\b|\bcomputational\b|\badaboost\b",
    re.I,
)
MEASUREMENT = re.compile(
    r"\bmeasured\b|\bexperiment(?:al|ally)?\b|\bin vivo\b|\bin vitro\b|"
    r"microdialysis|perfusion",
    re.I,
)
PREDICTION_FILTER_VERSION = "v2"
COMPUTATION = re.compile(
    r"\bin[ -]silico\b|\bcomputational\b|\bQSAR\b|"
    r"\bPBPK\b|\bmolecular dynamics\b|\bPK[ -]PD simulation\b|"
    r"\bQikProp\b|swiss\s*adme|admetlab|admet[s ]?sar|\bAdaBoost\b|"
    r"\bpkCSM\b|BOILED[ -]Egg|\bADMET\s+(?:prediction\s+)?model\b|"
    r"\b(?:BBB|PAMPA)\s+prediction\s+model\b|"
    r"\bpredicted\s+(?:PAMPA|in[ -]vitro|in[ -]vivo)\b",
    re.I,
)
ASSAY = re.compile(
    r"\bPAMPA(?:[ -]BBB)?\b|parallel artificial membrane permea(?:bility|tion)|"
    r"\bMDCK(?:[ -]MDR1)?\b|\bCaco[ -]?2\b|\bhCMEC/D3\b|"
    r"\btranswell\b|\bmicrodialysis\b|\bperfusion\b",
    re.I,
)
OBSERVED = re.compile(
    r"\b(?:measured|experimentally determined|experimentally tested)\b", re.I
)
NEGATED_EXPERIMENT = re.compile(
    r"\b(?:not|never|no|without)\s+"
    r"(?:(?:any|direct|actual|experimentally|experimental|been|being|was|were)\s+){0,3}"
    r"(?:measur\w*|test\w*|perform\w*|conduct\w*|experiment\w*|"
    r"PAMPA|MDCK|Caco[ -]?2|perfusion|microdialysis)\b|"
    r"\b(?:measurements?|experiments?|assays?|PAMPA)\s+"
    r"(?:(?:was|were|have|has|been)\s+)*(?:not|never)\b",
    re.I,
)


def prediction_only(record):
    """Remove prediction-only cards; retain experimental or mixed evidence.

    Recognizes assay-based outcome classifications and ignores negated or
    computational assay mentions. This remains a conservative lexical screen,
    not a source-paper semantic audit.
    """
    card = record["card"]
    # Keep fields separate: a computational context must not hide a separate
    # measured result, and a generic numeric endpoint is not proof of an assay.
    fields = [
        card.get(key, "")
        for key in (
            "assay_context",
            "support_text",
            "qualifying_conditions",
            "extra_details",
        )
    ]
    details = card.get("experimental_details") or {}
    fields.extend(
        details.get(key, "")
        for key in (
            "assay_type",
            "biological_system",
            "measurement_method",
            "evidence_type",
        )
    )
    text = "\n".join(str(value) for value in fields if value)
    prediction_text = "\n".join(str(value) for value in fields[:3] if value)
    if not (PREDICTION.search(prediction_text) or COMPUTATION.search(text)):
        return False
    computational_context = bool(COMPUTATION.search(text))
    assay_mention_only = computational_context or bool(NEGATED_EXPERIMENT.search(text))
    for field in fields:
        for clause in re.split(r"[;\n]|(?<=[.!?])\s+|\bbut\b", str(field)):
            if NEGATED_EXPERIMENT.search(clause):
                continue
            if OBSERVED.search(clause):
                return False
            if COMPUTATION.search(clause):
                continue
            if ASSAY.search(clause) and re.search(
                r"\b(?:showed|exhibited|yielded|tested|performed|conducted|monolayers?|studies)\b",
                clause,
                re.I,
            ):
                return False
            if MEASUREMENT.search(clause) and not computational_context:
                return False
            if ASSAY.search(clause) and not assay_mention_only:
                return False
    return True



def final_profile_validation_errors(
    content: Mapping[str, Any],
    *,
    profile: str,
) -> list[str]:
    """Enforce the selected candidate's explicit evidence gate."""
    if profile not in {
        "meaningful_cns_adjudication_v2",
        "meaningful_cns_adjudication_v3",
    }:
        return []
    prediction = str(content.get("bbb_prediction") or "")
    state = str(content.get("integrated_outcome_state") or "")
    basis = str(content.get("negative_evidence_basis") or "")
    errors: list[str] = []
    if prediction == "fail" and basis == "none":
        errors.append("fail requires affirmative negative_evidence_basis")
    if state == "transferable_positive" and prediction != "pass":
        errors.append("transferable_positive integrated outcome requires pass")
    return errors
