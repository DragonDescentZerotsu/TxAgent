"""Optional condition compatibility, reported separately from E * I."""

from __future__ import annotations

import re

from ._schema import _DILI_CONDITION_FIELDS, _DILI_SYSTEM_FIELDS, _text, selection_text

_ORGANISM_PATTERNS = {
    "rodent": r"\b(?:rats?|mice|mouse|hamsters?|rodents?|rodentia|rattus|mus musculus|mesocricetus|cricetulus|guinea pigs?|cavia porcellus|gerbils?|meriones|peromyscus)\b",
    "human": r"\b(?:humans?|patients?|workers?|volunteers?|women|woman|men|man|people|persons?|children|infants?|homo sapiens)\b",
    "dog": r"\b(?:dogs?|canine|beagles?|canis (?:lupus|familiaris))\b",
    "monkey": r"\b(?:monkeys?|macaques?|rhesus|cynomolgus|macaca|marmosets?|baboons?)\b",
    "rabbit": r"\b(?:rabbits?|oryctolagus)\b",
    "outside_five_groups": r"\b(?:fish|trout|zebrafish|medaka|danio|oryzias|oncorhynchus|chicken|chickens|quail|ducks?|pigs?|swine|porcine|sheep|ovine|goats?|cattle|bovine|drosophila|xenopus|frogs?|ferrets?|cats?|feline|chimpanzees?|apes?)\b",
}


def mentioned_groups(text):
    # Guinea pigs are rodents; do not independently interpret 'pig' here.
    text = re.sub(r"guinea[ -]pigs?", "cavia porcellus", str(text or ""), flags=re.I)
    text = re.sub(r"\bnon[ -]?human\b", "nonhuman", text, flags=re.I)
    groups = {
        group
        for group, pattern in _ORGANISM_PATTERNS.items()
        if re.search(pattern, text, re.I)
    }
    if re.search(r"\bprimates?\b", text, re.I) and not groups.intersection(
        {"monkey", "outside_five_groups"}
    ):
        groups.add("unresolved_primate")
    return groups


def _condition_rank(condition, fields):
    wanted = [part.split("=", 1) for part in condition.split("+") if "=" in part]
    context = _text(
        fields,
        (
            "qualifying_conditions",
            "study_context",
            "species_or_population",
            "evidence_population_or_model",
            "canonical_species_context",
            "species",
            "exposure_context",
            "population_context",
        ),
    )
    actual = set()
    exposure = selection_text(fields.get("exposure_context"))
    if exposure == "overdose_or_supratherapeutic":
        actual.add("exposure=" + exposure)
    for clause in re.split(r"[;|]", context):
        if re.search(r"\b(?:no|without|excluded)\b", clause):
            continue
        if re.search(r"\b(?:p[ae]diatric|children|infants?|neonates?)\b", clause):
            actual.add("age_group=pediatric")
        if re.search(r"(?:older than|over|age\s*>)\s*65\b", clause):
            actual.add("age_group=older_than_65")
    # Structured source values only; do not infer age from arbitrary doses.
    age = selection_text(fields.get("age"))
    match = re.fullmatch(r"(\d+)\s*(?:years?|y|yr)?", age, re.I)
    ages = [int(match[1])] if match else []
    # Explicit patient ages in condition fields, never arbitrary result
    # values, doses or years in a supporting-paper citation.
    ages.extend(int(n) for n in re.findall(r"\b(\d{1,3})[- ]year[- ]old\b", context))
    for years in ages:
        if not 0 <= years <= 120:
            continue
        if years < 18:
            actual.add("age_group=pediatric")
        elif years > 65:
            actual.add("age_group=older_than_65")
        else:
            actual.add("age_group=adult_up_to_65")
    ranks = []
    for key, value in wanted:
        native = selection_text(fields.get(key)).lower()
        if native:
            actual.add(f"{key}={native}")
        values = {a.split("=", 1)[1] for a in actual if a.startswith(key + "=")}
        if value in values:
            ranks.append(0 if values == {value} else 1)
        elif values and key in {
            "prandial_state",
            "age_group",
            "release_profile",
            "barrier_state",
        }:
            ranks.append(3)
        else:
            ranks.append(2)
    return max(ranks, default=2)


def _dili_simple_condition(condition, fields):
    if not condition or condition == "no_reported_external_condition":
        return {"status": "not_applicable", "score": 1.0, "axes": {}}
    axes = {}
    for atom in condition.split("+"):
        key, sep, value = atom.partition("=")
        if not sep or not value or key not in _DILI_CONDITION_FIELDS[1:] or key in axes:
            raise ValueError(f"Unsupported or duplicate DILI condition: {atom!r}")
        rank = _condition_rank(atom, fields)
        if key == "age_group":
            groups = mentioned_groups(_text(fields, _DILI_SYSTEM_FIELDS))
            if groups and "human" not in groups:
                rank = 2  # Animal age does not establish a human patient stratum.
        # Different baseline diseases can coexist: they are not mutually exclusive.
        # Only explicitly exclusive age/exposure/regimen values establish mismatch.
        if rank == 2 and key in {"exposure", "regimen"}:
            opposites = {
                "exposure": {"therapeutic_use", "overdose_or_supratherapeutic"},
                "regimen": {"single_dose_or_one_day", "repeated_dose"},
            }
            actual = fields.get(key) or (
                fields.get("exposure_context") if key == "exposure" else ""
            )
            if value in opposites[key] and actual in opposites[key] and actual != value:
                rank = 3
        axes[key] = {
            "requested": value,
            "status": ("matched", "limited", "unknown", "mismatched")[rank],
            "score": (1.0, 0.5, 0.5, 0.0)[rank],
        }
    worst = max(
        axes.values(),
        key=lambda a: ("matched", "limited", "unknown", "mismatched").index(
            a["status"]
        ),
    )
    return {"status": worst["status"], "score": worst["score"], "axes": axes}
