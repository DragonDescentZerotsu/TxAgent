"""Render frozen benchmark conditions for model-visible query payloads.

Both inference universes pass a benchmark row in and receive the same concise
natural-language condition sentence, or an empty string for the null condition.
"""

from __future__ import annotations

from typing import Any


NO_REPORTED_EXTERNAL_CONDITION = "no_reported_external_condition"
EXTERNAL_CONDITION_RENDERER_VERSION = "external_condition_natural_language.v1"

_CONDITION_VALUE_LABELS = {
    "atopic_dermatitis": "atopic dermatitis",
    "bacterial_meningitis": "bacterial meningitis",
    "brain_tumor_or_glioma": "a brain tumor or glioma",
    "cerebral_ischemia": "cerebral ischemia",
    "cirrhosis": "cirrhosis",
    "cystic_fibrosis": "cystic fibrosis",
    "disrupted": "a disrupted biological barrier",
    "fasted": "a fasted state",
    "fed_high_fat": "a high-fat fed state",
    "fed_unspecified": "a fed state",
    "meningitis_unspecified": "meningitis",
    "modified_release": "a modified-release formulation",
    "pneumococcal_meningitis": "pneumococcal meningitis",
    "tuberculous_meningitis": "tuberculous meningitis",
}


def external_condition_sentence(record: dict[str, Any]) -> str:
    """Render a frozen condition group as natural language for model input."""
    group = str(record.get("condition_group") or "").strip()
    if not group or group == NO_REPORTED_EXTERNAL_CONDITION:
        return ""
    clauses = [_render_condition_atom(atom) for atom in group.split(";") if atom]
    return "This prediction concerns the query molecule under " + " and ".join(clauses) + "."


def _render_condition_atom(atom: str) -> str:
    key, separator, raw_value = atom.partition("=")
    if not separator:
        return atom.replace("_", " ")
    value = _CONDITION_VALUE_LABELS.get(raw_value, raw_value.replace("_", " "))
    if key == "disease":
        return f"the disease condition {value}"
    if key == "co_treatment":
        return f"co-treatment with {value}"
    if key in {"prandial_state", "release_profile", "barrier_state"}:
        return value
    return f"{key.replace('_', ' ')}: {value}"
