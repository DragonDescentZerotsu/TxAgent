"""Project cached progressive cards into the model-visible score surface."""

from __future__ import annotations

from copy import deepcopy
import json
import re


MORGAN_LINE = re.compile(
    r"^Morgan fingerprint Tanimoto similarity:\s*([0-9.]+)\b.*$",
    re.MULTILINE,
)
PROXIMITY_TAG = re.compile(
    r"\b(?:very_)?(?:close|moderate|weak|distant)_analog\b",
    re.IGNORECASE,
)


def similarity_view(active, *, hidden=False, precision=2):
    """Return a detached view without rewriting stored tool receipts."""
    view = deepcopy(active)
    for molecule in view.values():
        numeric = molecule.get("morgan_similarity")
        for tool in molecule.get("query_analog_tool_summaries", []):
            content = tool.get("content") or ""
            match = MORGAN_LINE.search(content)
            if match and numeric is None:
                numeric = float(match.group(1))
            tool["content"] = "\n".join(
                line for line in content.splitlines() if not MORGAN_LINE.fullmatch(line)
            )
        if hidden:
            molecule.pop("morgan_similarity", None)
        else:
            if numeric is None or not 0 <= float(numeric) <= 1:
                raise ValueError("numeric view requires a valid frozen Morgan similarity")
            molecule["morgan_similarity"] = round(float(numeric), precision)
        for card in molecule.get("cards", {}).values():
            # Similarity is parent-scoped. Records grouped under that parent may
            # come from non-Morgan levels and need not carry a duplicate score.
            card.pop("morgan_similarity", None)
    return view


def validate_visible_messages(messages, *, hidden=False, stage_scoped=False):
    """Reject similarity metadata outside the active molecule-level score scope."""
    text = json.dumps(messages, ensure_ascii=False)
    if PROXIMITY_TAG.search(text) or "similarity_bucket" in text:
        raise ValueError("qualitative proximity label leaked into model input")
    if hidden and re.search(r"morgan|tanimoto", text, re.IGNORECASE):
        raise ValueError("Morgan similarity reference leaked into hidden-score input")
    payload = json.loads(messages[1]["content"])
    for molecule in payload["active_evidence"]:
        needs_morgan = not hidden and (
            not stage_scoped or bool(molecule.get("morgan_score_levels"))
        )
        if needs_morgan and "morgan_similarity" not in molecule:
            raise ValueError("numeric view lacks molecule-level similarity")
        if stage_scoped and not needs_morgan and "morgan_similarity" in molecule:
            raise ValueError("Morgan similarity lacks a visible Morgan-selected stage")
        if any(
            "morgan_similarity" in card for card in molecule["evidence_cards"]
        ):
            raise ValueError("Morgan similarity leaked into a record")
        tools = json.dumps(molecule.get("query_analog_tool_summaries", []))
        if re.search(r"morgan|tanimoto", tools, re.IGNORECASE):
            raise ValueError("duplicate Morgan similarity in tool summaries")
