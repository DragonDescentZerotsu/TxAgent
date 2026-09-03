"""Conservative direction recovery for experimental BBB outcome records.

The frozen BBB v3 contract rejects otherwise eligible experimental outcome
rows when ``bbb_permeability_label`` is null.  This module reviews only that
boundary.  It deliberately does not turn heterogeneous numeric measurements
into labels: absolute brain concentrations, PET uptake, CSF concentrations,
and total brain/plasma ratios do not share a defensible universal threshold.

Rows are recovered only when their own qualitative result text states an
unambiguous direction.  Every decision exposes a stable rule id so the full
source revision can be audited row by row.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .starling_benchmark import SOURCE_REVISION


REVIEW_VERSION = "bbb_experimental_metric_direction_review.v1"
_ADJUDICATION_PATH = Path(__file__).with_name(
    "experimental_metric_direction_review_adjudications.json"
)
_ADJUDICATION_SPEC = json.loads(_ADJUDICATION_PATH.read_text(encoding="utf-8"))
if _ADJUDICATION_SPEC["review_version"] != REVIEW_VERSION:
    raise RuntimeError("BBB metric-direction adjudications use another review version")
if _ADJUDICATION_SPEC["source_revision"] != SOURCE_REVISION:
    raise RuntimeError("BBB metric-direction adjudications use another source revision")
REVIEWED_PROPOSED_SOURCE_INDICES = frozenset(
    int(index) for index in _ADJUDICATION_SPEC["reviewed_proposed_source_indices"]
)
MANUAL_SOURCE_EXCLUSIONS = {
    int(index): reason for index, reason in _ADJUDICATION_SPEC["excluded"].items()
}
SOURCE_ARROW_SHA256 = _ADJUDICATION_SPEC["source_arrow_sha256"]


@dataclass(frozen=True)
class DirectionReview:
    """One auditable direction decision for a v3 missing-direction row."""

    label: int | None
    rule_id: str
    matched_text: str = ""


_SPACE_PATTERN = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<!\d)\.(?!\d)|[!?;|]+")
_ENDPOINT = r"(?:brain|cns|csf|cerebrospinal fluid|bbb|blood[- ]brain barrier)"
_OUTCOME = (
    r"(?:penetration|access|exposure|uptake|entry|distribution|concentration|"
    r"level|levels|ratio|delivery)"
)
_ENTRY_VERB = r"(?:cross(?:ed|es|ing)?|penetrat(?:ed|es|ing)?|enter(?:ed|s|ing)?)"


_RULES: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    (
        "explicit_negative_failed_entry",
        0,
        re.compile(
            rf"\b(?:cannot|could not|does not|did not|fails? to|failed to|"
            rf"unable to|was not able to)\b.{{0,55}}\b{_ENTRY_VERB}\b"
            rf".{{0,55}}\b{_ENDPOINT}\b|"
            rf"\b{_ENDPOINT}\b.{{0,55}}\b(?:cannot|could not|does not|did not|"
            rf"fails? to|failed to|unable to|was not able to)\b.{{0,55}}"
            rf"\b{_ENTRY_VERB}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_negative_outcome",
        0,
        re.compile(
            rf"\b(?:very low|low|poor|limited|minimal|negligible|restricted|"
            rf"insufficient)\b\s+{_ENDPOINT}\s+{_OUTCOME}\b|"
            rf"\b{_ENDPOINT}\b.{{0,12}}\b{_OUTCOME}\b\s*"
            rf"\b(?:was|were|is|remained|:)\s*(?:very low|low|poor|limited|"
            rf"minimal|negligible|restricted|insufficient)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_positive_modified_entry",
        1,
        re.compile(
            rf"\b(?:readily|easily|freely|rapidly|efficiently)\b.{{0,25}}"
            rf"\b{_ENTRY_VERB}\b.{{0,55}}\b{_ENDPOINT}\b|"
            rf"\b{_ENTRY_VERB}\b.{{0,55}}\b{_ENDPOINT}\b.{{0,25}}"
            rf"\b(?:readily|easily|freely|rapidly|efficiently)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_positive_outcome",
        1,
        re.compile(
            rf"\b(?:very high|high|good|excellent|extensive|substantial|robust|"
            rf"marked)\b\s+{_ENDPOINT}\s+{_OUTCOME}\b|"
            rf"\b{_ENDPOINT}\b.{{0,12}}\b{_OUTCOME}\b\s*"
            rf"\b(?:was|were|is|remained|:)\s*(?:very high|high|good|excellent|"
            rf"extensive|substantial|robust|marked)\b",
            re.IGNORECASE,
        ),
    ),
)

_QUALITATIVE_VALUES = {
    "not detected": (0, "qualitative_value_not_detected"),
    "undetectable": (0, "qualitative_value_not_detected"),
    "below limit of detection": (0, "qualitative_value_not_detected"),
    "below limit of quantification": (0, "qualitative_value_not_detected"),
    "negligible": (0, "qualitative_value_negative"),
    "minimal": (0, "qualitative_value_negative"),
    "very low": (0, "qualitative_value_negative"),
    "low": (0, "qualitative_value_negative"),
    "poor": (0, "qualitative_value_negative"),
    "restricted": (0, "qualitative_value_negative"),
    "very high": (1, "qualitative_value_positive"),
    "high": (1, "qualitative_value_positive"),
    "good": (1, "qualitative_value_positive"),
    "excellent": (1, "qualitative_value_positive"),
    "extensive": (1, "qualitative_value_positive"),
    "substantial": (1, "qualitative_value_positive"),
}

_CONTRAST_PATTERN = re.compile(
    r"\b(?:although|but|however|despite|whereas|while|except|exception|"
    r"contrary|challeng(?:e|ed|es|ing)|disput(?:e|ed|es|ing)|rather than|"
    r"on the other hand|compared to|relative to)\b",
    re.IGNORECASE,
)
_REPORTED_CLAIM_PATTERN = re.compile(
    r"\b(?:hypothesis|notion|suggestion|claim|assumption|idea)\b",
    re.IGNORECASE,
)
_POSITIVE_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|lack(?:s|ed|ing)?|unable|cannot|could not|does not|"
    r"did not|fails? to|failed to|inability|unlikely|neither|too low|poor|limited|minimal|negligible|"
    r"restricted|insufficient|impair(?:ed|ment)?)\b",
    re.IGNORECASE,
)
_NEGATIVE_COUNTEREVIDENCE_PATTERN = re.compile(
    rf"\b(?:detected|measured|quantified|observed|significant|high|good|"
    rf"substantial|extensive|readily|easily|freely|rapidly)\b.{{0,45}}"
    rf"\b{_ENDPOINT}\b|\b{_ENDPOINT}\b.{{0,45}}\b(?:detected|measured|"
    rf"quantified|observed|significant|high|good|substantial|extensive|"
    rf"readily|easily|freely|rapidly)\b",
    re.IGNORECASE,
)
_ENTRY_AFFIRMATION_PATTERN = re.compile(
    rf"\b(?:can|could|does|did|readily|easily|freely|known to|shown to|"
    rf"demonstrated to|was able to|is able to)?\s*{_ENTRY_VERB}\b.{{0,55}}"
    rf"\b{_ENDPOINT}\b|\b(?:detected|measured|quantified|observed)\b.{{0,35}}"
    rf"\b(?:in|within)\s+(?:the\s+)?(?:brain|csf|cerebrospinal fluid)\b",
    re.IGNORECASE,
)


def review_missing_direction(record: Mapping[str, Any]) -> DirectionReview:
    """Recover only an explicit qualitative direction from the source row."""

    value = _normalize(record.get("quant_value"))
    matches: list[tuple[str, int, str]] = []
    if value in _QUALITATIVE_VALUES:
        label, rule_id = _QUALITATIVE_VALUES[value]
        matches.append((rule_id, label, value))

    text_fields = tuple(
        part
        for part in (_normalize(record.get("support_text")),)
        if part
    )
    had_qualified_match = False
    for text in text_fields:
        for sentence in _SENTENCE_BOUNDARY.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            for rule_id, label, pattern in _RULES:
                match = pattern.search(sentence)
                if match is None:
                    continue
                if _ambiguous_sentence(sentence, rule_id, label):
                    had_qualified_match = True
                    continue
                matches.append((rule_id, label, match.group(0)))
    labels = {label for _, label, _ in matches}
    if len(labels) > 1:
        return DirectionReview(None, "conflicting_explicit_direction")
    if matches:
        rule_id, label, matched = matches[0]
        return DirectionReview(label, rule_id, matched)
    if had_qualified_match:
        return DirectionReview(None, "qualified_or_conflicting_direction")
    if value or text_fields:
        return DirectionReview(None, "no_unambiguous_qualitative_direction")
    return DirectionReview(None, "missing_measurement_and_result_text")


def adjudicate_missing_direction(
    record: Mapping[str, Any], *, source_index: int
) -> DirectionReview:
    """Apply the frozen source-index review to one proposed direction."""

    proposed = review_missing_direction(record)
    if proposed.label is None:
        return proposed
    if source_index not in REVIEWED_PROPOSED_SOURCE_INDICES:
        return DirectionReview(None, "unreviewed_proposed_direction")
    if source_index in MANUAL_SOURCE_EXCLUSIONS:
        return DirectionReview(
            None,
            f"manual_review_exclusion:{MANUAL_SOURCE_EXCLUSIONS[source_index]}",
            proposed.matched_text,
        )
    return DirectionReview(
        proposed.label,
        f"manual_review_approved:{proposed.rule_id}",
        proposed.matched_text,
    )


def _normalize(value: Any) -> str:
    return _SPACE_PATTERN.sub(" ", str(value or "").strip().lower())


def _ambiguous_sentence(sentence: str, rule_id: str, label: int) -> bool:
    """Reject matches whose local discourse reverses or qualifies direction."""

    if _CONTRAST_PATTERN.search(sentence) or _REPORTED_CLAIM_PATTERN.search(sentence):
        return True
    if label == 1 and _POSITIVE_NEGATION_PATTERN.search(sentence):
        return True
    if (
        label == 0
        and rule_id == "explicit_negative_failed_entry"
        and _NEGATIVE_COUNTEREVIDENCE_PATTERN.search(sentence)
    ):
        return True
    if label == 0 and rule_id == "explicit_negative_outcome" and _ENTRY_AFFIRMATION_PATTERN.search(sentence):
        return True
    return False
