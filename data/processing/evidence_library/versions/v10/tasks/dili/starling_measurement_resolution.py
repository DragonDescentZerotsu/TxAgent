"""DILI V10 measurement extraction and its fixed dgx027 provenance contract."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_literal_text,
    file_sha256,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    NO_DIGIT_RULE_ID,
    RouteDecision,
    SourceRoutingRules,
    has_digit,
    measurement_source_text,
    source_role_contract,
)
from data.processing.evidence_library.versions.v10.numeric_syntax import number_text
from data.processing.paths import REPO_ROOT, evidence_library_root
from data.processing.evidence_library.versions.v10.tasks.dili.mapping_registry import (
    mapping_path,
)


TASK_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = TASK_ROOT / "prompts/measurement_resolution_v11.jinja"

PROMPT_VERSION = "dili_measurement_resolution_prompt.v11"
MAPPING_VERSION = "dili_measurement_resolution.v4"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
ALLOW_REBATCH_UNATTEMPTED = True
STRATIFY_BATCHES = True
REQUIRE_SOURCE_ROW_UID = True
AUTO_VALIDATE_GENERATED_MAPPING = True
RETRY_VALIDATION_FEEDBACK = True
RETRY_SEMANTIC_ERRORS = True
REASONING_EFFORT = "low"
TEMPERATURE = 0.0
ASSIGNMENT_GUARD_VERSION = "dili_measurement_assignment_guard.v13"
DILI_MEASUREMENT_ROUTING_VERSION = "dili_schema_numeric_routing.v5"

DEEPSEEK_BASE_URL = "http://dgx027:50001/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_PROVIDER = "local"
DEEPSEEK_CREDENTIAL_ENV = ""
ENDPOINT_CONCURRENCY_BUDGET = 512
MAX_COMPLETION_TOKENS = 8_192
PROVIDER_POOL_CONFIG = REPO_ROOT / "predict/api_client/providers/current_endpoints.json"
PROVIDER_POOL_PARALLELISM = 1_152
PROVIDER_POOL_BASE_URLS = (
    "http://dgx005:50001/v1",
    "http://dgx011:50001/v1",
    "http://dgx014:50002/v1",
)
PROVIDER_POOL_NAMES = ("dgx005_50001", "dgx011_50001", "dgx014_50002")
OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_CREDENTIAL_ENVS = frozenset(
    {"OPENAI_API_KEY_ONE", "OPENAI_API_KEY_TWO"}
)

DEFAULT_CLEANED_RECORDS = evidence_library_root("dili", "v10") / "01_cleaned/records.parquet"
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
DEFAULT_MAPPING_PATH = (
    evidence_library_root("dili", "v10")
    / "measurement_resolution_v4/measurement_resolution.parquet"
)
DEFAULT_BASE_MAPPING_PATH = None
EXACT_UNIT_MAPPING_PATH = mapping_path("exact_measurement_units")
ENFORCE_EXACT_UNITS_DURING_EXTRACTION = True
DEFAULT_GOLD_FIXTURE = (
    REPO_ROOT
    / "tests/chembl_tool/common/measurement_resolution_quality/gold/dili.v10.1.jsonl"
)

SOURCE_MEASUREMENT_FIELDS = {
    "dili_base": ("causal_status", ""),
    "dili_v1": ("result_value", "result_unit"),
    "dili_v2": ("reported_result", ""),
    "dili_v3": ("result_value", "result_unit"),
    "dili_v4": ("result_value", "result_unit"),
    "dili_v5": ("quantitative_value", "quantitative_unit"),
}
SOURCE_IDS = tuple(SOURCE_MEASUREMENT_FIELDS)

_CONTEXT_FIELDS = {
    "dili_base": (),
    "dili_v1": ("assay_and_readout",),
    "dili_v2": ("assay_detail",),
    "dili_v3": (
        "endpoint_metric",
        "target_or_process",
        "probe_or_analyte",
        "effect_direction",
    ),
    "dili_v4": (
        "endpoint_and_comparator",
        "analytical_platform_and_normalization",
        "effect_direction",
    ),
    "dili_v5": (
        "specific_endpoint_name",
        "quantitative_measure_type",
        "assay_and_platform",
        "effect_direction",
    ),
}
UNIT_RECONCILIATION_CONTEXT_FIELDS = _CONTEXT_FIELDS

_MEASUREMENT_BASIS_FIELDS = {
    "dili_base": (),
    "dili_v1": ("assay_and_readout",),
    "dili_v2": ("assay_detail",),
    "dili_v3": ("endpoint_metric",),
    "dili_v4": ("endpoint_and_comparator",),
    "dili_v5": ("specific_endpoint_name",),
}

_SOURCE_KINDS = {
    "dili_base": "human agent-DILI relationship",
    "dili_v1": "integrated hepatocyte phenotype",
    "dili_v2": "bioactivation or mitochondrial energy-system assay",
    "dili_v3": "hepatobiliary homeostasis assay",
    "dili_v4": "immune-context or molecular-signature assay",
    "dili_v5": "redox, lipid, ER, calcium, lysosomal, or autophagy assay",
}

_RELATIVE_UNIT_MARKERS = (
    "fold_change",
    "fold_vs_",
    "of_baseline",
    "of_control",
    "of_time_zero",
    "of_vehicle",
    "percent_change",
    "percent_inhibition",
    "percent_of_",
    "relative_to_",
)
_RELATIVE_RESULT_UNIT = re.compile(
    r"(?:^|[^a-z])fold(?:[^a-z]|$)|\brelative\b"
    r"|\blog2\s*(?:fc|fold(?:[ _-]?change)?)\b|\bRQ\b|\bF\s*/\s*F0\b|\bSMD\b"
    r"|(?:%|percent)[_ ]*(?:change|increase|decrease|reduction|inhibition|suppression|recovery)"
    r"|(?:%|percent)[_ ]*loss\b"
    r"|(?:%|percent)(?:[_ ]+[A-Za-z][A-Za-z-]*){1,2}[_ ]+"
    r"(?:change|increase|decrease|loss|reduction|inhibition|suppression|recovery)\b"
    r"|(?:change|increase|decrease|reduction|inhibition|suppression|recovery)[_ ]*(?:%|percent)"
    r"|(?:%|percent)[_ ]+of[_ ]+(?:controls?|baseline|basal|vehicle|untreated|initial|normal|first|adult|GW4064)"
    r"|(?:^|[_ ])(?:of|vs)[_ ]*(?:controls?|baseline|basal|vehicle|untreated|initial|normal|first|adult|GW4064)"
    r"|percent_(?:growth|of|increase|decrease|change|inhibition|reduction)",
    re.IGNORECASE,
)
_INCOMPLETE_UNIT = re.compile(
    r"^(?:a\.?u\.?|arbitrary(?:\s+(?:fluorescence|signal|densitometric))?\s+units?"
    r"(?:\s*\([^)]*\))?|(?:relative\s+)?fluorescence(?:\s+intensity)?\s+units?"
    r"|(?:fluorescence|signal)\s+intensity(?:\s*\([^)]*\))?"
    r"|(?:%|percent)\s+(?:fluorescence|signal|intensity)"
    r"|intensity\s+units?|RFU|MFI|fraction|cells?|genes?|proteins?|transcripts?"
    r"|probe\s+sets?|spots?|puncta|events?)$",
    re.IGNORECASE,
)
# V1 and V2 declare free-text outcomes. V3 stores the upper endpoint of a range
# in result_value, V4 has no typed numeric role, and V5's generic absolute type
# includes normalized and incomplete metrics. Only V5's explicit potency roles
# are narrow enough for a deterministic exact-copy rule.
_V5_EXACT_POTENCY_RULE_IDS = {
    "ac50": "dili_v5_exact_ac50_pair.v1",
    "ec50": "dili_v5_exact_ec50_pair.v1",
    "ic50": "dili_v5_exact_ic50_pair.v1",
    "mec": "dili_v5_exact_mec_pair.v1",
}


def _typed_potency_pair_is_explicit(
    row: Mapping[str, Any], potency_type: str, measurement: str, unit: str
) -> bool:
    """Require the typed potency and exact pair in one source clause."""
    support = str(row.get("support_text") or "")
    metric = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(potency_type) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    metrics = list(metric.finditer(support))
    if len(metrics) != 1:
        return False
    for start, end, _ in _selected_number_spans(support, measurement):
        tail = re.sub(
            r"^\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",
            "",
            support[end:],
            flags=re.IGNORECASE,
        )
        unit_spans = _complete_unit_spans(tail.rstrip("."), unit)
        if not unit_spans or tail[: unit_spans[0][0]].strip():
            continue
        for label in metrics:
            if label.end() > start:
                continue
            bridge = support[label.end() : start]
            if (
                len(bridge) <= 120
                and not re.search(r";|\n|[.!?]\s+[A-Z]", bridge)
                and not _NUMBER.search(bridge)
            ):
                return True
    return False
_DOSE_ONLY_TEXT = re.compile(
    r"\b(?:effective|highest|lowest|nominal|optimal|optimum|selected|significant|subtoxic"
    r"|tested|test)\b[^;,.]{0,50}\b(?:dose|concentration)\b"
    r"|\b(?:dose|concentration)\b[^;,.]{0,40}"
    r"\b(?:effective|highest|lowest|selected|significant|tested)\b"
    r"|\b(?:oral|daily|working|test)\s+(?:dose|dosage|concentration)\b"
    r"|\bexposure\s+level\b",
    re.IGNORECASE,
)
_CONCENTRATION_ENDPOINT = re.compile(
    r"\b(?:(?:IC|EC|LC|I)50|LOEC|LOAEL|MEC|MIC|NOAEL?|NOEL"
    r"|K[_ ]?[im1]|(?-i:K[_ ]?[ad])|CRC)\b"
    r"|\b(?:half[- ]max(?:imal|imum)|onset\s+threshold|threshold\s+concentration)\b"
    r"|\b(?:effect|response|toxicity)\s+threshold\b"
    r"|\b(?:mean|minimal|minimum|inhibitory)\s+"
    r"(?:(?:effective|efficacious|inhibitory|uncoupling)\s+)?concentration\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(
    r"(?<![A-Za-z-])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
    r"(?:e[-+]?\d+)?",
    re.IGNORECASE,
)
_SCIENTIFIC_MULTIPLICATION = re.compile(
    r"(?<![A-Za-z0-9.])(?P<coefficient>[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))"
    r"\s*[x×*]\s*10\^?\s*(?P<exponent>[-+]\d+)",
    re.IGNORECASE,
)
_UNSUPPORTED_UNIT = re.compile(
    r"(?:^|\b)(?:arbitrary units?|arb\.?\s*units?(?:/s)?|signal intensity"
    r"|fluorescence intensity|redistribution units?)$"
    r"|(?:not specified|ambiguous|units\?|^[±+/-])",
    re.IGNORECASE,
)
_PHYSICAL_UNIT = re.compile(
    r"(?:%|/|\^|\b(?:[munpµ]?mol|[munpµ]?M|nA|U/L|cpm|cells?|copies|counts?"
    r"|transients?|units?|per)\b|_per_|_of_)",
    re.IGNORECASE,
)
_NAMED_METRIC = re.compile(
    r"\b(?:ratio|score|index|grade|coefficient|number|pEC50|pIC50)\b",
    re.IGNORECASE,
)


def _has_top_level_metric_separator(text: str, start: int, end: int) -> bool:
    """Return whether two metric names are separated outside parentheses."""

    round_depth = 0
    square_depth = 0
    visible: list[str] = []
    for index, character in enumerate(text[:end]):
        if character == "(":
            round_depth += 1
        elif character == "[":
            square_depth += 1
        elif character == ")":
            round_depth = max(0, round_depth - 1)
        elif character == "]":
            square_depth = max(0, square_depth - 1)
        if index >= start:
            visible.append(character if round_depth == square_depth == 0 else " ")
    return bool(re.search(r"\b(?:and|or)\b|[,/]", "".join(visible), re.IGNORECASE))
_RELATIVE_REFERENT = re.compile(
    r"\b(?:KO|knockout|mut|mutant|treated|treatment|exposed|drug)"
    r"\s*(?:/|:|-to-)\s*"
    r"(?:WT|wild[- ]?type|untreated|controls?|vehicle|baseline)\b"
    r"|\b(?:WT|wild[- ]?type|untreated|controls?|vehicle|baseline)"
    r"\s*(?:/|:|-to-)\s*"
    r"(?:KO|knockout|mut|mutant|treated|treatment|exposed|drug)\b",
    re.IGNORECASE,
)
_TIME_CONTEXT_IN_UNIT = re.compile(
    r"(?:\b(?:at|after|before|for)\s*|(?:^|[/\s(])\s*)\d+(?:\.\d+)?\s*"
    r"(?:s|sec|seconds?|min|minutes?|h|hr|hours?|d|days?)\b",
    re.IGNORECASE,
)


def _normalize_scientific_multiplication(text: str) -> str:
    return _SCIENTIFIC_MULTIPLICATION.sub(
        lambda match: (
            f"{match.group('coefficient')}e{int(match.group('exponent')):+d}"
        ),
        text,
    )


def _central_numbers(text: str, unit: str = "") -> list[str]:
    """Find outcome candidates after removing uncertainty and unit notation."""
    value = _normalize_scientific_multiplication(text)
    if unit:
        value = re.sub(re.escape(unit), "", value, flags=re.IGNORECASE)
    value = re.sub(r"\[[^\]]+\]\s*[-+]?\d*", "", value)
    value = re.sub(
        r"(?:±|\+/-)\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bP\s*(?:=|<|>|≤|≥)\s*\.?\d+(?:\.\d+)?", "", value, flags=re.I)
    value = re.sub(r"\bn\s*=\s*\d+", "", value, flags=re.I)
    value = re.sub(r"\bt\s*=\s*\d+(?:\.\d+)?", "", value, flags=re.I)
    value = re.sub(
        r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?\s*"
        r"(?:s|sec|seconds?|min|minutes?|h|hr|hours?|d|days?)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b(?:IC|EC|LC|LOEC|I)50\b", "metric", value, flags=re.I)
    return [match.group(0).replace(",", "") for match in _NUMBER.finditer(value)]


def _coefficient(value: str) -> str:
    """Return the printed coefficient without applying scientific notation."""
    return re.split(r"e", value, maxsplit=1, flags=re.I)[0]


def _selected_number_spans(
    text: str, measurement: str
) -> list[tuple[int, int, str | None]]:
    """Locate complete numeric tokens whose printed coefficient was returned."""
    value = text
    scientific_spans: list[tuple[int, int]] = []
    selected: list[tuple[int, int, str | None]] = []
    for match in _SCIENTIFIC_MULTIPLICATION.finditer(value):
        span = match.span()
        scientific_spans.append(span)
        if match.group("coefficient").replace(",", "") == measurement:
            selected.append(
                (span[0], span[1], f"10^{int(match.group('exponent'))}")
            )
    for match in _NUMBER.finditer(value):
        if any(start <= match.start() < end for start, end in scientific_spans):
            continue
        token = match.group(0)
        coefficient = _coefficient(token).replace(",", "")
        if coefficient != measurement:
            continue
        exponent = re.search(r"e([-+]?\d+)$", token, re.IGNORECASE)
        selected.append(
            (
                match.start(),
                match.end(),
                f"10^{int(exponent.group(1))}" if exponent else None,
            )
        )
    return sorted(selected)


def _selected_is_bound(text: str, measurement: str) -> bool:
    prefix = (
        r"(?:\b(?:up to|at least|at most|no more than|no less than)\s+"
        r"|\bin\s+(?:the\s+)?range\s+of\s+"
        r"|\b(?:as\s+)?(?:low|high)\s+as\s+"
        r"|\b(?:fewer|less|more|greater|higher|lower)\s+than\s+"
        r"|\b(?:not\s+exceeding|not\s+below|exceed(?:s|ed|ing)?|under|over)\s+"
        r"|\bbelow\s+(?:the\s+)?(?:limit|LOD|LOQ)[^;,.]{0,35}\bof\s+"
        r"|\b(?:LOD|LOQ)\s*(?:=|:)?\s*"
        r"|\b(?:minimum|maximum)(?:\s+of)?\s+"
        r"|\b(?:below|above)\s+"
        r"|(?:<=|>=|≤|≥|<|>)\s*[~≈]?)"
    )
    selected = re.escape(measurement) + r"(?![0-9.])"
    return bool(
        re.search(prefix + selected, text, re.IGNORECASE)
        or re.search(
            selected
            + r"[^;,.0-9]{0,40}\b(?:or\s+)?"
            r"(?:less|more|fewer|greater|higher|lower|below|above)\b",
            text,
            re.IGNORECASE,
        )
    )


def _selected_is_relative(text: str, measurement: str, unit: str) -> bool:
    lowered_unit = unit.lower()
    if any(marker in lowered_unit for marker in _RELATIVE_UNIT_MARKERS):
        return True
    if _RELATIVE_RESULT_UNIT.search(unit):
        return True
    if _has_relative_referent(unit) or _has_relative_referent(text):
        return True
    if re.search(
        r"\b(?:treat(?:ment|ed)?|drug|expos(?:ed|ure)?|vehicle|baseline|controls?)"
        r"\s*/\s*(?:treat(?:ment|ed)?|drug|expos(?:ed|ure)?|vehicle|baseline|controls?)\b",
        lowered_unit,
    ):
        return True
    if re.search(
        r"(?:/\s*(?:baseline|controls?|vehicle)\b"
        r"|\b(?:baseline|controls?|vehicle)\s*/)",
        lowered_unit,
    ):
        return True
    if re.search(
        r"\bof\b[^,;]{0,40}\b(?:controls?|vehicle|baseline)\b",
        lowered_unit,
    ):
        return True
    selected = r"(?<![0-9.])" + re.escape(measurement) + r"(?![0-9.])"
    patterns = (
        selected
        + r"[^;,.]{0,100}\b(?:more|less|higher|lower)\b[^;,.]{0,80}\bthan\b",
        r"\b(?:changed|decreased|declined|fell|increased|lowered|raised|reduced|rose)"
        r"\s+by\s+(?:about|approximately|approx\.?|around|~|≈)?\s*"
        + selected,
        selected
        + r"\s*%?\s*-?\s*(?:fold\s+)?(?:reduction|decrease|increase|change|inhibition)\b",
        r"\b(?:reduced|decreased|increased|inhibited)\s+by\s+[~≈]?"
        + selected
        + r"\s*%",
        selected
        + r"\s*(?:-?fold|times|[x×])\s+(?:the\s+)?"
        r"(?:controls?|vehicle|baseline|untreated|WT|wild[- ]?type)\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _has_relative_referent(text: str) -> bool:
    if _RELATIVE_REFERENT.search(text):
        return True
    if re.search(
        r"\b(?:T\s*[/:-]\s*C|(?:WT|wild[- ]?type|controls?|vehicle|baseline)"
        r"[-_ ]normalized)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    lowered = text.lower()
    if re.search(
        r"\bratio\b[^;,.]{0,60}\brelative\s+to\s+"
        r"(?:wt|wild[- ]?type|controls?|vehicle|baseline"
        r"|untreated|drug|treatment|exposed)",
        lowered,
    ):
        return True
    if re.search(r"\bratio\b", lowered) and re.search(
        r"\b(?:drug|dose|treat(?:ed|ment)?|exposed|group)[- A-Za-z0-9]*"
        r"\s*(?:/|:|-to-)\s*"
        r"(?:drug|dose|treat(?:ed|ment)?|exposed|group|low|high)\b",
        lowered,
    ):
        return True
    mutant = re.search(
        r"\b(?:ko|knockout|mut|mutant|deficient|deficiency|null)\b", lowered
    )
    wild_type = re.search(r"\b(?:wt|wild[- ]?type)\b", lowered)
    return bool(mutant and wild_type and re.search(r"\bratio\b", lowered))


def _is_dose_or_concentration_unit(unit: str) -> bool:
    value = unit.lower().replace("µ", "u").replace("μ", "u")
    value = re.sub(r"^10\^[-+]?\d+\s+", "", value)
    value = re.sub(r"\s+([fpnum]?l|kg|g)\s*\^\s*[−-]?1\b", r"/\1", value)
    value = re.sub(r"_per_|\s+per\s+", "/", value)
    value = value.replace("body weight", "bw")
    value = re.sub(r"\s+", "", value)
    return bool(
        re.fullmatch(r"[fpnum]?m", value)
        or re.fullmatch(
            r"[fpnum]?mol(?:/(?:[fpnum]?l|kg|g|m2)(?:/(?:d|day|week))*)?",
            value,
        )
        or re.fullmatch(
            r"[fpnumk]?g(?:/[a-z0-9^+-]+)*(?:/(?:d|day|week))?", value
        )
        or re.fullmatch(r"(?:iu|u)/(?:kg|g)(?:/(?:d|day|week))?", value)
    )


def _looks_like_named_concentration_endpoint(
    row: Mapping[str, Any], prefix: str, unit: str
) -> bool:
    matches = list(_CONCENTRATION_ENDPOINT.finditer(prefix))
    if not matches:
        return False
    marker = matches[-1]
    bridge = prefix[marker.end() :]
    if (
        len(bridge) > 60
        or re.search(
            r"[;,.]|\b(?:Cmax|Vmax|AUC|concentrations?)\b",
            bridge,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:lower|higher|less|more)\b[^;,.]*\bthan\b",
            bridge,
            re.IGNORECASE,
        )
        or re.search(r"\bproduced\s+by\b", bridge, re.IGNORECASE)
        or re.search(
            r"\b(?:not\s+(?:determined|determinable)|no\s+\w*\s*obtained)\b",
            prefix[max(0, marker.start() - 15) :],
            re.IGNORECASE,
        )
    ):
        return False
    marker_name = marker.group(0).casefold()
    basis = " ".join(_basis_values(row))
    if marker_name == "crc" and not re.fullmatch(
        r"\s*(?:=|:)\s*(?:about|approximately|roughly|~|≈)?\s*", bridge, re.I
    ):
        return False
    if marker_name == "mic" and re.search(
        r"\b(?:absorbance|fingerprint|spectra?|spectral|wavelength)\b", basis, re.I
    ):
        return False
    if marker_name == "mec":
        return True
    leading = prefix[max(0, marker.start() - 35) : marker.start()]
    return not re.search(
        r"\bat(?:\s+(?:its|the|a|an))?"
        r"(?:\s+\d+(?:\.\d+)?\s*(?:h|hr|hours?))?\s*$"
        r"|\b(?:versus|vs\.?|compared\s+with)\s*$",
        leading,
        re.IGNORECASE,
    )


_SUBJECT_STOPWORDS = {
    "activity",
    "after",
    "and",
    "assay",
    "as",
    "at",
    "average",
    "before",
    "blocked",
    "by",
    "capacity",
    "cell",
    "cells",
    "concentration",
    "control",
    "decreased",
    "death",
    "diminished",
    "effect",
    "affected",
    "exceeded",
    "exceeding",
    "exposure",
    "from",
    "high",
    "in",
    "increased",
    "inhibited",
    "inhibition",
    "killing",
    "level",
    "low",
    "maximal",
    "maximum",
    "of",
    "not",
    "or",
    "plasma",
    "prevention",
    "required",
    "respiration",
    "response",
    "result",
    "serum",
    "similar",
    "the",
    "tissue",
    "to",
    "treated",
    "treatment",
    "value",
    "viability",
    "versus",
    "vs",
    "with",
    "without",
    "was",
}

_ANALYTE_QUANTITY_BASIS = re.compile(
    r"\b(?:adduct|amount|concentration|content|formation|formed|level|oxidation"
    r"|oxidized|produced|production|quantification|quantified|quantitation"
    r"|measur(?:e|ed|ement)|methods?|rate|readout|uptake)\b",
    re.IGNORECASE,
)
_ANALYTE_SUBJECT_TOKEN = r"[A-Za-z0-9][A-Za-z0-9+/'-]*"
_ANALYTE_SUBJECT = (
    rf"(?P<subject>{_ANALYTE_SUBJECT_TOKEN}"
    rf"(?:\s+{_ANALYTE_SUBJECT_TOKEN}){{0,4}}?)"
)
_MASS_AMOUNT_UNIT = re.compile(r"^\s*[fpnumkµμ]?[gG](?:\b|/)")


def _subject_matches_measurement_basis(row: Mapping[str, Any], subject: str) -> bool:
    def tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", value)
            if token.casefold() not in _SUBJECT_STOPWORDS
        }

    subject_tokens = tokens(subject)
    basis_tokens = set().union(*(tokens(value) for value in _basis_values(row)))
    return bool(subject_tokens & basis_tokens)


def _basis_supports_analyte_quantity(
    row: Mapping[str, Any], subject: str
) -> bool:
    """Require the named subject to be a quantified assay result in the basis."""
    if re.search(
        r"\b(?:affected|after|as|at|before|blocked|by|diminished|dose|exceeded"
        r"|exceeding|exposure|from|in|inhibited|killing|not|required|through|to|treated"
        r"|treatment|until|was|with)\b",
        subject,
        re.IGNORECASE,
    ):
        return False
    subject_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", subject)
        if token.casefold() not in _SUBJECT_STOPWORDS
    }
    for value in _basis_values(row):
        for token in subject_tokens:
            for occurrence in re.finditer(
                r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])",
                value,
                re.IGNORECASE,
            ):
                local = value[
                    max(0, occurrence.start() - 80) : occurrence.end() + 80
                ]
                if not _ANALYTE_QUANTITY_BASIS.search(local):
                    continue
                if re.search(
                    re.escape(token)
                    + r"[^;,.]{0,35}\b(?:dose|exposure|titration)\b"
                    r"|\b(?:dose|exposure|titration)\b[^;,.]{0,35}"
                    + re.escape(token),
                    local,
                    re.IGNORECASE,
                ):
                    continue
                return True
    return False


def _looks_like_analyte_result(row: Mapping[str, Any], prefix: str) -> bool:
    """Recognize an analyte as the subject without accepting nearby conditions."""
    tail = prefix[-100:]
    if re.search(
        r"\b(?:and|or|vs\.?|versus|to|at|with|of|until|through|beyond|towards|from"
        r"|after|before|given|received|containing|treatment)\b\s*\(?\s*$"
        r"|\b(?:similar|close)\s+to\s*\(?\s*$"
        r"|\bin\s+(?:the\s+)?presence\s+of\s*\(?\s*$",
        tail,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:high|low|non[- ]?cytotoxic|subtoxic|tested|treatment)"
        r"(?:[- ]+\w+){0,3}\s+(?:dose|concentration)\s*\(?\s*$",
        tail,
        re.IGNORECASE,
    ):
        return False
    if tail.rstrip().endswith("("):
        return False
    if re.search(r"\b(?:close\s+to|/)\s*Cmax\s*$", tail, re.IGNORECASE):
        return False
    patterns = (
        _ANALYTE_SUBJECT
        + r"\s+"
        r"(?:concentration|content|amount|level|Cmax)\s*"
        r"(?:is|was|=)?\s*$",
        _ANALYTE_SUBJECT
        + r"\s+"
        r"(?:reached|oxidized|formed|produced)\s*(?:about|approximately|~|≈)?\s*$",
        _ANALYTE_SUBJECT + r"\s*(?:is|was|=|:|~|≈)?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, tail, re.IGNORECASE)
        if match and _basis_supports_analyte_quantity(row, match.group("subject")):
            return True
    return False


def _unit_embedded_subject_is_unrelated(
    row: Mapping[str, Any], unit: str
) -> bool:
    match = re.match(
        r"\s*[fpnumkµμ]?[gG]\s+(?P<subject>[^/]{1,40})/", unit
    )
    if not match:
        return False
    return not _subject_matches_measurement_basis(row, match.group("subject"))


def _looks_like_analyte_change(row: Mapping[str, Any], prefix: str) -> bool:
    tail = prefix[-120:]
    match = re.search(
        _ANALYTE_SUBJECT
        + r"\s+(?:concentration|content|amount|level)?\s*"
        r"(?:increased|decreased|changed|rose|fell)\s+from\s+"
        r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?:\s*(?:±|\+/-)\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))?\s+to\s*$",
        tail,
        re.IGNORECASE,
    )
    return bool(
        match
        and "concentration" not in match.group(0).casefold()
        and not re.search(r"\bas\s*$", tail[: match.start()], re.IGNORECASE)
        and _basis_supports_analyte_quantity(row, match.group("subject"))
    )


def _looks_like_absolute_concentration_result(
    row: Mapping[str, Any],
    prefix: str,
    suffix: str,
    measurement_text: str,
    measurement: str,
) -> bool:
    tail = prefix[-80:]
    if _looks_like_analyte_change(row, prefix):
        return True
    if re.search(r"\bpresent\s+at\s+(?:about|approximately|~|≈)?\s*$", tail, re.I) and re.search(
        r"\bin\s+(?:human\s+)?(?:blood|plasma|serum)\b", suffix, re.I
    ):
        return True
    if _measurement_is_bare_point(measurement_text, measurement):
        return any(
            re.search(r"\b(?:concentration|content|amount|level)\b", value, re.I)
            for value in _basis_values(row)
        )
    if _looks_like_analyte_result(row, prefix):
        return True
    if re.search(
        r"\b(?:at|with|using|via|through|from|after|except|equimolar|cumulative"
        r"|dose|dosage|concentration|exposure|treatment|required|selected)\b",
        tail,
        re.IGNORECASE,
    ):
        return False
    return False


def _visible_values(row: Mapping[str, Any]) -> list[str]:
    source_id = str(row.get("source_id") or "")
    return [
        str(row[field])
        for field in prompt_row_fields(source_id)
        if row.get(field) not in (None, "")
    ]


def _basis_values(row: Mapping[str, Any]) -> list[str]:
    source_id = str(row.get("source_id") or "")
    return [
        str(row[field])
        for field in _MEASUREMENT_BASIS_FIELDS[source_id]
        if row.get(field) not in (None, "")
    ]


def _complete_unit_spans(text: str, unit: str) -> list[tuple[int, int]]:
    """Find complete unit phrases while rejecting truncated denominators."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9µμ%/^_.+-])"
        + re.escape(unit)
        + r"(?![A-Za-z0-9µμ%/^_.+-])",
        re.IGNORECASE,
    )
    allowed_following_words = {
        "after",
        "and",
        "are",
        "at",
        "before",
        "by",
        "compared",
        "for",
        "from",
        "in",
        "is",
        "on",
        "or",
        "to",
        "under",
        "versus",
        "vs",
        "was",
        "were",
        "with",
        "without",
    }
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        remainder = text[match.end() :]
        if re.match(r"\s*\bof\b", remainder, re.IGNORECASE):
            continue
        if re.match(
            r"\s*\(\s*(?:[A-Za-z-]+\s+){0,3}"
            r"(?:DNA|protein|tissue|cells?|sample|wet\s+weight|dry\s+weight)\s*\)",
            remainder,
            re.IGNORECASE,
        ):
            continue
        following = re.match(r"\s+([A-Za-z][A-Za-z0-9+-]*)", remainder)
        if following and following.group(1).lower() not in allowed_following_words:
            continue
        spans.append(match.span())
    return spans


def _named_metric_component_occurs(text: str, phrase: str) -> bool:
    """Require a full named-metric component, not an arbitrary suffix."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        prefix = text[: match.start()].rstrip()
        suffix = text[match.end() :].lstrip()
        if re.match(r"(?:[A-Z]{1,5}|alpha|beta|gamma|[αβγ])\b", suffix):
            continue
        if re.match(r"\(\s*[A-Z]{1,5}\s*\)", suffix):
            continue
        left_complete = (
            not prefix
            or prefix[-1] in "(;:"
            or bool(re.fullmatch(r"(?:mean|median|average)", prefix, re.I))
            or bool(
                re.search(r"[/+:]", phrase)
                and re.search(r"\bratio\b", phrase, re.IGNORECASE)
            )
        )
        if left_complete:
            return True
    return False


def _phrase_occurs(text: str, phrase: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9])"
            + re.escape(phrase)
            + r"(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
    )


def _measurement_is_bare_point(text: str, measurement: str) -> bool:
    value = text
    selected = _selected_number_spans(value, measurement)
    if len(selected) != 1:
        return False
    start, end, _ = selected[0]
    remainder = value[:start] + value[end:]
    remainder = re.sub(
        r"(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",
        "",
        remainder,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(
        r"\b(?:about|approximately|approx\.?|around)\b",
        "",
        remainder,
        flags=re.IGNORECASE,
    )
    return not remainder.strip(" ~≈:;,()[]")


def _unit_immediately_follows_value(
    text: str, measurement: str, unit: str, scientific_scale: str | None
) -> bool:
    value = text
    literal_unit = unit
    if scientific_scale:
        prefix = f"{scientific_scale} "
        if not unit.startswith(prefix):
            return False
        literal_unit = unit[len(prefix) :]
    for _, end, scale in _selected_number_spans(value, measurement):
        if scale != scientific_scale:
            continue
        tail = re.sub(
            r"^\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",
            "",
            value[end:],
            flags=re.IGNORECASE,
        )
        spans = _complete_unit_spans(tail, literal_unit)
        if spans and not tail[: spans[0][0]].strip():
            return True
    return False


def _has_inline_physical_unit(text: str, measurement: str) -> bool:
    for _, end, _ in _selected_number_spans(text, measurement):
        tail = re.sub(
            r"^\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",
            "",
            text[end:],
            flags=re.IGNORECASE,
        )
        if re.match(
            r"\s*(?:%|[munpµμ]?[Mm]\b|[munpµμ]?mol\b|cpm\b|nA\b|U/L\b"
            r"|(?:cells?|copies|counts?|g|kg|mg|ng|units?)\b)",
            tail,
            re.IGNORECASE,
        ):
            return True
    return False


def _selected_dimension_is_result(
    row: Mapping[str, Any], measurement: str, unit: str
) -> bool | None:
    """Classify lowercase length or spectral units that resemble molarity."""
    if unit not in {"nm", "um", "µm", "μm"}:
        return None
    basis = " ".join(_basis_values(row))
    text = str(row.get("measurement_text") or "")
    has_length_basis = bool(
        re.search(
            r"\b(?:distance|diameter|length|size|thickness)\b",
            f"{basis} {text}",
            re.I,
        )
    )
    has_spectral_basis = bool(
        re.search(
            r"\b(?:absorbance|absorption|lambda[_ ]?max|spectra?|spectral|wavelength)\b",
            basis,
            re.I,
        )
    )
    if not (has_length_basis or has_spectral_basis):
        return None
    for start, end, _ in _selected_number_spans(text, measurement):
        local = text[max(0, start - 90) : min(len(text), end + 90)]
        if re.search(r"\bscale\s+bar\b", local, re.I):
            continue
        if re.search(
            r"\b(?:disappearance|loss)\s+of\s+(?:the\s+)?peak\s+at\b"
            r"|\bobserved\s*\(\s*(?:\w+\s+)?peak\s+at\b",
            local,
            re.I,
        ):
            continue
        if re.search(
            r"\b(?:range|ranged|ranges|ranging|varied|spanned)\b[^;,.]{0,50}$",
            text[max(0, start - 70) : start],
            re.I,
        ):
            continue
        if has_length_basis and re.search(
            r"\b(?:distance|diameter|length|size|thickness)\b"
            r"|\b(?:enlarged|expanded|giant)\b",
            local,
            re.I,
        ):
            if not re.search(
                r"\bcontrols?\b[^;,.]{0,35}[~≈]?\s*"
                + re.escape(measurement)
                + r"\b",
                local,
                re.I,
            ):
                return True
        if has_spectral_basis and re.search(
            r"\b(?:lambda[_ ]?max|soret\s+peak|absorption\s+peak|absorbance\s+peak"
            r"|spectral\s+peak|peak\s+(?:shifted|at|near))\b",
            local,
            re.I,
        ) and not re.search(
            r"\b(?:absorbance|read|measured|monitored|fingerprint)\s+at\b",
            local,
            re.I,
        ):
            return True
    return False


def _selected_is_exposure_condition(
    row: Mapping[str, Any], measurement: str, unit: str
) -> bool:
    """Reject a concentration unless the selected occurrence is a stated result."""
    if _unit_embedded_subject_is_unrelated(row, unit):
        return True
    dimension_result = _selected_dimension_is_result(row, measurement, unit)
    if dimension_result is not None:
        return not dimension_result
    concentration_scale = _is_dose_or_concentration_unit(unit)
    if not concentration_scale and not _MASS_AMOUNT_UNIT.search(unit):
        return False

    def marks_condition(text: str) -> bool:
        for start, end, _ in _selected_number_spans(text, measurement):
            prefix = text[max(0, start - 100) : start]
            suffix = text[end : min(len(text), end + 120)]
            if _looks_like_named_concentration_endpoint(row, prefix, unit):
                continue
            if re.search(
                r"\bpresent\s+at\s+(?:about|approximately|~|≈)?\s*$",
                prefix,
                re.IGNORECASE,
            ) and re.search(
                r"\bin\s+(?:human\s+)?(?:blood|plasma|serum)\b",
                suffix,
                re.IGNORECASE,
            ):
                continue
            if _looks_like_absolute_concentration_result(
                row, prefix, suffix, text, measurement
            ):
                continue
            if re.search(
                r"\b(?:doses?|dosages?|concentrations?)\b[^;,.]{0,80}$",
                prefix,
                re.IGNORECASE,
            ):
                return True
            if prefix.rfind("(") > prefix.rfind(")") and re.match(
                r"\s*" + re.escape(unit) + r"\b", suffix, re.IGNORECASE
            ):
                return True
            if re.match(
                r"\s*" + re.escape(unit) + r"\s*:", suffix, re.IGNORECASE
            ):
                return True
            if re.search(
                r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*"
                + re.escape(unit)
                + r"\s+to\s*$",
                prefix,
                re.IGNORECASE,
            ):
                return True
            if re.search(
                r"\b(?:after|at|before|following|with|using|via|through|up\s+to|starting(?:\s+at)?"
                r"|required|concentration\s+of|exposure\s+to"
                r"|dose(?:d)?(?:\s+at)?|exposed\s+to"
                r"|incubated\s+(?:at|with)|treated\s+(?:at|with)"
                r"|treatment\s+(?:at|with)|given|received|consumed"
                r"|gavaged\s+(?:at|with)|injected\s+(?:at|with))"
                r"\b[^;,.]{0,35}$",
                prefix,
                re.IGNORECASE,
            ):
                return True
            if re.search(
                r"\b(?:after|at|before|following)\s+"
                r"(?:about|approximately|approx\.?|around|~|≈)?\s*$",
                prefix,
                re.IGNORECASE,
            ):
                return True
            after_unit = re.sub(
                r"^\s*" + re.escape(unit), "", suffix, flags=re.IGNORECASE
            )
            if re.match(
                r"[^;]{0,60}\b(?:administered|added|caused|decreased|dose"
                r"|effective|exposure|increased|induced|inhibited|led|nontoxic"
                r"|produced|required|selected|stimulated|sufficient|treated|used"
                r"|given|received|consumed|gavaged|injected)\b",
                after_unit,
                re.IGNORECASE,
            ):
                return True
            if concentration_scale:
                return True
        return False

    measurement_text = str(row.get("measurement_text") or "")
    support_text = str(row.get("support_text") or "")
    return marks_condition(measurement_text) or (
        _measurement_is_bare_point(measurement_text, measurement)
        and marks_condition(support_text)
    )


def _selected_is_sample_size(text: str, measurement: str, unit: str) -> bool:
    if not re.fullmatch(
        r"(?:cells?|animals?|mice|rats?|dishes|wells|replicates|samples)",
        unit,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            re.escape(measurement)
            + r"\s+"
            + re.escape(unit)
            + r"\s+(?:were|was)\s+(?:analy[sz]ed|assayed|incubated|plated"
            r"|seeded|treated|used)",
            text,
            re.IGNORECASE,
        )
    )


def _selected_is_comparator(text: str, measurement: str, unit: str) -> bool:
    for start, end, _ in _selected_number_spans(text, measurement):
        prefix = text[max(0, start - 60) : start]
        full_prefix = text[:start]
        suffix = text[end:]
        if re.search(
            r"\b(?:baseline|comparator|controls?|initial(?:ly)?|placebo|reference|sham|untreated|vehicle)"
            r"(?:\s+(?:arm|cohort|group|level|mean|median|value)){0,3}"
            r"(?:\s+(?:of|was))?[\s,(:=-]*$",
            prefix,
            re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\bbasal(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3}\s*$",
            prefix,
            re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:blocked|inhibited|prevented|reversed)\s+by\b"
            r"[^;]{0,100}\(\s*(?:IC|EC|LC|I)50\b[^)]*$",
            full_prefix,
            re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:vs\.?|versus|compared\s+(?:to|with))\s+"
            r"[A-Za-z][A-Za-z0-9+/'-]*(?:\s+[A-Za-z][A-Za-z0-9+/'-]*){0,3}\s*$",
            prefix,
            re.IGNORECASE,
        ) and not _CONCENTRATION_ENDPOINT.search(prefix):
            return True
        if re.search(
            r"\b(?:more|less|higher|lower)\b[^;()]{0,100}\bthan\b",
            full_prefix,
            re.IGNORECASE,
        ) and re.match(
            r"\s*" + re.escape(unit) + r"\s+for\b",
            suffix,
            re.IGNORECASE,
        ):
            return True
        compared_group = re.search(
            r"\bthan\s+(?:to\s+)?(?P<group>[A-Za-z0-9-]+)"
            r"(?:\s+cells?)?[^;]*;[^;]*$",
            full_prefix,
            re.IGNORECASE,
        )
        if compared_group and re.match(
            r"\s*"
            + re.escape(unit)
            + r"\s+in\s+(?:the\s+)?"
            + re.escape(compared_group.group("group"))
            + r"\b",
            suffix,
            re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:mean|median)?\s*(?:baseline|comparator|control|initial(?:ly)?|placebo"
            r"|reference|sham|untreated|vehicle)\b[^;,.0-9]{0,35}$",
            prefix,
            re.IGNORECASE,
        ):
            return True
        if re.match(
            r"\s*(?:"
            + re.escape(unit)
            + r")?\s*[,;:]?\s*\(?\s*"
            r"(?:baseline|controls?|placebo|reference|sham|untreated|vehicle)\b",
            text[end:],
            re.IGNORECASE,
        ):
            return True
    return False


def _expected_declared_unit(measurement_text: str, unit_text: str) -> str:
    """Attach a printed scientific scale to the exact declared source unit."""
    scientific = re.search(
        r"[-+]?(\d+(?:\.\d+)?)\s*(?:e|[x×*]\s*10\^?)\s*([-+]\d+)",
        measurement_text,
        flags=re.I,
    )
    if scientific:
        return f"10^{int(scientific.group(2))} {unit_text}"
    return unit_text


def _unit_guard_reason(
    row: Mapping[str, Any], measurement: str, unit: str
) -> str | None:
    """Return why the proposed unit is not supported by the visible source row."""
    measurement_text = str(row.get("measurement_text") or "")
    unit_text = str(row.get("unit_text") or "").strip()
    if (
        not unit
        or unit == "m"
        or _UNSUPPORTED_UNIT.search(unit)
        or _INCOMPLETE_UNIT.fullmatch(unit.strip())
        or unit.strip().lower()
        in {
            "au",
            "arbitrary fluorescence units",
            "arbitrary fluorescence unit",
            "arbitrary signal units",
            "arbitrary signal unit",
        }
    ):
        return "unsupported_or_uncertain_unit"
    if "\n" in unit or ";" in unit or "|" in unit:
        return "unit_contains_additional_result"
    if re.search(
        r"(?:±|\+/-)|\b(?:p(?:-?value)?\s*[=<>≤≥]|SD|SE|CI)\b",
        unit,
        re.IGNORECASE,
    ):
        return "unit_contains_uncertainty"
    if re.search(
        r"\b(?:another|first|second)\s+(?:measurement|outcome|result|value)\b",
        unit,
        re.IGNORECASE,
    ):
        return "unit_contains_additional_result"
    physical_not_named = _PHYSICAL_UNIT.search(unit) and not _NAMED_METRIC.search(unit)
    if physical_not_named and (
        ":" in unit
        or "," in unit
        or re.search(r"[.!?]\s+\S", unit)
        or re.search(r"\s[-–—]\s", unit)
        or re.search(
            r"\b(?:caused|decreased|gave|increased|induced|inhibited|is|showed"
            r"|was|were|yielded)\b",
            unit,
            re.I,
        )
        or re.search(
            r"\b(?:after|at|before|during|following|for|from|injected"
            r"|in\s+(?:the\s+)?presence\s+of|reperfusion|remained|tested"
            r"|treated|treatment|up\s+to)\b",
            unit,
            re.I,
        )
        or re.search(
            r"\b(?:measurement|outcome|result|value|mean|median)\b", unit, re.I
        )
        or re.search(r"\([^)]*\d[^)]*\)", unit)
        or re.search(
            r"^\s*(?:[fpnumkµμ]?[gG]|[fpnumµμ]?mol)\s*/\s*[fpnumµμ]?[lL]"
            r"\s+[A-Za-z]",
            unit,
        )
    ):
        return "unit_contains_context_clause"
    if _TIME_CONTEXT_IN_UNIT.search(unit):
        return "unit_contains_time_context"
    if re.search(r"\bn\s*=\s*\d+", unit, re.IGNORECASE):
        return "unit_contains_sample_size"
    if re.search(
        r"\b(?:vs\.?|versus|compared|baseline|controls?|placebo|sham|untreated"
        r"|vehicle|treated|treatment|exposed|administered)\b",
        unit,
        re.IGNORECASE,
    ):
        return "unit_contains_comparator_context"
    if re.search(
        r"\b(?:and|or)\s*[~≈]?[-+]?(?:\d|\.\d)", unit, re.IGNORECASE
    ):
        return "unit_contains_additional_result"
    if re.search(
        r"\b(?:to|through)\s*[~≈]?[-+]?(?:\d|\.\d)", unit, re.IGNORECASE
    ):
        return "unit_contains_additional_result"
    if re.search(r",\s*[~≈]?[-+]?(?:\d|\.\d)", unit):
        return "unit_contains_additional_result"
    selected_spans = _selected_number_spans(measurement_text, measurement)
    scientific_scales = {scale for _, _, scale in selected_spans if scale}
    scientific_scale = next(iter(scientific_scales), None)
    if len(scientific_scales) > 1:
        return "ambiguous_scientific_scale"
    if scientific_scale and not unit.startswith(f"{scientific_scale} "):
        return "missing_printed_scientific_scale"
    if unit.strip().lower() in {"%", "percent", "per cent"}:
        return "unqualified_percent_unit"
    if unit.lower() in {
        "coefficient",
        "grade",
        "index",
        "number",
        "ratio",
        "score",
    } or re.match(r"^ratio\s*\(", unit, re.IGNORECASE):
        return "incomplete_named_metric"
    if unit_text:
        if unit != _expected_declared_unit(measurement_text, unit_text):
            return "unit_not_exact_declared_unit"
        return None

    basis_values = _basis_values(row)
    bare_point = _measurement_is_bare_point(measurement_text, measurement)
    literal_unit = (
        unit[len(scientific_scale) + 1 :] if scientific_scale else unit
    )
    exact_in_measurement = _phrase_occurs(measurement_text, literal_unit)
    exact_in_basis = any(_phrase_occurs(value, literal_unit) for value in basis_values)
    if unit.lower().startswith(("% of ", "percent of ")):
        if re.search(
            r"[<>≤≥]|\b(?:hours?|days?|minutes?|control|vehicle|baseline)\b",
            unit,
            re.IGNORECASE,
        ):
            return "percent_unit_contains_context"
        if not (exact_in_measurement or (bare_point and exact_in_basis)):
            return "unsupported_percent_scale"
        return None
    if _NAMED_METRIC.search(unit):
        complete_in_measurement = _named_metric_component_occurs(
            measurement_text, literal_unit
        )
        complete_in_basis = any(
            _named_metric_component_occurs(value, literal_unit)
            for value in basis_values
        )
        multiple_metric_basis = False
        for value in (*basis_values, unit):
            markers = list(_NAMED_METRIC.finditer(value))
            if any(
                _has_top_level_metric_separator(value, left.end(), right.start())
                for left, right in zip(markers, markers[1:])
            ):
                multiple_metric_basis = True
                break
        treatment_specific_basis = any(
            _phrase_occurs(value, literal_unit)
            and re.search(
                re.escape(literal_unit)
                + r"\s+(?:in|for|among)\s+(?:the\s+)?"
                r"(?:treated|treatment|exposed|controls?|vehicle|sham)\b",
                value,
                re.IGNORECASE,
            )
            for value in basis_values
        )
        if multiple_metric_basis or treatment_specific_basis:
            return "ambiguous_named_metric_basis"
        if complete_in_measurement or (bare_point and complete_in_basis):
            return None
        return "unit_not_tied_to_selected_value"
    if _PHYSICAL_UNIT.search(unit):
        if _unit_immediately_follows_value(
            measurement_text, measurement, unit, scientific_scale
        ):
            return None
        if _has_inline_physical_unit(measurement_text, measurement):
            return "unit_not_exact_measurement_text"
        support_text = str(row.get("support_text") or "")
        if _unit_immediately_follows_value(
            support_text, measurement, unit, scientific_scale
        ):
            return None
        if bare_point and any(
            _complete_unit_spans(value, literal_unit) for value in basis_values
        ):
            if re.search(r"\bcells?\b", unit, re.IGNORECASE):
                basis = " ".join(basis_values)
                if re.search(
                    r"\b(?:fraction|percent(?:age)?|proportion)\b",
                    basis,
                    re.IGNORECASE,
                ) and not re.search(
                    r"%|\b(?:fraction|percent(?:age)?|proportion)\b",
                    unit,
                    re.IGNORECASE,
                ):
                    return "missing_cell_fraction_scale"
            return None
        return "unit_not_tied_to_selected_value"
    count_pattern = re.compile(
        r"\b(?:number|count|percent|fraction)\s+of\s+" + re.escape(unit) + r"\b",
        re.IGNORECASE,
    )
    if (
        count_pattern.search(measurement_text)
        or bare_point
        and any(count_pattern.search(value) for value in basis_values)
    ):
        return None
    return "unsupported_literal_unit"


def _multiple_values_supported(
    text: str, measurement: str, unit: str
) -> bool:
    """Allow only source syntax that makes one of several numbers plainly focal."""
    numbers = _central_numbers(text, unit)
    coefficients = [_coefficient(number) for number in numbers]
    if len(numbers) == 1:
        return coefficients == [measurement]
    if measurement not in coefficients:
        return False
    if re.search(
        r"\bfrom\s+[-+]?\d[^;]*\bto\s+"
        + re.escape(measurement)
        + r"(?![0-9.])",
        text,
        re.IGNORECASE,
    ):
        directed_change = re.search(
            r"\b(?:changed|decreased|declined|fell|increased|lowered|raised"
            r"|reduced|rose)\b[^;,.]{0,60}\bfrom\b",
            text,
            re.IGNORECASE,
        )
        return bool(
            directed_change
            and len(coefficients) == 2
            and coefficients[-1] == measurement
        )

    comparator = re.search(r"\b(?:vs\.?|versus|compared\s+with)\b", text, re.I)
    if comparator:
        left = text[: comparator.start()]
        right = text[comparator.end() :]
        left_coefficients = [
            _coefficient(number) for number in _central_numbers(left, unit)
        ]
        right_coefficients = [
            _coefficient(number) for number in _central_numbers(right, unit)
        ]
        left_metric = re.search(
            r"\b([A-Za-z][A-Za-z0-9_-]*(?:\s+(?:activity|concentration|level"
            r"|score|value))?)\s+(?:is|was)\s+[-+]?\d",
            left,
            re.IGNORECASE,
        )
        right_metric = re.search(
            r"\b([A-Za-z][A-Za-z0-9_-]*(?:\s+(?:activity|concentration|level"
            r"|score|value))?)\s+(?:is|was)\s+[-+]?\d",
            right,
            re.IGNORECASE,
        )
        different_metrics = bool(
            left_metric
            and right_metric
            and left_metric.group(1).lower() != right_metric.group(1).lower()
        )
        first_is_comparator = re.search(
            r"\b(?:baseline|controls?|untreated|vehicle)\b",
            left,
            re.IGNORECASE,
        )
        second_is_explicit_comparator = re.search(
            r"\b(?:baseline|controls?|untreated|vehicle|without)\b",
            right,
            re.IGNORECASE,
        ) or (
            re.search(
                r"\b[A-Za-z][A-Za-z-]{2,}\s+group\b"
                r"[^;,.0-9]{0,30}[(:=]?\s*[-+]?\d",
                right,
                re.I,
            )
            and re.search(
                r"\b(?:decreased|increased|lower|reduced|higher|raised)\b",
                left,
                re.I,
            )
            and not re.search(r"\bgroup\b", left, re.I)
        )
        if (
            left_coefficients == [measurement]
            and len(right_coefficients) == 1
            and not first_is_comparator
            and not different_metrics
            and second_is_explicit_comparator
        ):
            return True

    parenthetical_comparator = re.search(
        r"\(\s*(?:baseline|controls?|untreated|vehicle)\b",
        text,
        re.IGNORECASE,
    )
    if parenthetical_comparator:
        left = text[: parenthetical_comparator.start()]
        right = text[parenthetical_comparator.end() :]
        if (
            len(coefficients) == 2
            and [
                _coefficient(number) for number in _central_numbers(left, unit)
            ]
            == [measurement]
            and len(_central_numbers(right, unit)) == 1
        ):
            return True

    without_time = re.sub(
        r"\(\s*[-+]?\d+(?:\.\d+)?\s*(?:h|hr|hours?|min|minutes?|d|days?)\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return bool(
        re.search(r"\b(?:IC|EC|LC|LOEC|I)50\b", text, re.IGNORECASE)
        and [
            _coefficient(number)
            for number in _central_numbers(without_time, unit)
        ]
        == [measurement]
    )


def _equivalent_number_pattern(measurement: str) -> str:
    """Match alternate decimal spellings of one already selected coefficient."""
    try:
        number = Decimal(measurement.replace(",", ""))
    except InvalidOperation:
        return re.escape(measurement)
    if not number.is_finite():
        return re.escape(measurement)
    sign = "-" if number.is_signed() else r"\+?"
    whole, _, fraction = format(abs(number), "f").partition(".")
    whole = str(int(whole))
    fraction = fraction.rstrip("0")
    if fraction:
        return sign + re.escape(whole) + r"\." + re.escape(fraction) + r"0*"
    return sign + re.escape(whole) + r"(?:\.0+)?"


def _equivalent_number_spans(text: str, measurement: str) -> list[re.Match[str]]:
    return list(
        re.finditer(_equivalent_number_token_pattern(measurement), text, re.I)
    )


def _equivalent_number_token_pattern(measurement: str) -> str:
    pattern = _equivalent_number_pattern(measurement)
    try:
        signed = Decimal(measurement.replace(",", "")).is_signed()
    except InvalidOperation:
        signed = measurement.startswith("-")
    normal = r"(?<![A-Za-z0-9.^+\-−])(?:" + pattern + r")"
    if signed:
        return normal + r"(?![\d.eE])"
    range_upper = r"(?<=[0-9][-–—])(?:" + pattern + r")"
    return r"(?:" + normal + "|" + range_upper + r")(?![\d.eE])"


def _has_conflicting_repeated_potency(
    support: str,
    potency_type: str,
    measurement: str,
    declared_unit: str,
) -> bool:
    """Find two point values for the same typed basis and post-value context."""
    try:
        selected_value = Decimal(measurement)
    except InvalidOperation:
        return False
    clause = re.compile(
        r"(?<![A-Za-z0-9])"
        + re.escape(potency_type)
        + r"\s+for\s+(?P<basis>[^.;]{1,100}?)\s+(?:was|is|=)\s*"
        r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))",
        re.I,
    )
    values_by_context: dict[tuple[str, str], set[Decimal]] = {}
    for match in clause.finditer(support):
        tail = re.sub(
            r"^\s*(?:±|\+/-)\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)",
            "",
            support[match.end() :],
        )
        unit_spans = _complete_unit_spans(tail, declared_unit)
        if not unit_spans or tail[: unit_spans[0][0]].strip():
            continue
        context = re.match(
            r"\s+for\s+(?:the\s+)?(?P<name>[A-Za-z][A-Za-z0-9+_.-]*)",
            tail[unit_spans[0][1] :],
            re.I,
        )
        if context is None:
            continue
        key = (
            " ".join(match.group("basis").casefold().split()),
            context.group("name").casefold().rstrip("."),
        )
        values_by_context.setdefault(key, set()).add(Decimal(match.group("value")))
    return any(
        selected_value in values and len(values) > 1
        for values in values_by_context.values()
    )


def _selected_in_labeled_series(support: str, measurement: str) -> bool:
    """Detect an unselected point copied from a multi-subject value series."""
    try:
        selected = Decimal(measurement)
    except InvalidOperation:
        return False
    values = [
        Decimal(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9.])([-+]?(?:\d+(?:\.\d+)?|\.\d+))"
            r"\s*%?\s*\([^)]{1,60}\)",
            support,
        )
    ]
    return len(values) >= 3 and selected in values


def _selected_in_scientific_labeled_series(
    support: str, measurement: str, declared_unit: str
) -> bool:
    """Detect one unfocused point from a same-scale labeled value series."""
    scale = re.match(r"\s*E(?P<exponent>[-+]\d+)\b", declared_unit, re.I)
    if scale is None:
        return False
    try:
        selected = Decimal(measurement)
    except InvalidOperation:
        return False
    exponent = int(scale.group("exponent"))
    values = [
        Decimal(match.group("coefficient").replace(",", ""))
        for match in _SCIENTIFIC_MULTIPLICATION.finditer(support)
        if int(match.group("exponent")) == exponent
        and re.match(r"\s+for\s+(?:the\s+)?[A-Za-z]", support[match.end() :], re.I)
    ]
    return len(values) >= 3 and selected in values


def _source_specific_guard_reason(
    row: Mapping[str, Any], measurement: str, unit: str
) -> str | None:
    """Reject source-declared relative, bounded, or incomplete DILI values."""
    source_id = str(row.get("source_id") or "")
    support = str(row.get("support_text") or "")
    measurement_text = str(row.get("measurement_text") or "")
    declared_unit = str(row.get("unit_text") or "").strip()
    basis = " ".join(
        str(row.get(field) or "")
        for field in (
            "assay_and_readout",
            "assay_detail",
            "endpoint_metric",
            "endpoint_and_comparator",
            "specific_endpoint_name",
            "analytical_platform_and_normalization",
            "assay_and_platform",
        )
    )

    if (
        re.search(
            r"%(?:\s+DPPH)?\s+(?:quenching|reversal|specific cytotoxicity)\b",
            declared_unit,
            re.I,
        )
        or (
            source_id == "dili_v5"
            and re.search(
                r"%\s+(?:inactivation|superoxide\s+radical\s+scavenged)\b",
                declared_unit,
                re.I,
            )
        )
        or (
            source_id == "dili_v5"
            and re.fullmatch(r"%\s+activity", declared_unit, re.I)
            and re.search(r"(?:inhibition\s*%|%\s*inhibition)", basis, re.I)
        )
        or re.search(
            r"%\s+of\s+(?:saline|(?:its\s+)?original\b|LLC-NTCP\b)",
            declared_unit,
            re.I,
        )
        or re.search(r"after subtracting (?:the )?basal rate", basis, re.I)
        or re.search(
            r"\b(?:net change|mean differences? in concentration"
            r"|percent increase \(reversal\))\b",
            basis,
            re.I,
        )
        or (
            "specific cytotoxicity" in declared_unit.lower()
            and re.search(r"\bcontrol sample\b", basis, re.I)
        )
        or (
            source_id == "dili_v3"
            and re.search(
                r"^\s*change in\b",
                str(row.get("endpoint_metric") or ""),
                re.I,
            )
        )
    ):
        return "relative_value"

    if (
        source_id == "dili_v3"
        and re.fullmatch(r"%\s+of\s+given\s+amount", declared_unit, re.I)
        and _selected_in_labeled_series(support, measurement)
    ):
        return "multiple_or_missing_point_candidates"
    if (
        source_id == "dili_v3"
        and _measurement_is_bare_point(measurement_text, measurement)
        and _selected_in_scientific_labeled_series(
            support, measurement, declared_unit
        )
    ):
        return "multiple_or_missing_point_candidates"

    if _measurement_is_bare_point(measurement_text, measurement):
        for selected in _equivalent_number_spans(support, measurement):
            prefix = support[max(0, selected.start() - 140) : selected.start()]
            suffix = support[selected.end() : selected.end() + 180]
            if source_id == "dili_v3" and re.search(
                r"\b(?:higher|lower|greater|less|above|below)\s+than\s*$",
                prefix,
                re.I,
            ):
                return "bound_not_point"
            interval = re.search(
                r"[-+]?\d+(?:\.\d+)?"
                r"(?:\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?)?"
                r"\s*%?\s*(?:to|[-–—])\s*$",
                prefix,
                re.I,
            )
            if interval and source_id == "dili_v3":
                lead = prefix[max(0, interval.start() - 100) : interval.start()]
                directed_change = re.search(
                    r"\b(?:changed|decreased|declined|fell|increased|lowered"
                    r"|raised|reduced|rose)\b[^;,.]{0,60}\bfrom\s*$",
                    lead,
                    re.I,
                )
                if not directed_change:
                    return "bound_not_point"
            series_scope = source_id == "dili_v3" or (
                source_id == "dili_v4"
                and re.search(r"\bvalues?\b[^;]{0,100}\bat\b", basis, re.I)
            )
            if series_scope and (
                re.search(
                    r"(?:[-+]?\d+(?:\.\d+)?"
                    r"(?:\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?)?\s*,\s*){2}"
                    r"[^;]{0,12}(?:and|or)\s*$",
                    prefix,
                    re.I,
                )
                or re.match(
                    r"\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?\s*,\s*"
                    r"[-+]?\d+(?:\.\d+)?"
                    r"(?:\s*(?:±|\+/-)\s*[-+]?\d+(?:\.\d+)?)?\s*,\s*"
                    r"[-+]?\d",
                    suffix,
                    re.I,
                )
            ):
                return "multiple_or_missing_point_candidates"

    quantitative_type = str(
        row.get("quantitative_measure_type") or ""
    ).strip().lower()
    if (
        source_id == "dili_v5"
        and quantitative_type in _V5_EXACT_POTENCY_RULE_IDS
        and re.match(r"^1e[-+]\d+\s+\S", declared_unit, re.I)
        and re.search(
            r"(?<![A-Za-z0-9])"
            + re.escape(quantitative_type)
            + r"\s*[<>≤≥]",
            support,
            re.I,
        )
    ):
        return "bound_not_point"
    if (
        source_id == "dili_v5"
        and quantitative_type in _V5_EXACT_POTENCY_RULE_IDS
        and declared_unit
        and _measurement_is_bare_point(measurement_text, measurement)
        and _has_conflicting_repeated_potency(
            support, quantitative_type, measurement, declared_unit
        )
    ):
        return "conflicting_focal_statement"

    band_abbreviation = re.fullmatch(r"(?:RBI|RI|RBD)", declared_unit, re.I) and re.search(
        r"\b(?:band|relative intensity|western blot|protein expression)\b",
        basis,
        re.I,
    )
    if (
        band_abbreviation
        or (
            source_id == "dili_v4"
            and re.fullmatch(r"%\s+of\s+total", declared_unit, re.I)
            and re.search(r"\bLDH\s+release\b", basis, re.I)
        )
        or (
            source_id == "dili_v5"
            and re.fullmatch(r"(?:CAT|enzyme)\s+activity\s+units?", declared_unit, re.I)
        )
        or (
            source_id == "dili_v5"
            and re.search(
                r"\breaction\s+strength\s+\+/\+\+/\+\+\+",
                declared_unit,
                re.I,
            )
        )
        or re.fullmatch(
            r"(?:AI value|index\s*\(dimensionless\)|IC50"
            r"|ALT\s*\(units not stated\))",
            declared_unit,
            re.I,
        )
        or re.fullmatch(
            r"%\s*immunostaining|%\s*/\s*g liver|cells_per_high-power_field",
            declared_unit,
            re.I,
        )
        or re.match(r"^%\s*U\s*/", declared_unit, re.I)
        or re.fullmatch(
            r"GPX-4 fluorescence intensity\s*\(arbitrary\)",
            declared_unit,
            re.I,
        )
        or re.match(
            r"^(?:H2O2(?:\s+decomposition)?|GSH-CDNB\s+conjugate|CDNB-GSH)\s*/",
            declared_unit,
            re.I,
        )
    ):
        return "unsupported_or_uncertain_unit"
    if re.search(
        r"\bby\s+(?:day|hour|hr|minute|min)\s*\d", declared_unit, re.I
    ):
        return "unit_contains_time_context"

    for selected in _equivalent_number_spans(support, measurement):
        if re.search(
            r"\b(?:reduced|decreased|fell|lowered)\s+from\s*$",
            support[max(0, selected.start() - 50) : selected.start()],
            re.I,
        ):
            return "control_or_comparator_not_outcome"

    if re.search(r"\b(?:survival|viability)\b", declared_unit, re.I):
        number = _equivalent_number_token_pattern(measurement)
        if re.search(
            number + r"\s*%\s*viability[^.;]{0,80}\bmortality reduction\b",
            support,
            re.I,
        ) or re.search(
            r"\bmortality\b[^.;]{0,100}" + number + r"\s*%",
            support,
            re.I,
        ):
            return "conflicting_focal_statement"

    if re.fullmatch(r"[fpnumµμ]?M\s*/\s*mL", declared_unit, re.I) and re.search(
        r"\b(?:IC|EC|LC)50\s+concentration\b[^.;]{0,60}"
        + _equivalent_number_token_pattern(measurement),
        support,
        re.I,
    ):
        return "dose_or_concentration_not_outcome"

    specific_endpoint = str(row.get("specific_endpoint_name") or "").strip()
    if (
        source_id == "dili_v5"
        and not declared_unit
        and "pearson" in specific_endpoint.casefold()
        and unit.casefold() != specific_endpoint.casefold()
        and re.fullmatch(
            r"Pearson(?:'s)?(?:\s+(?:correlation|colocalization))?"
            r"\s+coefficient(?:\s+R)?",
            unit,
            re.I,
        )
    ):
        return "incomplete_named_metric"
    return None


def _guard_reason(row: Mapping[str, Any], measurement: str, unit: str) -> str | None:
    """Return why an `ok` answer lacks support in its exact visible source row."""
    measurement_text = str(row.get("measurement_text") or "")
    visible_values = _visible_values(row)
    source_reason = _source_specific_guard_reason(row, measurement, unit)
    if source_reason is not None:
        return source_reason
    selected_named_endpoint = any(
        _looks_like_named_concentration_endpoint(
            row,
            measurement_text[max(0, start - 100) : start],
            unit,
        )
        for start, _, _ in _selected_number_spans(measurement_text, measurement)
    )
    if _DOSE_ONLY_TEXT.search(measurement_text) and not selected_named_endpoint:
        return "dose_or_concentration_not_outcome"
    if re.search(
        r"\bno\s+exact\b[^;,.]{0,30}\bvalue\s+(?:was\s+)?reported\b",
        measurement_text,
        re.IGNORECASE,
    ):
        return "missing_exact_focal_value"
    if _selected_is_exposure_condition(row, measurement, unit):
        return "dose_or_concentration_not_outcome"
    if _selected_is_sample_size(measurement_text, measurement, unit):
        return "sample_size_not_outcome"
    if _selected_is_comparator(measurement_text, measurement, unit):
        return "control_or_comparator_not_outcome"
    if _measurement_is_bare_point(measurement_text, measurement):
        supporting_values = [
            str(row.get("support_text") or ""),
        ]
        if any(
            value and _selected_is_bound(value, measurement)
            for value in supporting_values
        ):
            return "bound_not_point"
        if any(
            value and _selected_is_comparator(value, measurement, unit)
            for value in supporting_values
        ):
            return "control_or_comparator_not_outcome"
        if any(
            value and _selected_is_relative(value, measurement, unit)
            for value in supporting_values
        ):
            return "relative_value"
    if any(
        re.search(
            r"\b(?:differentially|differently|significantly)\s+"
            r"(?:expressed|affected)|\bderegulated\b",
            value,
            re.IGNORECASE,
        )
        for value in visible_values
    ):
        return "comparison_defined_count"
    if re.search(
        r";[^;]*\bno significant (?:change|effect) in\b",
        measurement_text,
        re.IGNORECASE,
    ):
        return "conflicting_focal_statement"
    if re.search(r";[^;]*\bno effect on\b", measurement_text, re.IGNORECASE):
        return "different_qualitative_focal_outcome"
    if re.match(
        r"\s*no\s+(?:significant\s+)?(?:change|effect|reduction)\b",
        measurement_text,
        re.IGNORECASE,
    ) and not _CONCENTRATION_ENDPOINT.search(measurement_text):
        return "different_qualitative_focal_outcome"
    if _selected_is_bound(measurement_text, measurement):
        return "bound_not_point"
    quantitative_type = str(row.get("quantitative_measure_type") or "").lower()
    if any(
        marker in quantitative_type
        for marker in ("relative", "fold", "change", "control_normalized")
    ):
        return "relative_value"
    if _selected_is_relative(measurement_text, measurement, unit):
        return "relative_value"
    if any(_has_relative_referent(value) for value in _basis_values(row)):
        return "relative_value"
    numbers = _central_numbers(measurement_text, unit)
    if measurement not in {_coefficient(number) for number in numbers}:
        return "measurement_not_exact_source_point"
    unit_reason = _unit_guard_reason(row, measurement, unit)
    if unit_reason is not None:
        return unit_reason
    if not _multiple_values_supported(measurement_text, measurement, unit):
        return "multiple_or_missing_point_candidates"
    return None


def guard_model_assignment(
    row: Mapping[str, Any], assignment: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when a model `ok` cannot be verified from its exact input."""
    if assignment.get("status") != "ok":
        assignment["assignment_guard_reason"] = None
        return assignment
    entries = json.loads(str(assignment.get("measurements_json") or "[]"))
    if len(entries) != 1:
        reason = "not_one_measurement"
    else:
        reason = _guard_reason(
            row,
            str(entries[0].get("measurement") or "").strip(),
            str(entries[0].get("unit") or "").strip(),
        )
    if reason is None:
        assignment["assignment_guard_reason"] = None
        return assignment
    assignment.update(
        status="unsure",
        measurements_json="[]",
        quantity_count=0,
        assignment_method="model_guarded_unsure",
        assignment_guard_reason=reason,
        rejected_response_json=None,
    )
    return assignment


def route_measurement(row: Mapping[str, Any]) -> RouteDecision:
    """Settle only explicitly typed V5 potency pairs without model inference."""
    source_id = str(row.get("source_id") or "")
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    source_measurement = measurement_source_text(row.get("measurement_text"))
    if not has_digit(source_measurement):
        return RouteDecision("reject", NO_DIGIT_RULE_ID)
    measurement = number_text(source_measurement)
    unit = str(row.get("unit_text") or "").strip()
    quantitative_type = str(row.get("quantitative_measure_type") or "").strip().lower()
    rule_id = (
        _V5_EXACT_POTENCY_RULE_IDS.get(quantitative_type)
        if source_id == "dili_v5"
        else None
    )
    if rule_id is None or not unit or measurement is None:
        return RouteDecision("extract")
    if not _is_dose_or_concentration_unit(unit):
        return RouteDecision("extract")
    if not _typed_potency_pair_is_explicit(
        row, quantitative_type, measurement, unit
    ):
        return RouteDecision("extract")
    if _guard_reason(row, measurement, unit) is not None:
        return RouteDecision("extract")
    return RouteDecision("accept", rule_id, measurement, unit)


def validate_persisted_measurement_route(
    row: Mapping[str, Any], persisted_route: str
) -> None:
    """Refuse generation from a cleaned artifact built with older DILI routing."""
    fields = {
        "measurement_resolution_rule_id",
        "measurement_resolution_exact_measurement",
        "measurement_resolution_exact_unit",
        "measurement_resolution_exact_unit_is_canonical",
    }
    missing = fields - set(row)
    if missing:
        raise ValueError(
            "DILI persisted measurement decision lacks fields: "
            f"{sorted(missing)}"
        )
    expected = route_measurement(row)
    found = RouteDecision(
        bucket=persisted_route,
        rule_id=row.get("measurement_resolution_rule_id"),
        measurement_text=row.get("measurement_resolution_exact_measurement"),
        unit_text=row.get("measurement_resolution_exact_unit"),
        unit_is_canonical=bool(
            row.get("measurement_resolution_exact_unit_is_canonical")
        ),
    )
    if found != expected:
        raise ValueError(
            "DILI persisted measurement decision differs from the active Stage-1 "
            f"policy: expected={expected!r}, found={found!r}, "
            f"cleaned_record_id={row.get('cleaned_record_id')!r}"
        )


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    """Route the normalized projection of each source's declared result fields."""
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            measurement_field="measurement_text",
            unit_field="unit_text" if unit_field else "",
            require_positive_value=False,
        )
        for source_id, (_, unit_field) in SOURCE_MEASUREMENT_FIELDS.items()
    }


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    """Return source fields that identify the raw endpoint and selected result."""
    if source_id not in SOURCE_MEASUREMENT_FIELDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    _, unit_field = SOURCE_MEASUREMENT_FIELDS[source_id]
    fields = ["endpoint_name", "measurement_text"]
    if unit_field:
        fields.append("unit_text")
    fields.extend(("support_text", *_CONTEXT_FIELDS[source_id]))
    return tuple(fields)


def canonical_endpoint_record(record: Mapping[str, Any]) -> str:
    """Derive the batching identity from the literal cleaned endpoint projection."""
    return canonical_endpoint_name(
        str(record.get("source_id") or ""), record.get("endpoint_name")
    )


def validate_candidate_endpoint_identity(
    record: Mapping[str, Any], selected_endpoint: str
) -> None:
    """Fail closed if a persisted alias could replace raw endpoint identity."""
    expected = canonical_endpoint_record(record)
    if selected_endpoint != expected:
        raise ValueError(
            "DILI endpoint batching identity differs from the raw endpoint: "
            f"expected={expected!r}, found={selected_endpoint!r}, "
            f"cleaned_record_id={record.get('cleaned_record_id')!r}"
        )


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    """Normalize spelling for identity batching without making endpoint aliases."""
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return clean_literal_text(endpoint_name) or "missing_endpoint"


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    """Render one source prompt without exposing endpoint-profile summaries."""
    del endpoint_profiles
    if source_id not in SOURCE_MEASUREMENT_FIELDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    if batch_size != BATCH_SIZE:
        raise ValueError(f"DILI measurement batches are frozen at {BATCH_SIZE}")
    measurement_field, unit_field = SOURCE_MEASUREMENT_FIELDS[source_id]
    return _environment().get_template(TEMPLATE_PATH.name).render(
        batch_size=batch_size,
        row_fields=list(prompt_row_fields(source_id)),
        source_kind=_SOURCE_KINDS[source_id],
        measurement_field=measurement_field,
        unit_field=unit_field or None,
        has_unit_column=bool(unit_field),
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    """Pin the task prompt and its source-specific rendered variants."""
    return {
        "prompt_version": PROMPT_VERSION,
        "candidate_routing_version": DILI_MEASUREMENT_ROUTING_VERSION,
        "template_path": TEMPLATE_PATH.relative_to(REPO_ROOT).as_posix(),
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "canonical_endpoint_name_role": "identity_batching_only",
        "endpoint_profiles_visible": False,
        "generation_reasoning_effort": REASONING_EFFORT,
        "generation_temperature": TEMPERATURE,
        "post_generation_assignment_guard": {
            "version": ASSIGNMENT_GUARD_VERSION,
            "module_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "module_sha256": file_sha256(Path(__file__).resolve()),
        },
        "source_measurement_fields": {
            source_id: {
                "measurement_field": measurement_field,
                "unit_field": unit_field or None,
            }
            for source_id, (measurement_field, unit_field) in SOURCE_MEASUREMENT_FIELDS.items()
        },
        "source_row_fields": {
            source_id: list(prompt_row_fields(source_id)) for source_id in SOURCE_IDS
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(render_prompt(source_id).encode("utf-8")).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


def source_role_manifest() -> dict[str, object]:
    from data.processing.evidence_library.versions.v10.tasks.dili.starling_schema import (
        RECORD_CONTRACT,
        UNIT_EXCEPTIONS,
    )

    return source_role_contract(RECORD_CONTRACT.sources, UNIT_EXCEPTIONS)


def _validate_frozen_gold(path: Path) -> None:
    """Refuse a replay when its source-only labels no longer match active inputs."""
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read DILI V10 gold manifest: {path}: {exc}") from exc
    if not rows:
        raise SystemExit(f"DILI V10 gold corpus is empty: {path}")
    manifest, cases = rows[0], rows[1:]

    mismatches: dict[str, Any] = {}
    expected_values = {
        "task_id": "dili",
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "labels_frozen_before_low_reasoning_gold_pilot": True,
        "model_outputs_used_as_label_evidence": False,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            mismatches[key] = {"expected": expected, "found": manifest.get(key)}
    if not isinstance(manifest.get("prompt_at_label_freeze"), dict):
        mismatches["prompt_at_label_freeze"] = "missing frozen label-time prompt receipt"
    case_payload = "".join(
        json.dumps(
            case, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
        for case in cases
    ).encode("utf-8")
    case_ids = [str(case.get("audit_case_id") or "") for case in cases]
    case_checks = {
        "cases": len(cases),
        "reviewed_cases": len(cases),
        "source_only_cases_sha256": hashlib.sha256(case_payload).hexdigest(),
    }
    for key, expected in case_checks.items():
        if manifest.get(key) != expected:
            mismatches[key] = {"expected": expected, "found": manifest.get(key)}
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        mismatches["audit_case_ids"] = "case IDs must be nonempty and unique"
    policy = manifest.get("deterministic_acceptance_policy") or {}
    if policy.get("version") != DILI_MEASUREMENT_ROUTING_VERSION:
        mismatches["deterministic_acceptance_policy"] = {
            "expected": DILI_MEASUREMENT_ROUTING_VERSION,
            "found": policy.get("version"),
        }

    stage1 = manifest.get("stage1") or {}
    artifacts = (
        (
            "cleaned_records",
            stage1,
            "cleaned_records_path",
            "cleaned_records_sha256",
            DEFAULT_CLEANED_RECORDS,
        ),
        (
            "clean_manifest",
            stage1,
            "clean_manifest_path",
            "clean_manifest_sha256",
            DEFAULT_CLEANED_RECORDS.with_name("manifest.json"),
        ),
        (
            "endpoint_profile",
            stage1,
            "endpoint_profile_path",
            "endpoint_profile_sha256",
            DEFAULT_PROFILE_PATH,
        ),
        (
            "release_manifest",
            stage1,
            "release_manifest_path",
            "release_manifest_sha256",
            DEFAULT_CLEANED_RECORDS.parents[1] / "manifest.json",
        ),
        (
            "source_snapshot",
            manifest,
            "source_snapshot",
            "source_snapshot_sha256",
            TASK_ROOT / "source_manifest.json",
        ),
    )
    for label, block, path_key, digest_key, expected_path in artifacts:
        declared = str(block.get(path_key) or "")
        declared_path = (REPO_ROOT / declared).resolve() if declared else None
        actual_digest = file_sha256(expected_path) if expected_path.is_file() else None
        expected_digest = block.get(digest_key)
        if declared_path != expected_path.resolve() or actual_digest != expected_digest:
            mismatches[label] = {
                "expected_path": str(expected_path),
                "declared_path": declared,
                "expected_sha256": expected_digest,
                "actual_sha256": actual_digest,
            }
    if mismatches:
        raise SystemExit(f"DILI V10 gold freeze provenance mismatch: {mismatches}")


def validate_generation_args(args: Any) -> None:
    """Accept only the reviewed paid phase or the reviewed local fallback."""
    common = {
        "task": "dili",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "provider_only": None,
        "base_mapping": None,
    }
    mismatches = {
        name: {"expected": value, "found": getattr(args, name)}
        for name, value in common.items()
        if getattr(args, name) != value
    }
    paid = args.model == OPENAI_MODEL
    if paid:
        expected = {
            "base_url": OPENAI_BASE_URL,
            "provider": "openai",
            "parallelism": None,
            "provider_pool_config": None,
            "no_token_ledger": False,
            "require_complete": False,
            "defer_publication": True,
        }
        if args.api_key_env not in OPENAI_CREDENTIAL_ENVS:
            mismatches["api_key_env"] = {
                "expected": sorted(OPENAI_CREDENTIAL_ENVS),
                "found": args.api_key_env,
            }
    else:
        expected = {
            "model": DEEPSEEK_MODEL,
            "base_url": None,
            "api_key_env": None,
            "provider": None,
            "parallelism": PROVIDER_POOL_PARALLELISM,
            "no_token_ledger": True,
            "require_complete": True,
        }
        if (
            args.provider_pool_config is None
            or Path(args.provider_pool_config).resolve() != PROVIDER_POOL_CONFIG.resolve()
        ):
            mismatches["provider_pool_config"] = {
                "expected": str(PROVIDER_POOL_CONFIG),
                "found": str(args.provider_pool_config),
            }
    mismatches.update(
        {
            name: {"expected": value, "found": getattr(args, name)}
            for name, value in expected.items()
            if getattr(args, name) != value
        }
    )
    if args.two_key_baidu_run:
        mismatches["two_key_baidu_run"] = {"expected": False, "found": True}
    if (
        args.gold_replay
        and Path(args.gold_fixture).resolve() != DEFAULT_GOLD_FIXTURE.resolve()
    ):
        mismatches["gold_fixture"] = {
            "expected": str(DEFAULT_GOLD_FIXTURE),
            "found": str(args.gold_fixture),
        }
    subset = args.gold_replay or args.limit is not None or args.source is not None
    if subset and (args.mapping_path is None or args.cache_dir is None):
        mismatches["subset_artifacts"] = (
            "explicit --mapping-path and --cache-dir are required"
        )
    if (
        subset
        and args.mapping_path is not None
        and Path(args.mapping_path).resolve() == DEFAULT_MAPPING_PATH.resolve()
    ):
        mismatches["mapping_path"] = "subset run cannot write DEFAULT_MAPPING_PATH"
    if mismatches:
        raise SystemExit(f"DILI V10 generation contract mismatch: {mismatches}")
    if args.gold_replay:
        _validate_frozen_gold(Path(args.gold_fixture))


def _stage1_candidate_uids(path: Path) -> dict[str, tuple[str, str]]:
    """Return each Stage 1 extraction candidate's source UID and source ID."""
    import pyarrow.parquet as pq

    required = {
        "cleaned_record_id",
        "measurement_text",
        "source_id",
        "source_row_uid",
        "measurement_resolution_route",
        "measurement_resolution_rule_id",
        "measurement_resolution_exact_measurement",
        "measurement_resolution_exact_unit",
        "measurement_resolution_exact_unit_is_canonical",
    }
    required.update(
        field
        for source_id in SOURCE_IDS
        for field in prompt_row_fields(source_id)
    )
    available = set(pq.read_schema(path).names)
    missing = required - available
    if missing:
        raise ValueError(f"DILI V10 Stage-1 records lack columns: {sorted(missing)}")
    candidates: dict[str, tuple[str, str]] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=sorted(required)):
        for row in batch.to_pylist():
            validate_persisted_measurement_route(
                row, str(row["measurement_resolution_route"] or "")
            )
            if (
                row["source_id"] not in SOURCE_IDS
                or row["measurement_resolution_route"] != "extract"
            ):
                continue
            record_id = str(row["cleaned_record_id"] or "")
            source_uid = str(row["source_row_uid"] or "")
            source_id = str(row["source_id"] or "")
            if not record_id or not source_uid or record_id in candidates:
                raise ValueError(
                    "DILI V10 Stage-1 extraction candidate has missing or duplicate identity"
                )
            candidates[record_id] = (source_uid, source_id)
    return candidates


def _validate_legacy_deepseek_mapping_provenance(
    mapping_path: str | Path,
    *,
    expected_record_ids: set[str] | None = None,
) -> None:
    """Reject incomplete mappings and assignments outside the fixed dgx027 run."""
    import pyarrow.parquet as pq

    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"DILI V10 extraction or manifest not found: {path}")
    if not DEFAULT_CLEANED_RECORDS.is_file():
        raise ValueError(f"DILI V10 Stage-1 records not found: {DEFAULT_CLEANED_RECORDS}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = pq.read_table(path)
    row_count = table.num_rows
    stage1_candidates = _stage1_candidate_uids(DEFAULT_CLEANED_RECORDS)
    selected_ids = set(stage1_candidates) if expected_record_ids is None else set(
        expected_record_ids
    )
    unknown_expected = selected_ids - set(stage1_candidates)
    observed_ids = {
        str(value or "") for value in table.column("cleaned_record_id").to_pylist()
    }
    expected = {
        "task_id": "dili",
        "mapping_version": MAPPING_VERSION,
        "model": DEEPSEEK_MODEL,
        "models": [DEEPSEEK_MODEL],
        "api_base_url": "mixed",
        "mapping_rows": row_count,
        "cleaned_records_sha256": file_sha256(DEFAULT_CLEANED_RECORDS),
        "mapping_sha256": file_sha256(path),
        "base_mapping": None,
        "prompt": prompt_manifest(),
        "inference_model_counts": {DEEPSEEK_MODEL: row_count},
        "credential_counts": {DEEPSEEK_CREDENTIAL_ENV: row_count},
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if set(manifest.get("api_base_urls") or []) != set(PROVIDER_POOL_BASE_URLS):
        mismatches["api_base_urls"] = {
            "expected": list(PROVIDER_POOL_BASE_URLS),
            "found": manifest.get("api_base_urls"),
        }
    base_url_counts = manifest.get("inference_base_url_counts") or {}
    if (
        set(base_url_counts) != set(PROVIDER_POOL_BASE_URLS)
        or sum(int(value) for value in base_url_counts.values()) != row_count
    ):
        mismatches["inference_base_url_counts"] = {
            "expected_urls": list(PROVIDER_POOL_BASE_URLS),
            "expected_rows": row_count,
            "found": base_url_counts,
        }
    if unknown_expected:
        mismatches["expected_record_ids"] = {
            "unknown_count": len(unknown_expected),
            "first_unknown": min(unknown_expected),
        }
    if observed_ids != selected_ids:
        missing_ids = selected_ids - observed_ids
        extra_ids = observed_ids - selected_ids
        mismatches["candidate_coverage"] = {
            "expected_rows": len(selected_ids),
            "observed_rows": len(observed_ids),
            "missing_rows": len(missing_ids),
            "extra_rows": len(extra_ids),
            "first_missing": min(missing_ids) if missing_ids else None,
            "first_extra": min(extra_ids) if extra_ids else None,
        }
    inference = manifest.get("inference") or {}
    expected_inference = {
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_mode": REASONING_EFFORT,
        "temperature": TEMPERATURE,
    }
    if inference != expected_inference:
        mismatches["inference"] = {
            "expected": expected_inference,
            "found": inference,
        }
    delta = manifest.get("delta_inference") or {}
    expected_delta = {"model": DEEPSEEK_MODEL, "models": [DEEPSEEK_MODEL], "rows": row_count}
    if delta != expected_delta:
        mismatches["delta_inference"] = {"expected": expected_delta, "found": delta}
    if row_count < 1:
        mismatches["mapping_rows"] = "expected at least one extraction row"
    if manifest.get("rejected_rows") != 0:
        mismatches["rejected_rows"] = {
            "expected": 0,
            "found": manifest.get("rejected_rows"),
        }
    for key in (
        "one_row_per_candidate",
        "unique_cleaned_record_ids",
        "only_ok_carries_measurements",
        "maximum_measurements_per_row",
    ):
        if (manifest.get("validations") or {}).get(key) is not True:
            mismatches[key] = "validation did not pass"

    required_columns = {
        "assignment_guard_reason",
        "cleaned_record_id",
        "source_id",
        "source_row_uid",
        "inference_source",
        "inference_model",
        "returned_model",
        "inference_base_url",
        "inference_credential_env",
        "rejected_response_json",
        "requested_provider",
        "served_provider",
    }
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        mismatches["mapping_columns"] = {"missing": sorted(missing_columns)}
    else:
        guarded_reasons: list[str] = []
        for row in table.select(
            sorted(required_columns | {"assignment_method", "quantity_count", "status", "raw_response_json"})
        ).to_pylist():
            if not row["source_row_uid"]:
                mismatches["source_row_uid"] = "every row must retain source identity"
                break
            observed = {
                "inference_source": row["inference_source"],
                "inference_model": row["inference_model"],
                "returned_model": row["returned_model"],
                "inference_credential_env": row["inference_credential_env"],
            }
            expected_row = {
                "inference_source": "delta_inference",
                "inference_model": DEEPSEEK_MODEL,
                "returned_model": DEEPSEEK_MODEL,
                "inference_credential_env": DEEPSEEK_CREDENTIAL_ENV,
            }
            if observed != expected_row:
                mismatches["row_inference_provenance"] = {
                    "expected": expected_row,
                    "found": observed,
                }
                break
            if row["inference_base_url"] not in PROVIDER_POOL_BASE_URLS:
                mismatches["row_inference_base_url"] = {
                    "expected": list(PROVIDER_POOL_BASE_URLS),
                    "found": row["inference_base_url"],
                }
                break
            record_id = str(row["cleaned_record_id"])
            expected_source = stage1_candidates.get(record_id)
            found_source = (str(row["source_row_uid"] or ""), str(row["source_id"] or ""))
            if found_source != expected_source:
                mismatches["source_identity"] = {
                    "cleaned_record_id": record_id,
                    "expected": expected_source,
                    "found": found_source,
                }
                break
            served_provider = str(row["served_provider"] or "")
            if (
                row["requested_provider"] not in (None, "")
                or served_provider not in PROVIDER_POOL_NAMES
            ):
                mismatches["row_provider_provenance"] = {
                    "served_provider": row["served_provider"],
                    "requested_provider": row["requested_provider"],
                }
                break
            if row["rejected_response_json"] not in (None, ""):
                mismatches["rejected_response_json"] = {
                    "cleaned_record_id": record_id,
                    "found": row["rejected_response_json"],
                }
                break
            guard_reason = str(row["assignment_guard_reason"] or "")
            is_guarded = row["assignment_method"] == "model_guarded_unsure"
            if bool(guard_reason) != is_guarded:
                mismatches["assignment_guard_row"] = {
                    "cleaned_record_id": record_id,
                    "assignment_method": row["assignment_method"],
                    "assignment_guard_reason": row["assignment_guard_reason"],
                }
                break
            if is_guarded and (
                row["status"] != "unsure"
                or int(row["quantity_count"]) != 0
                or not row["raw_response_json"]
            ):
                mismatches["assignment_guard_row"] = {
                    "cleaned_record_id": record_id,
                    "status": row["status"],
                    "quantity_count": row["quantity_count"],
                    "has_raw_response": bool(row["raw_response_json"]),
                }
                break
            if guard_reason:
                guarded_reasons.append(guard_reason)
        expected_guard = {
            "version": ASSIGNMENT_GUARD_VERSION,
            "guarded_rows": len(guarded_reasons),
            "reason_counts": {
                reason: guarded_reasons.count(reason)
                for reason in sorted(set(guarded_reasons))
            },
        }
        if manifest.get("assignment_guard") != expected_guard:
            mismatches["assignment_guard"] = {
                "expected": expected_guard,
                "found": manifest.get("assignment_guard"),
            }
    returned_usage = set((manifest.get("api_usage_by_returned_model") or {}).keys())
    if returned_usage != {DEEPSEEK_MODEL}:
        mismatches["api_usage_by_returned_model"] = {
            "expected_models": [DEEPSEEK_MODEL],
            "found_models": sorted(returned_usage),
        }
    served_provider_keys = set((manifest.get("served_provider_counts") or {}).keys())
    if served_provider_keys != set(PROVIDER_POOL_NAMES):
        mismatches["served_provider_counts"] = {
            "expected": list(PROVIDER_POOL_NAMES),
            "found": sorted(served_provider_keys),
        }
    if mismatches:
        raise ValueError(f"DILI V10 extraction provenance mismatch: {mismatches}")


def validate_mapping_provenance(
    mapping_path: str | Path,
    *,
    expected_record_ids: set[str] | None = None,
) -> None:
    """Validate the approved mixed GPT/local build and the DILI guard receipt."""
    import pyarrow.parquet as pq

    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        validate_full_mapping_provenance,
    )

    validate_full_mapping_provenance(
        mapping_path, task="dili", expected_record_ids=expected_record_ids
    )
    path = Path(mapping_path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    table = pq.read_table(
        path, columns=["assignment_guard_reason", "assignment_method"]
    )
    reasons = [
        str(row["assignment_guard_reason"])
        for row in table.to_pylist()
        if row["assignment_guard_reason"]
    ]
    expected_guard = {
        "version": ASSIGNMENT_GUARD_VERSION,
        "guarded_rows": len(reasons),
        "reason_counts": {
            reason: reasons.count(reason) for reason in sorted(set(reasons))
        },
    }
    if manifest.get("assignment_guard") != expected_guard:
        raise ValueError(
            "DILI V10 assignment-guard provenance mismatch: "
            f"expected={expected_guard}, found={manifest.get('assignment_guard')}"
        )


__all__ = [
    "ALLOW_REBATCH_UNATTEMPTED",
    "ASSIGNMENT_GUARD_VERSION",
    "AUTO_VALIDATE_GENERATED_MAPPING",
    "BATCH_SIZE",
    "DEFAULT_BASE_MAPPING_PATH",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_GOLD_FIXTURE",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "DILI_MEASUREMENT_ROUTING_VERSION",
    "ENDPOINT_CONCURRENCY_BUDGET",
    "MAPPING_VERSION",
    "MAX_COMPLETION_TOKENS",
    "MAX_MEASUREMENTS_PER_ROW",
    "DEEPSEEK_CREDENTIAL_ENV",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_PROVIDER",
    "PROMPT_VERSION",
    "REASONING_EFFORT",
    "RETRY_SEMANTIC_ERRORS",
    "RETRY_VALIDATION_FEEDBACK",
    "REQUIRE_SOURCE_ROW_UID",
    "SOURCE_IDS",
    "SOURCE_MEASUREMENT_FIELDS",
    "STRATIFY_BATCHES",
    "UNIT_RECONCILIATION_CONTEXT_FIELDS",
    "TEMPERATURE",
    "canonical_endpoint_name",
    "canonical_endpoint_record",
    "guard_model_assignment",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "route_measurement",
    "source_routing_rules",
    "source_role_manifest",
    "validate_mapping_provenance",
    "validate_generation_args",
    "validate_persisted_measurement_route",
]
