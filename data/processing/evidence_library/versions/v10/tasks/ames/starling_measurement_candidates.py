"""Deterministic source-anchored AMES scalar candidate grammar."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
SCI = re.compile(
    rf"(?:(?P<coefficient>{NUMBER})\s*[×xX*]\s*)?10\s*"
    r"(?:\^\s*(?P<exponent_caret>[−–+-]?\d+)|"
    r"(?P<exponent_signed>[−–+-]\d+))"
)
E_SCI = re.compile(rf"(?P<coefficient>{NUMBER})[eE](?P<exponent>[+-]?\d+)")
V4_SCI = re.compile(
    rf"(?:(?P<coefficient>{NUMBER})(?:\s*\[\s*{NUMBER}\s*"
    rf"(?:-|–|—|to)\s*{NUMBER}\s*\])?\s*[×xX*]\s*)?10\s*"
    r"(?:\^\s*(?P<exponent_caret>[−–+-]?\d+)|"
    r"(?P<exponent_signed>[−–+-]\d+))"
)
V4_COMPETING_METRIC = re.compile(
    r"\b(?:absorbance|coefficient|factor|fraction|frequenc(?:y|ies)|index|"
    r"km|potency|rates?|ratio|score|vmax)\b",
    re.IGNORECASE,
)
V4_CHANGED_TO = re.compile(
    r"\b(?:fell|rose|increased|decreased|dropped)\s+to\s*$", re.IGNORECASE
)
V4_RELATIVE_FOLLOW = re.compile(
    r"\s*(?:versus|vs\.?|relative\s+to|compared\s+(?:with|to))\b", re.IGNORECASE
)
V4_SCI_COMPARISON_FOLLOW = re.compile(
    rf"\s*(?:versus|vs\.?)\s*{NUMBER}\s*"
    r"(?:[×xX*]\s*10\s*(?:\^\s*[−–+-]?\d+|[−–+-]\d+)|[eE][+-]?\d+)",
    re.IGNORECASE,
)
V4_CLAUSE_END = re.compile(r"\r?\n|[;!?]|\.(?!\d)")
V4_NEW_RESULT = re.compile(
    r"(?:[;:!?\n—–]|\.(?!\d)|\b(?:but|however|whereas|while)\b|"
    r"\b(?:is|are|was|were|assessed|measured|recorded|evaluated|tested)\b"
    r"(?:[^.;!?\n]|\.(?=\d)){0,80}\band\b)"
    r"[\s\"'”’)\]]*(?P<subject>(?:the\s+)?"
    r"(?:[A-Za-z][A-Za-z0-9-]*\s+){1,8})"
    r"(?:=|:|\b(?:is|are|was|were|gave|produced|reported|showed|yielded|"
    r"measured|recorded|assessed|equaled|reached)\b)\s*$",
    re.IGNORECASE,
)
V4_ANAPHORIC_VALUE = re.compile(rf"\s*(?:is|was|were)\s+{NUMBER}\b", re.IGNORECASE)
V4_DIRECT_CUE = re.compile(
    r"(?:=|:|\b(?:is|was|were|of|to|at|about|around|approximately|"
    r"approx\.?|being|estimated\s+as|reported\s+as))\s*~?\s*$",
    re.IGNORECASE,
)
V4_FIXED_READOUT = re.compile(
    r"(?P<unit>\brecombination rate\b|\bGCR rate\b|"
    r"\b[A-Za-z][A-Za-z0-9-]*-inducing potency \([A-Za-z][A-Za-z0-9_-]{1,12}\)|"
    r"\b[A-Za-z0-9][A-Za-z0-9-]*-resistant mutant frequency\b|"
    r"\bmean [A-Za-z][A-Za-z0-9-]* mutation frequency\b|"
    r"\b[A-Za-z0-9][A-Za-z0-9-]*-induced mutation rate\b|"
    r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+){2,} "
    r"mutation-frequency constant [A-Za-z]\b|"
    r"\brates? of (?:reversion|mutation|recombination) to "
    r"[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4}?"
    r"(?=\s+(?:is|are|was|were)\b|\s+of\s+(?:cells?|groups?|samples?|strains?)\b|[,.;])|"
    r"\bfrequency of [A-Za-z][A-Za-z-]*"
    r"(?:\s+[A-Za-z][A-Za-z-]*){0,3}\s+recombinants\b)",
    re.IGNORECASE,
)
V4_FIXED_RESULT_CUE = re.compile(
    r"(?:=|:|\b(?:is|was|were|being))\s*"
    r"(?:approximately|about|around|probably\s+around)?\s*~?\s*$",
    re.IGNORECASE,
)
GROUPED_NUMBER = re.compile(r"(?<![\w,])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?![\w,])")
GROUPED_BOUND_PREFIX = re.compile(
    r"(?:[<>≤≥]|\b(?:less|fewer|more|greater)\s+than|\b(?:up|down)\s+to|"
    r"\b(?:at\s+least|at\s+most)|\b(?:over|under|above|below|exceed(?:s|ed|ing)?)|"
    r"\bto)\s*~?\s*$",
    re.IGNORECASE,
)
GROUPED_DOSE_UNIT = re.compile(
    r"(?:[fpnµμum]?mol(?:\s*/\s*l)?|[fpnµμum]?m|ppm|ppb|"
    r"[npµμum]?g(?:\s*/\s*(?:kg|l))?)\b",
    re.IGNORECASE,
)
V4_NONRESULT_SCI_SUFFIX = re.compile(
    rf"\s*(?:(?:degrees?\s*|°\s*)?(?:[CFK]\b|Celsius\b|Fahrenheit\b|Kelvin\b)|"
    r"%|percent\b|-?\s*(?:fold|times?)\b|of\s+(?:the\s+)?control\b|"
    r"(?:hours?|hrs?|days?|minutes?|mins?|seconds?|secs?)\b|"
    rf"pH(?:\s+units?)?\b|dose\s+units?\b|{GROUPED_DOSE_UNIT.pattern})",
    re.IGNORECASE,
)
V4_NONRESULT_SCI_PREFIX = re.compile(
    rf"(?:[<>≤≥±]\s*|\b(?:at\s+(?:least|most)|"
    r"(?:greater|less|fewer|more|higher|lower)\s+than|"
    r"(?:above|below|under|over)|(?:up|down)\s+to)\s*|"
    r"\b(?:between|from|rang(?:e[sd]?|ing)\s+from)\s*|"
    r"\b(?:standard\s+(?:error|deviation)|SE|SD|baseline)\s*"
    r"(?:=|:|is|was|were|of)?\s*|"
    r"\b(?:\d+(?:\.\d+)?%\s+)?confidence\s+interval\s*"
    r"(?:=|:|is|was|were|of)?\s*|"
    r"\b(?:lower|upper)\s+(?:\d+(?:\.\d+)?%\s+)?"
    r"(?:confidence\s+)?(?:bound|limit)\s*"
    r"(?:=|:|is|was|were|of)?\s*|"
    r"\b(?:\d+(?:\.\d+)?%\s+)?CI(?:\s+(?:lower|upper)\s+(?:bound|limit))?\s*"
    r"(?:=|:|is|was|were|of)?\s*|"
    rf"\b(?:between|(?:ranged?\s+)?from)\s+{NUMBER}\s+(?:and|to)\s*|"
    r"\b(?:assessed|determined|evaluated|measured|quantified|recorded|tested)\s+"
    r"(?:at|for|after)\s*)$",
    re.IGNORECASE,
)
V4_NONABSOLUTE_MEASUREMENT = re.compile(
    rf"[<>≤≥]|\b(?:at\s+(?:least|most)|"
    rf"(?:greater|less|fewer|more|higher|lower)\s+than|"
    rf"(?:above|below|under|over)|(?:up|down)\s+to)\b|"
    rf"\b(?:between\s+{NUMBER}\s+and|from\s+{NUMBER}\s+(?:to|-))",
    re.IGNORECASE,
)
INCOMPLETE_UNIT = re.compile(
    r"\b(?:denominator|scale|unit)\b[^()]{0,36}\b(?:not\s+(?:explicitly\s+)?stated|"
    r"missing|unknown|unspecified|unresolved)\b",
    re.IGNORECASE,
)
BARE_SCALE = re.compile(r"[×xX*]\s*10(?!\s*(?:\^\s*[−–+-]?\d+|[−–+-]\d+))")
TOKEN = re.compile(r"[^\s,;:=]+")
METRIC = re.compile(
    r"^(?:ratio|fraction|factor|index|frequency|rate|score|coefficient|constant|"
    r"units?|pka|pk|tosc|imax)$",
    re.IGNORECASE,
)
LEADING = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "decreased",
    "elevated",
    "enhanced",
    "for",
    "from",
    "gave",
    "giving",
    "had",
    "has",
    "have",
    "higher",
    "in",
    "increased",
    "indicated",
    "indicates",
    "indicating",
    "is",
    "its",
    "listed",
    "lists",
    "lower",
    "mean",
    "measured",
    "measuring",
    "of",
    "on",
    "remained",
    "reported",
    "reports",
    "rather",
    "significant",
    "significantly",
    "showed",
    "shows",
    "similar",
    "stable",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "versus",
    "vs",
}
TEXT_FIELDS = (
    "measurement_text",
    "unit_text",
    "support_text",
    "assay_method_and_endpoint",
    "endpoint_subtype",
    "assay_version",
    "exposure_and_mechanistic_conditions",
    "endpoint_class",
    "canonical_endpoint_name",
)
EVIDENCE_FIELDS = TEXT_FIELDS[:-2]
MAX_PRIMARY_VALUES = 5
CANDIDATE_GENERATOR_VERSION = "ames_measurement_candidates.v4"
MAX_CANDIDATES = 40
MAX_CANDIDATE_FIELD_BYTES = 1_024
MAX_CANDIDATE_JSON_BYTES = 65_536
GENERIC_INFERRED_UNITS = frozenset(
    {
        "factor",
        "fraction",
        "index",
        "rate",
        "ratio",
        "relative ratio",
        "score",
        "units",
    }
)
SOURCE_DEFINED_PAIR_SPECS = (
    (
        "(R)-PEG/(S)-PEG ratio",
        r"\bPEG enantiomer\s*\(R\)\s*/\s*\(S\)\s+ratio\b",
        r"\(R\)\s*/\s*\(S\)\s+ratio\s+(?:being|was|=|of)\s*{number}",
    ),
    (
        "GSH/GSSG ratio",
        r"\bGSH(?:\s+to|\s*[:/]\s*)GSSG\s+(?:redox\s+)?ratio\b",
        r"\bGdCl3\+CLP\s+{number}\b",
    ),
    (
        "GSH/GSSG ratio",
        r"\b(?:redox\s+)?ratio\s*\(\s*GSH\s*[:/]\s*GSSG\s*\)",
        r"\bfrom\s+" + NUMBER + r"\s+to\s+{number}\b",
    ),
    (
        "k_inh/k_p ratio",
        r"\bk_inh\s*/\s*k_p\b",
        r"\bk_inh\s*/\s*k_p\b.{0,180}?\b[A-Za-z][A-Za-z-]*\s*\({number}\)",
    ),
    (
        "ICE-Δ ratio",
        r"\bICE-Δ ratio\b",
        r"\bgave an ICE-Δ of\s+{number}\b",
    ),
    (
        "anaphase/metaphase ratio",
        r"\banaphase\s*:\s*metaphase\s*\(A/M\)\s+ratio\b",
        r"\bA/M(?:\s+ratio)?\s*(?:=|of|was|is)?\s*\(?{number}\b",
    ),
    (
        "GSH-adduct/internal-standard ratio",
        r"\bGSH-adduct\s*/\s*internal-standard\s*\(IS\)\s+ratio\b",
        (
            r"\bGSH-adduct\s*/\s*internal-standard\s*\(IS\)\s+ratio[^.;]{0,64}"
            r"\bwas\s+{number}\b"
        ),
    ),
    (
        "10^8 Δ(1/Mw)",
        r"Δ\(1/Mw\)\s*[×xX*]\s*10\s*\^?\s*8",
        (
            r"Δ\(1/Mw\)\s*[×xX*]\s*10\s*\^?\s*8\)?\s+was[^;]{0,160}"
            r"\b{number}\s+in\b"
        ),
    ),
    (
        "eucaryotic/procaryotic topoisomerase II MED ratio",
        r"\beucaryotic\s*/\s*procaryotic topoisomerase II MED ratio\b",
        r"\bvalues varied between\s+{number}\b",
    ),
    (
        "dose modifying factor",
        r"\bdose modifying factor\s*\(DMF\)",
        r"\b(?:DMF\s+uncorrected|corrected)\s*=\s*{number}\b",
    ),
    (
        "fraction of micronuclei containing one kinetochore",
        r"\bfraction of micronuclei containing one kinetochore\b",
        r"\bcontained one kinetochore\s*\([^)]*\bfraction\s+{number}\b",
    ),
    (
        "Nmn1+/Nmn fraction",
        r"\bone kinetochore\s*\(Nmn1\+/Nmn\)",
        r"\bcontained one kinetochore\s*\([^)]*\bfraction\s+{number}\b",
    ),
    (
        "treated/control factor",
        r"\bFactor\s*=\s*mean of treated\s*[÷/]\s*mean of concurrent control\b",
        r"\bfactor\s+{number}\b",
    ),
    (
        "mutant IC50 / wild-type IC50 ratio",
        r"\brelative resistance ratio\s*=\s*mutant IC50\s*/\s*wild-type IC50",
        r"\bRelative resistance ratio values?[^.;]{0,80}\b{number}\b",
    ),
    (
        "mutant/wild-type IC50 ratio",
        r"\brelative resistance ratio\s*=\s*mutant IC50\s*/\s*wild-type IC50",
        r"\bRelative resistance ratio values?[^.;]{0,80}\b{number}\b",
    ),
    (
        "treated/control DNA-synthesis-rate ratio",
        (
            r"\btreated-to-control ratios?\s+for DNA synthesis rates?\b|"
            r"\bDNA synthesis rate expressed as treated-to-control ratio\b"
        ),
        (
            r"\btreated-to-control ratios?\s+for DNA synthesis rates?\s+"
            r"(?:was|were|of)\s+{number}\b"
        ),
    ),
    (
        "fractional reduction index",
        r"\bfractional reduction index\s*=",
        r"\bfractional reduction index[^.;]{0,48}\bwas\s+{number}\b",
    ),
    (
        "pK",
        r"\bpK\s*\(pKa\)\s+determination\b",
        r"\bpK shift\s+from\s+" + NUMBER + r"\s+to\s+{number}\b",
    ),
    (
        "topoisomerase-II/gyrase IC50 ratio",
        r"\bratio of IC50 for .*topoisomerase II.* to IC50 for .*gyrase",
        r"\bto\s+{number}\s+for nalidixic acid\b",
    ),
    (
        "log k_GSH",
        r"\breported as log k_GSH\b",
        r"\blog k_GSH\s+" + NUMBER + r"\s+vs\s+{number}\b",
    ),
    (
        "GSH adduct/d4-LAP-OH internal-standard peak area ratio",
        r"\bGSH adduct\s*/\s*d4-LAP-OH internal standard\b",
        r"\bkcat\s*\(peak area ratio\)\s+for CYP3A4\s*\({number}\b",
    ),
    (
        "SOS induction factor",
        r"\bSOS (?:response|induction)|\binduction factors?\b|\bIF\b",
        (
            r"\bIF(?:\s+at)?(?:\s*" + NUMBER + r"\s*µM(?:\s+H2O2)?)?"
            r"\s*(?:=|:)\s*{number}\b"
        ),
    ),
    (
        "SOS induction factor",
        r"\bSOS (?:response|induction)|\binduction factors?\b",
        r"\binduction factor\s+(?:was\s+(?:about\s+)?|of\s+){number}\b",
    ),
    (
        "detected isomers",
        r"\bdetected isomers\b|\bisomers detected\b",
        r"\b{number}\s+of\s+" + NUMBER + r"\s+isomers detected\b",
    ),
    (
        "% repaired",
        r"%\s*repaired\b",
        (
            r"\b{number}\s*%\s+of\s+[^.;]{0,40}?\s+(?:was\s+)?repaired"
            r"\s+in\s+" + NUMBER + r"\s*(?:h|hours?)\b"
        ),
    ),
    (
        "O6-methylguanine/7-methylguanine ratio",
        r"\b7-methylguanine\b[^.;]{0,120}\bO6-methylguanine\b",
        r"\bO6\s*:\s*N7\s+ratio\s+of\s+{number}\b",
    ),
    (
        "relative resistance index (IC50 HL-60/MX2/IC50 HL-60)",
        (
            r"\brelative resistance index\s*\(RRI\s*=\s*IC50\(HL-60/MX2\)"
            r"\s*/\s*IC50\(HL-60\)\)"
        ),
        r"\bRRI\s+{number}\b",
    ),
)


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace("−", "-").replace("–", "-"))
    except InvalidOperation:
        return None


def _plain(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _row_text(row: dict[str, object]) -> str:
    return " ".join(str(row.get(field) or "") for field in TEXT_FIELDS)


def _source_text(row: dict[str, object]) -> str:
    return " ; ".join(str(row.get(field) or "") for field in EVIDENCE_FIELDS)


def _sci_exponent(match: re.Match[str]) -> int:
    raw = match.group("exponent_caret") or match.group("exponent_signed")
    return int(raw.replace("−", "-").replace("–", "-"))


def _numbers(text: str) -> list[str]:
    masked = SCI.sub(lambda m: (m.group("coefficient") or "1") + " ", text)
    masked = E_SCI.sub(lambda m: m.group("coefficient") + " ", masked)
    masked = BARE_SCALE.sub(" ", masked)
    masked = masked.replace(",", "")
    values = []
    for match in re.finditer(NUMBER, masked):
        raw = match.group().lstrip("+")
        if match.start() and masked[match.start() - 1].isalnum():
            continue
        if match.end() < len(masked) and masked[match.end()].isalnum():
            continue
        if (
            match.end() + 1 < len(masked)
            and masked[match.end()] in "-'"
            and masked[match.end() + 1].isalpha()
        ):
            continue
        if _decimal(raw) is not None and raw not in values:
            values.append(raw)
    return values


def _mask_parenthetical_followups(text: str) -> str:
    masked = list(text)
    for match in re.finditer(r"\([^()]*\)", text):
        if _numbers(text[: match.start()]) and _numbers(match.group()):
            masked[match.start() : match.end()] = " " * len(match.group())
    return "".join(masked)


def _measurement_value_text(text: str) -> str:
    text = re.sub(rf"±\s*{NUMBER}", " ", text)
    text = re.sub(
        rf"\b(?:n|p)\s*(?:=|<|>|≤|≥)\s*{NUMBER}", " ", text, flags=re.IGNORECASE
    )
    text = _mask_parenthetical_followups(text)
    text = re.sub(
        rf"(?<![\w.]){NUMBER}\s*(?:-|–|—)\s*"
        rf"(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])(?:\s*:\s*1)?",
        " ",
        text,
    )
    text = re.sub(rf"\b{NUMBER}\s*[µμ]M\s*(?=[:=])", " ", text)
    text = re.sub(rf"(\b{NUMBER}\s*:\s*){NUMBER}\b", r"\1", text)
    text = re.sub(rf"(/\s*){NUMBER}(?=\s*(?:[A-Za-zµμ]|$))", r"\1", text)
    return re.sub(rf"(\b{NUMBER}\s+of\s+){NUMBER}\b", r"\1", text)


def _value_variants(row: dict[str, object]) -> list[str]:
    measurement = _measurement_value_text(str(row.get("measurement_text") or ""))
    raw = _numbers(measurement)
    values: list[str] = []
    for item in raw:
        variants = [item, _plain(_decimal(item))]
        for value in variants:
            if value and value not in values:
                values.append(value)
    targets = {_decimal(value) for value in values}
    if any("." not in value for value in values):
        for item in _numbers(_row_text(row)):
            if _decimal(item) in targets and item not in values:
                values.append(item)
    return values


def _selected_sentences(row: dict[str, object]) -> list[str]:
    values = _value_variants(row)
    fields = ["assay_method_and_endpoint", "measurement_text", "support_text"]
    selected: list[str] = []
    for field in fields:
        text = str(row.get(field) or "")
        chunks = re.split(r"(?<=[.;])\s+", text)
        for chunk in chunks:
            visible = field != "support_text" or any(
                v in chunk.replace(",", "") for v in values
            )
            if visible and chunk and chunk not in selected:
                selected.append(chunk)
    return selected


def _suffix_metric_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    for clause in re.split(r"[.;]", text):
        matches = list(TOKEN.finditer(clause))
        tokens = [match.group().strip("[]{}'\"") for match in matches]
        for end, token in enumerate(tokens):
            if not METRIC.match(token.strip("()")):
                continue
            for width in range(1, min(3, end + 1) + 1):
                start = matches[end - width + 1].start()
                phrase = clause[start : matches[end].end()].strip(" -")
                words = {word.lower().strip("()") for word in phrase.split()}
                slash_ratio = "/" in phrase and token.lower().strip("()") == "ratio"
                if (
                    not slash_ratio
                    and not words.intersection(LEADING)
                    and _balanced_grouping(phrase)
                    and not _contains_standalone_number(phrase)
                    and not re.match(r"^[+*/=]", phrase)
                    and not re.search(r"[=<>±]", phrase)
                    and _complete_suffix_start(clause, start)
                    and _valid_suffix_delimiters(phrase)
                ):
                    phrases.add(phrase)
    return phrases


def _balanced_grouping(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in text:
        if character in pairs.values():
            stack.append(character)
        elif character in pairs and (not stack or stack.pop() != pairs[character]):
            return False
    return not stack


def _contains_standalone_number(text: str) -> bool:
    return bool(re.search(rf"(?:^|\s){NUMBER}(?:\s|$)", text))


def _complete_suffix_start(clause: str, start: int) -> bool:
    prefix = clause[:start].rstrip()
    if not prefix or prefix[-1] in "/:+,":
        return not prefix
    if any("/" in token for token in prefix.split()[-2:]):
        return False
    return not re.search(r"\b(?:form|to|versus|vs\.?)$", prefix, re.IGNORECASE)


def _valid_suffix_delimiters(phrase: str) -> bool:
    if "," in phrase or re.search(r"\s:|:\s", phrase):
        return False
    if re.fullmatch(r"[IVXLCDM]+\s+ratio", phrase, re.IGNORECASE):
        return False
    return not re.search(r":[^\s]+$", phrase)


def _normalize_metric(phrase: str) -> set[str]:
    phrase = re.sub(r"\[(?:\d+)?[A-Z][^]]*\]", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,;.")
    variants = {phrase}
    variants.add(re.sub(r"\s*\(IS\)(?=\s+ratio$)", "", phrase, flags=re.IGNORECASE))
    if phrase.startswith("(") and phrase.endswith(")"):
        variants.add(phrase[1:-1])
    variants.add(
        re.sub(r"\bredox\s+(ratio|index)$", r"\1", phrase, flags=re.IGNORECASE)
    )
    variants.add(
        re.sub(
            r"\b(metaphase|anaphase)-(?=(?:prophase|metaphase)\b)",
            r"\1/",
            phrase,
            flags=re.IGNORECASE,
        )
    )
    variants.add(re.sub(r"\s+to\s+", "/", phrase, flags=re.IGNORECASE))
    variants.add(re.sub(r"(?<=\w):(?=\w)", "/", phrase))
    if phrase.casefold() == "partition ratio":
        variants.add("partition ratio")
    cleaned = set()
    for value in variants:
        value = re.sub(r"(?<=\d)min\b", " min", value)
        value = re.sub(r"\s*/\s*", "/", value)
        value = re.sub(r"\s+", " ", value).strip(" ,;.")
        if value:
            cleaned.add(value)
    return cleaned


def _formula_units(text: str) -> set[str]:
    units: set[str] = set()
    formula = r"[^,;()]{1,70}(?:/|÷)[^,;()]{1,70}"
    label = r"[A-Za-z][A-Za-z -]{2,45}?(?:ratio|index|factor)"
    patterns = [
        rf"(?P<label>{label})\s*\([^=()]*=\s*(?P<formula>{formula})\)",
        rf"(?P<label>{label})\s*\([^()]*\)\s*=\s*(?P<formula>{formula})",
        rf"(?P<label>{label})\s*=\s*(?P<formula>{formula})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group("label").strip()
            expr = _clean_formula(match.group("formula"))
            if name.lower().endswith("index"):
                units.add(f"{name} ({expr})")
            elif name.lower().endswith("factor"):
                units.add(f"{expr} factor")
            else:
                units.add(f"{expr} ratio")
    return units


def _parenthetical_definition_units(text: str) -> set[str]:
    units: set[str] = set()
    pattern = re.compile(
        r"\((?P<label>[A-Za-z][A-Za-z -]{2,35}?(?:index|factor|ratio)),\s*"
        r"(?P<formula>[^()]{3,70}/[^()]{3,70})\)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        label = match.group("label").strip()
        formula = _clean_formula(match.group("formula"))
        units.add(f"{label} ({formula})")
    nested = re.compile(
        r"(?P<label>relative resistance index)\s*\(RRI\s*=\s*"
        r"IC50\((?P<a>[^()]+)\)/IC50\((?P<b>[^()]+)\)\)",
        re.IGNORECASE,
    )
    for match in nested.finditer(text):
        units.add(
            f"{match.group('label')} (IC50 {match.group('a')}/IC50 {match.group('b')})"
        )
    return units


def _common_suffix_ratios(units: set[str]) -> set[str]:
    variants = set(units)
    pattern = re.compile(r"^(.+?)\s+(\S+)\s*/\s*(.+?)\s+\2\s+ratio$", re.IGNORECASE)
    for unit in units:
        if re.search(r"[()]|\b[RS]-", unit, re.IGNORECASE):
            continue
        match = pattern.match(unit)
        if match:
            variants.add(f"{match.group(1)}/{match.group(3)} {match.group(2)} ratio")
    return variants


def _clean_formula(value: str) -> str:
    value = value.replace("÷", "/")
    value = re.sub(r"\bmean of\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bconcurrent\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*/\s*", "/", value)
    return re.sub(r"\s+", " ", value).strip(" .)")


def _ratio_of_units(text: str) -> set[str]:
    units: set[str] = set()
    pattern = re.compile(
        r"ratios? of (?P<left>[^,;.()]{1,55}?) to "
        r"(?P<right>[^,;.()]{1,65}?)(?=\s*(?:=|(?:ratio|are|were|was|than|at|in|is)\b|[,;.()]|$))",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        left, right = (part.strip() for part in match.group("left", "right"))
        if not _valid_ratio_parts(left, right):
            continue
        units.add(f"{left}/{right} ratio")
    return units


def _valid_ratio_parts(left: str, right: str) -> bool:
    forbidden = re.compile(
        r"\b(?:a|an|and|are|as|at|by|compared|for|from|gave|giving|had|in|"
        r"measured|of|produced|remained|reported|showed|than|the|to|was|were|"
        r"with|yielded)\b",
        re.IGNORECASE,
    )
    lower_is = re.compile(r"\bis\b")
    return (
        1 <= len(left.split()) <= 6
        and 1 <= len(right.split()) <= 6
        and _balanced_grouping(left)
        and _balanced_grouping(right)
        and not forbidden.search(left)
        and not forbidden.search(right)
        and not lower_is.search(left)
        and not lower_is.search(right)
        and not _contains_standalone_number(left)
        and not _contains_standalone_number(right)
        and not re.search(r"[=±]", left + right)
        and "/" not in left
        and "/" not in right
        and "." not in left
        and "." not in right
        and not (
            re.fullmatch(r"[RS]-", left, re.IGNORECASE)
            and re.match(r"[RS]-", right, re.IGNORECASE)
        )
    )


def _jc1_ratio_units(text: str) -> set[str]:
    match = re.search(
        r"\bJC-1\s+monomer\s*[:/]\s*aggregates?\s+(?:fluorescence\s+)?ratio\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return set()
    qualifier = " fluorescence" if "fluorescence" in match.group().lower() else ""
    return {f"JC-1 monomer/aggregate{qualifier} ratio"}


def _semantic_rewrites(text: str) -> set[str]:
    units = _jc1_ratio_units(text)
    isotope_free = re.sub(r"\[\d+[A-Z]\]", "", text)
    if re.search(r"PEG enantiomer \(R\)/\(S\) ratio", text, re.IGNORECASE):
        units.add("(R)-PEG/(S)-PEG ratio")
    if re.search(
        r"\bO6-methylguanine\s*/\s*7-methylguanine\s+ratio\b|"
        r"\bO6\s*:\s*N7\s+ratio\b",
        isotope_free,
        re.IGNORECASE,
    ):
        units.add("O6-methylguanine/7-methylguanine ratio")
    if re.search(r"GSH(?:\s+to|:|/)\s*GSSG", text, re.IGNORECASE):
        units.add("GSH/GSSG ratio")
    if "k_inh/k_p" in text:
        units.add("k_inh/k_p ratio")
    if re.search(r"GSH-adduct\s*/\s*IS ratio", text, re.IGNORECASE):
        units.add("GSH-adduct/IS ratio")
        if re.search(r"internal[- ]standard|\(IS\)", text, re.IGNORECASE):
            units.add("GSH-adduct/internal-standard ratio")
    if "A/M" in text and "anaphase:metaphase" in text.lower():
        units.add("anaphase/metaphase ratio")
    if re.search(r"\bform II\s*/\s*form I ratio\b", text, re.IGNORECASE):
        units.add("form II/form I ratio")
    for match in re.finditer(
        r"\b\d{3}/\d{3}\s+nm fluorescence ratio\b", text, re.IGNORECASE
    ):
        units.add(match.group())
    for match in re.finditer(
        r"\b\dR,\dS:\dS,\dR(?:\s+enantiomer)?\s+ratio\b",
        text,
        re.IGNORECASE,
    ):
        units.add(match.group())
    for match in re.finditer(
        r"\b([A-Za-z0-9-]+)\s+(red/green fluorescence ratio)\b",
        text,
        re.IGNORECASE,
    ):
        units.add(f"{match.group(1)} {match.group(2)}")
    for match in re.finditer(
        r"ratio of (?P<left>[RS])- to (?P<right>[RS])-"
        r"(?P<compound>[A-Za-z][A-Za-z -]{1,45}?)"
        r"(?=\s+(?:in|is|produced|was|were)\b|[,;.])",
        text,
        re.IGNORECASE,
    ):
        compound = re.sub(
            r"^enantiomers? of\s+", "", match.group("compound"), flags=re.IGNORECASE
        ).strip()
        units.add(
            f"({match.group('left').upper()})-{compound}/"
            f"({match.group('right').upper()})-{compound} ratio"
        )
    return units


def _count_and_comparison_units(text: str) -> set[str]:
    units: set[str] = set()
    per = re.search(
        r"molecules? of (\S+) metabolized per molecule of ([^,;)]+) inactivated",
        text,
        re.IGNORECASE,
    )
    if per:
        units.add(
            f"{per.group(1)} molecules metabolized per {per.group(2)} molecule inactivated"
        )
    rate = re.search(
        r"([A-Za-z -]+ rate) expressed as treated-to-control ratio", text, re.IGNORECASE
    )
    if rate:
        metric = re.sub(r"\s+", "-", rate.group(1).strip())
        units.add(f"treated/control {metric} ratio")
    if "Factor = mean of treated ÷ mean of concurrent control" in text:
        units.add("treated/control factor")
    return units


def _physical_pairs(text: str, values: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    symbols = r"(?:nA|µm|μm|min|hours?|fmol/µg DNA|U/mg dry wt)"
    for value in values:
        for match in re.finditer(rf"{re.escape(value)}\s*({symbols})\b", text):
            pairs.add((value, match.group(1)))
    return pairs


def _literal_named_units(text: str) -> set[str]:
    units: set[str] = set()
    patterns = (
        r"\bindex of induction\b",
        r"\blog\s+(?:K\b|k_[A-Za-z0-9]+\b)",
        r"\blog tail moment\b",
        r"\brelative pseudo-first-order rate constant\b",
        r"\barbitrary rate units\b",
        r"\bfold change\b",
        r"\bmtDNA break frequency \(Bf\)",
        r"\bpartition ratio\b",
        r"\bdose modifying factor\b",
        r"\bSOS induction factor\b",
        r"\bpK\b",
    )
    for pattern in patterns:
        units.update(
            match.group() for match in re.finditer(pattern, text, re.IGNORECASE)
        )
    return units


def _linked_percent_pairs(text: str, values: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for value in values:
        number = rf"(?<![\d.]){re.escape(value)}(?![\d.])"
        direct = rf"{number}\s*(?:%|percent\b)"
        shared = rf"{number}\s*(?:versus|vs\.?)\s*{NUMBER}\s*(?:%|percent\b)"
        if re.search(direct, text, re.IGNORECASE) or re.search(
            shared, text, re.IGNORECASE
        ):
            pairs.add((value, "%"))
    return pairs


def _fraction_units(text: str) -> set[str]:
    units: set[str] = set()
    for match in re.finditer(r"fraction of ([^,;.()]{3,65})", text, re.IGNORECASE):
        tail = match.group(1)
        if re.search(
            r"\b(?:are|be|been|being|could|had|has|have|is|may|might|must|"
            r"should|to|was|were|will|would)\b",
            tail,
            re.IGNORECASE,
        ):
            continue
        phrase = re.split(r"\s+(?:and|at|in|from|with)\s+", tail)[0]
        units.add(f"fraction of {phrase.strip()}")
    for match in re.finditer(
        r"fraction of [^()]{3,80}\(([^()]+/[^()]+)\)", text, re.IGNORECASE
    ):
        units.add(f"{_clean_formula(match.group(1))} fraction")
    return units


def _ratio_label_units(text: str) -> set[str]:
    units: set[str] = set()
    for match in re.finditer(
        r"label ratio\s+([^,;.()]+/[^,;.()]+)", text, re.IGNORECASE
    ):
        formula = re.sub(r"(?<=\d)min\b", " min", match.group(1))
        units.add(f"{_clean_formula(formula)} ratio")
    match = re.search(
        r"ratio\s+([A-Za-z0-9-]+)/([A-Za-z0-9-]+)\s*=", text, re.IGNORECASE
    )
    if match and "activity" in text.lower():
        units.add(f"{match.group(1)}/{match.group(2)} activity ratio")
    return units


def _change_series_ratio_units(text: str) -> set[str]:
    pattern = re.compile(
        r"\b(?P<label>[A-Za-z][A-Za-z0-9µμΔ^+_./-]{0,45}\s+ratio)\s+"
        r"(?:declined|decreased|dropped|fell|increased|rose)\s+from\b",
        re.IGNORECASE,
    )
    return {match.group("label") for match in pattern.finditer(text)}


def _slash_ratio_units(text: str) -> set[str]:
    atom = r"[A-Za-zΑ-Ωα-ω0-9µμΔ^+()._\[\]-]+"
    pattern = re.compile(
        rf"(?P<left>{atom})\s*/\s*"
        rf"(?P<right>{atom}(?:\s+{atom}){{0,4}})\s+ratio\b",
        re.IGNORECASE,
    )
    units: set[str] = set()
    for match in pattern.finditer(text):
        prefix = text[: match.start()].rstrip()
        if prefix.endswith("/") or re.search(r"\bform$", prefix, re.IGNORECASE):
            continue
        left, right = match.group("left", "right")
        left = re.sub(r"\[\d+[A-Z][^]]*\]", "", left)
        right = re.sub(r"\[\d+[A-Z][^]]*\]", "", right)
        left = re.sub(r"\[([^]]+)\]", r"\1", left)
        right = re.sub(r"\[([^]]+)\]", r"\1", right)
        if left.startswith("(") and not _balanced_grouping(left):
            left = left[1:]
        if right.endswith(")") and not _balanced_grouping(right):
            right = right[:-1]
        if not _valid_ratio_parts(left, right):
            continue
        units.add(f"{left}/{right} ratio")
    return units


def _valid_unit_syntax(unit: str) -> bool:
    if not _balanced_grouping(unit):
        return False
    if re.search(r"(?:^|\s)[=±](?:\s|$)", unit):
        return False
    malformed_leading = re.compile(
        r"^(?:measuring|method|the|yielding|(?:in[- ]?)?vitro)\b|"
        r"^change\s+ratio$|"
        r"^(?:h|hr|hours?)\s+(?:control\s+)?ratio$|^control\s+ratio$",
        re.IGNORECASE,
    )
    if malformed_leading.search(unit) or re.search(
        r"\bindex\s*\(ratio of [^)]*/[^)]*\)$|^its\b.*\bitself\b",
        unit,
        re.IGNORECASE,
    ):
        return False
    if (
        unit.startswith("~")
        or re.match(r"^number\s*\(", unit, re.IGNORECASE)
        or (unit.startswith("(") and unit.endswith(")") and "/" not in unit)
    ):
        return False
    result_bearing = re.compile(
        r"\b(?:and|are|gave|giving|had|reported|showed|was|were)\b.*"
        r"\b(?:factor|fraction|index|ratio)\b",
        re.IGNORECASE,
    )
    return not result_bearing.search(unit)


def _scale_expression_units(text: str) -> set[str]:
    units: set[str] = set()
    for match in re.finditer(r"(Δ\([^()]+\))\s*[×xX*]\s*(10\s*\^\s*[+-]?\d+)", text):
        scale = re.sub(r"\s+", "", match.group(2))
        units.add(f"{scale} {match.group(1)}")
    return units


def _compressed_ratio_units(text: str) -> set[str]:
    units: set[str] = set()
    if re.search(
        r"ratio of IC50 for .*topoisomerase II.* to IC50 for .*gyrase",
        text,
        re.IGNORECASE,
    ):
        units.add("topoisomerase-II/gyrase IC50 ratio")
    match = re.search(
        r"ratio of total (\w+) metabolic consumption .*? to loss of (P450\s+\S+)",
        text,
        re.IGNORECASE,
    )
    if match:
        units.add(f"{match.group(1)} consumed per {match.group(2)} lost")
    return units


def _peak_area_units(text: str) -> set[str]:
    units: set[str] = set()
    match = re.search(
        r"([A-Za-z0-9-]+ adduct) peak area ratio \(([^()]+ internal standard)\)",
        text,
        re.IGNORECASE,
    )
    if match:
        formula = match.group(2).replace(" internal standard", " internal-standard")
        units.add(f"{formula} peak area ratio")
    return units


def _slash_spacing_units(units: set[str]) -> set[str]:
    variants = set(units)
    for unit in units:
        if "/" in unit:
            variants.add(re.sub(r"\s*/\s*", "/", unit))
            variants.add(re.sub(r"\s*/\s*", " / ", unit))
        variants.add(re.sub(r"203Hg\(II\)", "203Hg", unit))
    return {unit.strip() for unit in variants if unit.strip()}


def _descriptor_units(row: dict[str, object]) -> set[str]:
    units: set[str] = set()
    if str(row.get("unit_text") or "").strip():
        return units
    for field in ("endpoint_subtype", "assay_method_and_endpoint"):
        text = str(row.get(field) or "").strip()
        if " (" in text:
            units.add(text.split(" (", 1)[0])
    return units


def _correlation_pairs(text: str, values: list[str]) -> set[tuple[str, str]]:
    lower = text.lower()
    if "correlat" not in lower and not re.search(r"\br\s*=", text):
        return set()
    cue = (
        r"(?:correlation coefficient(?:\s*\(r\))?|"
        r"spearman(?:'s)?(?:\s+correlation)?(?:\s+coefficient)?(?:\s*\(r\))?|"
        r"spearman\s+r|\br|rho)"
    )
    linked: list[str] = []
    for value in values:
        number = rf"(?<![\d.]){re.escape(value)}(?!\d|\.\d)"
        pattern = rf"{cue}\s*(?:=|:|of|was|is)?\s*{number}"
        if re.search(pattern, text, re.IGNORECASE):
            linked.append(value)
    if not linked:
        return set()
    units = {
        "correlation coefficient",
        "correlation coefficient (r)",
        "correlation coefficient r",
    }
    if "spearman" in lower:
        units.add("Spearman correlation coefficient (r)")
    if "dimensionless" in lower:
        units.add("correlation coefficient (dimensionless)")
    return {(value, unit) for value in linked for unit in units}


def _named_rewrites(text: str) -> set[str]:
    units: set[str] = set()
    substitutions = {
        "A/M": "anaphase/metaphase ratio",
        "C-metaphase count": "C-metaphases",
        "polyploidy cells": "polyploid cells",
    }
    for cue, unit in substitutions.items():
        if cue.lower() in text.lower():
            units.add(unit)
    if re.search(r"\bmicronuclei\b", text, re.IGNORECASE):
        units.add("micronuclei")
    if re.search(r"\bmicronucleus counts?\b", text, re.IGNORECASE):
        units.add("micronuclei")
    if "micronucleus" in text.lower() and "frequency" in text.lower():
        units.add("micronucleus frequency")
    if "dominant lethality" in text.lower() and "calculated as" in text.lower():
        units.add("dominant lethality fraction")
    if "binucleated cells with micronuclei" in text.lower():
        units.add("binucleated cells with micronuclei")
    if "chromosomal aberrations" in text.lower():
        units.add("chromosomal aberrations")
    if re.search(r"\brecombination events?\b", text, re.IGNORECASE):
        units.add("recombination events")
    if re.search(r"\bone repeat\b", text, re.IGNORECASE):
        units.add("repeat")
    if "mutation rates" in text.lower() or "reported rate" in text.lower():
        units.add("mutation rate")
    return units


def _event_pairs(text: str, values: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    nouns = r"(?:point mutations?|resistant mutants?|recombination events?|chromosomal aberrations?|repeat)"
    for value in values:
        escaped = re.escape(value)
        for match in re.finditer(rf"{escaped}\s+({nouns})\b", text, re.IGNORECASE):
            pairs.add((value, match.group(1)))
    return pairs


def _authoritative_units(row: dict[str, object]) -> set[str]:
    unit = str(row.get("unit_text") or "").strip()
    if not unit:
        return set()
    units = {unit}
    if " (" in unit:
        units.add(unit.split(" (", 1)[0])
    if unit.lower().startswith("percent "):
        units.add("% " + unit[8:])
    if unit.startswith("% "):
        units.add("percent " + unit[2:])
    return units


def _scientific_pairs(row: dict[str, object], units: set[str]) -> set[tuple[str, str]]:
    text = _source_text(row)
    base = {_decimal(v) for v in _value_variants(row)}
    pairs: set[tuple[str, str]] = set()
    stems = _scientific_stems(text, units)
    for match in SCI.finditer(text):
        coefficient = match.group("coefficient") or "1"
        exponent = _sci_exponent(match)
        coeff_dec = _decimal(coefficient)
        if abs(exponent) > 30:
            continue
        product = coeff_dec * (Decimal(10) ** exponent)
        if coeff_dec not in base and product not in base:
            continue
        for stem in _associated_scientific_stems(text, match, stems):
            pairs.add((coefficient, f"10^{exponent} {stem}"))
    for match in E_SCI.finditer(text):
        coefficient = match.group("coefficient")
        exponent = int(match.group("exponent"))
        coeff_dec = _decimal(coefficient)
        if abs(exponent) > 30:
            continue
        if coeff_dec not in base and coeff_dec * (Decimal(10) ** exponent) not in base:
            continue
        for stem in _associated_scientific_stems(text, match, stems):
            pairs.add((coefficient, f"10^{exponent} {stem}"))
    return pairs


def _group_scientific_pairs(
    row: dict[str, object], units: set[str]
) -> set[tuple[str, str]]:
    text = _source_text(row)
    targets = set(_value_variants(row))
    stems = _scientific_stems(text, units)
    pairs: set[tuple[str, str]] = set()
    pattern = re.compile(r"\(([^()]*)\)\s*[×xX*]\s*10\s*(?:\^\s*)?([−–+-]?\d+)")
    for match in pattern.finditer(text):
        exponent = int(match.group(2).replace("−", "-").replace("–", "-"))
        if abs(exponent) > 30:
            continue
        for value in _numbers(match.group(1)):
            if value in targets:
                linked = _associated_scientific_stems(text, match, stems)
                pairs.update((value, f"10^{exponent} {stem}") for stem in linked)
    return pairs


def _near_terms(text: str, left: str, right: str) -> bool:
    gap = r"(?:\W+\w+){0,8}\W+"
    forward = rf"\b(?:{left})\w*\b{gap}\b(?:{right})\w*\b"
    reverse = rf"\b(?:{right})\w*\b{gap}\b(?:{left})\w*\b"
    return bool(
        re.search(forward, text, re.IGNORECASE)
        or re.search(reverse, text, re.IGNORECASE)
    )


def _stem_marker_spans(text: str, stem: str) -> list[tuple[int, int]]:
    marker = next(
        (
            word
            for word in ("frequency", "rate", "level", "fraction")
            if word in stem.lower()
        ),
        stem,
    )
    spans = [
        match.span() for match in re.finditer(re.escape(marker.lower()), text.lower())
    ]
    if "frequency" in stem.lower():
        spans.extend(
            match.span()
            for match in re.finditer(r"\b(?:F\s*=|MFs?\b)", text, re.IGNORECASE)
        )
    if "rate" in stem.lower():
        spans.extend(match.span() for match in re.finditer(r"(?:\bmu|µ)\s*=", text))
    return spans


def _associated_scientific_stems(
    text: str, match: re.Match[str], stems: set[str]
) -> set[str]:
    if len(stems) <= 1:
        return stems
    distances = {}
    unanchored = set()
    for stem in stems:
        markers = _stem_marker_spans(text, stem)
        if markers:
            distances[stem] = min(
                _span_distance(match.span(), marker) for marker in markers
            )
        else:
            unanchored.add(stem)
    if not distances:
        return unanchored
    nearest = min(distances.values())
    return unanchored | {
        stem for stem, distance in distances.items() if distance == nearest
    }


def _scientific_stems(text: str, units: set[str]) -> set[str]:
    del units
    stems: set[str] = set()
    lower = text.lower()
    basics = {
        "mutation frequency": _near_terms(lower, "mutat|mutant", "frequenc"),
        "mutation rate": _near_terms(lower, "mutat|mutant", "rate"),
        "reversion frequency": _near_terms(lower, "reversion", "frequenc"),
        "reversion rate": _near_terms(lower, "reversion", "rate"),
        "crossover rate": _near_terms(lower, "crossover", "rate"),
        "recombination frequency": _near_terms(lower, "recombin", "frequenc"),
        "relative adduct level": "relative adduct level" in lower,
    }
    stems.update(unit for unit, present in basics.items() if present)
    if "mfs" in lower:
        stems.add("mutation frequency")
    if "resistant mutant" in lower and "frequenc" in lower:
        stems.update(
            {
                "mutation frequency",
                "resistant mutant frequency",
                "resistant-mutant frequency",
            }
        )
    if "mutation was the basis" in lower and "frequenc" in lower:
        stems.add("mutation frequency")
    if "mutant" in lower and "frequenc" in lower:
        stems.add("mutation frequency")
    if "revertant" in lower and "scored male" in lower:
        stems.add("revertants per scored male")
    if "mutation-frequency constant m" in lower:
        stems.add("mutation-frequency constant m")
    if "mms-induced rate" in lower:
        stems.add("MMS-induced mutation rate")
    if re.search(r"\bscs\b", text, re.IGNORECASE) and "mutation rate" in lower:
        stems.add("SCS mutation rate")
    if re.search(r"\bhprt mutation frequency\b", text, re.IGNORECASE):
        stems.add("hprt mutation frequency")
    if "hgt rate" in lower or "hgt rates" in lower:
        stems.add("HGT rate")
    if "deletion" in lower and "allele" in lower and "frequenc" in lower:
        stems.add("deletion allele frequency")
    if "mutants" in lower and "cell" in lower and "spontaneous" in lower:
        stems.add("mutants/cell")
    allele = re.search(r"(tet\(A\)ΔtetR) allele", text, re.IGNORECASE)
    if allele:
        stems.add(f"{allele.group(1)} allele fraction")
    return stems


def _scaled_authoritative_pairs(row: dict[str, object]) -> set[tuple[str, str]]:
    unit = str(row.get("unit_text") or "").strip()
    text = str(row.get("measurement_text") or "")
    pairs: set[tuple[str, str]] = set()
    if not unit:
        return pairs
    match = SCI.match(text.lstrip())
    if match:
        coefficient = match.group("coefficient") or "1"
        exponent = _sci_exponent(match)
        pairs.add((coefficient, f"10^{exponent} {unit}"))
    bare = re.match(rf"\s*({NUMBER})\s*\(?\s*{BARE_SCALE.pattern}", text)
    if bare:
        pairs.add((bare.group(1).lstrip("+"), f"×10 {unit}"))
    return pairs


def _strong_unit_candidates(row: dict[str, object], source: str) -> set[str]:
    all_text = _source_text(row)
    units = _authoritative_units(row)
    if source != "premutagenic_damage":
        units.update(_named_rewrites(all_text))
        units.update(_formula_units(all_text))
        units.update(_parenthetical_definition_units(all_text))
        units.update(_semantic_rewrites(all_text))
        units.update(_count_and_comparison_units(all_text))
        units.update(_ratio_label_units(all_text))
        units.update(_scale_expression_units(all_text))
        units.update(_compressed_ratio_units(all_text))
        units.update(_peak_area_units(all_text))
        units.update(_literal_named_units(all_text))
    else:
        units.update(_descriptor_units(row))
    return _normalized_units(units, source)


def _suffix_unit_candidates(row: dict[str, object], source: str) -> set[str]:
    if source == "premutagenic_damage":
        return set()
    units: set[str] = set()
    for text in _selected_sentences(row):
        units.update(_suffix_metric_phrases(text))
    return _normalized_units(units, source)


def _association_unit_candidates(row: dict[str, object], source: str) -> set[str]:
    if source == "premutagenic_damage":
        return set()
    text = _source_text(row)
    units = _ratio_of_units(text)
    units.update(_fraction_units(text))
    units.update(_slash_ratio_units(text))
    units.update(_change_series_ratio_units(text))
    units.update(_suffix_unit_candidates(row, source))
    return _normalized_units(units, source)


def _filter_partition_units(row: dict[str, object], units: set[str]) -> set[str]:
    assay = str(row.get("assay_method_and_endpoint") or "").lower()
    if str(row.get("source_id") or "") != "mutagenicity_mechanism":
        return units
    if not re.search(r"\bpartition[- ]ratio\b", assay):
        return units
    return {
        unit
        for unit in units
        if unit.lower() == "partition ratio"
        or "metabolized per" in unit.lower()
        or "consumed per" in unit.lower()
    }


def _unit_candidates(
    row: dict[str, object], source: str, values: list[str]
) -> set[str]:
    del values
    units = _strong_unit_candidates(row, source)
    units.update(_association_unit_candidates(row, source))
    return _filter_partition_units(row, units)


def _normalized_units(units: set[str], source: str) -> set[str]:
    if source == "premutagenic_damage":
        normalized = units
    else:
        normalized = set(units)
        for unit in units:
            normalized.update(_normalize_metric(unit))
        normalized = _common_suffix_ratios(normalized)
    units = (
        _slash_spacing_units(normalized)
        if source != "premutagenic_damage"
        else normalized
    )
    return {unit for unit in units if _valid_unit_syntax(unit)}


def _direct_pairs(row: dict[str, object], values: list[str]) -> set[tuple[str, str]]:
    text = _source_text(row)
    pairs = _linked_percent_pairs(text, values)
    pairs.update(_correlation_pairs(text, values))
    pairs.update(_physical_pairs(text, values))
    pairs.update(_event_pairs(text, values))
    return pairs


def _assay_declares_competing_metrics(text: str) -> bool:
    formulas = {
        re.sub(r"\s+", "", match.group()).casefold()
        for match in re.finditer(r"\b[A-Za-z0-9()-]+\s*[:/]\s*[A-Za-z0-9()-]+\b", text)
        if re.search(
            r"\b(?:ratio|state)\b",
            text[match.end() : match.end() + 56],
            re.IGNORECASE,
        )
    }
    listed_glutathione = re.search(
        r"\btotal\s+glutathione\b.*\bGSH\b.*\bGSSG\b", text, re.IGNORECASE
    )
    ambiguous_ratio = re.search(
        r"\bratio of\b[^.;]{0,100}\bto\b[^.;]{0,100}\brelative to\b",
        text,
        re.IGNORECASE,
    )
    secondary_partition = re.search(
        r"\bpartition[- ]ratio\b[^.;]{0,48}\b(?:as well|also)\b",
        text,
        re.IGNORECASE,
    )
    return len(formulas) > 1 or bool(
        listed_glutathione or ambiguous_ratio or secondary_partition
    )


def _informative_declared_units(units: set[str]) -> set[str]:
    known = re.compile(
        r"^(?:partition ratio|dose modifying factor|SOS induction factor|"
        r"fold change|arbitrary rate units|mutation rate|pK|log\s+\S+)$",
        re.IGNORECASE,
    )
    generic = {"ratio", "factor", "fraction", "index", "rate", "score", "units"}
    selected = set()
    for unit in units:
        terms = [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9µμΔ-]+", unit)
            if token.casefold() not in generic and not token.isdigit()
        ]
        if "/" in unit or ":" in unit or len(terms) >= 2 or known.match(unit):
            selected.add(unit)
    return selected


def _assay_declared_units(row: dict[str, object], source: str) -> set[str]:
    assay = str(row.get("assay_method_and_endpoint") or "").strip()
    if not assay or _assay_declares_competing_metrics(assay):
        return set()
    declared = {field: "" for field in TEXT_FIELDS}
    declared.update(source_id=source, assay_method_and_endpoint=assay)
    units = _unit_candidates(declared, source, [])
    return _informative_declared_units(units)


def _explicit_named_pairs(
    row: dict[str, object], values: list[str], units: set[str]
) -> set[tuple[str, str]]:
    text = _source_text(row)
    patterns = {
        "micronucleus frequency": r"\bfrequency\s*(?:=|of|was|is)?\s*{number}",
        "binucleated cells with micronuclei": r"\bBNMN\s*=\s*{number}",
    }
    pairs: set[tuple[str, str]] = set()
    for unit, pattern in patterns.items():
        if unit not in units:
            continue
        for value in values:
            number = rf"(?<![\d.]){re.escape(value)}(?![\d.])"
            if re.search(pattern.format(number=number), text, re.IGNORECASE):
                pairs.add((value, unit))
    return pairs


def _semantic_pattern_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    text = _source_text(row)
    specs = (
        (
            "dominant lethality fraction",
            r"\bdominant lethality\b",
            r"\bmean\s+of\s+{number}",
        ),
        (
            "micronuclei",
            r"\bmicronucle(?:us|i)\b",
            r"\b(?:counts?[^.;]{{0,80}}(?:were\s+)?all|all)\s+{number}",
        ),
        (
            "anaphase/metaphase ratio",
            r"\b(?:anaphase:metaphase|A/M)\b",
            r"\bA/M(?:\s+ratio)?\s*(?:=|of|was|is)?\s*{number}",
        ),
        (
            "O6-methylguanine/7-methylguanine ratio",
            r"\bO6[^.;]{0,80}\b7[^.;]{0,80}\bratio\b",
            (
                r"\b(?:O6[^.;]{{0,100}}\b7[^.;]{{0,100}}\bratio|O6:N7\s+ratio)"
                r"[^.;]{{0,36}}{number}"
            ),
        ),
        (
            "desulfuration/dearylation activity ratio",
            r"\bdesulfuration\b[^.;]{0,80}\bdearylation\b",
            r"\bratio\s+desulfuration/dearylation\s*=\s*{number}",
        ),
        (
            "arbitrary rate units",
            r"\barbitrary rate units\b",
            r"{number}\s*(?:±\s*" + NUMBER + r")?\s*\(arbitrary rate units",
        ),
        ("fold change", r"\bfold change\b", r"{number}\s*-\s*fold change\b"),
    )
    pairs: set[tuple[str, str]] = set()
    for value in values:
        number = rf"(?<![\d.]){re.escape(value)}(?![\d.])"
        for unit, cue, result in specs:
            if re.search(cue, text, re.IGNORECASE) and re.search(
                result.format(number=number), text, re.IGNORECASE
            ):
                pairs.add((value, unit))
    return pairs


def _special_semantic_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    text = _source_text(row)
    specs = (
        (
            "O-6/N-7 guanine alkylation ratio",
            r"\bO-?6\s*(?::|versus|vs\.?)\s*N-?7\b[^.;]{0,48}\bguanine\b",
            (
                r"\bO-?6\s*(?::|versus|vs\.?)\s*N-?7\b[^.;]{0,80}"
                r"\bguanine\b[^.;]{0,48}\bof\s+{number}"
            ),
        ),
        (
            "TR_GSH ratio",
            r"\bTR_GSH\b",
            r"\bTR_GSH\s*=\s*{number}",
        ),
    )
    pairs: set[tuple[str, str]] = set()
    for value in values:
        for unit, cue, result in specs:
            if re.search(cue, text, re.IGNORECASE) and _pattern_links_value(
                text, result, value
            ):
                pairs.add((value, unit))
    return pairs


def _pattern_links_value(text: str, template: str, value: str) -> bool:
    target = _decimal(value)
    marker = "__FOCAL_VALUE__"
    pattern = template.replace("{number}", marker)
    for start, end, number in _number_spans(text):
        if number != target:
            continue
        marked = text[:start] + marker + text[end:]
        if re.search(pattern, marked, re.IGNORECASE):
            return True
    return False


def _source_defined_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    text = _source_text(row)
    pairs: set[tuple[str, str]] = set()
    for unit, cue, result in SOURCE_DEFINED_PAIR_SPECS:
        if not re.search(cue, text, re.IGNORECASE):
            continue
        pairs.update(
            (value, unit)
            for value in values
            if _pattern_links_value(text, result, value)
        )
    return pairs


def _apparent_pka_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    if row.get("source_id") != "mutagenicity_mechanism":
        return set()
    result = (
        r"\bapparent\s+pKa\b[^;]{0,360}?\b(?:value\s+(?:of|was\s+estimated\s+"
        r"to\s+be)|estimated\s+to\s+be)\s*{number}"
    )
    pairs: set[tuple[str, str]] = set()
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        if not re.search(r"\bapparent\s+pKa\b", text, re.IGNORECASE):
            continue
        pairs.update(
            (value, "apparent pKa")
            for value in values
            if _pattern_links_value(text, result, value)
        )
    return pairs


def _formula_ratio_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    if row.get("source_id") != "mutagenicity_mechanism":
        return set()
    pattern = re.compile(
        r"\b(?:molar\s+)?ratio\s+(?P<formula>[A-Za-z0-9][A-Za-z0-9._-]*\s*/\s*"
        r"\([A-Za-z0-9+._-]+\))",
        re.IGNORECASE,
    )
    assay = str(row.get("assay_method_and_endpoint") or "")
    support = str(row.get("support_text") or "")
    declared = {
        re.sub(r"\s*/\s*", "/", match.group("formula")).strip()
        for match in pattern.finditer(assay)
    }
    pairs: set[tuple[str, str]] = set()
    for match in pattern.finditer(support):
        formula = re.sub(r"\s*/\s*", "/", match.group("formula")).strip()
        if formula not in declared:
            continue
        for value in values:
            target = _decimal(value)
            linked = any(
                start > match.end() and start - match.end() <= 320 and number == target
                for start, _, number in _number_spans(support)
            )
            if linked:
                pairs.add((value, formula))
    return pairs


def _half_life_time_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    unit = str(row.get("unit_text") or "")
    if row.get("source_id") != "premutagenic_damage" or not re.search(
        r"half[- ]life|t1/2", unit, re.IGNORECASE
    ):
        return set()
    descriptor = re.sub(r"t1/2", "", unit, flags=re.IGNORECASE)
    time_unit = r"(?:h|hr|hours?|min|minutes?|s|sec|seconds?|d|days?)"
    if re.search(rf"(?<![A-Za-z]){time_unit}(?![A-Za-z])", descriptor, re.IGNORECASE):
        return set()
    if re.search(r"[;/]", descriptor):
        return set()
    text = str(row.get("measurement_text") or "")
    if re.search(r"[<>≤≥]|\b(?:vs\.?|versus|to)\b", text, re.IGNORECASE):
        return set()
    adjacent: list[tuple[Decimal, str]] = []
    pattern = re.compile(rf"\s*(?P<unit>{time_unit})\b", re.IGNORECASE)
    for _, end, number in _number_spans(text):
        match = pattern.match(text[end:])
        if match:
            adjacent.append((number, match.group("unit")))
    if len(adjacent) != 1:
        return set()
    return {
        (value, adjacent[0][1]) for value in values if _decimal(value) == adjacent[0][0]
    }


def _dna_length_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    if row.get("source_id") != "premutagenic_damage":
        return set()
    unit = str(row.get("unit_text") or "").strip()
    pattern = re.compile(
        r"^(?P<core>(?:(?:mt|n)DNA\s+)?lesions?\s*(?:/|per)\s*\d+(?:\^\d+)?"
        r"\s*(?:bp|kb|kbp|mb))(?P<suffix>\s+(?:of\s+)?(?:DNA|mtDNA|nDNA|segment))$",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(unit)
    if not match:
        return set()
    core = re.escape(match.group("core")).replace(r"\ ", r"\s*")
    text = str(row.get("measurement_text") or "")
    pairs: set[tuple[str, str]] = set()
    for value in values:
        target = _decimal(value)
        for start, end, number in _number_spans(text):
            bounded = re.search(
                r"(?:[<>≤≥]|\bto)\s*$", text[max(0, start - 20) : start]
            )
            if (
                number == target
                and not bounded
                and re.match(rf"\s*{core}\b", text[end:], re.IGNORECASE)
            ):
                pairs.add((value, unit))
    return pairs


def _stereochemical_ratio_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    if row.get("source_id") != "mutagenicity_mechanism":
        return set()
    assay = str(row.get("assay_method_and_endpoint") or "")
    units = {
        re.sub(r"\s*([:/])\s*", r"\1", match.group()).strip()
        for match in re.finditer(
            r"\b[RS]\s*[:/]\s*[RS]\s+ratio\b", assay, re.IGNORECASE
        )
    }
    support = str(row.get("support_text") or "")
    pairs: set[tuple[str, str]] = set()
    for value in values:
        target = _decimal(value)
        direct = any(
            number == target and re.match(r"\s*:\s*1\b", support[end:])
            for _, end, number in _number_spans(support)
        )
        if direct:
            pairs.update((value, unit) for unit in units)
    return pairs


def _partition_result_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    ending = re.compile(
        r"(?:\b(?:was|is)\s+(?:(?:calculated|estimated|found|determined)\s+)?"
        r"(?:to be\s+|as\s+)?(?:approximately|about|around)?\s*~?\s*|"
        r"^\s*(?:of|=)\s*~?\s*)$",
        re.IGNORECASE,
    )
    source = str(row.get("source_id") or "")
    units = _assay_declared_units(row, source) if source else set()
    units.add("partition ratio")
    pairs: set[tuple[str, str]] = set()
    for field in EVIDENCE_FIELDS:
        chunks = re.split(r"(?<=[.;])\s+|;", str(row.get(field) or ""))
        for chunk in chunks:
            anchors = list(re.finditer(r"\bpartition[- ]ratio\b", chunk, re.IGNORECASE))
            for value in values:
                target = _decimal(value)
                for start, _, observed in _number_spans(chunk):
                    if observed != target:
                        continue
                    for anchor in anchors:
                        if anchor.end() >= start:
                            continue
                        bridge = chunk[anchor.end() : start]
                        repeated = re.search(
                            r"\bpartition[- ]ratio\b", bridge, re.IGNORECASE
                        )
                        if len(bridge) > 220 or repeated or re.match(r"\s*\+", bridge):
                            continue
                        if ending.search(bridge):
                            pairs.update((value, unit) for unit in units)
    return pairs


def _number_spans(text: str) -> list[tuple[int, int, Decimal]]:
    pattern = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)")
    spans: list[tuple[int, int, Decimal]] = []
    for match in pattern.finditer(text):
        if match.start() and text[match.start() - 1].isalnum():
            continue
        if match.end() < len(text) and text[match.end()].isalnum():
            continue
        if (
            match.end() + 1 < len(text)
            and text[match.end()] in "-'"
            and text[match.end() + 1].isalpha()
        ):
            continue
        value = _decimal(match.group().replace(",", ""))
        if value is not None:
            spans.append((match.start(), match.end(), value))
    return spans


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(left[0] - right[1], right[0] - left[1], 0)


def _ratio_series_tail(between: str) -> bool:
    competing = re.compile(
        r"\b(?:ratios?|coefficient|frequency|rate|score|index|factor|fraction|"
        r"turnover\s+number|auc|km|vmax)\b",
        re.IGNORECASE,
    )
    if competing.search(between):
        return False
    series_starts = (
        r"^\s*(?:\([^.;]{0,80}\)\s*)?(?:values?\s+)?(?:are|were)\b",
        r"^\s*:\s*",
        r"^\s*from\b[^.;]{0,180}\bto\b",
        r"^\s*(?:significantly\s+)?(?:declined|decreased|dropped|fell|increased|rose)\s+from\b",
        r"^\s*was\b[^.;]{0,180}\b(?:and|versus|vs\.?|to)\b",
    )
    return any(re.search(pattern, between, re.IGNORECASE) for pattern in series_starts)


def _ratio_series_patterns(unit: str) -> list[str]:
    if not unit.lower().endswith(" ratio"):
        return []
    escaped = lambda value: re.escape(value).replace(r"\ ", r"\s+")
    if "/" not in unit:
        return [escaped(unit) + r"\b"]
    body = unit[: -len(" ratio")].strip()
    left, right = (part.strip() for part in body.split("/", 1))
    separator = r"\s*[:/]\s*"
    patterns = [rf"{escaped(body).replace('/', separator)}\s+ratio\b"]
    patterns.append(rf"ratios?\s+of\s+{escaped(left)}\s+to\s+{escaped(right)}\b")
    if re.fullmatch(r"[A-Za-z0-9µμΔ^+_.-]+", left) and re.fullmatch(
        r"[A-Za-z0-9µμΔ^+_.-]+", right
    ):
        patterns.append(
            rf"\[?{escaped(left)}\]?\s*/\s*\[?{escaped(right)}\]?\s+ratio\b"
        )
    return patterns


def _explicit_ratio_series_link(text: str, target: Decimal, unit: str) -> bool:
    targets = [span for span in _number_spans(text) if span[2] == target]
    for pattern in _ratio_series_patterns(unit):
        for cue in re.finditer(pattern, text, re.IGNORECASE):
            for target_span in targets:
                if target_span[0] <= cue.end():
                    continue
                if _span_distance(cue.span(), target_span[:2]) > 256:
                    continue
                if _ratio_series_tail(text[cue.end() : target_span[0]]):
                    return True
    return False


def _anaphoric_ratio_link(text: str, target: Decimal, unit: str) -> bool:
    if "/" not in unit or not unit.lower().endswith(" ratio"):
        return False
    targets = [span for span in _number_spans(text) if span[2] == target]
    for anchor in _unit_anchors(unit):
        if "/" not in anchor and ":" not in anchor:
            continue
        for cue in re.finditer(re.escape(anchor), text, re.IGNORECASE):
            for target_span in targets:
                if not cue.end() < target_span[0]:
                    continue
                between = text[cue.end() : target_span[0]]
                sentence_break = re.search(r";|(?<!\d)\.(?!\d)", between)
                if (
                    len(between) <= 300
                    and not sentence_break
                    and re.search(
                        r"\b(?:raised|increased|decreased|reduced|dropped)\s+"
                        r"the\s+ratio\s+to\s*$",
                        between,
                        re.IGNORECASE,
                    )
                ):
                    return True
    return False


def _direct_link_changes_metric(
    text: str, unit_span: tuple[int, int], target_span: tuple[int, int]
) -> bool:
    if target_span[1] <= unit_span[0]:
        between = text[target_span[1] : unit_span[0]]
    else:
        between = text[unit_span[1] : target_span[0]]
    competing = re.search(
        r"\b(?:auc|coefficient|difference|factor|fold|fraction|index|km|log|"
        r"odds ratio|ratio|score|survival|times?|turnover number|viability|vmax)\b|"
        r"\bR[²2]\s*=",
        between,
        re.IGNORECASE,
    )
    reported = re.search(
        r"\b(?:gave|produced|reported|showed|yielded)\b[^.;]{0,80}"
        r"\b(?:frequency|rate)\b",
        between,
        re.IGNORECASE,
    )
    new_result = V4_NEW_RESULT.search(between)
    fixed = V4_FIXED_READOUT.fullmatch(text[slice(*unit_span)])
    same_subject = False
    if new_result and fixed:
        subject = re.sub(
            r"^(?:the|this|that)\s+", "", new_result.group("subject").casefold()
        ).strip()
        unit_name = text[slice(*unit_span)].casefold()
        same_subject = subject in {
            "it",
            "its frequency",
            "its rate",
            "its value",
            "rate",
            "frequency",
            unit_name,
        }
        if not same_subject:
            return True
    if fixed and V4_COMPETING_METRIC.search(between) and not same_subject:
        return True
    return bool(competing or reported)


def _dimensionless_metric(unit: str) -> bool:
    return bool(
        re.search(
            r"\b(?:ratio|factor|fraction|index|coefficient|score|pK)\b",
            unit,
            re.IGNORECASE,
        )
        or unit.lower().startswith("log ")
    )


def _result_bridge(bridge: str, *, target_before_unit: bool) -> bool:
    if len(bridge) > 120 or re.search(r"[.;]", bridge):
        return False
    if target_before_unit:
        return bool(re.fullmatch(r"\s*(?:±\s*" + NUMBER + r"\s*)?", bridge))
    competing = re.compile(
        r"\b(?:turnover\s+number|frequency|rate|score|coefficient|"
        r"ratio|fraction|factor|index|auc|km|vmax)\b",
        re.IGNORECASE,
    )
    if competing.search(bridge):
        return False
    if re.fullmatch(r"\s*~?\s*", bridge):
        return True
    return bool(
        re.search(
            r"(?:=|:|\b(?:is|was|were|of|to|at|approximately|approx\.?|"
            r"estimated\s+as|reported\s+as))\s*(?:~\s*)?$",
            bridge,
            re.IGNORECASE,
        )
    )


def _metric_is_operand(
    text: str, unit_span: tuple[int, int], target_span: tuple[int, int]
) -> bool:
    if target_span[0] <= unit_span[1]:
        return False
    if re.match(r"\s*[+*/-]", text[unit_span[1] : target_span[0]]):
        return True
    prefix = text[max(0, unit_span[0] - 96) : unit_span[0]]
    if re.search(
        r"\b(?:coefficients?|factors?|fractions?|indices|ratios?)\s+(?:for|of)\b"
        r"[^.;]{0,48}$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:absolute\s+)?(?:difference|variation|variability|change)\s+"
            r"(?:in|between|of)\b[^.;]{0,48}$",
            prefix,
            re.IGNORECASE,
        )
    )


def _explicit_metric_anchor_link(
    text: str, target: Decimal, unit_span: tuple[int, int]
) -> bool:
    for start, end, value in _number_spans(text):
        if value != target or not (end <= unit_span[0] or start >= unit_span[1]):
            continue
        if end <= unit_span[0]:
            bridge = text[end : unit_span[0]]
            if _result_bridge(bridge, target_before_unit=True):
                return True
        else:
            bridge = text[unit_span[1] : start]
            if not _metric_is_operand(text, unit_span, (start, end)) and _result_bridge(
                bridge, target_before_unit=False
            ):
                return True
    return False


def _nearest_unit_link(text: str, target: Decimal, unit: str) -> bool:
    numbers = _number_spans(text)
    if not any(span[2] == target for span in numbers):
        return False
    if _explicit_ratio_series_link(text, target, unit):
        return True
    if _anaphoric_ratio_link(text, target, unit):
        return True
    lower = text.lower()
    anchors = []
    for anchor in _unit_anchors(unit):
        anchors.extend(
            (len(anchor), match.start(), match.end())
            for match in re.finditer(re.escape(anchor.lower()), lower)
        )
    if not anchors:
        return False
    longest = max(length for length, _, _ in anchors)
    for _, start, end in (item for item in anchors if item[0] == longest):
        unit_span = (start, end)
        if _dimensionless_metric(unit):
            if _explicit_metric_anchor_link(text, target, unit_span):
                return True
            continue
        external = [
            span
            for span in numbers
            if span[1] <= unit_span[0] or span[0] >= unit_span[1]
        ]
        targets = [span for span in external if span[2] == target]
        if not targets or not external:
            continue
        nearest = min(_span_distance(unit_span, span[:2]) for span in external)
        target_distance = min(_span_distance(unit_span, span[:2]) for span in targets)
        direct = target_distance == nearest and any(
            not _direct_link_changes_metric(text, unit_span, span[:2])
            and not _metric_is_operand(text, unit_span, span[:2])
            for span in targets
            if _span_distance(unit_span, span[:2]) == target_distance
        )
        if nearest <= 96 and direct:
            return True
    return False


def _linked_suffix_pairs(
    row: dict[str, object], values: list[str], units: set[str]
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for field in EVIDENCE_FIELDS:
        chunks = re.split(r"(?<=[.;])\s+|;", str(row.get(field) or ""))
        for value in values:
            target = _decimal(value)
            if target is None:
                continue
            for unit in units:
                if any(_nearest_unit_link(chunk, target, unit) for chunk in chunks):
                    pairs.update((value, item) for item in _slash_spacing_units({unit}))
    return pairs


def _relative_value_markers(row: dict[str, object], value: str) -> set[str]:
    text = str(row.get("measurement_text") or "")
    target = _decimal(value)
    markers: set[str] = set()
    for start, end, number in _number_spans(text):
        if number != target:
            continue
        after = text[end : end + 16]
        if re.match(r"\s*(?:%|percent\b)", after, re.IGNORECASE):
            markers.add("percent")
        if re.match(r"\s*-?\s*fold\b", after, re.IGNORECASE):
            markers.add("fold")
    return markers


def _authoritative_pairs(
    row: dict[str, object], values: list[str], units: set[str], source: str
) -> set[tuple[str, str]]:
    pairs = _linked_suffix_pairs(row, values, units)
    distinct = {_decimal(value) for value in values}
    fallback = values if len(distinct) == 1 else []
    for value in fallback:
        markers = _relative_value_markers(row, value)
        for unit in units:
            lower = unit.casefold()
            if "percent" in markers and "%" not in unit and "percent" not in lower:
                continue
            if "fold" in markers and "fold" not in lower:
                continue
            pairs.add((value, unit))
    return pairs


def _prune_incomplete_metric_pairs(
    pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    output = set(pairs)
    for measurement, short in pairs:
        lowered = short.casefold()
        for other_measurement, long in pairs:
            longer = long.casefold()
            if measurement != other_measurement or lowered == longer:
                continue
            ratio_fragment = longer.endswith(" ratio") and "/" in longer
            rate_fragment = longer.endswith(" rate constant")
            if lowered in longer and (
                (ratio_fragment and lowered.endswith(" ratio"))
                or (rate_fragment and lowered.endswith((" rate", "constant")))
            ):
                output.discard((measurement, short))
    return output


def _v4_named_metric(unit: str) -> bool:
    bare = (
        r"(?:factor|fraction|frequenc(?:y|ies)|index|potency|rates?|ratio|score|"
        r"constant|units?)"
    )
    return not (
        re.fullmatch(bare + r"(?:\s+[A-Za-z])?", unit.strip(), re.IGNORECASE)
        or re.search(r"\b(?:dose|exposure)\s+rates?\b", unit, re.IGNORECASE)
    )


def _v4_slash_metrics(text: str) -> set[str]:
    token = r"[A-Za-z][A-Za-z0-9-]*"
    pattern = re.compile(
        rf"\b(?:{token}\s+)?{token}\s*/\s*{token}\s+ratio\b",
        re.IGNORECASE,
    )
    return {re.sub(r"\s*/\s*", "/", match.group()) for match in pattern.finditer(text)}


def _v4_suffix_phrases(text: str) -> set[str]:
    heads = re.compile(
        r"^(?:ratio|fraction|factor|index|frequenc(?:y|ies)|rates?|potency|"
        r"score|coefficient|constant|units?|pka|pk|tosc|imax)$",
        re.IGNORECASE,
    )
    phrases: set[str] = set()
    for clause in re.split(r"[.;]", text):
        matches = list(TOKEN.finditer(clause))
        tokens = [match.group().strip("[]{}'\"") for match in matches]
        for end, token in enumerate(tokens):
            head = token.casefold().strip("()")
            if not heads.match(head):
                continue
            suffix = clause[matches[end].end() :]
            extra = ""
            if head == "potency":
                match = re.match(r"\s*\([A-Za-z][A-Za-z0-9_-]{1,12}\)", suffix)
                extra = match.group() if match else ""
            elif head == "constant":
                match = re.match(r"\s+[A-Za-z]\b", suffix)
                extra = match.group() if match else ""
            for width in range(1, min(4, end + 1) + 1):
                start = matches[end - width + 1].start()
                phrase = clause[start : matches[end].end()].strip(" -")
                for candidate in {phrase, phrase + extra} if extra else {phrase}:
                    words = {word.casefold().strip("()") for word in candidate.split()}
                    leading = words.intersection(LEADING | {"highest", "initial"})
                    if candidate.casefold().startswith("mean "):
                        leading.discard("mean")
                    slash_ratio = "/" in candidate and head == "ratio"
                    valid = (
                        not slash_ratio
                        and not leading
                        and _balanced_grouping(candidate)
                        and not _contains_standalone_number(candidate)
                        and not re.match(r"^[+*/=]", candidate)
                        and not re.search(r"[=<>±]", candidate)
                        and _complete_suffix_start(clause, start)
                        and _valid_suffix_delimiters(candidate)
                    )
                    if valid and _v4_named_metric(candidate):
                        phrases.add(candidate)
    return phrases


def _v4_safe_bridge(text: str) -> str:
    dose = rf"\b(?:pH\s*)?{NUMBER}\s*(?:{GROUPED_DOSE_UNIT.pattern}|"
    dose += r"days?|h|hours?|min|minutes?|s|seconds?|years?)\b"
    text = re.sub(dose, " ", text, flags=re.IGNORECASE)
    identifier = r"\b(?:[A-Za-z][A-Za-z0-9]*-\d+|"
    identifier += rf"(?:ATCC|clone|isolate|line|strain)\s+{NUMBER})\b"
    return re.sub(identifier, " ", text, flags=re.IGNORECASE)


def _v4_unit_anchors(unit: str) -> list[str]:
    anchors = set(_unit_anchors(unit))
    match = re.match(
        r"^(rates?|frequenc(?:y|ies))\s+of\s+([A-Za-z][A-Za-z0-9-]*)",
        unit,
        re.IGNORECASE,
    )
    if match:
        noun = match.group(2)
        heads = (
            ("rate", "rates")
            if match.group(1).lower().startswith("rate")
            else (
                "frequency",
                "frequencies",
            )
        )
        anchors.update(f"{noun} {head}" for head in heads)
    return sorted(anchors, key=lambda value: (-len(value), value.casefold(), value))


def _v4_metric_span_link(
    text: str, target_span: tuple[int, int], unit: str, *, scientific: bool
) -> bool:
    before = text[max(0, target_span[0] - 48) : target_span[0]]
    after = text[target_span[1] : target_span[1] + 24]
    if re.search(r"[<>≤≥]\s*$", before):
        return False
    if re.search(r"\bto\s*$", before, re.IGNORECASE) and not V4_CHANGED_TO.search(
        before
    ):
        return False
    if scientific and GROUPED_DOSE_UNIT.match(after.lstrip()):
        return False
    relative = V4_RELATIVE_FOLLOW.match(after)
    if scientific and relative and not V4_SCI_COMPARISON_FOLLOW.match(after):
        return False
    anchors = [
        match.span()
        for anchor in _v4_unit_anchors(unit)
        for match in re.finditer(re.escape(anchor), text, re.IGNORECASE)
    ]
    if not anchors and unit.casefold().endswith(" mutation rate"):
        short = unit[: -len(" mutation rate")] + " rate"
        anchors = [
            match.span() for match in re.finditer(re.escape(short), text, re.IGNORECASE)
        ]
    for start, end in anchors:
        if target_span[1] <= start:
            if re.fullmatch(
                r"\s*(?:±\s*" + NUMBER + r"\s*)?", text[target_span[1] : start]
            ):
                return True
            continue
        if end > target_span[0]:
            continue
        bridge = text[end : target_span[0]]
        if V4_COMPETING_METRIC.search(bridge):
            continue
        anaphoric = bool(
            V4_ANAPHORIC_VALUE.match(bridge) and V4_CHANGED_TO.search(bridge)
        )
        safe = _v4_safe_bridge(bridge)
        masked = E_SCI.sub(" ", V4_SCI.sub(" ", safe)) if scientific else safe
        series = (
            scientific
            and "respectively" in text.casefold()
            and not _number_spans(masked)
        )
        direct = not _number_spans(masked) and V4_DIRECT_CUE.search(masked)
        if anaphoric or series or direct:
            return True
    return False


def _v4_scientific_spans(
    text: str,
) -> list[tuple[tuple[int, int], Decimal, Decimal]]:
    spans: list[tuple[tuple[int, int], Decimal, Decimal]] = []
    for match in V4_SCI.finditer(text):
        coefficient = _decimal(match.group("coefficient") or "1")
        exponent = _sci_exponent(match)
        if coefficient is not None and abs(exponent) <= 30:
            product = coefficient * Decimal(10) ** exponent
            spans.append((match.span(), coefficient, product))
    for match in E_SCI.finditer(text):
        coefficient = _decimal(match.group("coefficient"))
        exponent = int(match.group("exponent"))
        if coefficient is not None and abs(exponent) <= 30:
            product = coefficient * Decimal(10) ** exponent
            spans.append((match.span(), coefficient, product))
    return spans


def _v4_scientific_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    source = str(row.get("source_id") or "")
    if source != "mutagenicity_mechanism":
        return set()
    pairs: set[tuple[str, str]] = set()
    targets = {_decimal(value): value for value in values}
    for field in EVIDENCE_FIELDS:
        chunks = re.split(r"(?<=[.;])\s+|;", str(row.get(field) or ""))
        for chunk in chunks:
            units = _v4_suffix_phrases(chunk)
            units.update(_v4_slash_metrics(chunk))
            units = {unit for unit in units if _dimensionless_metric(unit)}
            units = {unit for unit in units if _v4_named_metric(unit)}
            for span, coefficient, product in _v4_scientific_spans(chunk):
                value = targets.get(product)
                if (
                    value is None
                    or coefficient in targets
                    or not _v4_safe_scientific_result(chunk, span)
                ):
                    continue
                pairs.update(
                    (value, unit)
                    for unit in units
                    if _v4_metric_span_link(chunk, span, unit, scientific=True)
                )
    return pairs


def _v4_fixed_result_link(
    text: str, unit: re.Match[str], span: tuple[int, int]
) -> bool:
    bridge = text[unit.end() : span[0]]
    after = text[span[1] :]
    name = unit.group("unit").casefold()
    alias = re.search(
        r"[.;]\s*(?:the\s+)?reversion rates?\b[^.;]{0,160}"
        r"\b(?:was|were)\s*(?:approximately|around|about|probably\s+around)?\s*$",
        bridge,
        re.IGNORECASE,
    )
    if name.startswith(("rate of reversion", "rates of reversion")) and alias:
        return True
    boundary = V4_CLAUSE_END.search(bridge)
    if (
        boundary
        or re.search(r"\b(?:r|rho)\s*=", bridge, re.IGNORECASE)
        or _direct_link_changes_metric(text, unit.span(), span)
    ):
        return False
    relative = re.match(
        rf"\s*(?:%|percent\b|-?\s*fold\b|(?:-|–|—|to)\s*{NUMBER}\b|"
        rf"{GROUPED_DOSE_UNIT.pattern})",
        after,
        re.IGNORECASE,
    )
    condition = re.search(
        r"\b(?:dose|duration|exposure|pH|temperature|time|year|cell line)\s*"
        r"(?:=|:|is|was|were)\s*$",
        bridge,
        re.IGNORECASE,
    )
    if relative or condition:
        return False
    series = name.startswith("frequency of ") and name.endswith(" recombinants")
    if series and re.match(r"\s+(?:was|were)\b", bridge, re.IGNORECASE):
        return bool(
            re.search(r"\bfor\b[^.;]{0,160}\brespectively\b", after, re.IGNORECASE)
        )
    changed = re.search(
        rf"\b(?:was|were)\s+{NUMBER}\b[^.;]{{0,120}}"
        r"\b(?:fell|dropped|rose|increased|decreased)\s+to\s*$",
        bridge,
        re.IGNORECASE,
    )
    presence = name == "recombination rate" and re.search(
        rf"\b(?:was|were)\s+{NUMBER}\b[^.;]{{0,120}}\band\s*$",
        bridge,
        re.IGNORECASE,
    )
    return bool(
        V4_FIXED_RESULT_CUE.search(bridge)
        or changed
        or (presence and re.match(r"\s+in the presence\b", after, re.IGNORECASE))
    )


def _v4_fixed_pairs(row: dict[str, object], values: list[str]) -> set[tuple[str, str]]:
    targets = {_decimal(value): value for value in values}
    pairs: set[tuple[str, str]] = set()
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        scientific = _v4_scientific_spans(text)
        for unit in V4_FIXED_READOUT.finditer(text):
            name = unit.group("unit")
            if name.casefold().startswith(("rate of ", "rates of ", "frequency of ")):
                name = name[0].lower() + name[1:]
            for span, _, product in scientific:
                if (
                    product in targets
                    and _v4_safe_scientific_result(text, span)
                    and _v4_fixed_result_link(text, unit, span)
                ):
                    pairs.add((targets[product], name))
            for start, end, number in _number_spans(text):
                overlap = any(
                    start < span[1] and end > span[0] for span, _, _ in scientific
                )
                if (
                    number in targets
                    and not overlap
                    and _v4_fixed_result_link(text, unit, (start, end))
                ):
                    pairs.add((targets[number], name))
    return pairs


def _v4_comparison_values(text: str, anchor_end: int) -> list[Decimal]:
    tail = text[anchor_end : anchor_end + 300]
    cue = re.compile(
        r"(?:=|:|\b(?:are|is|was|were)|\bcompared\s+to|\bversus|\band)\s*"
        r"(?:approximately|about|around)?\s*~?\s*$",
        re.IGNORECASE,
    )
    values: list[Decimal] = []
    for start, end, value in _number_spans(tail):
        if (
            not V4_NEW_RESULT.search(tail[:start])
            and cue.search(tail[:start])
            and re.match(r"\s+(?:for|in)\s+[A-Za-z]", tail[end:], re.IGNORECASE)
        ):
            values.append(value)
    return values


def _v4_comparison_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    support = str(row.get("support_text") or "")
    assay = str(row.get("assay_method_and_endpoint") or "")
    targets = {_decimal(value): value for value in values}
    pairs: set[tuple[str, str]] = set()
    prose = re.compile(
        r"\bratio of [RS]-\s+to\s+[RS]-(?:enantiomers?\s+of\s+)?"
        r"[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,5}?"
        r"(?=\s+(?:are|formed|generated|is|produced|was|were)\b)",
        re.IGNORECASE,
    )
    for match in prose.finditer(support):
        linked = _v4_comparison_values(support, match.end())
        if len(linked) >= 2:
            pairs.update(
                (targets[value], match.group()) for value in linked if value in targets
            )
    vmax = re.search(
        r"(?P<unit>\bVmax ratio of (?P<left>.+?) versus (?P<right>.+?) activity)\s*$",
        assay,
        re.IGNORECASE,
    )
    if not vmax:
        return pairs
    left = re.sub(r"\s*\([^()]*\)\s*", " ", vmax.group("left")).strip()
    right = re.sub(r"\s*\([^()]*\)\s*", " ", vmax.group("right")).strip()
    pattern = re.compile(
        rf"\bVmax ratios?\s+for\s+{re.escape(left)}\s+versus\s+{re.escape(right)}\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(support):
        linked = _v4_comparison_values(support, match.end())
        if len(linked) >= 2:
            for number in linked:
                if number in targets:
                    pairs.update(
                        {
                            (targets[number], vmax.group("unit")),
                            (targets[number], match.group()),
                        }
                    )
    return pairs


def _v4_direct_mechanism_pairs(
    row: dict[str, object], values: list[str]
) -> set[tuple[str, str]]:
    support = str(row.get("support_text") or "")
    targets = {_decimal(value): value for value in values}
    pairs: set[tuple[str, str]] = set()
    cardinal_words = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    ]
    cardinals = {word: Decimal(index) for index, word in enumerate(cardinal_words)}
    if len(targets) == 1:
        pattern = re.compile(
            r"\b(" + "|".join(cardinals) + r")\s+(different peptide adducts?)\b",
            re.IGNORECASE,
        )
        for match in pattern.finditer(support):
            number = cardinals[match.group(1).casefold()]
            if number in targets:
                pairs.add((targets[number], match.group(2)))
    direct = (
        (
            r"\b((?:[A-Za-z][A-Za-z0-9-]*\s+){2}induction index)\s+(?:of|=|:|was|is)\s*",
            1,
        ),
        (r"\b(absorbance\s+[A-Za-z][A-Za-z0-9]?)\s*(?:=|:)\s*", 1),
    )
    for prefix, group in direct:
        for match in re.finditer(
            prefix + rf"(?P<value>{NUMBER})\b", support, re.IGNORECASE
        ):
            number = _decimal(match.group("value"))
            if number in targets:
                pairs.add((targets[number], match.group(group)))
    for chunk in re.split(r"(?<=[.;])\s+|;", support):
        if not any(
            word in chunk.casefold() for word in ("associat", "correlat", "regress")
        ):
            continue
        for match in re.finditer(rf"\bR\s*=\s*({NUMBER})\b", chunk):
            number = _decimal(match.group(1))
            if number in targets:
                pairs.add((targets[number], "R"))
    return pairs


def _v4_dna_pairs(row: dict[str, object], values: list[str]) -> set[tuple[str, str]]:
    if (
        str(row.get("unit_text") or "").strip()
        or len({_decimal(v) for v in values}) != 1
    ):
        return set()
    measurement = str(row.get("measurement_text") or "")
    if not re.fullmatch(rf"\s*{NUMBER}\s*", measurement):
        return set()
    if re.search(
        r"[<>≤≥]|\b(?:between|relative|versus|vs\.?|to)\b",
        measurement,
        re.IGNORECASE,
    ):
        return set()
    pattern = re.compile(
        r"\bDNA lesions in (?:an?\s+)?\d+(?:\.\d+)?\s*"
        r"(?:bp|kb|kbp|mb)\s+(?:fragment|segment)\b",
        re.IGNORECASE,
    )
    assay = {
        match.group()
        for match in pattern.finditer(str(row.get("assay_method_and_endpoint") or ""))
    }
    support = {
        match.group() for match in pattern.finditer(str(row.get("support_text") or ""))
    }
    matched = {
        unit
        for unit in support
        if any(unit.casefold() == item.casefold() for item in assay)
    }
    return {(value, unit) for value in values for unit in matched}


def _v4_repair_pairs(row: dict[str, object], values: list[str]) -> set[tuple[str, str]]:
    source = str(row.get("source_id") or "")
    measurement = str(row.get("measurement_text") or "")
    if V4_NONABSOLUTE_MEASUREMENT.search(measurement):
        return set()
    if source == "fixed_mutation":
        return _v4_fixed_pairs(row, values)
    if source != "mutagenicity_mechanism":
        return set()
    pairs = _v4_direct_mechanism_pairs(row, values)
    pairs.update(_v4_comparison_pairs(row, values))
    pairs.update(_v4_dna_pairs(row, values))
    pairs.update(_v4_scientific_pairs(row, values))
    return pairs


def _v4_safe_scientific_result(text: str, span: tuple[int, int]) -> bool:
    before = text[max(0, span[0] - 96) : span[0]]
    after = text[span[1] : span[1] + 48]
    if V4_NONRESULT_SCI_PREFIX.search(before):
        return False
    bounded = re.search(rf"{NUMBER}\s*(?:-|–|—|to)\s*$", before, re.IGNORECASE)
    changed = re.search(
        r"\b(?:fell|rose|increased|decreased|dropped|reduced)\b",
        before,
        re.IGNORECASE,
    )
    condition = re.search(
        r"\b(?:concentration|dose|exposure|pH|temperature|time)\s*"
        r"(?:=|:|of|at|was|were|is)?\s*$",
        before,
        re.IGNORECASE,
    )
    relative = V4_RELATIVE_FOLLOW.match(after)
    comparison = V4_SCI_COMPARISON_FOLLOW.match(after)
    nonabsolute = V4_NONRESULT_SCI_SUFFIX.match(after)
    event_count = re.match(r"\s+mutations?\s+occurred\b", after, re.IGNORECASE)
    return not (
        condition
        or event_count
        or nonabsolute
        or (relative and not comparison)
        or (bounded and not changed)
    )


def _v4_projection_keys(
    row: dict[str, object], *, safe: bool
) -> set[tuple[Decimal, int, str]]:
    text = _source_text(row)
    stems = _scientific_stems(text, set())
    keys: set[tuple[Decimal, int, str]] = set()
    for match in V4_SCI.finditer(text):
        coefficient = _decimal(match.group("coefficient") or "1")
        exponent = _sci_exponent(match)
        if (
            coefficient is None
            or _v4_safe_scientific_result(text, match.span()) != safe
        ):
            continue
        keys.update(
            (coefficient, exponent, stem)
            for stem in _associated_scientific_stems(text, match, stems)
        )
    for match in E_SCI.finditer(text):
        coefficient = _decimal(match.group("coefficient"))
        exponent = int(match.group("exponent"))
        if (
            coefficient is None
            or _v4_safe_scientific_result(text, match.span()) != safe
        ):
            continue
        keys.update(
            (coefficient, exponent, stem)
            for stem in _associated_scientific_stems(text, match, stems)
        )
    return keys


def _v4_scaled_replacements(
    row: dict[str, object], values: list[str], pairs: set[tuple[str, str]]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    targets = {_decimal(value): value for value in values}
    keys = _v4_projection_keys(row, safe=True)
    unsafe_keys = _v4_projection_keys(row, safe=False)
    primary_bound = V4_NONABSOLUTE_MEASUREMENT.search(
        str(row.get("measurement_text") or "")
    )
    eligible: dict[tuple[Decimal, str], list[tuple[str, str]]] = {}
    replacements: set[tuple[str, str]] = set()
    removals: set[tuple[str, str]] = set()
    source = " ; ".join(
        str(row.get(field) or "")
        for field in EVIDENCE_FIELDS
        if field != "measurement_text"
    )
    plain_source = E_SCI.sub(" ", V4_SCI.sub(" ", source))
    for measurement, unit in pairs:
        match = re.fullmatch(r"10\^([+-]?\d+) (.+)", unit)
        coefficient = _decimal(measurement)
        if not match or coefficient is None:
            continue
        stem = match.group(2)
        exponent = int(match.group(1))
        key = coefficient, exponent, stem
        if primary_bound or (key in unsafe_keys and key not in keys):
            removals.add((measurement, unit))
            plain_link = _nearest_unit_link(plain_source, coefficient, stem)
            if plain_link and coefficient in targets and not primary_bound:
                replacements.add((targets[coefficient], stem))
            if primary_bound or not plain_link:
                removals.update(
                    pair
                    for pair in pairs
                    if _decimal(pair[0]) == coefficient and pair[1] == stem
                )
            continue
        if coefficient in targets:
            continue
        product = coefficient * Decimal(10) ** exponent
        if product not in targets:
            continue
        if key not in keys:
            continue
        eligible.setdefault((product, stem), []).append((measurement, unit))
    for (product, stem), old_pairs in eligible.items():
        if len(old_pairs) != 1:
            continue
        replacements.add((targets[product], stem))
        removals.add(old_pairs[0])
    return replacements, removals


def _v4_merge_pairs(
    row: dict[str, object], values: list[str], pairs: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    if str(row.get("source_id") or "") not in {
        "fixed_mutation",
        "mutagenicity_mechanism",
    }:
        return pairs
    repairs = _v4_repair_pairs(row, values)
    replacements, removals = _v4_scaled_replacements(row, values, pairs)
    remaining = pairs - removals
    scaled_targets: set[tuple[Decimal, str]] = set()
    for measurement, unit in remaining:
        match = re.fullmatch(r"10\^([+-]?\d+) (.+)", unit)
        coefficient = _decimal(measurement)
        if match and coefficient is not None:
            scaled_targets.add(
                (coefficient * Decimal(10) ** int(match.group(1)), match.group(2))
            )
    repairs = {
        pair for pair in repairs if (_decimal(pair[0]), pair[1]) not in scaled_targets
    }
    return remaining | replacements | repairs


def _candidate_outcome(
    row: dict[str, object],
) -> tuple[set[tuple[str, str]], str]:
    """Return the whole bounded set or one explicit atomic disposition."""
    source = str(row.get("source_id") or "")
    values = _value_variants(row)
    distinct_values = {_decimal(value) for value in values}
    if len(distinct_values) > MAX_PRIMARY_VALUES:
        return set(), "ambiguous_multiple_values"
    strong_units = _filter_partition_units(row, _strong_unit_candidates(row, source))
    linked_units = _filter_partition_units(
        row, _association_unit_candidates(row, source)
    )
    authoritative = _normalized_units(_authoritative_units(row), source)
    strong_units = {
        unit
        for unit in strong_units
        if unit.casefold() not in GENERIC_INFERRED_UNITS or unit in authoritative
    }
    linked_units = {
        unit
        for unit in linked_units
        if unit.casefold() not in GENERIC_INFERRED_UNITS or unit in authoritative
    }
    units = strong_units | linked_units
    direct_pairs = _direct_pairs(row, values)
    pairs = _authoritative_pairs(row, values, authoritative, source)
    pairs.update(_linked_suffix_pairs(row, values, units - authoritative))
    pairs.update(_explicit_named_pairs(row, values, strong_units))
    pairs.update(_semantic_pattern_pairs(row, values))
    pairs.update(_special_semantic_pairs(row, values))
    pairs.update(_source_defined_pairs(row, values))
    pairs.update(_apparent_pka_pairs(row, values))
    pairs.update(_formula_ratio_pairs(row, values))
    pairs.update(_half_life_time_pairs(row, values))
    pairs.update(_dna_length_pairs(row, values))
    pairs.update(_stereochemical_ratio_pairs(row, values))
    pairs.update(_partition_result_pairs(row, values))
    pairs.update(direct_pairs)
    pairs.update(_scientific_pairs(row, units))
    pairs.update(_group_scientific_pairs(row, units))
    pairs.update(_scaled_authoritative_pairs(row))
    pairs = _prune_incomplete_metric_pairs(pairs)
    pairs = _v4_merge_pairs(row, values, pairs)
    if len(pairs) > MAX_CANDIDATES:
        return set(), "ambiguous_candidate_cross_product"
    disposition = "candidates" if pairs else "no_source_grounded_candidate"
    return pairs, disposition


def _materialized_outcome(
    row: dict[str, object],
) -> tuple[list[dict[str, object]], str]:
    pairs, disposition = _candidate_outcome(row)
    if disposition != "candidates":
        return [], disposition
    if any(
        len(value.encode("utf-8")) > MAX_CANDIDATE_FIELD_BYTES
        for pair in pairs
        for value in pair
    ):
        return [], "candidate_text_too_large"
    grounded = [
        (
            measurement,
            unit,
            _evidence(row, measurement, unit),
        )
        for measurement, unit in sorted(pairs)
    ]
    grounded = [item for item in grounded if item[2]]
    output = [
        {
            "candidate_id": str(candidate_id),
            "measurement": measurement,
            "unit": unit,
            "rule_id": _rule_id(row, unit),
            "evidence": evidence,
            "hints": _hints(unit),
        }
        for candidate_id, (measurement, unit, evidence) in enumerate(grounded, 1)
    ]
    payload = json.dumps(
        output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(payload.encode("utf-8")) > MAX_CANDIDATE_JSON_BYTES:
        return [], "candidate_payload_too_large"
    return output, "candidates" if output else "no_source_grounded_candidate"


def candidate_generation_disposition(row: dict[str, object]) -> str:
    """Explain why a row has candidates or was rejected atomically."""
    _, disposition = _materialized_outcome(row)
    return disposition


def _rule_id(row: dict[str, object], unit: str) -> str:
    if unit.startswith(("10^", "×10 ")):
        return "scientific_scale.v1"
    if unit in _authoritative_units(row):
        return "authoritative_unit_text.v1"
    if "correlation coefficient" in unit.lower():
        return "value_linked_correlation.v1"
    if unit == "%":
        return "value_linked_percent.v1"
    if unit in {
        "h",
        "hr",
        "hour",
        "hours",
        "min",
        "minute",
        "minutes",
        "nA",
        "s",
        "sec",
        "second",
        "seconds",
        "µm",
        "μm",
    }:
        return "value_adjacent_physical_unit.v1"
    if any(word in unit.lower() for word in ("ratio", "fraction", "factor", "index")):
        return "defined_dimensionless_metric.v1"
    return "source_phrase_metric.v1"


def _hints(unit: str) -> list[str]:
    hints = ["point"]
    if unit.startswith(("10^", "×10 ")):
        hints.append("scientific_scale")
    if any(word in unit.lower() for word in ("ratio", "fraction", "factor", "index")):
        hints.append("defined_dimensionless_metric")
    return hints


def _excerpt(text: str, needle: str, limit: int = 96) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    position = clean.lower().find(needle.lower())
    if position < 0:
        return ""
    start = max(0, position - 32)
    return clean[start : start + limit]


def _numeric_evidence(
    row: dict[str, object], measurement: str, unit: str
) -> str | None:
    target = _decimal(measurement)
    measurement_text = str(row.get("measurement_text") or "")
    unit_text = str(row.get("unit_text") or "").strip()
    support = str(row.get("support_text") or "")
    if unit == unit_text:
        for match in GROUPED_NUMBER.finditer(support):
            raw = match.group()
            position = measurement_text.find(raw)
            if position < 0:
                continue
            prefix = measurement_text[:position]
            suffix = measurement_text[position + len(raw) :]
            denominator = re.search(r"(?:/|\bper)\s*$", prefix, re.IGNORECASE)
            dose = re.match(rf"\s*{GROUPED_DOSE_UNIT.pattern}", suffix, re.IGNORECASE)
            cross_metric = "fold" in unit.casefold() and re.match(
                rf"\s+[^.;]{{1,48}}?\bper\s+{NUMBER}\s*-?\s*fold\b",
                suffix,
                re.IGNORECASE,
            )
            relative_percent = re.search(
                r"(?:percent|%)\s+of\s+control", unit, re.IGNORECASE
            ) and re.match(r"\s*%\s*(?:higher|lower)\b", suffix, re.IGNORECASE)
            table_list = GROUPED_NUMBER.fullmatch(
                measurement_text.strip()
            ) and re.search(r"\bvalues\s*$", support[: match.start()], re.IGNORECASE)
            unsafe = (
                GROUPED_BOUND_PREFIX.search(prefix)
                or re.match(r"\s*\bto\b", suffix, re.IGNORECASE)
                or denominator
                or (dose and not GROUPED_DOSE_UNIT.search(unit))
                or cross_metric
                or relative_percent
                or INCOMPLETE_UNIT.search(unit)
                or table_list
            )
            if (
                raw.count(",") <= 2
                and not unsafe
                and _decimal(raw.replace(",", "")) == target
            ):
                excerpt = _excerpt(support, raw)
                if excerpt:
                    return f"support_text: {excerpt}"
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        for raw in _numbers(text):
            if _decimal(raw) == target:
                excerpt = _excerpt(text, raw)
                if excerpt:
                    return f"{field}: {excerpt}"
    return None


def _field_span(row: dict[str, object], needle: str) -> str | None:
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        excerpt = _excerpt(text, needle)
        if excerpt:
            return f"{field}: {excerpt}"
    return None


def _term_stem(value: str) -> str:
    value = value.lower().replace("μ", "µ")
    if value == "polyploidy":
        return "polyploid"
    if value in {"micronuclei", "micronucleus"}:
        return "micronucleus"
    if value.startswith("consum"):
        return "consum"
    if value in {"loss", "lost"}:
        return "loss"
    if value.startswith("metabol"):
        return "metabol"
    if value.startswith("inactivat"):
        return "inactivat"
    if value.endswith("ies") and len(value) > 5:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 4:
        return value[:-1]
    return value


def _unit_terms(unit: str) -> set[str]:
    stem = re.sub(r"^(?:10\^[+-]?\d+|×10)\s+", "", unit)
    ignored = {"and", "of", "or", "per", "the", "to"}
    terms = {
        _term_stem(token)
        for token in re.findall(r"[A-Za-z0-9µμΔ]+", stem)
        if len(token) > 1 and not token.isdigit() and token.lower() not in ignored
    }
    return terms


def _term_evidence(row: dict[str, object], unit: str, rule_id: str) -> str | None:
    required = _unit_terms(unit)
    if not required:
        return None
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        observed = {
            _term_stem(token)
            for token in re.findall(r"[A-Za-z0-9µμΔ]+", text)
            if len(token) > 1 and not token.isdigit()
        }
        if not required.issubset(observed):
            continue
        anchor = min(required, key=lambda value: (-len(value), value.casefold(), value))
        excerpt = _excerpt(text, anchor)
        if not excerpt:
            excerpt = re.sub(r"\s+", " ", text).strip()[:96]
        if excerpt:
            return f"{rule_id}: {field}: {excerpt}"
    return None


def _semantic_unit_evidence(
    row: dict[str, object], unit: str, rule_id: str
) -> str | None:
    stem = re.sub(r"^(?:10\^[+-]?\d+|×10)\s+", "", unit).lower()
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        lower = text.lower()
        marker = ""
        if stem == "mutation frequency" and (
            re.search(r"\bmfs?\b", text, re.IGNORECASE)
            or ("frequenc" in lower and ("mutat" in lower or "mutant" in lower))
        ):
            marker = "MF" if re.search(r"\bmfs?\b", text, re.IGNORECASE) else "frequenc"
        elif stem == "mms-induced mutation rate" and all(
            cue in lower for cue in ("mms", "induced", "rate", "mutagen")
        ):
            marker = "MMS"
        elif (
            "tet(a)δtetr allele fraction" in stem
            and all(cue in lower for cue in ("tet(a)", "tetr", "allele"))
            and ("proportion" in lower or "frequenc" in lower)
        ):
            marker = "tet(A)"
        elif stem == "dominant lethality fraction" and all(
            cue in lower for cue in ("dominant lethality", "calculated as")
        ):
            marker = "dominant lethality"
        if marker:
            excerpt = _excerpt(text, marker)
            if excerpt:
                return f"{rule_id}: {field}: {excerpt}"
    return None


def _unit_anchors(unit: str) -> list[str]:
    stem = re.sub(r"^(?:10\^[+-]?\d+|×10)\s+", "", unit)
    if "/" in stem:
        compact = re.sub(r"\s*/\s*", "/", stem)
        base = re.sub(
            r"\s+(?:ratio|fraction|factor|index)$", "", compact, flags=re.IGNORECASE
        )
        variants = {
            stem,
            compact,
            base,
            compact.replace("/", ":"),
            compact.replace("/", "-"),
        }
        variants.update(
            re.sub(r"(?<=\d)\s+(?=min\b)", "", item) for item in tuple(variants)
        )
        if base.count("/") == 1:
            left, right = base.split("/", 1)
            variants.add(f"ratio of {left} to {right}")
        match = re.fullmatch(r"(.+?) \((.+)\)", stem)
        if match:
            variants.add(f"{match.group(1)}, {match.group(2)}")
        return sorted(
            {value for value in variants if len(value) > 1},
            key=lambda value: (-len(value), value.casefold(), value),
        )
    anchors = [stem]
    if stem.lower() == "polyploid cells":
        anchors.append("polyploidy cells")
    if "micronuclei" in stem.lower():
        anchors.append("micronucleus")
    return sorted(
        {value for value in anchors if len(value) > 1},
        key=lambda value: (-len(value), value.casefold(), value),
    )


def _unit_evidence(row: dict[str, object], unit: str, rule_id: str) -> str | None:
    exact = _field_span(row, unit)
    if exact:
        return f"{rule_id}: {exact}"
    if rule_id == "authoritative_unit_text.v1":
        unit_text = str(row.get("unit_text") or "").strip()
        if unit_text:
            return f"{rule_id}: unit_text: {unit_text[:180]}"
    if rule_id == "value_linked_correlation.v1":
        for marker in ("r =", "r=", "rho"):
            span = _field_span(row, marker)
            if span:
                return f"{rule_id}: {span}"
    return _term_evidence(row, unit, rule_id) or _semantic_unit_evidence(
        row, unit, rule_id
    )


def _evidence(row: dict[str, object], measurement: str, unit: str) -> list[str]:
    if unit == "R" and row.get("source_id") == "mutagenicity_mechanism":
        return _v4_r_evidence(row, measurement)
    rule_id = _rule_id(row, unit)
    value_span = _numeric_evidence(row, measurement, unit)
    unit_span = _unit_evidence(row, unit, rule_id)
    return [value_span, unit_span] if value_span and unit_span else []


def _v4_r_evidence(row: dict[str, object], measurement: str) -> list[str]:
    target = _decimal(measurement)
    value_span = _numeric_evidence(row, measurement, "R")
    for field in EVIDENCE_FIELDS:
        text = str(row.get(field) or "")
        for chunk in re.split(r"(?<=[.;])\s+|;", text):
            context = re.search(
                r"\b(?:associat\w*|correlat\w*|regress\w*)\b",
                chunk,
                re.IGNORECASE,
            )
            for match in re.finditer(rf"\bR\s*=\s*(?P<value>{NUMBER})\b", chunk):
                if not context or _decimal(match.group("value")) != target:
                    continue
                excerpt = _excerpt(chunk, match.group())
                if value_span and excerpt:
                    return [value_span, f"source_phrase_metric.v1: {field}: {excerpt}"]
    return []


def candidates_for_record(row: dict[str, object]) -> list[dict[str, object]]:
    """Materialize canonical numbered candidates without partial truncation."""
    output, _ = _materialized_outcome(row)
    return output


def candidate_json_for_record(row: dict[str, object]) -> str:
    """Serialize candidates in the canonical candidate-hash representation."""
    return json.dumps(
        candidates_for_record(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_set_sha256_for_record(row: dict[str, object]) -> str:
    """Hash the canonical candidate list for exact plan and merge joins."""
    return hashlib.sha256(candidate_json_for_record(row).encode("utf-8")).hexdigest()


def candidates(row: dict[str, object]) -> set[tuple[str, str]]:
    """Compatibility helper for offline pair-coverage audits."""
    return {
        (str(candidate["measurement"]), str(candidate["unit"]))
        for candidate in candidates_for_record(row)
    }


if __name__ == "__main__":
    assert _plain(Decimal("1.500")) == "1.5"
    assert ("1.95", "10^-3 x") in _scaled_authoritative_pairs(
        {"measurement_text": "1.95 × 10^-3", "unit_text": "x"}
    )


__all__ = [
    "CANDIDATE_GENERATOR_VERSION",
    "MAX_CANDIDATES",
    "MAX_CANDIDATE_FIELD_BYTES",
    "MAX_CANDIDATE_JSON_BYTES",
    "candidate_generation_disposition",
    "candidate_json_for_record",
    "candidate_set_sha256_for_record",
    "candidates_for_record",
]
