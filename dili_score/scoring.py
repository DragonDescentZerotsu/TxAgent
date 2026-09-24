"""DILI relevance and structured completeness; pure standard-library rules."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from ._conditions import _dili_simple_condition
from ._schema import (
    _DILI_CONDITION_FIELDS,
    _DILI_SYSTEM_FIELDS,
    CALL_FIELDS,
    GROUP_FIELDS,
    native_source,
    normalize_endpoint,
    selection_text,
)

_REDOX = (
    r"oxidative|reactive oxygen|reactive species|redox|antioxid|peroxid|"
    r"glutathione|superoxide|catalase|malondialdehyde|protein carbonyl|"
    r"radical|nitric oxide|\b(?:ros|rns|gsh|gssg|sod\d*|gpx\d*|gst\w*|mda|tbar[s]?|"
    r"nrf2|nqo1|ho 1|hmox1|h2o2)\b|\bcat (?:activity|enzyme)\b"
)

_REACTIVE_METABOLISM = (
    r"reactive metabol|reactive intermediates?|thiol conjugat|bioactivat|metabolic activation|electrophil|covalent|"
    r"detoxification|conjugation|epoxide|glutathione|\bgsh\b|"
    r"\berod\b|aryl hydrocarbon hydroxylase"
)

_TRANSPORT = (
    r"p glycoprotein|\bp ?gp\b|\b(?:bcrp\d*|mdr\d*|mrp\d*|abc[bgc]\d+|"
    r"slc\w*|oatp?\w*|oct\d*|lat ?[12]|mct\d*|pept\d*|glut\d*|cnt\d*|rfc|lnaa)\b|"
    r"antiport|\bsystem l\b|monocarboxylic acid carrier|"
    r"(?:apical|basolateral|luminal|abluminal|mucosal|serosal).{0,35}transport"
)

_CELL = (
    r"cytotox|apopto|necro|viabil|cell(?:ular)? (?:death|survival|growth|proliferation|"
    r"number|cycle|morphology)|caspase|membrane integrity|\bldh\b|trypan blue|"
    r"adenylate kinase release|dna (?:synthesis|fragmentation)|viable cell|"
    r"lactate dehydrogenase|parp.{0,10}cleavage|\bbax\b|bcl 2|brdu|thymidine incorporation"
)

_IMMUNE = (
    r"cytokine|chemokine|interleukin|interferon|\b(?:il ?\d+[αβ]?|tnf[αβ]?|ifn|nf ?κb|"
    r"nf kappa b|pge2|inos)\b|immune|inflamm|t cell|lymphocyte|macrophage"
)

_ENDPOINTS = r"\bdili\b|liver|hepatic|hepatotox|hepatitis|cholesta|bilirubin|\b(?:alt|ast|alp|ggt|bsep)\b|bile|mitochondri|oxidative|reactive oxygen|apoptosis|cell (?:death|viability)|cytotox"

_ALIASES = (
    _REDOX,
    _REACTIVE_METABOLISM,
    _CELL,
    _IMMUNE,
    _TRANSPORT,
    r"hepat|biliar|canalicular|taurocholat|\bntcp\b|phospholipid transport|"
    r"jaundice|steatosis|cirrhosis|transamin|aminotransferase|alkaline phosphatase|"
    r"prothrombin|albumin|urea|gluconeogen|glucose production|protein synthesis|rbp secretion|"
    r"respirat|oxygen (?:consumption|uptake)|\b(?:atp|adp|nadh|ocr|oxphos|rcr|fao)\b|"
    r"\bp/o\b|proton leak|complex (?:i|ii|iii|iv|v)\b|cytochrome c|fatty acid oxidation|"
    r"palmitate oxidation|lipid|endoplasmic|er stress|autophag|lysosom|calcium|"
    r"\b(?:chop|atf[46]|grp78|bip|xbp1|ire1|perk|eif2|lc3|p62)\w*\b|"
    r"transcriptom|proteom|metabolom|gene expression|mrna|protein signall?ing|"
    r"mtdna|\bp:o\b|mptp|extracellular acidification|\becar\b|ketogen|ketone body|"
    r"beta oxidation|jc 1|succinate oxidase|o2 uptake|taurocholic|membrane blebbing|"
    r"beclin|triglyceride|ca2\+|nlrp3|hepcidin|\bp38\b|"
    r"^(?:clinical phenotype|biochemical pattern|maximum reported severity|causal status)$",
)

_OFF_TASK = r"renal toxicity|nephrotox|visual disturbance|skin pigmentation|ototoxic"

_ENDPOINT_PATTERN = re.compile("|".join((_ENDPOINTS, *_ALIASES)))

DILI_SIMPLE_VERSION = "dili_relevance_completeness.v5"

_DILI_ASSAY_FIELDS = (
    "canonical_assay_context",
    "assay_type",
    "assay_method",
    "assay_detail",
    "assay_system",
    "assay_model",
    "assay_format",
    "assay_and_readout",
    "assay_and_platform",
    "assay_method_and_endpoint",
    "assay_and_detection_method",
    "study_design",
)

_DILI_HEPATIC = re.compile(
    r"\b(?:livers?|hepatic|hepat\w*|biliar\w*|bile|cholesta\w*|kupffer|"
    r"nafld|nash|masld|hep ?g ?2|hep ?3 ?b|hep ?arg|huh ?7|l[0o] ?2|thle\w*|aml ?12)\b"
)

_DILI_LIVER_ENDPOINT = re.compile(
    r"\b(?:dili|hepat\w*|steatohepatitis|livers?|biliar\w*|bile|cholesta\w*|bilirubin|"
    r"transamin\w*|aminotransferase\w*|alt|ast|bsep|abcb11|ntcp|slc10a1)\b"
)

_DILI_EXTRA_READOUT = re.compile(
    r"\b(?:fxr|nr1h4|shp|nr0b2|hmgb ?1|tlr4|cpt\d*|carnitine palmitoyltransferase|"
    r"proliferation|regenerative reserve|dcf\w*|intracellular acidosis|"
    r"cytosolic ph|intracellular na\+?|nucleophil(?:e|ic) trapping|"
    r"cysteine adducts?|cyp\d\w*|p450|cellular stress homeostasis|"
    r"xenobiotic metabolic or transport function|transporter substrate or kinetics)\b"
)


def _information_parts(value, *, split_semicolons=True):
    """Remove missing values and field-name wrappers, retaining literal content."""
    separator = r"[|;]" if split_semicolons else r"\||;\s*(?=[a-zA-Z][a-zA-Z0-9_]*\s*=)"
    for part in re.split(separator, selection_text(value)):
        match = re.fullmatch(r"\s*([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(.*)", part, re.S)
        key, content = (
            (match[1].lower(), match[2].strip()) if match else ("", part.strip())
        )
        text = selection_text(content)
        normalized = normalize_endpoint(text)
        if not text or normalized in {
            "missing endpoint",
            "unknown endpoint",
            "not reported",
            "not applicable",
            "not available",
            "not specified",
            "not stated",
            "unclear",
        }:
            continue
        yield key, text


def _dili_result_status(value):
    """Recognize reported outcomes, explicit descriptions, or unresolved prose.

    Citation/method words alone never veto a result. Unrecognized prose retains
    field-presence credit but is exposed as unresolved, not certified as an outcome.
    """
    text = selection_text(value).lower().replace("_", " ")
    if re.fullmatch(
        r"\s*(?:ros(?: production| level)?|cell viability|apoptotic rate|"
        r"mitochondrial membrane potential|ic50|ec50|fold change)\s*",
        text,
    ):
        return "description_only"
    # Drop reference-only parentheses before interpretation. Removing a citation
    # must not turn "altered morphology" into a missing result (or vice versa).
    text = re.sub(
        r"\((?:qualitative,?\s*)?(?:fig(?:ure)?s?\.?|table)\b[^()]*\)", "", text
    )
    # Remove purpose clauses: asking whether a treatment increases ROS is not
    # a reported increase. Other independent result fields can still count.
    observation = re.sub(
        r"\b(?:to|for) (?:determine|assess|evaluate|measure|detect)\b.*", "", text
    )
    observation = re.sub(r"\b(?:whether|ability to)\b.*", "", observation)
    observation = re.sub(
        r"\b(?:increased|decreased|reduced|elevated)\s+(?:dose|concentration|exposure|duration)s?\b",
        "",
        observation,
    )
    qualitative = re.search(
        r"\b(?:increas(?:e|ed|es)|decreas(?:e|ed|es)|reduced|elevated|unchanged|"
        r"up[ -]?regulat(?:ed|ion)|down[ -]?regulat(?:ed|ion)|"
        r"inhibited|stimulated|attenuated|abolished|prevented|"
        r"enhanced|suppressed|restored|induced|affected|changed|modified|"
        r"detected|formed|caused|catalyzed|restrained|showed|equivalent|positive|negative|"
        r"lower|higher|reversed|depleted|altered|observed|identified|preserved)\b|"
        r"\bno (?:significant |detectable |measurable )?(?:changes?|effects?|differences?|increase|decrease)\b|"
        r"\b(?:weak|potent|strong) (?:inhibitor|inducer)\b|"
        r"\b(?:low|weak|minimal) effect\b|\bfell into\b.{0,50}\bpattern\b|"
        r"\b(?:significant|measurable|marked) (?:inhibition|increase|decrease)\b|"
        r"\b(?:concentration|dose|time)[ -]dependent (?:effect|toxicity)\b|"
        r"\b(?:not|non)[ -]?(?:cytotoxic|toxic|inhibitory)\b|"
        r"\bno (?:measurable|detectable)\b.{0,40}\b(?:binding|activity|toxicity)\b|"
        r"\b(?:significant|significantly)\s+(?:cyto)?toxicity\b|"
        r"\bnot (?:significantly )?(?:changed|altered|affected|detected)\b|"
        r"\b(?:significant|significantly|statistically significant)\b.{0,40}\b(?:differences?|different)\b|"
        r"\bchange\b.{0,80}\b(?:observed|detected)\b",
        observation,
    )
    # Scalar-led values and explicitly attributed numeric readouts; no generic
    # "contains a digit" rule (e.g. JC-1, A520, Fig. 7, 10 uM exposure).
    numeric = re.search(
        r"^\s*[<>~≤≥]?\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s|%|$)|"
        r"\b(?:ic50|ec50|ki|kinact)\b[^.;()]{0,30}?[<>=:~≤≥]\s*[+-]?(?:\d|\.\d)|"
        r"\b(?:ic50|ec50|ki|kinact|rate|activity|viability|absorbance|ratio|value)"
        r"\s*(?:was|were|of|=|:)\s*[<>~≤≥]?\s*[+-]?(?:\d|\.\d)|"
        r"\b(?:determined|measured) concentration\s+[<>~≤≥]?\s*\d|"
        r"\b(?:present|found)\s+(?:at|in concentrations? of)\s*[<>~≤≥]?\s*\d|"
        r"\bp\s*[<≤]\s*0\.\d+\s*(?:vs\.?|versus)\s+control\b|"
        r"\b(?:ic50|ec50)\b.{0,20}\b\d+\s*[×x]|"
        r"\b\d+(?:\.\d+)?\s*(?:%|[-–]?fold\b)",
        observation,
    )
    if qualitative or numeric:
        return "reported"
    # Only affirmative method/preparation predicates or explicit pointers to
    # unavailable results establish description-only. "Assay" or a figure ID
    # anywhere in otherwise unfamiliar prose is insufficient.
    # Parenthetical qualifications can say "no numeric value reported" while
    # the main clause reports a real qualitative outcome. They cannot veto it.
    # Do not require every valid outcome to be in our recognition vocabulary.
    main_clause = re.sub(r"\([^()]*\)", "", text).strip()
    description = re.search(
        r"\b(?:measured|quantified|monitored|assessed|evaluated|determined|"
        r"collected|prepared|depicted|calculated|measurement|assessment)\b|"
        r"\b(?:ic50|ec50)\s+generated\b|"
        r"^cell[ -]cycle phase percentages\s*$|"
        r"\bno (?:quantitative )?results? (?:is |was |were )?(?:provided|reported|given)\b|"
        r"^\s*percent cell viability\b.*\breported\s*$|"
        r"\b(?:values?|results?|outcome|findings|direction)\b.{0,40}\b(?:not (?:provided|stated|available|present)|"
        r"outside|elsewhere|results section)\b|"
        r"\b(?:values?|results?|effect|ic50|ec50)\b.{0,40}\b(?:in|shown in|see)\s+(?:fig|figure|table)\b|"
        r"^\s*(?:see\s+(?:fig(?:ure)?s?\.?|table)\b|"
        r"(?:fig(?:ure)?s?\.?|table)\s*\d|flow cytometry\b|western blot\b)",
        main_clause,
    )
    return "description_only" if description else "unresolved"


def score_dili_record(record: Mapping, *, condition: str = "") -> dict:
    """Pure relevance × field completeness for one DILI record, without an LLM.

    Accept a native/canonical row (including raw/reviewed JSON), or a prepared
    record. Prefer the source row: old prepared inputs may have lost conditions.
    Condition compatibility is reported separately and never changes this score.
    Completeness measures structured fields plus support presence, with a bounded
    method/figure-only result screen. It does not recover every fact from prose.
    Evidence keys and excluded descriptions explain the checks.
    This is a coarse information score, not reliability or molecule transfer.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of field names to values")
    if not isinstance(condition, str):
        raise TypeError("condition must be a string")
    if "input" in record:
        if not isinstance(record["input"], Mapping) or not isinstance(
            record["input"].get("source_fields"), Mapping
        ):
            raise TypeError("prepared record input.source_fields must be a mapping")
        source = dict(record["input"]["source_fields"])
        support = record["input"].get("support_text", "")
        context_review = "source_requests_context_review" in record.get(
            "llm_reasons", []
        )
        input_kind = "prepared"
    else:
        # Seed native_source with the row to preserve age/population/regimen,
        # while retaining its existing raw/reviewed JSON precedence (including null).
        source = native_source(record, record)
        if record.get("reviewed_record_json"):
            reviewed = record["reviewed_record_json"]
            source.update(
                json.loads(reviewed) if isinstance(reviewed, str) else reviewed
            )
        support = source.get("support_text", "")
        context_review = str(source.get("needs_more_context", "")).lower() == "true"
        input_kind = "source"
    relevant_fields = set(sum(GROUP_FIELDS.values(), [])) | set(_DILI_CONDITION_FIELDS)
    # Canonical-only cards retain typed values as key=value fragments. Recover
    # those scientific fields without overwriting explicit native/reviewed nulls.
    for name in (
        "canonical_endpoint_name",
        "canonical_measurement_text",
        "canonical_assay_context",
        "canonical_species_context",
    ):
        for key, value in _information_parts(
            source.get(name), split_semicolons=name != "canonical_measurement_text"
        ):
            if (
                key in relevant_fields
                and not key.startswith("canonical_")
                and key not in source
            ):
                source[key] = value
    fields = {
        k: " | ".join(v for _, v in _information_parts(source.get(k)))
        for k in relevant_fields
        if source.get(k) is not None
    }

    # Recover exact typed qualifiers only, not arbitrary disease mentions or
    # result prose. Explicit top-level/reviewed values, even null, take priority.
    for name in ("qualifying_conditions", "study_context", "exposure_context"):
        for key, value in _information_parts(source.get(name)):
            if key in _DILI_CONDITION_FIELDS and key not in source:
                fields[key] = value
    for key in _DILI_CONDITION_FIELDS[1:]:
        if fields.get(key):
            fields[key] = fields[key].lower().replace(" ", "_")

    endpoints = {k: fields[k] for k in GROUP_FIELDS["endpoint"] if fields.get(k)}
    endpoint = normalize_endpoint(" | ".join(endpoints.values()))
    endpoint = re.sub(r"\bnon hepatic\b", "nonhepatic", endpoint)
    # Bibliographic/evidence-basis/exposure-only wrappers do not constitute
    # an assay or tested system, even inside canonical_assay_context.
    systems = {k: fields[k] for k in _DILI_SYSTEM_FIELDS if fields.get(k)}
    for k in _DILI_ASSAY_FIELDS:
        parts = [
            v
            for name, v in _information_parts(source.get(k))
            if not name
            or name in (*_DILI_SYSTEM_FIELDS, *_DILI_ASSAY_FIELDS, "culture_format")
        ]
        if parts:
            systems[k] = " | ".join(parts)
    system = normalize_endpoint(" | ".join(systems.values()))
    positive_system = re.sub(
        r"\b(?:no|without|non|not(?: a| an)?)\s+(?:hepatic|livers?|hepatocytes?)\b",
        "",
        system,
    )
    # A source's "liver microsomal/vitro system implied" is not an identified
    # hepatic system. Remove bounded qualified phrases, not other known models.
    positive_system = re.sub(
        r"\b(?:liver|hepatic|hepatocytes?)\b(?:[\s/]+\w+){0,5}\s+(?:implied|inferred|presumed|assumed)\b|"
        r"\b(?:implied|inferred|presumed|assumed)\s+(?:liver|hepatic|hepatocytes?)\b",
        "",
        positive_system,
    )
    # Missing liver-specific competence is often boilerplate on non-liver cells.
    # It must not confer tissue identity; another explicit liver model still can.
    positive_system = re.sub(
        r"\bhepatic (?:metabolic )?competence\s+(?:(?:is|was)\s+)?"
        r"(?:not (?:stated|reported|specified|established|documented)|unknown|unclear)\b",
        "",
        positive_system,
    )
    positive_system = re.sub(
        r"\b(?:hepatocytes?|hepatic|liver)\s+(?:identity|origin)"
        r"(?:\s+and species)?\s+(?:(?:is|was)\s+)?"
        r"(?:not (?:confirmed|established|verified|reported)|unknown|uncertain)\b",
        "",
        positive_system,
    )
    # A disease background is not the anatomical site of the measured endpoint.
    positive_system = re.sub(
        r"\b(?:hepatic|liver) (?:stress |disease )?context\b", "", positive_system
    )
    hepatic = bool(_DILI_HEPATIC.search(positive_system))
    # Specific native severity can disambiguate a generic clinical phenotype;
    # background support paragraphs never rescue an unrelated readout.
    severity = normalize_endpoint(fields.get("maximum_reported_severity", ""))
    clinical = bool(fields.get("clinical_phenotype")) and bool(
        _DILI_LIVER_ENDPOINT.search(severity)
    )
    recognized = bool(
        _ENDPOINT_PATTERN.search(endpoint) or _DILI_EXTRA_READOUT.search(endpoint)
    )
    liver_endpoint = bool(_DILI_LIVER_ENDPOINT.search(endpoint))
    off_task = bool(re.search(_OFF_TASK, endpoint))
    if off_task:
        relevance, status = (
            (0.5, "mixed_scope") if liver_endpoint or clinical else (0.0, "unrelated")
        )
        reason = (
            "mixed_liver_and_offtask_readouts"
            if relevance
            else "explicit_offtask_endpoint"
        )
    elif liver_endpoint or clinical:
        relevance, status, reason = 1.0, "related", "explicit_liver_endpoint"
    elif recognized and hepatic:
        relevance, status, reason = (
            1.0,
            "related",
            "mechanism_in_identified_hepatic_system",
        )
    else:
        relevance, status = 0.5, "unknown"
        reason = (
            "mechanism_without_identified_hepatic_scope"
            if recognized
            else "unrecognized_endpoint"
        )
    reasons = [reason]
    if re.search(r"\bmisidentified\b", system) and re.search(r"\bhela\b", system):
        relevance = min(relevance, 0.5)
        status = "uncertain_system" if relevance else status
        reasons.append("source_disclaims_hepatic_cell_identity")
    if re.search(
        r"\b(?:plant|leaves|seedlings?|cotyledons?|arabidopsis|pelargonium|chloroplast)\b",
        system,
    ):
        relevance = min(relevance, 0.25)
        reasons.append("plant_system_distant_from_task")
        if relevance:
            status = "distant_system"

    result_fields = (
        "canonical_measurement_text",
        "measurement_text",
        "reported_value",
        "reported_result",
        "result_value",
        "response_value",
        "endpoint_result",
        "quantitative_value",
        "quantitative_readout_value",
        *CALL_FIELDS,
    )
    results, excluded_results, unresolved_results = {}, {}, {}
    reported_result = False
    for k in result_fields:
        kept, excluded = [], []
        for name, value in _information_parts(source.get(k), split_semicolons=False):
            if name and name not in (*result_fields, "maximum_reported_severity"):
                continue
            result_status = _dili_result_status(value)
            # An explicit structured categorical call is already a reported
            # conclusion. This does not interpret its polarity or validity.
            if (name or k) in CALL_FIELDS and re.fullmatch(r"[a-zA-Z_]+", value):
                result_status = "reported"
            (excluded if result_status == "description_only" else kept).append(value)
            reported_result |= result_status == "reported"
            if result_status == "unresolved":
                unresolved_results.setdefault(k, []).append(value)
        if kept:
            results[k] = " | ".join(kept)
        if excluded:
            excluded_results[k] = excluded
    evidence = {
        "endpoint": sorted(endpoints),
        "result": sorted(k for k, value in results.items() if value),
        "support": ["support_text"] if list(_information_parts(support)) else [],
        "assay_or_system": sorted(systems),
    }
    checks = {key: bool(keys) for key, keys in evidence.items()}
    if excluded_results and not checks["result"]:
        reasons.append("result_description_without_reported_outcome")
    completeness = sum(checks.values()) / len(checks)
    compatibility = _dili_simple_condition(condition, fields)
    return {
        "version": DILI_SIMPLE_VERSION,
        "score": relevance * completeness,
        "components": {"relevance": relevance, "completeness": completeness},
        "statuses": {"relevance": status, "condition": compatibility["status"]},
        "condition": compatibility,
        "completeness_checks": checks,
        "completeness_basis": "structured_fields_and_support_presence",
        "completeness_evidence": evidence,
        "excluded_result_descriptions": excluded_results,
        "result_content_status": (
            "reported"
            if reported_result
            else "unresolved"
            if unresolved_results
            else "description_only"
            if excluded_results
            else "missing"
        ),
        "unresolved_result_descriptions": unresolved_results,
        "missing_fields": [key for key, present in checks.items() if not present],
        "reasons": reasons,
        "context_review_flag": context_review,
        "input_kind": input_kind,
    }
