"""Task-specific oral relevance × structured completeness; no model or data loading."""

from collections.abc import Mapping
import json
import re
import unicodedata

from optimization.dili_score.scoring import _dili_result_status, _information_parts

BIOAVAILABILITY_SIMPLE_VERSION = "bioavailability_relevance_completeness.v3"
RELEVANCE_TIERS = {1.: "direct", .9: "availability_component", .8: "proximal_oral_proxy",
                   .7: "oral_concentration", .6: "gi_dissolution_or_stability", .5: "ambiguous_proxy",
                   .4: "absorption_timing", .25: "distant_proxy", 0.: "unrelated"}

# Aliases are source fields, not inferred labels or a record's source membership.
_ENDPOINT = tuple("""endpoint endpoint_name endpoint_category canonical_endpoint_name
canonical_endpoint_concept canonical_endpoint exposure_measure gut_wall_process metric_type""".split())
_RESULT = tuple("""reported_result result_value reported_value measured_value parameter_value
oral_bioavailability_value measurement_text canonical_measurement_text quantitative_value
response_value endpoint_result result_label outcome_label effect_direction substrate_status""".split())
_ASSAY = tuple("""assay_system assay_method assay_type assay_detail assay_or_test assay_and_readout
experimental_system biological_system biological_context biological_model biological_model_context
canonical_assay_context study_design study_context""".split())
_SPECIES = ("species", "species_or_population", "canonical_species_context")
_CONTEXT = tuple(dict.fromkeys((*_ASSAY, *_SPECIES, *"""intestinal_site biological_matrix
canonical_biological_matrix canonical_exposure_context qualifying_conditions experimental_conditions
exposure_route route route_of_administration oral_exposure_mode oral_dose dose
transporter_or_enzyme enzyme_or_pathway transport_mechanism canonical_measurement_target_id""".split())))
_TARGET = ("transporter_or_enzyme", "enzyme_or_pathway")
_CALLS = {"substrate_status", "result_label", "outcome_label", "effect_direction"}
_ANATOMICAL_FIELDS = set(_ASSAY) - {"study_context"} | {
    "intestinal_site", "biological_matrix", "canonical_biological_matrix",
}
_ROUTE_FIELDS = _ANATOMICAL_FIELDS | {
    "study_context", "exposure_route", "route", "route_of_administration",
    "oral_exposure_mode", "oral_dose",
}
_POPULATION_BACKGROUND = re.compile(r"\b(?:patients?|recipients?|transplant|disease)\b")
_TESTED_MODEL = re.compile(r"\b(?:cell\w*|monolayer|microsom\w*|hepatocyt\w*|perfus\w*|tissue|membrane|caco ?2|mdck|pampa|hela)\b")
_ROUTE_REFERENCE = re.compile(r"\b(?:reference|comparator|calculated from|derived from|prior to)\b")
_SYSTEM_INFORMATION = re.compile(
    r"\b(?:human|patient|volunteer|clinical|rat|rats|mouse|mice|murine|dog|beagle|pig|porcine|"
    r"rabbit|monkey|macaque|cynomolgus|in vivo|in vitro|perfus\w*|cell\w*|monolayer|"
    r"hepatocyt\w*|microsom\w*|caco ?2|pampa|mdck|trial|crossover|cross over)\b"
)


def _normalized(value):
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("∞", "inf").replace("τ", "tau").replace("pampа", "pampa")
    text = re.sub(r"[\s_\-‐‑‒–—−]+", " ", text).strip()
    return {"solubidity": "solubility", "aauc0 t": "auc0 t"}.get(text, text)


# Families stay separate so a shared transporter name cannot establish its site.
_BIOAVAILABILITY = re.compile(r"\bbioavailabilit\w*\b|\bsystemic availability\b|\bextent (?:of )?f\b")
_AVAILABILITY_COMPONENT = re.compile(
    r"\b(?:fraction absorbed|f ?[agh]|(?:intestinal|hepatic|gut wall) availability|"
    r"(?:hepatic|intestinal|gut wall) extraction ratio)\b|"
    r"\bfraction (?:escaping|that escapes|escaping first pass)\b[^;|.]{0,60}"
    r"\b(?:gut|intestinal|hepatic|liver)\b"
)
_ORAL = re.compile(r"\boral(?:ly)?\b|\bper ?oral\b|\bper os\b|\bp\.?\s?o\.?\b|\bintra ?gastric\b|\bgavage\b")
_NONORAL = re.compile(
    r"\bintravenous\b|\bi\.?\s?v\.?\b|\bintraperitoneal\b|\bsubcutaneous\b|\bintramuscular\b|"
    r"\b(?:intra)?nasal\b|\binhal(?:ed|ation)\b|\b(?:trans)?dermal\b|\btopical\b|\bocular\b|\bbuccal\b|"
    r"\bsublingual\b"
)
_GUT = re.compile(
    r"\b(?:intestin\w*|enterocyt\w*|enteric|gut|gastrointestinal|gastric|duoden\w*|jejun\w*|ileal|ileum|"
    r"colonic|colon)\b|\bcaco ?2\b|\bgi\b|\bussing\b|\bsingle pass intestinal perfusion\b|\bspip\b"
)
_LIVER = re.compile(
    r"\b(?:liver|hepatic|biliary|hepatobiliary|bile|hepatocyt\w*|hep ?g ?2|hep ?arg|hlm|rlm|mlm|dlm)\b"
)
_GI_MEDIUM = re.compile(
    r"\b(?:fassif|fessif|fassgf|fessgf|sgf|sif|biorelevant)\b|\bsimulated (?:intestinal|gastric|"
    r"gastrointestinal) fluid\b"
)
_REMOTE = re.compile(
    r"\b(?:brain|bbb|blood brain|skin|dermal|ocular|retinal|corneal|renal|kidney|hela|plant|arabidopsis|"
    r"chloroplast)\b"
)
_PREDICTED = re.compile(
    r"\b(?:predict(?:ed|ion|ions)|predictive (?:models?|algorithms?|simulations?|values?|estimates?|outputs?|scores?)|"
    r"in silico|qsar|computational|docking|simulated bioavailability|qikprop|swissadme|"
    r"admetlab|admetsar|pkcsm|boiled egg)\b"
)
_NEGATED_PREDICTION = re.compile(
    r"\b(?:not|no|without|rather than)\s+(?:a |an |any )?(?:model )?" + _PREDICTED.pattern
    + r"(?:\s+(?:models?|predictions?|results?|estimates?))?"
)
_MISSING_OBSERVATION = re.compile(
    r"\b(?:not|never)\s+(?:directly |experimentally )?(?:measured|tested|performed|determined|assessed)\b|"
    r"\b(?:no|without)\s+(?:reported |experimental |measured )?(?:measurement|measurements|assay|test|data|results)\b"
)
_NEGATED_SCOPE = re.compile(
    r"\b(?:no|without|not(?: a| an)?)\s+(?:identified\s+)?(?:intestinal|gut|hepatic|liver|"
    r"oral)\s+(?:model|system|context|administration|measurement)\b|"
    r"\bnon (?:intestinal|gut|hepatic|liver|oral)\b"
)
_UNCERTAIN_SCOPE = re.compile(
    r"\b(?:intestinal|gut|hepatic|liver|oral)\b(?:\s+\w+){0,4}\s+(?:implied|inferred|presumed|assumed|unknown|"
    r"uncertain)\b|\b(?:implied|inferred|presumed|assumed)\s+(?:intestinal|gut|hepatic|liver|oral)\b|"
    r"\b(?:hepatic|liver|intestinal|gut|oral)\s+(?:origin|identity|competence|context|"
    r"scope)\b.{0,35}\bnot (?:confirmed|established|verified|reported|specified|documented)\b"
)
_NO_MEASUREMENT = re.compile(
    r"\b(?:no|without)\s+(?:measured\s+)?(?:oral )?bioavailability measurements?\b|"
    r"\b(?:oral )?bioavailability\b[^;|.]{0,30}\bnot (?:measured|reported|determined)\b"
)
_NON_ADME = re.compile(
    r"\b(?:optical|uv|ultraviolet|visible|infrared|light) absorption\b|"
    r"\babsorption (?:spectrum|spectra|spectroscopy)\b|\bdissolved oxygen\b|"
    r"\b(?:radiation|occupational|environmental) exposure\b"
)

_ABSORPTION = re.compile(r"\b(?:absorp\w*|absorbed|fraction absorbed|fa)\b")
_PERMEABILITY = re.compile(
    r"\b(?:permeab\w*|permiab\w*|permeib\w*|perability|papp|pampa|caco ?2|mdck|transcellular transport|"
    r"accumulated mass)\b"
)
_DISSOLUTION = re.compile(
    r"\b(?:solubil\w*|dissol\w*|dissolved|gi stability|gastrointestinal stability|gastric stability|"
    r"intestinal stability)\b"
)
_TRANSPORT = re.compile(
    r"\b(?:transport\w*|intestinal (?:excretion|elimination)|efflux|secretory|uptake|absorptive|substrate|"
    r"p ?gp|p glycoprotein|bcrp|mrp\d*|mdr\d*|abcb\d+|abcg\d+|abcc\d+|oatp\w*|slco\w*|pept\d*|slc15a\d+)\b"
)
_METABOLISM = re.compile(
    r"\b(?:metaboli\w*|clearance|clint|extraction ratio|hepatic extraction|gut wall extraction|"
    r"substrate depletion|disappearance rate|microsom\w* stability|s9 stability|hepatocyte stability|"
    r"first pass|hydrolys\w*|glucuronid\w*|oxidation|dealkylation|cyp(?:\d[a-z0-9]*)?|ugt(?:\d[a-z0-9]*)?|"
    r"fmo(?:\d+)?|esteras\w*|carboxylesteras\w*)\b"
)
_PK = re.compile(
    r"\bauc(?:\d\w*|inf|last|ss|tau|t|b|all|exp|obs|oral)?\b|\baumc\b|\b[ct] ?max\b|\bc ?(?:min|avg|av|tau|"
    r"last|ss|trough)\b|\bc\d+(?:h)?\b|\bconcentration\b|\bexposure\b|\bfood effect\b|\belimination rate\b|"
    r"\boral iv\b"
)
_EXPOSURE_EXTENT = re.compile(r"\bauc\w*\b|\boral iv (?:exposure|auc)\b")
_ABSORPTION_TIMING = re.compile(r"\bt ?max\b|\btime to (?:maximum|maximal|peak) concentration\b|"
                               r"\babsorption rate(?: constant)?\b")
_WEAK = re.compile(
    r"\blog ?[pd]\b|\bpka\b|\blipophilic\w*\b|\bpartition coefficient\b|\b(?:plasma )?protein binding\b|"
    r"\bfraction unbound\b|\bmolecular weight\b|\bpolar surface area\b|\bionization\b"
)
_OFF_TASK = re.compile(
    r"\b(?:nephrotox\w*|renal toxicity|hepatotox\w*|liver injury|mutagen\w*|ames|cytotox\w*|cell viability|"
    r"apoptos\w*|antitumou?r|tumou?r growth|anticancer|receptor binding|analges\w*|antibacterial|antiviral|"
    r"blood brain barrier|bbb penetration|brain penetration|brain plasma|kp brain|skin (?:permeation|"
    r"permeability|flux)|(?:trans)?dermal delivery|ocular (?:delivery|permeability)|corneal (?:permeability|"
    r"flux))\b"
)
_FAMILIES = {
    "bioavailability": _BIOAVAILABILITY, "absorption": _ABSORPTION,
    "availability_component": _AVAILABILITY_COMPONENT,
    "permeability": _PERMEABILITY, "dissolution_solubility": _DISSOLUTION,
    "transport": _TRANSPORT, "metabolism_clearance": _METABOLISM,
    "exposure_pk": _PK, "weak_descriptor": _WEAK, "off_task": _OFF_TASK,
}
_SCOPE = {"oral": _ORAL, "nonoral": _NONORAL, "gut": _GUT, "liver": _LIVER,
          "gi_medium": _GI_MEDIUM, "remote": _REMOTE, "predicted": _PREDICTED}


def _values(source, names):
    """Read literal values and known typed fragments; reviewed nulls stay absent."""
    values = {}
    for field in names:
        for key, value in _information_parts(source.get(field), split_semicolons=False):
            if field == "support_text":
                key = ""
            if key and (key not in names or (key in source and source[key] is None)):
                continue
            values.setdefault(key or field, []).append(value)
    return {key: " | ".join(parts) for key, parts in values.items()}


def _has_prediction(text):
    return bool(_PREDICTED.search(_NEGATED_PREDICTION.sub("", text)))


def _relevance(endpoint, scope):
    no_measurement = bool(_NO_MEASUREMENT.search(endpoint) or _MISSING_OBSERVATION.search(endpoint))
    predicted = _has_prediction(endpoint)
    endpoint = _UNCERTAIN_SCOPE.sub("", _NEGATED_SCOPE.sub("", endpoint))
    families = [name for name, pattern in _FAMILIES.items() if pattern.search(endpoint)]
    oral = bool(scope["oral"] or _ORAL.search(endpoint))
    nonoral = bool(scope["nonoral"] or _NONORAL.search(endpoint))
    # A Caco-2/MDCK/PAMPA umbrella does not identify the assay actually used.
    endpoint_scope = endpoint
    if re.search(r"\b(?:mdck|pampa)\b", endpoint):
        endpoint_scope = re.sub(r"\bcaco ?2\b", "", endpoint_scope)
    gut = bool(scope["gut"] or _GUT.search(endpoint_scope))
    liver = bool(scope["liver"] or _LIVER.search(endpoint))
    remote = bool(scope["remote"] or _REMOTE.search(endpoint))
    if _NON_ADME.search(endpoint):
        tier, reason = 0.0, "non_adme_measurement"
    elif "bioavailability" in families:
        if no_measurement:
            tier, reason = 0.5, "bioavailability_measurement_not_established"
        elif oral:
            tier, reason = 1.0, "direct_oral_bioavailability"
        elif nonoral:
            tier, reason = 0.0, "nonoral_bioavailability"
        else:
            tier, reason = 0.5, "bioavailability_route_unknown"
    elif "availability_component" in families:
        tier, reason = .9, "measured_availability_component"
    elif "off_task" in families and not ((gut or liver or oral) and any(
        family in families for family in ("absorption", "permeability", "transport", "dissolution_solubility", "metabolism_clearance", "exposure_pk")
    )):
        tier, reason = 0.0, "explicit_offtask_endpoint"
    elif "weak_descriptor" in families:
        tier, reason = 0.25, "weak_physicochemical_or_binding_proxy"
    elif _ABSORPTION_TIMING.search(endpoint) and not _EXPOSURE_EXTENT.search(endpoint):
        tier, reason = (.4, "oral_absorption_timing") if oral or gut else (.5, "absorption_timing_context_incomplete")
    elif "metabolism_clearance" not in families and any(f in families for f in ("absorption", "permeability", "transport")):
        if gut:
            tier, reason = 0.8, "identified_intestinal_absorption_or_transport"
        elif liver and "transport" in families:
            tier, reason = 0.8, "identified_hepatic_transport"
        elif remote:
            tier, reason = 0.25, "transport_in_distant_system"
        elif oral:
            tier, reason = 0.8, "explicit_oral_absorption_proxy"
        else:
            tier, reason = 0.5, "absorption_or_transport_context_incomplete"
    elif "dissolution_solubility" in families:
        if gut or scope["gi_medium"] or oral:
            tier, reason = 0.6, "gi_dissolution_solubility_or_stability"
        else:
            tier, reason = 0.5, "dissolution_or_solubility_context_incomplete"
    elif "metabolism_clearance" in families:
        if gut:
            tier, reason = 0.8, "identified_intestinal_disposition"
        elif liver:
            tier, reason = 0.8, "identified_hepatic_disposition"
        elif oral:
            tier, reason = 0.8, "explicit_oral_disposition_proxy"
        elif remote or scope["nonoral"]:
            tier, reason = 0.25, "distant_or_nonoral_disposition"
        else:
            tier, reason = 0.5, "metabolic_site_unknown"
    elif "exposure_pk" in families:
        if oral:
            tier, reason = (.8, "oral_exposure_extent") if _EXPOSURE_EXTENT.search(endpoint) else (.7, "oral_concentration_proxy")
        elif nonoral or remote:
            tier, reason = 0.25, "nonoral_or_remote_exposure_proxy"
        else:
            tier, reason = 0.5, "exposure_route_unknown"
    else:
        tier, reason = 0.5, "unrecognized_endpoint"
    reasons = [reason]
    if no_measurement and tier > 0:
        tier = min(tier, .5)
        reasons.append("measurement_not_established")
    if "off_task" in families and tier > 0:
        tier = min(tier, 0.5)
        reasons.append("inseparable_mixed_endpoint_scope")
    if "bioavailability" in families and oral and nonoral and not scope["oral"]:
        tier = min(tier, 0.5)
        reasons.append("conflicting_route_context")
    if (scope["predicted"] or predicted) and tier > 0.5:
        tier = 0.5
        reasons.append("predicted_instead_of_measured")
    return tier, reasons, families


def score_bioavailability_record(record: Mapping) -> dict:
    """Score information relevance, not the molecule's bioavailability or reliability.

    Explicit source/reviewed fields take precedence over canonical aliases. Source
    passages provide support credit only, never missing endpoint/result facts.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of field names to values")
    source = dict(record)
    input_kind = "source"
    if "input" in record:
        prepared = record["input"]
        if not isinstance(prepared, Mapping) or not isinstance(prepared.get("source_fields"), Mapping):
            raise TypeError("prepared record input.source_fields must be a mapping")
        source = dict(prepared["source_fields"])
        source.setdefault("support_text", prepared.get("support_text"))
        input_kind = "prepared"
    for field in ("raw_record_json", "reviewed_record_json"):
        value = source.get(field)
        if value is None or value == "":
            continue
        native = json.loads(value) if isinstance(value, str) else value
        if not isinstance(native, Mapping):
            raise TypeError(f"{field} must contain an object")
        # Equivalent stale aliases must not resurrect explicit source corrections.
        for aliases in (_ENDPOINT, _RESULT, _ASSAY, _SPECIES):
            if any(key in native for key in aliases):
                for key in aliases:
                    source.pop(key, None)
        source.update(native)
    endpoints = _values(source, _ENDPOINT)
    if not endpoints and not any(key in source for key in _ENDPOINT) and "oral_bioavailability_value" in source:
        endpoints = {"oral_bioavailability_value": "oral bioavailability"}
    if not endpoints and not any(key in source for key in _ENDPOINT):
        endpoints = _values(source, _TARGET)
    contexts = _values(source, _CONTEXT)
    normalized_contexts = {key: _normalized(value) for key, value in contexts.items()}
    scope = {}
    for name, pattern in _SCOPE.items():
        scope[name] = []
        for key, value in normalized_contexts.items():
            if name in {"gut", "liver", "remote"}:
                if key not in _ANATOMICAL_FIELDS:
                    continue
                if _POPULATION_BACKGROUND.search(value) and not _TESTED_MODEL.search(value):
                    continue
            if name in {"oral", "nonoral"}:
                if key not in _ROUTE_FIELDS:
                    continue
                if name == "nonoral" and key == "study_context" and _ROUTE_REFERENCE.search(value):
                    continue
            asserted = _UNCERTAIN_SCOPE.sub("", _NEGATED_SCOPE.sub("", value))
            if (_has_prediction(asserted) if name == "predicted" else pattern.search(asserted)):
                scope[name].append(key)
    if _values(source, ("oral_dose",)):
        scope["oral"].append("oral_dose")
    # MDCK is an established permeability assay, not proof of an intestinal site.
    if any(re.search(r"\bmdck\w*\b", value) for value in normalized_contexts.values()):
        scope["remote"] = [key for key in scope["remote"]
                           if not re.search(r"\b(?:renal|kidney)\b", normalized_contexts[key])]
    matched_rules = []
    for field, value in endpoints.items():
        endpoint = _normalized(value)
        tier, reasons, families = _relevance(endpoint, scope)
        matched_rules.append({"field": field, "relevance": tier, "rule_ids": reasons, "families": families,
                              "context_fields": sorted({key for keys in scope.values() for key in keys})})
    known = [row for row in matched_rules if "unrecognized_endpoint" not in row["rule_ids"]]
    candidates = known if any(row["relevance"] > 0 for row in known) else matched_rules
    relevance = max((row["relevance"] for row in candidates), default=.5)
    # An unqualified alias cannot erase an explicit prediction/missing measurement.
    if any(row["relevance"] > 0 and any(reason in row["rule_ids"] for reason in
           ("predicted_instead_of_measured", "measurement_not_established")) for row in known):
        relevance = min(relevance, .5)
    flags = []
    if not endpoints or any("unrecognized_endpoint" in row["rule_ids"] for row in matched_rules):
        flags.append("unrecognized_endpoint")
    if any("unknown" in reason or "incomplete" in reason or "not_established" in reason
           for row in matched_rules for reason in row["rule_ids"]):
        flags.append("incomplete_task_context")
    if any("conflicting_route_context" in row["rule_ids"] for row in matched_rules):
        flags.append("conflicting_route_context")
    if any("inseparable_mixed_endpoint_scope" in row["rule_ids"] for row in matched_rules):
        flags.append("mixed_endpoint_scope")
    if any("predicted_instead_of_measured" in row["rule_ids"] for row in matched_rules):
        flags.append("predicted_evidence")
    results, excluded, unresolved = {}, {}, {}
    reported = False
    for field, value in _values(source, _RESULT).items():
        status = _dili_result_status(value)
        if field in _CALLS and status != "description_only":
            status = "reported"
        if status == "description_only":
            excluded[field] = [value]
        else:
            results[field] = value
            reported |= status == "reported"
            if status == "unresolved":
                unresolved[field] = [value]
    assay_information = _values(source, (*_ASSAY, *_SPECIES))
    if "study_context" in assay_information and not _SYSTEM_INFORMATION.search(_normalized(assay_information["study_context"])):
        assay_information.pop("study_context")
    evidence = {
        "endpoint": sorted(endpoints), "result": sorted(results),
        "support": list(_values(source, ("support_text",))),
        "assay_or_system": sorted(assay_information),
    }
    checks = {key: bool(value) for key, value in evidence.items()}
    completeness = sum(checks.values()) / 4
    reasons = list(dict.fromkeys(reason for row in matched_rules for reason in row["rule_ids"]))
    if not matched_rules:
        reasons.append("unrecognized_endpoint")
    return {
        "version": BIOAVAILABILITY_SIMPLE_VERSION,
        "score": relevance * completeness,
        "components": {"relevance": relevance, "completeness": completeness},
        "statuses": {"relevance": RELEVANCE_TIERS[relevance]},
        "reasons": reasons, "matched_rules": matched_rules, "ambiguity_flags": flags,
        "completeness_checks": checks, "completeness_evidence": evidence,
        "completeness_basis": "structured_fields_and_support_presence",
        "missing_fields": [key for key, present in checks.items() if not present],
        "result_content_status": "reported" if reported else "unresolved" if unresolved else "description_only" if excluded else "missing",
        "excluded_result_descriptions": excluded, "unresolved_result_descriptions": unresolved,
        "input_kind": input_kind,
    }
