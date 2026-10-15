"""Offline BBB, Ames and carcinogenicity relevance × completeness scores.

These are information scores, not toxicity probabilities or reliability estimates.
No datasets, benchmark labels or model services enter the single-record API.
"""

from collections.abc import Mapping
from functools import lru_cache, partial
import json
import re

from optimization.bioavailability_score.scoring import _has_prediction, _MISSING_OBSERVATION, _normalized, _values
from optimization.dili_score.scoring import _dili_result_status

VERSIONS = {task: f"{task}_relevance_completeness.v3"
            for task in ("bbb_martins", "ames", "carcinogens")}
RELEVANCE_TIERS = {
    "bbb_martins": {1.: "direct", .85: "barrier_permeability", .7: "brain_exposure_or_transport",
                    .6: "csf_proxy", .5: "ambiguous_proxy", .4: "barrier_integrity",
                    .25: "distant_proxy", 0.: "unrelated"},
    "ames": {1.: "direct", .85: "other_bacterial_mutation", .75: "other_gene_mutation",
             .6: "physical_dna_damage", .5: "chromosomal_or_ambiguous", .35: "damage_response",
             .25: "remote_mechanism", 0.: "unrelated"},
    "carcinogens": {1.: "direct", .85: "transformation", .75: "fixed_mutation",
                    .65: "genomic_damage", .5: "ambiguous_proxy", .4: "regulatory_response",
                    .25: "remote_mechanism", 0.: "unrelated"},
}

_ENDPOINT = tuple("""endpoint endpoint_name canonical_endpoint_name canonical_endpoint_concept
assay_family endpoint_class endpoint_category endpoint_subtype endpoint_measure endpoint_detail
assay_domain phenotype_domain cancer_or_tumor quant_metric metric_name quantitative_metric transport_endpoint""".split())
_RESULT = tuple("""measurement_text canonical_measurement_text reported_result result_value
response_value endpoint_result quantitative_readout_value measured_value reported_value result_call
result_status result_direction result_interpretation effect_direction mutagenicity_result
carcinogenicity_conclusion classification_label bbb_permeability_label bbb_transport_label
interaction_conclusion passive_bbb_interpretation quant_value metric_value quantitative_value""".split())
_ASSAY = tuple("""assay_model assay_system assay_type assay_method assay_format assay_version
assay_and_detection_method assay_and_readout assay_method_and_endpoint biological_system
biological_model biological_test_system test_system experimental_context canonical_assay_context
species canonical_species_context evidence_population_or_model study_context study_design""".split())
_CONTEXT = (*_ASSAY, "evidence_basis", "evidence_type", "exposure_route",
            "metabolic_activation", "metabolic_activation_status", "metabolic_activation_system")
_TARGET = ("transporter_identifier", "normalized_transporter_identifier", "canonical_transporter_identifier",
           "mediator_identifier", "mediator_name")
_CALLS = {"result_call", "result_direction", "effect_direction", "mutagenicity_result",
          "carcinogenicity_conclusion", "classification_label", "bbb_permeability_label",
          "bbb_transport_label", "interaction_conclusion", "passive_bbb_interpretation"}
_normalize = lru_cache(maxsize=32768)(_normalized)
_result_status = lru_cache(maxsize=8192)(_dili_result_status)


def _pattern(text):
    return re.compile(r"\b(?:" + text + r")\b")


_UNCERTAIN = _pattern(r"implied|inferred|presumed|possibly|uncertain|unspecified|unqualified|context dependent")
_UNESTABLISHED = re.compile(r"\b(?:no|without|not(?: a| an)?)\s+"
                            r"(?:(?:identified|established) )?(?:bbb|blood brain(?: barrier)?|"
                            r"brain endothelial|bacterial reverse mutation|ames)\s+"
                            r"(?:model|assay|context|measurement|test)\b|"
                            r"\b(?:presumed|assumed|hypothetical)\s+(?:bbb|blood brain|ames)\b|"
                            r"\b(?:bbb|blood brain|ames|reverse mutation)\b[^;|.]{0,35}"
                            r"\bnot (?:tested|measured|established|identified|performed)\b")
_SYSTEM = _pattern(r"human|patients?|cohort|case control|animal\w*|rats?|mice|mouse|murine|dogs?|"
                   r"cells?|culture|monolayer|in vivo|in vitro|bioassay|bacter\w*|salmonella|"
                   r"escherichia|e coli|ta\s?\d+|wp2|pampa|mdck|bcrp|caco\s?2|perfus\w*")
_BBB_SITE = _pattern(r"bbb|blood brain|brain endothelial|brain microvascular|cerebral microvascular|"
                     r"brain capillar\w*|hcmec|bmv\w*|bcec\w*|rbec\w*|rbe4|b end ?3|bend3")
_BBB_MODEL = _pattern(r"in vitro|monolayer|cell\w*|pampa|mdck|transwell|co culture|coculture|div bbb")
_BBB_PERMEABILITY = _pattern(r"permeab\w*|papp\w*|pampa|log ?pe|pe|ps product|log ps")
_BARRIER_INTEGRITY = _pattern(r"evans blue (?:extravasation|content)|eb (?:extravasation|content)|"
                             r"teer|transendothelial electrical resistance|tight junction integrity|"
                             r"barrier (?:integrity|disruption|leakage)")
_BBB_REMOTE = _pattern(r"intestin\w*|caco\s?2|gut|renal|kidney|hepatic|hepatocyt\w*|cornea\w*|skin")
_DIRECT_BBB = _pattern(r"(?:bbb|blood brain barrier|brain) (?:permeab\w*|penetrat\w*|entry|"
                       r"distribution|access|cross\w*)|(?:cross\w*|penetrat\w*) (?:the )?(?:bbb|blood brain barrier)|"
                       r"blood to brain transport|brain (?:to )?(?:blood|plasma|serum) (?:ratio|partition\w*)|"
                       r"brain (?:to )?(?:blood|plasma) (?:auc|concentration) ratio|"
                       r"(?:unbound )?brain (?:to )?(?:unbound )?plasma ratio|log ?bb|kp ?uu|"
                       r"(?:apparent )?(?:blood brain|brain blood|brain plasma) partition coefficient|kp brain|"
                       r"cns penetration")
_BBB_PROXY = _pattern(r"brain (?:concentration|exposure|uptake|level\w*|accumulation)|brain endothelial cell uptake|"
                      r"csf|cerebrospinal|bbb substrate relationship|transporter limited brain exposure|"
                      r"luminal to abluminal|abluminal to luminal|brain uptake index|bbb level|"
                      r"evans blue (?:extravasation|content)|eb (?:extravasation|content)")
_TRANSPORT = _pattern(r"permeab\w*|papp\w*|pampa|log ?pe|pe|efflux\w*|influx\w*|transport\w*|"
                      r"substrate|uptake|"
                      r"p ?gp|p glycoprotein|abcb1|bcrp|abcg2")
_POTENCY = _pattern(r"inhibition potency|(?:michaelis menten )?km|ic50|ec50")
_BBB_TRANSPORTER = _pattern(r"p ?gp|p glycoprotein|abcb1[ab]?|bcrp|abcg2")
_DESCRIPTOR = _pattern(r"log ?[pd]\w*|lipophilic\w*|pka|polar surface area|molecular weight|"
                       r"protein binding|fraction unbound|solubil\w*|clearance|metaboli\w*|oral bioavailabilit\w*")
_BBB_OFF = _pattern(r"ames|mutagen\w*|dna damage|carcinogen\w*|tumou?r (?:growth|response)|"
                    r"cytotox\w*|cell viability|apoptos\w*|hepatotox\w*|nephrotox\w*|"
                    r"antibacterial|antiviral|analges\w*|receptor binding")
_EXPOSURE = _pattern(r"auc\w*|cmax|tmax|concentration|exposure|suv|percent id per g|"
                     r"extraction ratio|ps product|permeability surface area product|k1|ki|kp|vmax|vt")
_BRAIN_DOSING = _pattern(r"intracerebral|intracerebroventricular|intrathecal|direct brain injection")

_AMES_DIRECT = _pattern(r"ames|bacterial reverse mutation|salmonella reverse mutation|"
                        r"bacterial reversion|bacterial back mutation|oecd (?:tg )?471")
_BACTERIA = _pattern(r"salmonella|typhimurium|escherichia|e\.? coli|ta\s?(?:97\w*|98|100|102|104|153\d)|wp2\w*")
_REVERSAL = _pattern(r"reverse mutation|reversion|revertant\w*|back mutation|his\+|trp\+")
_NONBACTERIAL = _pattern(r"yeast|saccharomyces|fungal|(?<!non )mammalian cells?|mouse lymphoma|cho|v79")
_PROTECTIVE = _pattern(r"anti ?mutagen\w*|anti ?genotox\w*|mutation (?:prevention|suppression)|"
                        r"protection against (?:mutagen\w*|dna damage)")
_MUTATION = _pattern(r"(?:gene|somatic|germ cell|heritable|forward|point|microbial|microorganism|"
                     r"mammalian cell|drosophila|yeast|fungal|eukaryote|fixed|induced) mutation|"
                     r"mutation frequenc\w*|hprt|hgprt|xprt|mouse lymphoma|tk mouse|"
                     r"sequencing based induced mutation|recombination|gene conversion|transgenic rodent|"
                     r"specific locus|pig a mutation|dominant lethal|heritable translocation|germ cell heritable")
_DNA_DAMAGE = _pattern(r"dna (?:damage|adduct\w*|crosslink\w*|repair|strand\w*)|"
                       r"(?:targeted )?dna adduct measurement|oxidized base|oxidative dna damage|"
                       r"covalent binding or adduct|32p postlab\w*|comet|strand break\w*|"
                       r"micronucle\w*|chromosom\w* (?:aberration|damage|loss|malsegregation)|"
                       r"clastogen\w*|aneuploid\w*|malsegregation|sister chromatid|"
                       r"h2ax|h2a x|ddr|sos|umu\w*|p53|repair (?:activity|synthesis|deficient|kinetics)|"
                       r"alkaline (?:elution|unwinding)|genotox\w*|mutagen\w*|mutation assay\w*|"
                       r"other direct damage|topoisomerase|dna (?:polymerase|fibre|replication)|"
                       r"kinetochore|chromosome segregation|(?:ap|abasic) site|host cell reactivation|"
                       r"tubulin|microtubule|polyploidy|spindle|centrosome|gadd45a|cytogenetic\w*|"
                       r"multiflow|toxtracker")
_PHYSICAL_DNA_DAMAGE = _pattern(r"dna (?:damage|adduct\w*|crosslink\w*|strand\w*)|"
                                r"targeted dna adduct measurement|oxidized base|oxidative dna damage|"
                                r"32p postlab\w*|comet|strand break\w*|alkaline (?:elution|unwinding)|"
                                r"topoisomerase cleavage complex")
_CHROMOSOMAL = _pattern(r"micronucle\w*|chromosom\w* (?:aberration|damage|loss|malsegregation)|"
                        r"clastogen\w*|aneuploid\w*|malsegregation|sister chromatid|polyploidy|"
                        r"kinetochore|chromosome segregation|cytogenetic\w*")
_DNA_RESPONSE = _pattern(r"repair|h2ax|h2a x|ddr|sos|umu\w*|p53|gadd45a|multiflow|toxtracker|"
                         r"topoisomerase|dna (?:polymerase|fibre|replication)|"
                         r"tubulin|microtubule|spindle|centrosome")
_WEAK_MECHANISM = _pattern(r"antioxida\w*|thiol\w*|ros|rns|redox|oxidative (?:stress|damage|protein)|"
                           r"lipid peroxidation|glutathione|gsh|reactive (?:species|metabolite)|"
                           r"metabol\w*|detoxif\w*|conjugation|enzyme activity|cyp\w*|"
                           r"electrophil\w*|nucleophil\w*|covalent (?:adduct|binding)|"
                           r"cytotox\w*|apoptos\w*|cell (?:death|viability|survival)|necrosis")
_AMES_OFF = _pattern(r"bbb|blood brain|oral bioavailabilit\w*|skin permeab\w*|"
                     r"antibacterial|antiviral|analges\w*|receptor binding|tumou?r response")

_CARC_DIRECT = _pattern(r"carcinogenicity|carcinogenic potential|carcinogenic outcome|carcinogenic classification|"
                        r"cancer (?:risk|incidence|mortality)|tumou?r (?:incidence|multiplicity|latency|induction)|"
                        r"(?:induced|exposure induced) tumou?r|chemical carcinogenesis|oncogenicity")
_CARC_BASIS = _pattern(r"carcinogenicity bioassay|animal carcinogenicity|chemical carcinogenesis|"
                       r"human epidemiolog\w*|exposure related cancer|authoritative classification")
_TUMOR = _pattern(r"cancer\w*|carcinoma\w*|tumou?r\w*|neoplas\w*|sarcoma\w*|leuk(?:a?emia)\w*|lymphoma\w*|"
                  r"adenoma\w*|melanoma\w*|mesothelioma\w*|papilloma\w*|malignan\w*|oncogenesis|"
                  r"\w*carcinoma\w*|\w*sarcoma\w*|\w*carcinogenesis|hepatoma\w*|glioma\w*|myeloma\w*")
_THERAPY = _pattern(r"antitumou?r|anti tumou?r|anticancer|anti cancer|chemotherap\w*|"
                    r"therapeutic|tumou?r (?:treatment|regression|shrink\w*|response)|"
                    r"(?:implanted|established) tumou?r")
_IMPLANT = _pattern(r"xenograft|implanted tumou?r|established tumou?r")
_TRANSFORMATION = _pattern(r"cell transformation|transformed foci|anchorage independent|soft agar|"
                           r"preneoplastic|pre neoplastic|tumou?r promotion|tumou?r initiation|"
                           r"bhas\s?42|balb\s?3t3|syrian hamster embryo|immortalization|growth factor independence")
_GROWTH = _pattern(r"proliferation|dna synthesis|cell cycle|mitosis|receptor mediated growth|"
                   r"injury driven regeneration|injured regeneration|chromatin\w*|chromat\w* state|"
                   r"dna methylation|epigen\w*|histone\w*|gap junction|contact inhibition|"
                   r"three dimensional architecture|inflamm\w*|senescence|wound repair|oncogene expression|other direct growth control")


def _bbb(endpoint, context, *, target=""):
    site = bool(_BBB_SITE.search(endpoint) or _BBB_SITE.search(context))
    if _BRAIN_DOSING.search(context) and (_DIRECT_BBB.search(endpoint) or _BBB_PROXY.search(endpoint)):
        return .25, "brain_exposure_bypasses_barrier"
    if _BARRIER_INTEGRITY.search(endpoint):
        if site or re.search(r"\bbrain\b", endpoint + " | " + context):
            return .4, "barrier_integrity_not_compound_penetration"
        if _BBB_REMOTE.search(endpoint + " | " + context):
            return .25, "barrier_integrity_in_distant_system"
        return .5, "barrier_integrity_site_not_established"
    if _DIRECT_BBB.search(endpoint):
        return (.85, "bbb_model_permeability") if _BBB_MODEL.search(context) else (1., "direct_bbb_outcome")
    if _BBB_PROXY.search(endpoint):
        if re.search(r"\b(?:csf|cerebrospinal)\b", endpoint):
            return .6, "csf_exposure_proxy"
        return .7, "brain_exposure_or_barrier_proxy"
    if _POTENCY.search(endpoint) and _BBB_OFF.search(endpoint):
        return 0., "explicit_offtask_potency"
    if _TRANSPORT.search(endpoint) or (_POTENCY.search(endpoint) and _BBB_TRANSPORTER.search(context + " | " + target)):
        if site:
            if _BBB_PERMEABILITY.search(endpoint):
                return .85, "identified_barrier_permeability"
            return .7, "identified_barrier_transport"
        if _BBB_REMOTE.search(context) or _BBB_REMOTE.search(endpoint):
            return .25, "transport_in_distant_system"
        return .5, "transport_context_incomplete"
    if _POTENCY.search(endpoint):
        if _BBB_REMOTE.search(context) or _BBB_REMOTE.search(endpoint):
            return .25, "potency_in_distant_system"
        return .5, "potency_target_not_established"
    if _DESCRIPTOR.search(endpoint):
        return .25, "distant_adme_or_descriptor"
    if _BBB_OFF.search(endpoint):
        return 0., "explicit_offtask_endpoint"
    if _EXPOSURE.search(endpoint):
        return (.7, "identified_barrier_exposure_proxy") if site else (.5, "exposure_context_incomplete")
    return .5, "unrecognized_endpoint"


def _ames(endpoint, context):
    if _PROTECTIVE.search(endpoint):
        return .25, "protective_modulation_not_mutagenicity"
    if (_AMES_DIRECT.search(endpoint) or (_REVERSAL.search(endpoint) and _BACTERIA.search(context))
            or (endpoint in {"unspecified mutagenicity", "mutagenicity"} and _AMES_DIRECT.search(context))):
        return 1., "direct_bacterial_reverse_mutation"
    if re.search(r"\b(?:reverse|back) mutation\b", endpoint) and _NONBACTERIAL.search(endpoint + " | " + context):
        return .75, "nonbacterial_reverse_mutation_proxy"
    if _MUTATION.search(endpoint):
        if (re.search(r"\b(?:bacterial|microbial|microorganism) (?:forward |gene |point )?mutation\b", endpoint)
                and _BACTERIA.search(endpoint + " | " + context) and not _NONBACTERIAL.search(endpoint)):
            return .85, "other_bacterial_gene_mutation"
        return .75, "identified_gene_mutation_proxy"
    if _CHROMOSOMAL.search(endpoint):
        return .5, "chromosomal_damage_proxy"
    if _PHYSICAL_DNA_DAMAGE.search(endpoint):
        return .6, "physical_dna_damage_proxy"
    if _DNA_RESPONSE.search(endpoint):
        return .35, "dna_damage_response_proxy"
    if _DNA_DAMAGE.search(endpoint) or _REVERSAL.search(endpoint):
        return .5, "damage_or_unspecified_mutagenicity_proxy"
    if _WEAK_MECHANISM.search(endpoint):
        return .25, "remote_mechanistic_proxy"
    if _AMES_OFF.search(endpoint):
        return 0., "explicit_offtask_endpoint"
    return .5, "unrecognized_endpoint"


def _carcinogens(endpoint, context):
    if _THERAPY.search(endpoint) or (_IMPLANT.search(endpoint) and "growth" in endpoint):
        return 0., "treatment_or_implanted_tumor_response"
    if _IMPLANT.search(endpoint) and "formation" in endpoint:
        return .85, "cell_transformation_proxy"
    if _PROTECTIVE.search(endpoint):
        return .25, "protective_modulation_not_carcinogenicity"
    mechanistic = any(pattern.search(endpoint) for pattern in
                      (_TRANSFORMATION, _MUTATION, _DNA_DAMAGE, _GROWTH, _WEAK_MECHANISM))
    if _CARC_DIRECT.search(endpoint) and mechanistic:
        return .5, "mixed_carcinogenicity_mechanism_scope"
    if _CARC_DIRECT.search(endpoint) or (_TUMOR.search(endpoint) and _CARC_BASIS.search(context) and not mechanistic):
        if _THERAPY.search(context) or _IMPLANT.search(context):
            return .5, "carcinogenicity_context_conflicts_with_treatment"
        return 1., "direct_carcinogenicity_evidence"
    if _TRANSFORMATION.search(endpoint):
        return .85, "cell_transformation_proxy"
    if _MUTATION.search(endpoint) or _AMES_DIRECT.search(endpoint):
        return .75, "fixed_mutation_proxy"
    if _PHYSICAL_DNA_DAMAGE.search(endpoint) or _CHROMOSOMAL.search(endpoint):
        return .65, "genomic_damage_proxy"
    if _DNA_RESPONSE.search(endpoint):
        return .4, "dna_damage_response_proxy"
    if _DNA_DAMAGE.search(endpoint):
        return .5, "genotoxicity_type_not_established"
    if _GROWTH.search(endpoint):
        return .4, "growth_or_regulatory_proxy"
    if "clonogenic survival" in endpoint:
        return .25, "generic_reactivity_or_survival_proxy"
    if _WEAK_MECHANISM.search(endpoint):
        return .25, "generic_reactivity_or_survival_proxy"
    if _TUMOR.search(endpoint):
        return .5, "tumor_endpoint_context_incomplete"
    if _AMES_OFF.search(endpoint):
        return 0., "explicit_offtask_endpoint"
    return .5, "unrecognized_endpoint"


def score_record(record: Mapping, *, task: str) -> dict:
    """Return a task-specific information score; explicit reviewed aliases take priority."""
    if task not in VERSIONS:
        raise ValueError(f"Unsupported record-scoring task: {task}")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    source, input_kind = dict(record), "source"
    if "input" in source:
        prepared = source["input"]
        if not isinstance(prepared, Mapping) or not isinstance(prepared.get("source_fields"), Mapping):
            raise TypeError("prepared record input.source_fields must be a mapping")
        source = dict(prepared["source_fields"])
        source.setdefault("support_text", prepared.get("support_text"))
        input_kind = "prepared"
    for name in ("raw_record_json", "reviewed_record_json"):
        value = source.get(name)
        if value is None or value == "":
            continue
        native = json.loads(value) if isinstance(value, str) else value
        if not isinstance(native, Mapping):
            raise TypeError(f"{name} must contain an object")
        for aliases in (_ENDPOINT, _RESULT, _ASSAY, _TARGET):
            if any(key in native for key in aliases):
                for key in aliases:
                    source.pop(key, None)
        source.update(native)
    endpoints = _values(source, _ENDPOINT)
    # Only named task conclusions identify an endpoint; a source ID never does.
    direct_fields = {"bbb_martins": {"bbb_permeability_label": "bbb permeability outcome",
                                    "passive_bbb_interpretation": "passive bbb permeability outcome"},
                     "ames": {"mutagenicity_result": "unspecified mutagenicity"},
                     "carcinogens": {"carcinogenicity_conclusion": "carcinogenicity conclusion",
                                     "classification_label": "carcinogenic classification"}}[task]
    endpoints.update({key: label for key, label in direct_fields.items() if _values(source, (key,))})
    contexts = _values(source, _CONTEXT)
    context = " | ".join(_normalize(value) for value in contexts.values())
    # A negated assay/site cannot establish scope for another endpoint.
    scope = " | ".join(value for value in re.split(r"\s*[;|]\s*", context) if not _UNESTABLISHED.search(value))
    predictions = context + " | " + " | ".join(_normalize(value) for value in _values(source, direct_fields).values())
    classify = {"bbb_martins": _bbb, "ames": _ames, "carcinogens": _carcinogens}[task]
    if task == "bbb_martins":
        target = " | ".join(_normalize(value) for value in _values(source, _TARGET).values())
        classify = partial(classify, target=target)
    matched, flags = [], []
    for field, value in endpoints.items():
        endpoint = _normalize(value)
        tier, reason = classify(endpoint, scope)
        reasons = [reason]
        uncertain = bool(_UNCERTAIN.search(endpoint))
        if reason == "direct_bacterial_reverse_mutation" and endpoint == "unspecified mutagenicity":
            uncertain = False  # An explicitly identified Ames assay resolves the generic call.
        if uncertain or _UNESTABLISHED.search(endpoint):
            tier = min(tier, .5)
            reasons.append("task_scope_not_established")
        if _MISSING_OBSERVATION.search(endpoint):
            tier = min(tier, .5)
            reasons.append("measurement_not_established")
        if _has_prediction(endpoint) or _has_prediction(predictions):
            tier = min(tier, .5)
            reasons.append("predicted_instead_of_observed")
        # Check separated clauses, not the words of a compound assay name.
        clauses = re.split(r"\s*(?:;|\||\band\b|\bplus\b)\s*", endpoint)
        if tier > .5 and any(classify(clause, scope)[0] == 0 for clause in clauses if clause):
            tier = .5
            reasons.append("inseparable_mixed_endpoint_scope")
        matched.append({"field": field, "relevance": tier, "rule_ids": reasons,
                        "context_fields": sorted(contexts)})
    known = [row for row in matched if "unrecognized_endpoint" not in row["rule_ids"]]
    # Unknown aliases cannot promote an identified weak proxy. With no positive
    # identified evidence, preserve the unknown fallback rather than inventing zero.
    candidates = known if any(row["relevance"] > 0 for row in known) else matched
    relevance = max((row["relevance"] for row in candidates), default=.5)
    if any(row["relevance"] > 0 and any(reason in row["rule_ids"] for reason in
           ("predicted_instead_of_observed", "measurement_not_established")) for row in known):
        relevance = min(relevance, .5)
    reasons = list(dict.fromkeys(reason for row in matched for reason in row["rule_ids"]))
    if not endpoints:
        reasons.append("unrecognized_endpoint")
    for token, flag in (("unrecognized", "unrecognized_endpoint"), ("incomplete", "incomplete_task_context"),
                        ("not_established", "incomplete_task_context"), ("predicted", "predicted_evidence"),
                        ("mixed", "mixed_endpoint_scope"), ("conflicts", "conflicting_task_context")):
        if any(token in reason for reason in reasons) and flag not in flags:
            flags.append(flag)
    results, excluded, unresolved = {}, {}, {}
    reported = False
    for field, value in _values(source, _RESULT).items():
        status = _result_status(value)
        if field in _CALLS and status != "description_only":
            status = "reported"
        if status == "description_only":
            excluded[field] = [value]
        else:
            results[field] = value
            reported |= status == "reported"
            if status == "unresolved":
                unresolved[field] = [value]
    assay = _values(source, _ASSAY)
    if "study_context" in assay and not _SYSTEM.search(_normalize(assay["study_context"])):
        assay.pop("study_context")
    evidence = {"endpoint": sorted(endpoints), "result": sorted(results),
                "support": list(_values(source, ("support_text",))), "assay_or_system": sorted(assay)}
    checks = {key: bool(value) for key, value in evidence.items()}
    completeness = sum(checks.values()) / 4
    return {"version": VERSIONS[task], "task_id": task, "score": relevance * completeness,
            "components": {"relevance": relevance, "completeness": completeness},
            "statuses": {"relevance": RELEVANCE_TIERS[task][relevance]},
            "reasons": reasons, "matched_rules": matched, "ambiguity_flags": flags,
            "completeness_checks": checks, "completeness_evidence": evidence,
            "completeness_basis": "structured_fields_and_support_presence",
            "missing_fields": [key for key, present in checks.items() if not present],
            "result_content_status": "reported" if reported else "unresolved" if unresolved else "description_only" if excluded else "missing",
            "excluded_result_descriptions": excluded, "unresolved_result_descriptions": unresolved,
            "input_kind": input_kind}
