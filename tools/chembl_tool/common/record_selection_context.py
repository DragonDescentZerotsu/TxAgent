"""Source-only priorities for molecule-local record selection.

These bounded compatibility hints neither change family assignment nor infer
labels. Unrecognized endpoints retain soft features but receive no hard outcome
bonus. Unreported conditions stay unknown. Raw cards are never rewritten here.
"""

from __future__ import annotations

import re
from collections import Counter

from dili_score import DILI_SIMPLE_VERSION, score_dili_record

from .record_diversity import (
    ames_condition_rank, normalize_endpoint, select_level_records, selection_text,
)

VERSION = "record_selection_context.v3"
QUALITY_VERSION = "record_quality.v3"

# Family assignment has already restricted the candidate records to this level.
# This second gate applies to EACH observation, not to the whole parent record.
# These are readout vocabularies, NOT final-label rules. A negative mutation
# assay and a positive mutation assay receive the same eligibility. Share
# mechanism vocabulary so, e.g., chromosome damage cannot disappear in one task.
_GENETIC = (
    r"ames|mutagen|mutation|mutant|reverta|genotox|chromosom|chromatid|micronucl|"
    r"\bdna\b|adduct|repair|aneugen|aneuploid|clastogen|dominant lethal|"
    r"recombin|gene conversion|sequence level gene change|heritable lethal|"
    r"strand break|alkali labile|comet|tail moment|tail length|tail intensity|"
    r"topoisomerase|spindle|kinetochore|segregation|tubulin|microtubul|"
    r"\b(?:sos|umu|hprt|sce|uds|fpg|chk[12]|p53|gadd45\w*)\b|"
    r"h2ax|8 (?:oh|oxo|nitroguanine)|endo iii|abasic|pyrimidine dimer|"
    r"recessive lethal|nondisjunction|polyploid|heritable translocation|nuclear bud|"
    r"single nucleotide variants?|\bindels?\b|\bcnvs?\b|copy number variants?|"
    r"\b(?:mnpce|mn|sces|ner|ddr|53bp1|rad51|o6mgmt)\b|\bap sites\b"
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
_ENDPOINTS = {
    "bioavailability_ma": r"bioavailability|oral (?:exposure|absorption)|\b(?:auc|cmax|papp)\b|absorption|solubility|dissolution|permeability|efflux|intestinal|hepatic|clearance|metaboli|cytochrome|\bcyp\w*|p glycoprotein|\bpgp\b",
    "bbb_martins": r"\bbbb\b|blood brain|brain|\bcns\b|cerebrospinal|\bcsf\b|permeability|efflux|influx|uptake|p glycoprotein|\b(?:pgp|bcrp|mdr1|abcb1|abcg2)\b|neuro|central nervous",
    "ames": r"ames|mutagen|reverse mutation|reverta|genotox|chromosom|micronucl|\bdna\b|adduct|repair|oxidative|reactive oxygen|sister chromatid|aneugen|clastogen",
    "carcinogens": r"carcinogen|tumou?r|cancer|neoplas|adenom|carcinom|sarcom|lymphom|leuk[ae]m|cell proliferation|cell transformation|genotox|mutagen|\bdna\b|adduct|epigen|oxidative|apoptosis|cytotox",
    "dili": r"\bdili\b|liver|hepatic|hepatotox|hepatitis|cholesta|bilirubin|\b(?:alt|ast|alp|ggt|bsep)\b|bile|mitochondri|oxidative|reactive oxygen|apoptosis|cell (?:death|viability)|cytotox",
    "skin_reaction": r"sensiti[sz]|contact (?:allergy|dermatitis)|\bllna\b|\bdpra\b|protein (?:adduct|binding)|peptide (?:depletion|reactivity)|keratinocyte|dendritic|t cell|\b(?:nrf2|are|cd54|cd86)\b",
}

# Observed aliases supplement the original vocabulary. Do not match the support
# passage to rescue an unrelated observation (e.g. parasite load in a BA card).
_ALIASES = {
    "ames": (_GENETIC, _REDOX, _REACTIVE_METABOLISM),
    "carcinogens": (_GENETIC, _REDOX, _REACTIVE_METABOLISM, _CELL,
        r"transform|clonogen|soft agar|anchorage independent|senescen|immortali|"
        r"methylat|histone|chromatin|gap junction|\bgjic\b|contact inhibition|intercellular|"
        r"regenerat|hyperplasia|differentiation|morphogenesis|tissue architecture|organoid architecture|receptor mediated|"
        r"hormon|endocrin|compensatory|organ growth|hepatoma|plasmacytoma|papilloma|"
        r"hepatoblastoma|mesothelioma|blastoma|pheochromocytoma|proliferat|mitotic index|colony formation|"
        r"(?:s|g1|g0/g1|g2/m) phase|growth inhibition|h3k\d+|5mc|dnasei hypersensitiv|"
        r"dye coupling|dye migration|uterine.*weight|vaginal (?:cornification|epithelial height)|"
        r"epidermal thickness|ornithine decarboxylase"),
    "bbb_martins": (_TRANSPORT,
        r"cerebr|hippocamp|striat|cortex|parenchym|gray (?:matter|tissues)|"
        r"\blog ?bb\b|\bkp(?:,?uu)?\b|\bpapp\b|passive diffusion|"
        r"\b(?:pe|ps|qppmdck)\b|sedation|anticonvuls|seizure|catalepsy|receptor occupancy|"
        r"(?:large )?neutral amino acid (?:transport|carrier)|nucleoside transporter|"
        r"sodium fluorescein (?:extravasation|labelling)|passage of sodium fluorescein"),
    "bioavailability_ma": (_TRANSPORT,
        r"\bauc(?:\d|inf|last|tau|τ|l\b|ss\b|t\b)|area under (?:the )?(?:concentration time )?curve|"
        r"\b(?:tmax|cmin|c max|c24|mrt)\b|"
        r"(?:plasma|serum|systemic|dose normalized|total) (?:exposure|concentration|availability|elimination)|"
        r"(?:peak|maximum) (?:plasma |serum )?(?:concentration|level)|time to peak|"
        r"half life|\bt1/2\b|\bcl/f\b|mean residence time|fraction absorbed|absorbability|"
        r"(?:gut wall|first pass) (?:extraction|elimination)|\bfg substrate outcome:|"
        r"(?:gi|digestive|microsomal|hepatocyte|s9) stability|permeation|transepithelial|"
        r"transcellular transport|absorptive transport|supersaturation|disintegration|precipitation|"
        r"glucuronid|glucuronyltransferase|\bugt\w*|demethylat|demethylase|deethylat|hydroxylat|"
        r"substrate depletion|extraction ratio|(?:ap to bl|bl to ap) transport|absorptive flux|"
        r"secretory flux|\bp eff\b|systemic drug exposure"),
    "dili": (_REDOX, _REACTIVE_METABOLISM, _CELL, _IMMUNE, _TRANSPORT,
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
        r"^(?:clinical phenotype|biochemical pattern|maximum reported severity|causal status)$"),
    "skin_reaction": (_REDOX, _IMMUNE,
        r"contact hypersensitiv|allergenicity|patch test|lymph node|lymphocyte proliferat|ear swelling|"
        r"stimulation index|\b(?:ec3|nesil|h clat|musst|keratinosens|lusens)\b|"
        r"(?:cysteine|lysine|peptide|protein|hsa|gsh).{0,25}(?:adduct|deplet|reactiv|bind|modif|conjugat)|"
        r"hapten|\b(?:cd40|cd69|cd80|cd83|cd54|cd86|hla dr)\b|"
        r"cell surface thiol|schiff base|dcya depletion|thiol adduct|\bp38\b"),
}
_PATTERNS = {task: re.compile("|".join((pattern, *_ALIASES[task])))
             for task, pattern in _ENDPOINTS.items()}

# A few source schemas separate the readout from its assay/target. Only these
# bounded generic names can borrow explicit assay scope, never arbitrary outcomes
# from the same passage. The supporting text and result value are not searched.
_SCOPED_READOUTS = {
    "ames": (
        (r"(?:n |o )?hydroxylation|cyp\w+ (?:expression|activity)", _REACTIVE_METABOLISM),
    ),
    "carcinogens": (
        (r"(?:n |o )?hydroxylation|cyp\w+ (?:expression|activity)", _REACTIVE_METABOLISM),
    ),
    "bbb_martins": (
        (r"inhibition potency|other quantitative endpoint",
         _TRANSPORT + r"|permeability|efflux|influx|brain|blood brain|\bbbb\b"),
    ),
    "bioavailability_ma": (
        (r"(?:cellular )?uptake(?: rate)?|transport|flux|vmax|km|ki|ic50",
         _TRANSPORT + r"|permeability|efflux|absorptive|intestinal|microsom|metaboli|\bcyp\w*|\bugt\w*"),
    ),
    "skin_reaction": (
        (r"induction(?: threshold)?|elicitation(?: reaction| threshold)?|erythema(?: incidence| score)?|"
         r"erythema and swelling score|skin reaction|positive reaction|allergic reaction|noel",
         r"skin sensiti[sz]ation|contact hypersensitiv|\b(?:hript|hmt|gpmt|buehler)\b|guinea pig sensiti[sz]ation"),
        (r"cell proliferation|brdu incorporation(?: index)?",
         r"\bllna\b|lymph node|t cell|lymphocyte"),
        (r"adduct formation|reactivity|depletion",
         r"protein binding|peptide|cysteine|lysine"),
    ),
}


def relevant_outcomes(task, outcomes, source_fields=None):
    """A conservative task-endpoint gate, never a positive/negative label gate."""
    pattern = _PATTERNS[task]  # Fail closed on unsupported tasks.
    fields = source_fields or {}
    scope = normalize_endpoint(_text(fields, (
        "canonical_endpoint_name", "canonical_assay_context", "assay_type",
        "assay_system", "transporter_name", "aop_event",
    )))
    scoped_names = [name for name, assay in _SCOPED_READOUTS.get(task, ())
                    if re.search(assay, scope)]
    return frozenset((e, k) for e, k in outcomes
                     if pattern.search(normalize_endpoint(e))
                     or any(re.fullmatch(name, normalize_endpoint(e)) for name in scoped_names))


def retrieval_ames_rank(condition, fields):
    """Recognize explicitly named standard strains amidst variant descriptions.

    TA98NR alone does not imply TA98. A mixed/partial panel is compatible, never
    exact, and a pooled outcome is never assigned to an individual strain.
    The strict gold parser and the historical selector remain unchanged.
    """
    rank = ames_condition_rank(condition, fields)
    if rank != 2 or fields.get("canonical_endpoint_name") != "bacterial_reverse_mutation":
        return rank
    panel = str(fields.get("test_system") or "")
    strains = re.findall(r"(?<!\w)TA\s*(?:1535|1537|1538|100|102|104|97a?|98)(?![\w/,-]\d|[\w/])", panel, re.I)
    if not strains:
        return rank
    patched = {**fields, "test_system": ", ".join(s.replace(" ", "").upper() for s in strains)}
    return max(1, ames_condition_rank(condition, patched))


def _text(fields, names):
    return " | ".join(value for k in names if (value := selection_text(fields.get(k)))).lower()


def _condition_rank(task, condition, fields, support):
    if task == "ames":
        return retrieval_ames_rank(condition, fields)
    wanted = [part.split("=", 1) for part in condition.split("+") if "=" in part]
    if not wanted:
        return 2  # No condition is not evidence of healthy/standard conditions.
    context = _text(fields, ("qualifying_conditions", "study_context", "species_or_population",
                            "evidence_population_or_model", "canonical_species_context",
                            "species", "exposure_context", "population_context"))
    actual = set()
    if task == "bioavailability_ma":
        from .starling.external_condition import normalize_condition_text
        from ..tasks.bioavailability_ma.condition_ontology import classify_external_condition

        # Classify individual clauses, not a combined result/comparison passage.
        for clause in re.split(r"[;|]", normalize_condition_text(context)):
            result = classify_external_condition(clause)
            if result.signature:
                actual.update(result.signature.split("+"))
    elif task in {"bbb_martins", "skin_reaction"}:
        from .starling.external_condition import propose_pattern_condition
        if task == "bbb_martins":
            from ..tasks.bbb_martins.prepare_condition_review import RULES
        else:
            from ..tasks.skin_reaction.context_conditioned_benchmark import RULES
        for clause in re.split(r"[;|]", context):
            if re.search(r"\b(?:no|without|absence of|excluded)\b", clause):
                continue
            proposal = propose_pattern_condition(clause, rules=RULES)
            actual.update(atom.key for atom in proposal.atoms)
    elif task == "dili":
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
        if key == "species":
            from ..tasks.carcinogens.reviewed_conditions import mentioned_groups

            # Attribute species to the tested population/system. Qualifiers can
            # mention human-equivalent doses or assay platforms without testing
            # humans, and must not override explicit animal specimen fields.
            groups = mentioned_groups(_text(fields, (
                "species", "species_or_population", "canonical_species_context",
                "evidence_population_or_model", "test_system",
                "biological_test_system", "biological_system",
                "biological_system_context", "biological_model",
                "biological_model_context", "experimental_system",
            )))
            # Cross-species limitations are useful without being direct studies
            # in the query species. Preserve that distinction in the rank.
            limitations = [s for s in re.split(r"[.;]", support)
                           if re.search(r"not |absen|lack|only |specific|except|unlike", s, re.I)]
            if value in groups:
                ranks.append(0 if len(groups) == 1 else 1)
            elif groups and any(value in mentioned_groups(s) for s in limitations):
                ranks.append(1)
            else:
                ranks.append(3 if groups else 2)
            continue
        native = selection_text(fields.get(key)).lower()
        if native:
            actual.add(f"{key}={native}")
        values = {a.split("=", 1)[1] for a in actual if a.startswith(key + "=")}
        if value in values:
            ranks.append(0 if values == {value} else 1)
        elif values and key in {"prandial_state", "age_group", "release_profile", "barrier_state"}:
            ranks.append(3)
        else:
            ranks.append(2)
    return max(ranks, default=2)


def selection_context(task, condition, prepared, card, outcomes):
    fields = prepared["input"]["source_fields"]
    support = prepared["input"].get("support_text", "")
    qualifiers = _text(fields, ("qualifying_conditions", "molecule_role", "tested_entity_role",
                               "tested_entity_form", "chemical_entity_type", "oral_exposure_mode",
                               "formulation_or_solid_form", "formulation_vehicle", "extra_details"))
    rank = _condition_rank(task, condition, fields, support)
    reasons = []
    if task == "ames" and re.search(r"reaction mixture|nitrite.{0,20}treat|treat.{0,30}nitrite|nitrosat", qualifiers):
        rank = max(rank, 3)
        reasons.append("chemically_transformed_specimen")
    scoped = relevant_outcomes(task, outcomes, fields)
    endpoint = fields.get("canonical_endpoint_name") or card.get("endpoint", "")
    related = bool(scoped or relevant_outcomes(task, [(endpoint, "numeric_only")], fields))
    # These are anchor tie-breaks within compatibility, not deletion rules.
    direct = 1
    if task == "bioavailability_ma":
        from .record_numeric_features import absolute_oral_f

        measurement = absolute_oral_f(prepared)
        if measurement["status"] == "eligible":
            binding = measurement["measurement_binding"]
            direct = 0 if binding["analyte_scope"] == "reported_focal_f" and not binding["labelled_measurement"] else 2
            if direct == 2:
                reasons.append("qualified_measurement_scope")
        report_type = selection_text(fields.get("bioavailability_report_type")).lower()
        if "relative" in report_type:
            direct = 2
            reasons.append("relative_measurement")
        if re.search(r"smedds|snedds|self.{0,8}emuls|\bemulsion\b|nano|liposom|solid dispersion", qualifiers):
            direct = max(direct, 2)
            reasons.append("special_formulation")
    elif task == "ames" and normalize_endpoint(endpoint) == "bacterial reverse mutation":
        direct = 0
    elif task == "skin_reaction" and re.search(
        r"elicitation|(?:previously|already) sensiti[sz]ed|"
        r"\b[\w-]+[ ‐‑–—]allergic (?:individual|patient|subject)|"
        r"\b(?:individuals?|patients?|subjects?) allergic to\b|"
        r"\b\w+ positive (?:hairdressers?|patients?|subjects?).{0,30}\b(?:acd|allergic contact dermatitis)\b",
        normalize_endpoint(_text(fields, ("qualifying_conditions", "assay_context",
            "canonical_assay_context", "study_context", "species_or_population",
            "biological_system_context", "experimental_conditions"))),
    ):
        direct = 2
        reasons.append("elicitation_in_previously_sensitized_subjects")
    elif re.search(r"predict|in silico", _text(fields, ("evidence_basis", "evidence_type"))):
        direct = 2
        reasons.append("predicted_evidence")
    # Some BBB sources encode the delivery system as the assay model rather
    # than a formulation field. Keep this separate from the exposure route.
    if task == "bbb_martins" and re.search(
        r"\b(?:nanocapsules?|nanoparticles?|liposomes?)\b",
        _text(fields, ("canonical_assay_context", "assay_model", "formulation_or_solid_form",
                       "formulation_vehicle", "tested_entity_form")),
    ):
        direct = 2
        reasons.append("special_formulation")
    return {"condition_rank": (rank, int(not related)), "anchor_rank": direct,
            "outcomes": scoped, "context_audit": {"version": VERSION, "reasons": reasons,
                "outcomes_without_hard_bonus": sorted(set(outcomes) - scoped)}}


# Explicitly identified off-task readouts, not a complement of the relevance
# vocabulary. A vocabulary miss must remain unknown rather than score zero.
_OFF_TASK = {
    "bioavailability_ma": r"parasite load|parasitemia|antimicrobial|tumou?r size|viral load",
    "bbb_martins": r"oral bioavailability|intestinal absorption|renal clearance|antimicrobial",
    "ames": r"antimicrobial activity|bacterial growth inhibition|minimum inhibitory concentration|body weight|analgesia",
    "dili": r"renal toxicity|nephrotox|visual disturbance|skin pigmentation|ototoxic",
    "carcinogens": r"^body weight$|antimicrobial activity|analgesia",
    "skin_reaction": r"hyperpigmentation|skin hydration|hair growth|skin irritation only",
}


def endpoint_relevance(task, endpoint, source_fields=None):
    """Return related/unrelated/unknown using endpoint-bound source information."""
    name = normalize_endpoint(endpoint)
    if re.search(_OFF_TASK[task], name):
        return "unrelated"
    if relevant_outcomes(task, {(name, "numeric_only")}, source_fields):
        return "related"
    return "unknown"


def record_quality(task, condition, prepared, card, annotation=None):
    """Auditable rule score; no query labels, structural similarity or LLM calls.

    Q = E * (1 + C + O + I) / 4. Scores are an initial ordinal rubric, not
    calibrated probabilities. Outcome sign never changes record quality.
    For the separate, simple native DILI E*I score, use score_dili_record().
    """
    if annotation and any('evidence_basis' in o for o in annotation.get('outcomes', [])):
        return _assessed_record_quality(task, condition, prepared, card, annotation)
    fields = prepared["input"]["source_fields"]
    observations = (annotation or {}).get("outcomes", [])
    focal = [o for o in observations if o.get("focal_scope") == "focal"]
    endpoints = [o["endpoint"] for o in focal] or [prepared.get("endpoint") or card.get("endpoint", "")]
    statuses = [endpoint_relevance(task, e, fields) for e in endpoints]
    status = "related" if "related" in statuses else (
        "unrelated" if statuses and all(s == "unrelated" for s in statuses) else "unknown")
    relevance = {"related": 1.0, "unknown": 0.5, "unrelated": 0.0}[status]
    reasons = ["endpoint_" + status]
    system = _text(fields, ("canonical_species_context", "biological_test_system",
        "biological_system", "biological_context", "assay_system", "test_system",
        "biological_model_context", "biological_model", "biological_system_context",
        "experimental_system", "canonical_assay_context", "assay_model",
        "species_or_population", "evidence_population_or_model", "study_context"))
    endpoint_text = " ".join(map(normalize_endpoint, endpoints))
    # Negated tissue scope is not positive evidence of a hepatic model. Do not
    # apply this to the endpoint: 'no liver injury' is still liver evidence.
    positive_system = re.sub(
        r'\b(?:no|without|non[- ]?)\s*(?:hepatic|liver|hepatocyte)[-\w ]{0,35}(?=[;,.]|$)',
        '', system,
    )
    if task == "dili" and status == "related" and not re.search(
        r"liver|hepat|biliar|bile|cholest|dili|\b(?:alt|ast|bsep|ntcp)\b", endpoint_text + " " + positive_system
    ):
        relevance = 0.5
        reasons.append("mechanism_without_identified_hepatic_scope")
    if task in {'carcinogens', 'dili', 'bbb_martins', 'skin_reaction'} and re.search(
        r'\b(?:plant|leaves|seedlings?|arabidopsis|pelargonium|chloroplast)\b', system
    ):
        relevance = min(relevance, .25)
        reasons.append('plant_system_distant_from_task')
    outcomes = {(o["endpoint"], o["kind"]) for o in focal}
    context = selection_context(task, condition, prepared, card, outcomes)
    rank = context["condition_rank"][0]
    if not condition or condition == "no_reported_external_condition":
        compatibility, condition_status = 1.0, "not_applicable"
    else:
        compatibility = (1.0, 0.5, 0.5, 0.0)[rank]
        condition_status = ("matched", "limited", "unknown", "mismatched")[rank]
    reasons.extend(context["context_audit"]["reasons"])
    role = _text(fields, ("molecule_role", "tested_entity_role", "tested_entity_form",
                          "chemical_entity_type", "probe_or_analyte"))
    from .record_diversity import DIRECT_ROLES, MISSING
    roles = [selection_text(fields.get(k)).lower().replace(" ", "_")
             for k in ("molecule_role", "tested_entity_role", "chemical_entity_type") if fields.get(k)]
    attribution = 1.0 if (focal or any(r in DIRECT_ROLES for r in roles)) else 0.5
    object_status = "focal" if attribution == 1 else "unknown"
    if context["anchor_rank"] == 2 or "chemically_transformed_specimen" in reasons or re.search(
        r"metabolite|reaction mixture|radiolabel|total radioactivity|modifier|protectant", role
    ):
        attribution, object_status = 0.5, "qualified_scope"
    if observations and not focal:
        # v3 has focal/unclear, not an explicit non-focal outcome class.
        # Failure to resolve attribution is not evidence of the wrong molecule.
        attribution, object_status = 0.5, "unknown"
    if (annotation or {}).get("unresolved"):
        attribution = min(attribution, 0.5)
        object_status = "unresolved"
    def present(value):
        text = selection_text(value)
        return bool(text and not MISSING.fullmatch(text))
    # A qualitative source call is a result; a numerical unit is not required
    # for a classification. Missing arbitrary metadata/word count is not quality.
    reported = card.get("reported_value") or fields.get("canonical_measurement_text")
    has_result = present(reported) or bool(prepared.get("outcome_kinds")) or any(
        o.get("kind") not in {"not_reported", "unclear"} for o in focal)
    checks = {"endpoint": any(present(e) for e in endpoints), "result": has_result,
              "support": present(prepared["input"].get("support_text")),
              "assay_or_system": present(system) or present(fields.get("canonical_assay_context"))}
    completeness = sum(checks.values()) / len(checks)
    if "predicted_evidence" in reasons or "source_requests_context_review" in prepared.get("llm_reasons", []):
        completeness = min(completeness, 0.5)
    score = relevance * (1 + compatibility + attribution + completeness) / 4
    return {"version": QUALITY_VERSION, "score": score,
            "components": {"relevance": relevance, "condition": compatibility,
                           "object": attribution, "completeness": completeness},
            "statuses": {"relevance": status, "condition": condition_status, "object": object_status},
            "completeness_checks": checks, "reasons": sorted(set(reasons))}


def _assessed_record_quality(task, condition, prepared, card, annotation):
    """Evaluate each bound observation; unrelated background cannot downgrade it.

    V explicitly represents evidence basis/limitations instead of treating every
    nonempty result as a complete experiment. Q = V*E*(1+C+O+I)/4. The record
    receives its strongest usable observation, with all alternatives audited.
    """
    from copy import deepcopy
    from .record_diversity import GROUP_FIELDS

    basis_weights = {'measured': 1., 'predicted': .5, 'inferred': .35,
                     'classification': .6, 'methods_only': .1, 'unclear': .4}
    caps = {'insufficient_evaluation': .2, 'method_limitation': .5,
            'conflicting_results': .35, 'missing_experimental_details': .75,
            'inconclusive_result': .5}
    scores = []
    context_keys = set(GROUP_FIELDS['system'] + GROUP_FIELDS['exposure'])
    for index, observation in enumerate(annotation['outcomes']):
        source = deepcopy(prepared)
        fields = source['input']['source_fields']
        # Override the reported aspect, not every condition. A dose-only quote
        # must not erase the source's explicitly linked species or strain.
        if observation.get('conditions'):
            spans = observation['conditions']
            quoted_fields = {}
            for span in spans:
                if span['field'] in context_keys or span['field'] == 'canonical_assay_context':
                    quoted_fields.setdefault(span['field'], []).append(span['quote'])
            fields.update({key:' | '.join(values) for key,values in quoted_fields.items()})
            bound_context = ' | '.join(s['quote'] for s in spans)
            from ..tasks.carcinogens.reviewed_conditions import mentioned_groups
            system_quotes = [s['quote'] for s in spans if mentioned_groups(s['quote'])
                             and not re.search(r'human.equivalent|equivalent.{0,12}dose',s['quote'],re.I)]
            if system_quotes:
                for key in GROUP_FIELDS['system']:
                    fields.pop(key, None)
                fields['biological_system_context'] = ' | '.join(system_quotes)
            if task == 'ames':
                fields['canonical_endpoint_name'] = observation['endpoint']
                if re.search(r'\bTA\s*\d+', bound_context, re.I):
                    fields['test_system'] = bound_context
                absent = re.search(r'without S9|no S9|activation\s*=\s*absent', bound_context, re.I)
                present = re.search(r'\bwith S9|activation\s*=\s*present', bound_context, re.I)
                if absent or present:
                    fields['metabolic_activation'] = 'both' if absent and present else 'absent' if absent else 'present'
            fields['qualifying_conditions'] = bound_context
        simple = {k: v for k,v in observation.items()
                  if k not in {'evidence_basis','basis_evidence','result_scope','limitations'}}
        value = record_quality(task, condition, source, card, {'outcomes':[simple], 'unresolved':[]})
        components = value['components']
        limitations = {l['kind'] for l in observation.get('limitations', [])}
        strength = basis_weights[observation['evidence_basis']]
        basis_quotes = observation.get('basis_evidence', [])
        if (observation['evidence_basis'] == 'classification' and basis_quotes
                and all(s['field'] in {'carcinogenic_role', 'evidence_basis'} for s in basis_quotes)
                and any(s['quote'] == 'review_list_or_narrative_assertion' for s in basis_quotes)):
            strength = min(strength, basis_weights['inferred'])
            limitations.add('classification_supported_only_by_source_codes')
        for limitation in limitations:
            strength = min(strength, caps.get(limitation, 1.))
        consistency = _replicate_count_conflict(prepared['input']['source_fields'])
        if consistency:
            strength = min(strength, .35)
            limitations.add('replicate_threshold_count_conflict')
        if observation['kind'] == 'not_reported':
            strength = min(strength, .1)
            value['completeness_checks']['result'] = False
            components['completeness'] = min(components['completeness'], sum(value['completeness_checks'].values())/4)
        if observation['result_scope'] != 'focal_effect' or 'attribution_ambiguity' in limitations:
            components['object'] = min(components['object'], .5)
            value['statuses']['object'] = observation['result_scope']
        components['evidence_strength'] = strength
        value['score'] = strength * components['relevance'] * (1+components['condition']+components['object']+components['completeness'])/4
        value.update(observation_index=index, endpoint=observation['endpoint'], evidence_basis=observation['evidence_basis'],
                     result_scope=observation['result_scope'], limitations=sorted(limitations))
        value['reasons'] = sorted(set(value['reasons']) | {'basis_'+observation['evidence_basis']} | limitations)
        scores.append(value)
    best = max(scores, key=lambda s:s['score'])
    return {**best, 'aggregation':'strongest_bound_observation', 'observation_scores': scores}


def _replicate_count_conflict(fields):
    """Check explicitly reported replicate arithmetic, never infer a task label."""
    detail = selection_text(fields.get('extra_details'))
    values = re.search(r'\b(?:ratios|values|responses)\s*:\s*([\d., +\-]+)', detail, re.I)
    count = re.search(r'(?:significant|positive) in (\d+)\s*/\s*(\d+) experiments', detail, re.I)
    threshold = re.search(r'\bratio\s*>\s*(\d+(?:\.\d+)?)',
                          selection_text(fields.get('experimental_conditions')), re.I)
    if not (values and count and threshold):
        return False
    numbers = [float(x) for x in re.findall(r'[+-]?\d+(?:\.\d+)?', values[1])]
    return len(numbers) == int(count[2]) and sum(x > float(threshold[1]) for x in numbers) != int(count[1])



def select_contextual_records(cards, encoded, prepared_by_record, *, task,
                              query_condition="", previous_records=(), per_molecule=5):
    """Shared preparation adapter; cacheable source features need no re-extraction.

    cards are this level's eligible raw cards for already-retrieved molecules.
    Return encoded selected records plus audit; caller retains the raw cards.
    """
    from .progressive_assay_reasoning import _card_order

    counts = Counter(c["analog_id"] for c in cards)
    history = {}
    for c in previous_records:
        history.setdefault(c["molecule_id"], []).append(c["features"])
    by_molecule = {}
    context_audit = {}
    for card in cards:
        cid = card["card_id"]
        c = {**encoded[cid], "reported_value": card.get("reported_value"),
             "support_text": card.get("support_text")}
        if c["molecule_id"] != card["analog_id"]:
            raise ValueError("Feature/card molecule identity mismatch")
        if counts[card["analog_id"]] > per_molecule and per_molecule:
            c.update(selection_context(task, query_condition, prepared_by_record[card["record_id"]], card, c["outcomes"]))
            context_audit[cid] = {**c["context_audit"], "condition_rank": c["condition_rank"], "anchor_rank": c["anchor_rank"]}
        by_molecule.setdefault(c["molecule_id"], []).append((card, c))
    candidates = []
    for group in by_molecule.values():
        if len(group) > per_molecule and per_molecule:
            group.sort(key=lambda pair: (pair[1]["condition_rank"], pair[1]["anchor_rank"],
                                         _card_order(pair[0], measured_values_only=True)))
        candidates.extend(c for _, c in group)
    selected, audit = select_level_records(candidates, per_molecule=per_molecule,
        previous_features_by_molecule=history, selection_policy="molecule_local.v1")
    audit.update(context_version=VERSION,
                 record_context={c["id"]: context_audit[c["id"]] for c in selected if c["id"] in context_audit},
                 candidate_context_reasons=dict(Counter(reason for a in context_audit.values() for reason in a["reasons"])))
    return selected, audit
