"""Source-claim triage and condition normalization for reviewed new-task gold.

Rules produce auditable candidates, never evidence of human review. Unknown
qualifiers remain exact atoms; split feasibility is handled only downstream.
"""
import hashlib
import json
import re


def payload_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def passage(row):
    return ' '.join(str(row.get(k) or '') for k in ('support_text', 'qualifying_conditions', 'extra_details'))


def canonical_text(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().casefold()).rstrip('.')


def study_identity(row):
    """Use explicit original-study references, otherwise the reporting PMID.

    A cited study may support a claim, but another citing PMID cannot add a vote.
    Unknown secondary origins remain pending, not a new independent study.
    """
    text = passage(row)
    named = [(name, pattern) for name, pattern in (
        ('explore_xa', r'\bEXPLORE[- ]Xa\b'), ('re_ly', r'\bRE[- ]LY\b'), ('precision_aprocitentan', r'\bPRECISION\b'),
        ('road_rosuvastatin', r'\bROAD study\b'), ('opal_beyond', r'\bOPAL Beyond\b'),
        ('opal_broaden', r'\bOPAL Broaden\b'), ('medal', r'\bMEDAL\b'),
        ('scandinavian_simvastatin_survival', r'\bScandinavian Simvastatin Survival Study\b')) if re.search(pattern, text, re.I)]
    # Explicit named original trials are stable across citing articles. Mixtures
    # of named trials need a record-level decision rather than one invented vote.
    if len(named) > 1:
        return None, 'multiple_original_studies_require_review'
    if named:
        return 'trial:' + named[0][0], ''
    support = str(row.get('support_text') or '')
    if re.search(r'\b(?:our (?:patients?|study)|their patient|present (?:case|study)|goal of this study|the reported case|we (?:found|observed|studied)|were studied in sufficient detail to warrant inclusion)\b', support, re.I):
        pmid = str(row.get('pmid') or '')
        if re.fullmatch(r'[1-9][0-9]*', pmid):
            return 'pmid:' + pmid, ''
    citation_text = re.sub(r'[^.;]*(?:according to|classified according|classification of|methods? of)[^.;]*', '', text, flags=re.I)
    citations = re.findall(r'\b([A-Z][A-Za-z]{2,})(?:\s+et\s+al\.?)?(?:,\s*[A-Z][a-z]+)*(?:,?\s+(?:and|&)\s+[A-Z][A-Za-z]+)?\s*[,;(]?\s*((?:19|20)\d{2})[a-z]?\b', citation_text)
    stop = {'in','since','until','between','from','year','during','the','study','phase','fda','ntp','group','table','figure','fct'}
    citations += re.findall(r'\b(NTP)\s*[,;(]?\s*((?:19|20)\d{2})\b', citation_text)
    stop.discard('ntp')
    citations = sorted({(a.casefold(), y) for a,y in citations if a.casefold() not in stop})
    if re.search(r'\b(?:19|20)\d{2}[a-z]\s*[,/]\s*[a-z]\b', citation_text):
        return None, 'multiple_original_studies_require_review'
    if len(citations) == 1:
        return 'cited:' + ':'.join(citations[0]), ''
    secondary = re.search(r'\b(?:review\w*|meta.analysis|pooled|integrated (?:safety )?database|phase [IV12]+ and [IV23]+ studies|summariz\w*|summaris\w*|summar(?:y|ies) of|personal communication|and co.workers|referenc(?:e|es)\s*(?:\[\s*[1-9]\d*(?:\s*[-,]\s*[1-9]\d*)*(?=\s*\])|\(\s*[1-9]\d*(?=\s*\))|[1-9]\d*\b)|\w+ et al\.? (?:reported|found|showed)|previous studies|earlier studies|limited long.term.{0,12}studies|all .{0,30}trials|consensus relative risk|cited (?:in|work|studies)|refs?\.?\s*[([]?\d+|panel|re.evaluation|literature|label.s? (?:carcinogenesis|section)|NTP evaluation|National Toxicology Program|\d+ (?:clinical |randomized )?trials)\b', text, re.I)
    author_attribution = re.search(r'\b(?:study|experiment) by [A-Z][a-z]+\b', text)
    if len(citations) > 1 or secondary or author_attribution:
        return None, 'secondary_or_pooled_study_origin_requires_review'
    # Dose/duration alone cannot establish an original experiment: reviews also
    # quote doses. Require a study, cohort, case or experimental exposure, not dose alone.
    primary = re.search(r'\b(?:this study|our study|we (?:found|observed|reported)|case (?:report|series|[0-9])|\d+.year.old|(?:one|two|three|four|five|six|seven|eight|nine|ten) (?:patients|cases|women|men)|\d+ (?:[a-z]+[ -]){0,3}(?:patients|subjects|participants|volunteers|cases|rats|mice|animals)|(?:prospective|retrospective|randomized|controlled|crossover|double.blind|clinical|two.year|2.year) (?:[a-z]+[ -]){0,3}(?:study|trial)|(?:study|trial) population|(?:DILIN|registry) cohort|(?:rats|mice|patients) (?:were|received))\b', text, re.I)
    quantified_exposure = re.search(r'\d', text) and re.search(r'\b(?:(?:administered|given|injected|exposed|treated|fed).{0,100}\b(?:rats|mice|animals)|(?:rats|mice|animals).{0,100}\b(?:administered|given|injected|exposed|treated|fed))\b', text, re.I)
    if not (primary or quantified_exposure):
        return None, 'original_experimental_study_not_established'
    if not re.fullmatch(r'[1-9][0-9]*', str(row.get('pmid') or '')):
        return None, 'missing_valid_pmid'
    return 'pmid:' + str(row['pmid']), ''


def source_conditions(row, task, atoms):
    """Normalize known effect-modifying axes, retain all unparsed qualifiers.

    Exact dose/duration remains source provenance unless declared as a material
    restriction in qualifying_conditions, in which case its text is retained.
    """
    text = passage(row)
    qualifier = canonical_text(row.get('qualifying_conditions'))
    if qualifier in {'null', 'none', 'not reported', 'not applicable'}:
        qualifier = ''
    if task == 'dili':
        patterns = (
            ('healthy_volunteers', r'\bhealthy (?:adult |human |male |female )?(?:volunteers|subjects|participants)\b'),
            ('type_2_diabetes', r'\b(?:type (?:2|II) diabet\w*|T2DM)\b'),
            ('alzheimer_disease', r"\bAlzheimer"),
            ('atrial_fibrillation', r'\batrial fibrillation\b'),
            ('resistant_hypertension', r'\bresistant hypertension\b'),
            ('psoriasis', r'\bpsoria\w*\b'),
            ('inflammatory_bowel_disease', r"\b(?:Crohn|ulcerative colitis|inflammatory bowel)"),
            ('osteoarthritis', r'\bosteoarthritis\b'),
            ('rheumatoid_arthritis', r'\brheumatoid arthritis\b'),
            ('axial_spondyloarthritis', r'\b(?:ankylosing spondylitis|axial spondyloarthritis)\b'),
            ('metabolic_fatty_liver_disease', r'\b(?:NAFLD|NASH|MASLD|MASH|non.alcoholic fatty liver|steatohepatitis)\b'),
            ('renal_transplant', r'\b(?:renal|kidney) (?:allograft|transplant)\b'),
            ('hiv', r'\bHIV(?:.\d)?(?:.infected|.positive)?\b'),
            ('hepatitis_c', r'\b(?:HCV|hepatitis C)\b'),
            ('hepatitis_b', r'\b(?:HBV|hepatitis B)\b'),
            ('alcohol_dependence', r'\b(?:alcohol.dependent|alcohol dependence|alcohol use disorder)\b'),
            ('migraine', r'\bmigraine\b'),
            ('adhd', r'\b(?:ADHD|attention.deficit)\b'),
        )
        # Do not turn tests excluding competing infections into infected hosts.
        host_text = re.sub(r'[^.;]*(?:negative|excluded|absent|no history)[^.;]*', '', str(row.get('support_text') or ''), flags=re.I)
        if not qualifier and re.search(r'\bliver[- ]transplant(?:ed)? (?:recipients|patients)\b', host_text, re.I):
            atoms.append('population_context=liver_transplant')
        populations = [name for name, pattern in patterns if not qualifier and re.search(pattern, host_text, re.I)]
        if len(populations) == 1:
            atoms.append('population_context=' + populations[0])
        elif populations:
            atoms.append('unresolved_population_context=' + '|'.join(sorted(populations)))
        if re.search(r'\b(?:children|pediatric|paediatric|infants|neonates)\b', host_text, re.I):
            atoms.append('age_group=pediatric')
        if re.search(r'\b(?:intranasal|nasal spray)\b', text, re.I):
            atoms.append('route=intranasal')
        if re.search(r'\b(?:intramuscular|long.acting injectable|extended.release naltrexone)\b', text, re.I):
            atoms.append('formulation=long_acting_injectable')
        if re.search(r'\b(?:one.day course|single.day|single dose|single oral dose)\b', text, re.I):
            atoms.append('regimen=single_dose_or_one_day')
    # Unknown qualifiers are not silently discarded or converted to null. Their
    # semantic normalization is supplied by the hash-bound review ledger.
    if qualifier:
        atoms.append('reported_condition=' + qualifier)
    return sorted(set(a.replace('%', '%25').replace('+', '%2B') for a in atoms))


def outcome_in_condition(row, task):
    """Never publish an unreviewed answer-bearing extraction as query context."""
    qualifier = str(row.get('qualifying_conditions') or '')
    if task == 'dili':
        pattern = r'\b(?:DILI|VOD|hepatotox\w*|hepatitis|cholestasis|(?:hepatic|liver|hepato.renal) failure|injur\w*|toxic\w*|fatal\w*|adverse|negative|positive|not associated|caus\w*|attribut\w*|rucam|probable|co.suspect\w*|elevat\w*)\b'
    else:
        pattern = r'\b(?:negative|positive|tumou?r\w*|carcinog\w*|neoplas\w*|cancer\w*|malignan\w*|leuka?emi\w*|lymphom\w*|adenom\w*|sarcom\w*|mesotheliom\w*|hazard|affected|elevations? observed|found in|(?:increased|raised|reduced)\s+(?:incidence|survival|mean life))\b'
    return bool(re.search(pattern, qualifier, re.I))


def answer_bearing_atoms(atoms, task):
    """Check the actual query surface, including hash-reviewed replacements.

    Baseline diagnoses in a reviewed model/population are legitimate context;
    explicit outcome predicates are not. Unparsed qualifiers use the broad
    triage guard; semantic review must replace them with pre-outcome axes.
    """
    for atom in atoms:
        key, _, value = atom.partition('=')
        if key == 'reported_condition' and outcome_in_condition({'qualifying_conditions': value}, task):
            return True
        if re.search(r'\b(?:developed|developing|induced|raised|increased|reduced|negative|positive)\b.{0,60}\b(?:tumou?r\w*|leuka?emi\w*|lymphom\w*|adenom\w*|incidence|survival|life span|liver injury|hepatotoxicity)\b', value, re.I):
            return True
    return False
