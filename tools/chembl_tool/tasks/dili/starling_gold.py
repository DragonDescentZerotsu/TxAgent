"""Human DILI outcome candidates; source-text review is a separate ledger."""
import re
from tools.chembl_tool.common.starling.source_gold_review import passage, source_conditions

VERSION = 'dili_human_conditioned_source_votes.v2.1'
TARGET = 'reported clinically meaningful human DILI outcome under the stated exposure and population; a negative is not universal absence of DILI risk'
PRIMARY = {'individual_case','case_series','registry_or_adjudicated_cohort','clinical_trial','observational_epidemiology','explicit_negative_evidence'}
INJURY = re.compile(r'\b(?:DILI|hepatotoxic\w*|(?:liver|hepatic) (?:injur\w*|failure|toxicity)|hepatitis|cholestasis)\b', re.I)
NEGATIVE = re.compile(
    r'\b(?:non[- ]hepatotoxic|not hepatotoxic|'
    r'no (?:reports of |evidence of |signs of |cases of |clinically significant |significant |drug.induced |drug.related )*(?:hepatotoxicity|liver injury|hepatic injury|DILI)|'
    r'no (?:patients|subjects|participants) (?:experienced|developed|had|showed) (?:any |significant |clinically significant |drug.related )*(?:hepatotoxicity|liver injury|hepatic injury|DILI)|'
    r'neither hepatotoxicity|'
    r'(?:hepatotoxicity|liver injury|hepatic injury|DILI)\b.{0,55}(?:not (?:observed|detected|seen|reported)|absent)|'
    r'not associated with (?:any |significant |clinically significant )*(?:hepatotoxicity|liver injury|DILI)|'
    r'without (?:evidence of )?(?:hepatotoxicity|liver injury|DILI))\b', re.I)



def decide(row):
    if row.get('agent_type') != 'small_molecule_or_chemical':
        return None,'non_single_small_molecule',[]
    status=row.get('causal_status'); basis=row.get('human_evidence_basis')
    if status not in {'established_or_definite','highly_probable','probable','not_supported'}:
        return None,'uncertain_or_indirect_causal_status',[]
    if basis not in PRIMARY:
        return None,'nonprimary_basis_requires_source_review',[]
    text=passage(row); support=row.get('support_text') or ''
    if not INJURY.search(text):return None,'no_explicit_injury_support',[]
    if row.get('maximum_reported_severity')=='liver_test_abnormality_only':return None,'laboratory_signal_only',[]
    exposure=row.get('exposure_context')
    if exposure not in {'therapeutic_use','overdose_or_supratherapeutic'}:
        return None,'unresolved_exposure_context',[]
    if status=='not_supported':
        if basis=='individual_case' or re.search(r'\b(?:single patient|this patient|this human case|the patient who|one patient)\b', text, re.I):return None,'individual_no_event_not_negative_hazard_evidence',[]
        if not NEGATIVE.search(support):return None,'negative_endpoint_requires_semantic_review',[]
        if re.search(r'\b(?:no (?:cases of )?hepatotoxicity.related fatal\w*|no.{0,45}fatal (?:drug.induced liver injury|DILI)|no.{0,45}Hy.s law|no increased rate of hepatotoxicity)\b',support,re.I):
            return None,'negative_limited_severity_or_comparison_requires_review',[]
        observed_cohort = bool(re.search(r'\b(?:in (?:this|the) (?:retrospective )?study|in this (?:trial|cohort|case series)|\d+ patients)\b', support, re.I))
        if re.search(r'\b(?:not (?:yet )?been reported|reports.{0,30}not appeared|rare(?:ly)?|few cases|infrequent)\b',support,re.I) or (re.search(r'\bno reports\b',support,re.I) and not observed_cohort):
            return None,'no_reports_or_residual_injury_risk_requires_review',[]
        if re.search(r'\b(?:unlikely cause|not the cause|cochrane|APRI|no (?:association|difference)|no (?:statistically )?significant (?:difference|association))\b',support,re.I):
            return None,'causal_exclusion_or_null_comparison_requires_review',[]
        if basis=='explicit_negative_evidence' and not re.search(r'\b(?:trial|patients|participants|cohort|subjects|volunteers)\b',text,re.I):
            return None,'negative_without_observed_human_study',[]
    if re.search(r'\b(?:unresolved co.suspect|competing contribution|alternative culprit)\b',text,re.I):
        return None,'unresolved_agent_attribution',[]
    if exposure=='therapeutic_use' and re.search(r'\b(?:overdos\w*|poisoning|intoxication)\b',support,re.I):
        return None,'source_exposure_contradiction_requires_review',[]
    atoms=source_conditions(row,'dili',['population=human','exposure='+exposure])
    return int(status!='not_supported'),'explicit_human_outcome_candidate',atoms
