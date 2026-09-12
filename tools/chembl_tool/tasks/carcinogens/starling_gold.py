"""Conditioned any-site carcinogenicity candidates, distinct from site nulls."""
import re
from tools.chembl_tool.common.starling.source_gold_review import passage, source_conditions, canonical_text

VERSION='carcinogens_conditioned_source_votes.v2.1'
TARGET='reported any-site carcinogenic hazard under the stated species and exposure; site-only negatives do not imply overall absence of hazard'
NEGATIVE=re.compile(r'\b(?:non[- ]carcinogenic|not carcinogenic|not tumou?rigenic|not a carcinogen|no (?:evidence (?:of|for) )?carcinogenic\w*|(?:carcinogenic\w*|tumou?rigenic\w*) (?:was |were )?not (?:observed|seen|detected|demonstrated)|(?:no|without) (?:evidence (?:of|for) )?(?:\w+ ){0,3}(?:tumou?rs?|neoplasms?)|did not (?:induce|cause) (?:\w+ ){0,3}(?:tumou?rs?|neoplasms?|cancer))\b',re.I)


def decide(row):
    if row.get('agent_category')!='individual_substance_or_drug':return None,'non_single_substance',[]
    label=row.get('carcinogenicity_conclusion')
    if label not in {'positive','established','negative'}:return None,'uncertain_or_nonbinary_conclusion',[]
    if row.get('carcinogenic_role') not in {None,'general_or_unspecified_carcinogen'}:return None,'initiator_promoter_or_cocarcinogen_role',[]
    basis,scope=row.get('evidence_basis'),row.get('evidence_scope');model=canonical_text(row.get('evidence_population_or_model'))
    if basis=='animal_carcinogenicity_bioassay' and scope=='experimental_animal':
        species=[s for s,pat in [('rat',r'\brats?\b'),('mouse',r'\b(?:mice|mouse)\b'),('hamster',r'\bhamsters?\b'),('dog',r'\b(?:dogs?|canine)\b'),('rabbit',r'\brabbits?\b'),('monkey',r'\b(?:monkeys?|rhesus|cynomolgus)\b')] if re.search(pat,model)]
        if len(species)!=1:return None,'animal_species_unresolved_or_mixed',[]
        species=species[0]
    elif basis in {'human_epidemiology','human_clinical_or_case_evidence'} and scope=='human':species='human'
    else:return None,'not_direct_human_or_animal_outcome',[]
    support=row.get('support_text') or '';text=passage(row)
    if re.search(r'\b(?:target cancer risk|THQ|TTHQ|hazard quotient|risk (?:indices|index|models?)|calculated.{0,30}cancer risk)\b',text,re.I):
        return None,'modeled_risk_not_observed_carcinogenicity',[]
    if re.search(r'\b(?:technical|nanoparticle|nanofiber|rutile|anatase|mixture)\b',str(row.get('agent_name') or ''),re.I):
        return None,'material_test_article_requires_identity_review',[]
    if not re.search(r'\b(?:carcinogen\w*|tumou?r\w*|cancer\w*|neoplas\w*|leuk[ae]mia|lymphoma)\b',support,re.I):return None,'no_explicit_carcinogenicity_support',[]
    if label=='negative':
        if re.search(r'\bno malignant tumou?rs?\b',support,re.I) and not re.search(r'\b(?:benign|not carcinogenic|no carcinogenic)\b',support,re.I):
            return None,'malignant_only_negative_not_all_neoplasms',[]
        if row.get('cancer_or_tumor') and re.search(r'not carcinogenic.{0,50}\b(?:kidney|liver|lung|bladder|skin|renal|intestinal)',support,re.I) and not re.search(r'any (?:other )?tissue|all tissues',support,re.I):
            return None,'site_limited_negative_not_overall_hazard',[]
        if not NEGATIVE.search(support):return None,'negative_endpoint_requires_semantic_review',[]
        if row.get('cancer_or_tumor') and not re.search(r'\b(?:non[- ]carcinogenic|not carcinogenic|no evidence (?:of|for) carcinogenicity|any (?:other )?(?:tumou?r|tissue)|all (?:tumou?r|tissue))\b',support,re.I):return None,'site_limited_negative_not_overall_hazard',[]
    atoms=['species='+species]
    simple={'rat','rats','mouse','mice','mouse (mus musculus)','rat (rattus norvegicus)','human','humans','hamster','hamsters','dog','dogs','rabbit','rabbits','monkey','monkeys'}
    if model and model not in simple:
        model=re.sub(r'\b(?:mice|rats)\b',lambda m:'mouse' if m[0]=='mice' else 'rat',model)
        model=re.sub(r'\b(?:f[- ]?344|fischer ?344|fisher ?344)\b','f344',model)
        model=re.sub(r'sprague[- ]dawley','sprague-dawley',model)
        atoms.append('model='+model)
    if row.get('exposure_route'):atoms.append('route='+row['exposure_route'])
    if re.search(r'\b(?:heterozygous|knockout|knock.out|transgenic|genetically modified)\b',text,re.I) and not row.get('qualifying_conditions'):
        return None,'genotype_condition_missing_requires_review',[]
    return int(label!='negative'),'explicit_carcinogenicity_candidate',source_conditions(row,'carcinogens',atoms)
