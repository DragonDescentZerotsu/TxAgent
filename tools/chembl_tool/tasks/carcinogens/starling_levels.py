"""Record-semantic carcinogenicity families with conservative direct containment.

This module assigns retrieval families, never gold labels. Cancer cell lines
alone are not direct carcinogenicity evidence; explicit hazard assertions,
tumour outcomes and descriptions of carcinogens are contained in L2, including
negative, predicted, secondary and other-agent passages.
"""
import json
import re

VERSION = 'carcinogens_record_semantic_levels.v1'
FAMILIES = {
    1: 'carcinogenicity_direct',
    2: 'carcinogenicity_near_direct',
    3: 'transformation_growth_escape',
    4: 'genotoxicity_dna_damage_repair',
    5: 'reactive_metabolism_oxidative_injury',
    6: 'epigenetic_intercellular_growth_restraint',
    7: 'regenerative_receptor_proliferation',
}
LEVEL_DESCRIPTIONS = {
    1: 'Actual source records supporting accepted study-level carcinogenicity votes',
    2: 'Nonvoter carcinogenicity assertions and tumour outcomes, preserving role and limitations',
    3: 'Cell transformation, growth escape, survival, apoptosis and cell-cycle phenotypes',
    4: 'Mutations, chromosome damage, DNA lesions and damage/repair responses',
    5: 'Reactive metabolism, electrophilic reactivity, oxidative injury and redox homeostasis',
    6: 'Epigenetic state, gap junction communication and intercellular growth restraint',
    7: 'Injury-associated regeneration and receptor-driven growth or differentiation',
}
_DIRECT = re.compile(
    r'carcinogen\w*|tumou?rigen\w*|neoplas\w*|'
    r'\btumou?r(?:s)?[ _-]+(?:incidence|induction|formation|growth|volume|weight|burden|development|latency|promotion|outcome|bearing)|'
    r'\b(?:induc\w*|caus\w*|develop\w*|produc\w*|prevent\w*|suppress\w*|inhibit\w*|no|without)\b.{0,70}\btumou?rs?\b|'
    r'\b(?:induc\w*|caus\w*|develop\w*|risk|incidence)\b.{0,50}\b(?:cancer|carcinoma|adenoma|sarcoma|lymphoma|leuk[ae]mia|mesothelioma)\b', re.I | re.S)
_FIELDS = ('canonical_endpoint_name', 'canonical_measurement_text',
           'canonical_assay_context', 'canonical_species_context',
           'qualifying_conditions', 'support_text', 'extra_details',
           'assay_interpretation_conditions', 'interpretation_conditions',
           'injury_or_growth_context', 'endpoint_result', 'reported_result')
_ENDPOINTS = ('canonical_endpoint_name', 'canonical_assay_context',
              'assay_family', 'assay_domain', 'endpoint_category', 'phenotype_domain',
              'endpoint', 'endpoint_detail', 'endpoint_measure', 'endpoint_and_method',
              'assay_method', 'assay_type', 'assay_and_detection_method')
_MECHANISMS = (
    (6, re.compile(r'methylat|chromatin|histone|epigenet|gap[ _-]?junction|\bGJIC\b|contact[ _-]inhibition|intercellular|heterochromatin', re.I)),
    (4, re.compile(r'mutat|genotox|clastogen|chromosom|micronucle|comet|sister[ _-]chromatid|dna[ _-](?:damage|repair|adduct|lesion|strand|crosslink)|oxidative[ _-]dna|dominant[ _-]lethal|recombination|gene[ _-]conversion|\bHPRT\b|\bAmes\b|\bSCE\b', re.I)),
    (5, re.compile(r'reactive[ _-]metabol|metaboli\w*[ _-](?:activ|depend)|covalent|electrophil|redox|glutathione|\bGSH\b|\bROS\b|\bRNS\b|peroxid|oxidative|antioxidant|radical|reduct\w*[ _-]potential', re.I)),
    (7, re.compile(r'regenerat|compensatory|receptor[ _-]mediated|hormon|endocrin|injury[ _-]driven|wound|differentiation|morphogenesis|hyperplasia|organ[ _-]growth', re.I)),
    (3, re.compile(r'transform|anchor\w*[ _-]independent|soft[ _-]agar|clonogen|proliferat|dna[ _-]synthesis|viabil|surviv|apopto|cell[ _-]cycle|mitos|senescen|immortali|cell[ _-]growth', re.I)),
)


def _text(row, fields):
    return ' '.join(str(row.get(k) or '') for k in fields)


def direct_guard(row):
    """Independent broad guard over complete structured/passage surfaces."""
    text = _text(row, _FIELDS + _ENDPOINTS)
    # Stage-03 adapters retain extra_details in canonical qualifiers. Also read
    # raw payload where available so no source-specific field escapes the guard.
    raw = row.get('raw_record_json')
    if raw:
        text += ' ' + raw
    return bool(_DIRECT.search(text.replace('_', ' ')))


def classify(row, is_voter=False):
    if row.get('retrieval_eligible') is False:
        return dict(level=0, family_key='', reason='source_identity_hold', direct_signal=False)
    direct = direct_guard(row)
    if is_voter:
        return dict(level=1, family_key=FAMILIES[1], reason='actual_voter_membership', direct_signal=True)
    if direct:
        return dict(level=2, family_key=FAMILIES[2], reason='broad_direct_containment', direct_signal=True)
    # A source-native domain alone is not an observed endpoint. In particular,
    # motor behaviour after injury is not a regeneration/proliferation assay.
    canonical_endpoint = str(row.get('canonical_endpoint_name') or '')
    if canonical_endpoint.startswith('phenotype_domain=') and 'endpoint_and_method=' not in canonical_endpoint:
        for level, pattern in _MECHANISMS:
            if pattern.search(str(row.get('support_text') or '')):
                return dict(level=level, family_key=FAMILIES[level], reason='sparse_endpoint_substantive_passage', direct_signal=False)
        return dict(level=0, family_key='', reason='domain_without_substantive_endpoint', direct_signal=False)
    endpoint = _text(row, _ENDPOINTS)
    if not endpoint.strip() and row.get('raw_record_json'):
        endpoint = _text(json.loads(row['raw_record_json']), _ENDPOINTS)
    for level, pattern in _MECHANISMS:
        if pattern.search(endpoint):
            return dict(level=level, family_key=FAMILIES[level], reason='endpoint_semantics', direct_signal=False)
    # Source membership alone cannot grant a family. Substantive text supplies
    # recall for sparse endpoints, with the same cross-source ordering.
    text = _text(row, _FIELDS)
    for level, pattern in _MECHANISMS:
        if pattern.search(text):
            return dict(level=level, family_key=FAMILIES[level], reason='passage_semantics', direct_signal=False)
    return dict(level=0, family_key='', reason='no_declared_endpoint_family', direct_signal=False)
