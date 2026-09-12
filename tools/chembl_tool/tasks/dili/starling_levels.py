"""Record-level DILI retrieval families, separate from strict human gold voting.

The source run never determines a family. Direct-containing mixed/protective,
negative and predicted passages are contained in L2; this grants no vote.
"""
from __future__ import annotations
import re

VERSION = 'dili_record_semantic_levels.v1'
FAMILIES = {1: 'dili_actual_voter', 2: 'dili_direct_outcome', 3: 'dili_hepatic_cell_function',
            4: 'dili_hepatobiliary_homeostasis', 5: 'dili_bioactivation_mitochondria',
            6: 'dili_immune_perturbational_signature', 7: 'dili_organelle_redox_stress'}
LEVEL_DESCRIPTIONS = {
    1: 'Actual records contributing to the current strict human DILI source votes.',
    2: 'Nonvoter liver injury outcomes, predictions, clinical signals and mixed direct-containing passages; role, system and uncertainty remain explicit.',
    3: 'Hepatic cell survival, membrane integrity, functional reserve and synthetic or secretory function.',
    4: 'Bile-acid transport, canalicular function, hepatobiliary homeostasis and bile-acid pools.',
    5: 'Metabolic bioactivation, reactive metabolites or binding, mitochondrial respiration and cellular energy.',
    6: 'Hepatic immune interactions, inflammatory responses and perturbational molecular signatures.',
    7: 'Hepatic redox, lipid, ER/calcium, lysosomal and autophagic stress.',
}

def _rx(s): return re.compile(s, re.I | re.S)
_DIRECT = _rx(r'\b(?:DILI|ATTIH|hepatotoxic\w*|hepatotox\w*|(?:liver|hepatic|hepatocellular)[ -]+(?:injur\w*|failure|damage|toxic\w*|necrosis|fibrosis|cirrhosis)|drug[ -]induced[ -]hepatitis|Hy.s[ -]law|cholestatic[ -](?:injury|hepatitis)|(?:clinical|toxic)[ -]+hepatitis|acute[ -]hepatitis|fulminant[ -]hepatitis)\b')
_DIRECT_CONTEXT = _rx(r'\b(?:serum|plasma|patients?|clinical|in vivo|in-vivo|histopatholog\w*|liver tissue)\b')
_DIRECT_LAB = _rx(r'\b(?:(?:ALT|AST|transaminases?|aminotransferases?|bilirubin)\b.{0,55}(?:elevat\w*|increas\w*|reduc\w*|normal|unchanged)|(?:elevat\w*|increas\w*|reduc\w*)\b.{0,55}(?:ALT|AST|transaminases?|aminotransferases?))\b')
_HEPATIC = _rx(r'\b(?:liver|hepatic|hepat\w*|Hep[ -]?G[ -]?2|Hep[ -]?aRG|Huh[ -]?7|L[ -]?O[ -]?2|L[ -]?0[ -]?2|THLE\w*|AML[ -]?12|Kupffer|biliar\w*|bile|cholesta\w*|D[ -]?GalN|DILI)\b')
_NONHEPATIC = _rx(r'\b(?:glioma|glioblastoma|astrocytes?|brain|neuronal|spinal cord|embryo\w*|embryoblast|lung|A549|KG[ -]?1A|leukemi\w*|leukaemi\w*|endothelial|cardiac|cardiomyocyt\w*|kidney|renal|neutrophils?|keratinocyt\w*|melanoma|breast|prostate|colon|HeLa|HEK[ -]?293)\b')
_EFFICACY = _rx(r'\b(?:anti[ -]?cancer|anti[ -]?tumou?r|tumou?r[ -](?:growth|regression)|cancer[ -]therapy|anticancer|anti[ -]?proliferative efficacy)\b')
_BILE = _rx(r'\b(?:bile|biliary|hepatobiliary|canalicular|BSEP|ABCB11|ABCB4|MDR3|NTCP|ASBT|SLC10A[12]|MRP[234]|OATP|cholesta\w*|taurocholat\w*|cholyl\w*)\b')
_BIOENERGY = _rx(r'\b(?:mitochondri\w*|mitotox\w*|bioactiv\w*|reactive metabolite\w*|nucleophile trapping|covalent bind\w*|adduct\w*|ATP|energy charge|cellular energy|oxphos|oxidative phosphorylation|oxygen consumption|respiration|electron transport|CYP time dependent|CYP.*inactivation|nitroso|hydroxylamine intermediate)\b')
_IMMUNE = _rx(r'\b(?:immune|immunotoxic\w*|immunomodulat\w*|cytokine\w*|chemokine\w*|eicosanoid\w*|Tregs?|T[ -]?cells?|iNKT|Kupffer|HLA|transcriptom\w*|proteom\w*|metabolom\w*|gene expression|mRNA|microarray|protein signaling|protein signalling|inflamm\w*|TNF|IL[ -]?\d+|interleukin\w*)\b')
_STRESS = _rx(r'\b(?:reactive species|ROS|redox|oxid\w* stress|antioxid\w*|lipid\w*|peroxid\w*|glutathione|GSH|GSSG|thiol\w*|Nrf2|lysosom\w*|autophag\w*|endoplasmic|ER stress|UPR|calcium|Ca2|ferropto\w*|cellular stress|phospholipidos\w*)\b')
_CELL = _rx(r'\b(?:viability|cell number|cell death|apopto\w*|necro\w*|caspas\w*|cellular morphology|membrane integrity|LDH|cellular injury|cytotoxic\w*|synthetic|secretory|secretion|albumin|urea|gluconeogen\w*|functional reserve|recovery|proliferation|MTT|trypan)\b')

def _surfaces(row):
    # Canonical unmapped fields preserve all raw study content in qualifying_conditions.
    endpoint = str(row.get('canonical_endpoint_name') or '').replace('_', ' ')
    context = str(row.get('canonical_assay_context') or '').replace('_', ' ')
    full = ' '.join(str(row.get(k) or '') for k in ('canonical_endpoint_name', 'canonical_measurement_text',
        'canonical_assay_context', 'canonical_species_context', 'qualifying_conditions', 'support_text')).replace('_', ' ')
    return endpoint, context, full

def direct_guard(row):
    _, _, text = _surfaces(row)
    return bool(_DIRECT.search(text) or (_DIRECT_CONTEXT.search(text) and _DIRECT_LAB.search(text))
                or ('human evidence basis=' in text and 'causal status=' in text))

def _result(level, reason, direct=False):
    return {'level': level, 'family_key': FAMILIES.get(level, ''), 'reason': reason, 'direct_signal': bool(direct)}

def classify(row, is_voter=False):
    """Classify canonical row; L1 membership is supplied by actual vote provenance."""
    direct = direct_guard(row)
    if not row.get('retrieval_eligible', True) or row.get('identity_review_reason'):
        return _result(0, row.get('identity_review_reason') or 'source_identity_ineligible', direct)
    if is_voter:
        return _result(1, 'actual_source_vote_membership', direct)
    if direct:
        return _result(2, 'direct_containment_in_full_record', True)
    endpoint, context, text = _surfaces(row)
    hepatic = bool(_HEPATIC.search(context) or _HEPATIC.search(endpoint))
    if _NONHEPATIC.search(context) and not hepatic:
        return _result(0, 'explicit_nonhepatic_assay_without_hepatic_endpoint')
    if _EFFICACY.search(text) and not _DIRECT.search(text):
        # A normal-hepatocyte safety comparator remains relevant even in oncology work.
        if not _rx(r'\b(?:normal|primary|non[ -]tumou?r|non[ -]cancer)\b').search(context):
            return _result(0, 'cancer_efficacy_without_liver_injury_outcome')
    # Prefer the measured endpoint over incidental co-reported pathways.
    for surface, qualifier in ((endpoint, 'endpoint'), (text, 'support')):
        for pattern, level in ((_BILE, 4), (_BIOENERGY, 5), (_IMMUNE, 6), (_STRESS, 7), (_CELL, 3)):
            if pattern.search(surface):
                return _result(level, 'semantic_' + qualifier + '_family')
    return _result(0, 'unresolved_endpoint_requires_semantic_review')
