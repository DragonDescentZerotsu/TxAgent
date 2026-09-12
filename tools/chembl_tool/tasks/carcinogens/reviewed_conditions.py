"""Five broad organism groups; original exposure/model details stay in records."""
import re

VERSION = 'carcinogens_five_organism_groups.v1'
GROUPS = ('rodent', 'human', 'dog', 'monkey', 'rabbit')
_PATTERNS = {
    'rodent': r'\b(?:rats?|mice|mouse|hamsters?|rodents?|rodentia|rattus|mus musculus|mesocricetus|cricetulus|guinea pigs?|cavia porcellus|gerbils?|meriones|peromyscus)\b',
    'human': r'\b(?:humans?|patients?|workers?|volunteers?|women|woman|men|man|people|persons?|children|infants?|homo sapiens)\b',
    'dog': r'\b(?:dogs?|canine|beagles?|canis (?:lupus|familiaris))\b',
    'monkey': r'\b(?:monkeys?|macaques?|rhesus|cynomolgus|macaca|marmosets?|baboons?)\b',
    'rabbit': r'\b(?:rabbits?|oryctolagus)\b',
    'outside_five_groups': r'\b(?:fish|trout|zebrafish|medaka|danio|oryzias|oncorhynchus|chicken|chickens|quail|ducks?|pigs?|swine|porcine|sheep|ovine|goats?|cattle|bovine|drosophila|xenopus|frogs?|ferrets?|cats?|feline|chimpanzees?|apes?)\b',
}
_CELL = re.compile(r'\b(?:cells?|cell lines?|in vitro|cultured|organoids?)\b', re.I)
_PEOPLE = re.compile(r'\b(?:patients?|workers?|volunteers?|women|men|people|children|infants?|beneficiaries|cohort|population)\b', re.I)
_OUTCOME = re.compile(r'carcinogen|tumou?r|neoplas|cancer|carcinoma|adenoma|leuk[ae]mia|lymphoma', re.I)


def mentioned_groups(text):
    # Guinea pigs are rodents; do not independently interpret 'pig' here.
    text = re.sub(r'guinea[ -]pigs?', 'cavia porcellus', str(text or ''), flags=re.I)
    text = re.sub(r'\bnon[ -]?human\b', 'nonhuman', text, flags=re.I)
    groups = {group for group, pattern in _PATTERNS.items() if re.search(pattern, text, re.I)}
    if re.search(r'\bprimates?\b', text, re.I) and not groups.intersection({'monkey','outside_five_groups'}):
        groups.add('unresolved_primate')
    return groups


def classify_condition(raw):
    """Resolve focal model first. Mixed/unknown claims are never duplicated."""
    scope = str(raw.get('evidence_scope') or '').casefold()
    basis = str(raw.get('evidence_basis') or '').casefold()
    model = str(raw.get('evidence_population_or_model') or '')
    groups = mentioned_groups(model)
    result = {'group': None, 'basis': '', 'quote': model, 'detected_groups': sorted(groups)}
    if scope == 'in_vitro' or basis.startswith('in_vitro'):
        return {**result, 'basis': 'in_vitro_not_organism_outcome'}
    if scope == 'human':
        groups.add('human')
    if groups:
        if len(groups) > 1:
            return {**result, 'basis': 'mixed_organism_groups', 'detected_groups': sorted(groups)}
        group = next(iter(groups))
        if group == 'unresolved_primate':
            return {**result, 'basis': 'organism_unresolved'}
        if group not in GROUPS:
            return {**result, 'basis': 'outside_five_groups'}
        if _CELL.search(model) and not (scope == 'human' and _PEOPLE.search(model)):
            return {**result, 'basis': 'cell_model_not_organism_outcome'}
        return {**result, 'group': group, 'basis': 'explicit_model_or_human_scope', 'detected_groups': [group]}
    # Recover a missing field only from a uniquely stated, focal outcome clause.
    # A second group anywhere in the support leaves attribution unresolved.
    support = str(raw.get('support_text') or '')
    all_groups = mentioned_groups(support)
    if len(all_groups) == 1 and not _CELL.search(support):
        group = next(iter(all_groups))
        if group in GROUPS and not (scope == 'experimental_animal' and group == 'human'):
            for clause in re.split(r'(?<=[.!?;])\s+', support):
                # Human exposure or relevance alone does not locate an outcome
                # in humans. Require an explicitly attributed human claim.
                if group == 'human' and not re.search(
                    r'(?:carcinogen(?:s|ic|icity)?|tumou?rs?|cancers?|carcinomas?|leuk[ae]mia|lymphoma)\s+'
                    r'(?:(?:risk|effects?|incidence|observed|reported|found|were|was|are|is)\s+){0,4}'
                    r'(?:in|to|among)\s+(?:the\s+)?(?:humans?|man|people|patients?|workers?|women|men|children)\b', clause, re.I
                ):continue
                if mentioned_groups(clause) == {group} and _OUTCOME.search(clause) and re.search(
                    r'\b(?:in|among|to|of|exposed|treated|administered|fed|induced|developed|observed|showed)\b', clause, re.I
                ):
                    return {**result, 'group': group, 'basis': 'unique_outcome_clause', 'quote': clause,
                            'detected_groups': [group]}
    return {**result, 'basis': 'organism_unresolved', 'detected_groups': sorted(all_groups)}
