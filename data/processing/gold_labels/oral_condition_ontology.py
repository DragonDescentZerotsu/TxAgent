"""Canonical external-condition ontology for context-conditioned gold labels.

The ontology is deliberately conservative: it recognizes concrete exposure
settings, patient states, formulations, and co-treatments.  Intrinsic PK
mechanisms and result/comparison statements never become condition groups.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


ONTOLOGY_VERSION = "bioavailability_external_conditions.v2"
NO_REPORTED_CONDITION = "no_reported_external_condition"


@dataclass(frozen=True)
class ConditionAtom:
    family: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.family}={self.value}"


@dataclass(frozen=True)
class ConditionClassification:
    scope: str
    signature: str | None
    atoms: tuple[ConditionAtom, ...]
    reason: str
    normalized_text: str
    mechanism_flags: tuple[str, ...] = ()


_RESULT_OR_COMPARISON = re.compile(
    r"\b(?:not|un)[- ]?(?:affected|influenced|altered)\b|"
    r"\b(?:independent|irrespective|regardless) of\b|"
    r"\bwith or without\b|\bversus\b|\bvs\.?\b|\bcompared with\b|"
    r"\bbefore and after\b|\bnot truly at steady[- ]state\b|"
    r"\b(?:unchanged|increase(?:d|s)?|decrease(?:d|s)?|reduce(?:d|s)?|"
    r"enhance(?:d|s)?|improve(?:d|s)?)\b[^;,.]{0,60}"
    r"\b(?:with|by|after|when|if|from|compared)\b|"
    r"\bdoes not change after\b|"
    r"\bno (?:significant )?(?:food|formulation|dose) effect\b|"
    r"\b(?:normal|partial) cyclosporin(?:e)? (?:absorption|malabsorption)\b",
    re.IGNORECASE,
)

_MECHANISM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "first_pass_or_clearance",
        re.compile(
            r"\bfirst[- ]pass\b|\bpresystemic\b|\bhepatic extraction\b|"
            r"\b(?:hepatic|metabolic) clearance\b|\bextensive metabolism\b|"
            r"\bhigh clearance\b",
            re.IGNORECASE,
        ),
    ),
    (
        "solubility_absorption_or_permeability",
        re.compile(
            r"\bpoor solubility\b|\binsoluble\b|\blow solubility\b|"
            r"\bpoor absorption\b|\bpoor permeability\b|\blow permeability\b|"
            r"\befflux\b|\bp[- ]?gp substrate\b",
            re.IGNORECASE,
        ),
    ),
)


def _pattern(text: str) -> re.Pattern[str]:
    return re.compile(text, re.IGNORECASE)


# Specific concepts only.  Broad families such as ``disease`` or ``DDI`` are
# never themselves condition values.
_ATOM_PATTERNS: tuple[tuple[ConditionAtom, re.Pattern[str]], ...] = (
    # Prandial state.  More specific meals precede generic fed state.
    (ConditionAtom("prandial_state", "fed_high_fat"), _pattern(r"\bhigh[- ]?fat (?:meal|food|diet)\b|\bfatty (?:meal|food)\b")),
    (ConditionAtom("prandial_state", "fed_low_fat"), _pattern(r"\blow[- ]?fat (?:meal|food|diet)\b")),
    (ConditionAtom("prandial_state", "fed_high_protein"), _pattern(r"\bhigh[- ]?protein meal\b")),
    (ConditionAtom("prandial_state", "fasted"), _pattern(r"\bfast(?:ed|ing)\b|\bovernight[- ]fast(?:ed|ing)?\b|\bempty stomach\b|\bwithout food(?: intake)?\b|\bunfed\b|\bstarved\b")),
    (ConditionAtom("prandial_state", "fed_unspecified"), _pattern(r"\bnon[- ]?fast(?:ed|ing)\b|\bfed(?: state| conditions?| subjects?| animals?)?\b|\bpost[- ]?prandial(?:ly)?\b|\bwith food\b|\bafter (?:a )?meals?\b|\bafter food(?: intake)?\b|\bpresence of food\b|\bfood ingestion\b|\bfood intake\b|\bbreakfast\b")),
    # Disease, surgery, and organ-function states.
    (ConditionAtom("disease", "cirrhosis"), _pattern(r"\b(?:liver|hepatic)?\s*cirrhosis\b|\bcirrhosis of the liver\b")),
    (ConditionAtom("disease", "cystic_fibrosis"), _pattern(r"\bcystic fibrosis\b|^CF$")),
    (ConditionAtom("disease", "congestive_heart_failure"), _pattern(r"\bcongestive heart failure\b|\bheart failure\b|\bCHF\b")),
    (ConditionAtom("renal_function", "normal"), _pattern(r"\bnormal renal function\b|\bnormal kidney function\b")),
    (ConditionAtom("disease", "renal_impairment"), _pattern(r"\b(?:chronic )?renal (?:failure|impairment|insufficiency|disease|dysfunction)\b|\bimpaired renal function\b|\bchronic kidney disease\b|\bkidney (?:failure|impairment|insufficiency|disease|dysfunction)\b|\bura?emic\b|\bura?emia\b")),
    (ConditionAtom("disease", "hepatic_impairment_noncirrhotic"), _pattern(r"\b(?:chronic )?(?:liver|hepatic) (?:failure|impairment|disease|dysfunction)\b")),
    (ConditionAtom("disease", "crohns_disease"), _pattern(r"\bcrohn'?s disease\b")),
    (ConditionAtom("disease", "hiv_aids"), _pattern(r"\bHIV\b|\bAIDS\b")),
    (ConditionAtom("physiologic_state", "obesity"), _pattern(r"\bobes(?:e|ity)\b")),
    (ConditionAtom("disease", "diabetes"), _pattern(r"\b(?:type [12] )?diabet(?:es|ic)\b|\bT2DM\b")),
    (ConditionAtom("disease", "hypertension"), _pattern(r"\bhypertension\b")),
    (ConditionAtom("disease", "malaria"), _pattern(r"\bmalaria\b")),
    (ConditionAtom("disease", "acute_coronary_syndrome"), _pattern(r"\bacute coronary syndromes?\b")),
    (ConditionAtom("disease", "rheumatoid_arthritis"), _pattern(r"\brheumatoid arthritis\b")),
    (ConditionAtom("disease", "inflammatory_bowel_disease"), _pattern(r"\binflammatory bowel disease\b|\bulcerative colitis\b")),
    (ConditionAtom("disease", "multiple_myeloma"), _pattern(r"\bmultiple myeloma\b")),
    (ConditionAtom("disease", "breast_cancer"), _pattern(r"\b(?:metastatic )?breast cancer\b")),
    (ConditionAtom("disease", "pancreatic_cancer"), _pattern(r"\bpancreatic cancer\b")),
    (ConditionAtom("disease", "cancer_unspecified"), _pattern(r"\badvanced cancer\b|\bcancer patients?\b|\bpatients? with cancer\b|^cancer$")),
    (ConditionAtom("population_state", "critically_ill"), _pattern(r"\bcritically ill\b")),
    (ConditionAtom("thyroid_state", "hyperthyroid"), _pattern(r"\bhyperthyroid(?:ism)?\b")),
    (ConditionAtom("thyroid_state", "hypothyroid"), _pattern(r"\bhypothyroid(?:ism)?\b")),
    (ConditionAtom("thyroid_state", "euthyroid"), _pattern(r"\beuthyroid\b")),
    (ConditionAtom("dialysis", "peritoneal"), _pattern(r"\bperitoneal dialysis\b")),
    (ConditionAtom("surgery", "gastrectomy"), _pattern(r"\bgastrectom\w*\b")),
    (ConditionAtom("surgery", "bariatric_surgery"), _pattern(r"\bbariatric\b|\bgastric bypass\b")),
    (ConditionAtom("surgery_status", "pre_roux_en_y_gastric_bypass"), _pattern(r"\bpre[- ]RYGBS\b|\bbefore roux[- ]en[- ]y gastric bypass\b")),
    (ConditionAtom("surgery_status", "post_roux_en_y_gastric_bypass"), _pattern(r"\bpost[- ]RYGBS\b|\bafter roux[- ]en[- ]y gastric bypass\b")),
    (ConditionAtom("surgery", "ileostomy"), _pattern(r"\bileostom(?:y|ies)\b")),
    (ConditionAtom("surgery", "short_bowel_or_jejunoileal_bypass"), _pattern(r"\bshort bowel\b|\bjejunoileal bypass\b")),
    (ConditionAtom("gi_state", "malabsorption"), _pattern(r"\bmalabsorb\w*\b")),
    (ConditionAtom("transplant", "liver"), _pattern(r"\bliver transplant\w*\b")),
    (ConditionAtom("transplant", "kidney"), _pattern(r"\b(?:kidney|renal) transplant\w*\b")),
    (ConditionAtom("transplant_status", "pre_kidney_transplant"), _pattern(r"\bbefore (?:kidney|renal) transplant\w*\b|\bpretransplantation\b[^;,.]{0,40}\b(?:kidney|renal)\b|\b(?:kidney|renal) transplant candidates?\b")),
    (ConditionAtom("transplant", "hematopoietic_stem_cell"), _pattern(r"\bHSCT\b|\bhematopoietic stem[- ]cell transplant\w*\b")),
    # Demographic and physiologic states.
    (ConditionAtom("age_group", "elderly"), _pattern(r"\belderly\b|\bgeriatric\b|\baged (?:6[5-9]|[7-9]\d)\b")),
    (ConditionAtom("physiologic_state", "pregnancy"), _pattern(r"\bpregnan\w*\b|\bgestation\b")),
    (ConditionAtom("physiologic_state", "nonpregnant"), _pattern(r"\bnon[- ]?pregnan\w*\b")),
    (ConditionAtom("age_group", "pediatric"), _pattern(r"\bpediatric\b|\bpaediatric\b|\bchildren\b|\binfants?\b|\badolescen\w*\b|\bneonat\w*\b")),
    (ConditionAtom("population_state", "healthy"), _pattern(r"\bhealthy (?:state|subjects?|volunteers?|adults?|(?:male|female) volunteers?)\b")),
    (ConditionAtom("age_group", "young_adult"), _pattern(r"\byoung (?:healthy )?(?:adult|man|men|woman|women|male|female)s?\b|\byoung,? (?:white|black|asian) (?:men|women)\b")),
    (ConditionAtom("race", "white"), _pattern(r"\bwhite (?:subjects?|volunteers?|adults?|men|women|males?|females?)\b")),
    (ConditionAtom("race", "black"), _pattern(r"\bblack (?:subjects?|volunteers?|adults?|men|women|males?|females?)\b")),
    (ConditionAtom("race", "asian"), _pattern(r"\basian (?:subjects?|volunteers?|adults?|men|women|males?|females?)\b")),
    (ConditionAtom("physiologic_state", "postmenopausal"), _pattern(r"\bpostmenopausal\b")),
    (ConditionAtom("sex", "female"), _pattern(r"\bfemale\b|\bwomen\b")),
    (ConditionAtom("sex", "male"), _pattern(r"\bmale\b|\bmen\b")),
    # Release profile, formulation, and solid state.
    (ConditionAtom("release_profile", "modified_release"), _pattern(r"\b(?:sustained|extended|controlled|prolonged|slow)[- ]release\b|\bER formulation\b")),
    (ConditionAtom("release_profile", "immediate_release"), _pattern(r"\bimmediate[- ]release\b|\bIR formulation\b")),
    (ConditionAtom("formulation", "enteric_coated"), _pattern(r"\benteric[- ]coated\b|\benteric coating\b")),
    (ConditionAtom("formulation", "microemulsion"), _pattern(r"\bmicroemulsion\b")),
    (ConditionAtom("formulation", "self_emulsifying"), _pattern(r"\bself[- ]emulsif\w*\b|\bSEDDS\b|\bSED\b")),
    (ConditionAtom("formulation", "lipid_based"), _pattern(r"\blipid[- ]based\b|\bliposom\w*\b")),
    (ConditionAtom("formulation", "nanoparticle"), _pattern(r"\bnano(?:particle|carrier|crystal|suspension|emulsion)s?\b")),
    (ConditionAtom("solid_state", "micronized"), _pattern(r"\bmicroni[sz]ed\b")),
    (ConditionAtom("solid_state", "amorphous"), _pattern(r"\bamorphous\b")),
    (ConditionAtom("solid_state", "crystalline"), _pattern(r"\bcrystal(?:line|lized|lised)?\b|\bpolymorph\b")),
    (ConditionAtom("dosage_form", "solution"), _pattern(r"\bsolution formulation\b|\boral solution\b")),
    (ConditionAtom("dosage_form", "suspension"), _pattern(r"\bsuspension formulation\b|\boral suspension\b")),
    (ConditionAtom("dosage_form", "tablet"), _pattern(r"\btablets?\b")),
    (ConditionAtom("dosage_form", "capsule"), _pattern(r"\bcapsules?\b")),
    # Molecular administered form.
    (ConditionAtom("salt", "hydrochloride"), _pattern(r"\bhydrochloride salt\b|\bHCl salt\b")),
    (ConditionAtom("salt", "sodium"), _pattern(r"\bsodium salt\b|\bsodium form\b")),
    (ConditionAtom("salt", "calcium"), _pattern(r"\bcalcium salt\b|\bcalcium form\b")),
    (ConditionAtom("salt", "mesylate"), _pattern(r"\bmesylate salt\b")),
    (ConditionAtom("salt", "phosphate"), _pattern(r"\bphosphate salt\b")),
    (ConditionAtom("salt", "sulfate"), _pattern(r"\bsulph?ate salt\b")),
    (ConditionAtom("salt", "tromethamine"), _pattern(r"\btromethamine salt\b")),
    (ConditionAtom("administered_form", "free_base"), _pattern(r"\bfree base\b")),
    (ConditionAtom("administered_form", "free_acid"), _pattern(r"\bfree acid\b")),
    (ConditionAtom("administered_form", "racemic"), _pattern(r"\bracemic\b")),
    (ConditionAtom("formulation", "buffered"), _pattern(r"\bbuffered\b")),
    (ConditionAtom("salt", "bromide"), _pattern(r"\bbromide salt\b")),
    (ConditionAtom("administered_form", "prodrug_unspecified"), _pattern(r"\bpro[- ]?drug\b")),
    # Specific co-treatments.  Absence/comparator statements are rejected below.
    (ConditionAtom("co_treatment", "rifampin"), _pattern(r"\brifamp(?:in|icin)\b")),
    (ConditionAtom("co_treatment", "grapefruit_juice"), _pattern(r"\bgrapefruit(?: juice)?\b")),
    (ConditionAtom("co_treatment", "cimetidine"), _pattern(r"\bcimetidine\b")),
    (ConditionAtom("co_treatment", "ritonavir"), _pattern(r"\britonavir\b")),
    (ConditionAtom("co_treatment", "omeprazole"), _pattern(r"\bomeprazole\b")),
    (ConditionAtom("co_treatment", "antacid"), _pattern(r"\bantacids?\b|\bMaalox\b|\bAmphojel\b")),
    (ConditionAtom("co_treatment", "cyclosporine"), _pattern(r"\bcyclosporin(?:e)?\b")),
    (ConditionAtom("co_treatment", "ketoconazole"), _pattern(r"\bketoconazole\b")),
    (ConditionAtom("co_treatment", "elacridar"), _pattern(r"\belacridar\b")),
    (ConditionAtom("co_treatment", "cedazuridine"), _pattern(r"\bcedazuridine\b")),
    (ConditionAtom("co_treatment", "st_johns_wort"), _pattern(r"\bSt\.? John'?s wort\b")),
    (ConditionAtom("co_treatment", "voriconazole"), _pattern(r"\bvoriconazole\b")),
    (ConditionAtom("co_treatment", "ginkgo_extract"), _pattern(r"\bGBE ?741\b|\bginkgo biloba\b")),
    (ConditionAtom("co_treatment", "pectin"), _pattern(r"\baddition of pectin\b|\bwith pectin\b")),
    # Regimen.
    (ConditionAtom("regimen", "single_dose"), _pattern(r"\bsingle[- ](?:oral )?doses?\b|\bfirst (?:oral )?dose\b")),
    (ConditionAtom("regimen", "steady_state"), _pattern(r"\bsteady[- ]state(?: conditions?)?\b")),
    (ConditionAtom("regimen", "repeated_dosing"), _pattern(r"\bmultiple doses?\b|\bmultiple dosing\b|\brepeated dosing\b|\brepeated administration\b|\bchronic dosing\b|\bchronic oral administration\b|\blong[- ]term (?:treatment|therapy)\b")),
    (ConditionAtom("regimen", "microdose"), _pattern(r"\bmicrodos(?:e|ing)\b")),
    # GI environment.
    (ConditionAtom("gi_environment", "achlorhydria"), _pattern(r"\bachlorhydria\b|\bhypochlorhydria\b")),
    (ConditionAtom("gi_environment", "normal_gastric_ph"), _pattern(r"\bnormal stomach pH\b|\bnormal gastric pH\b")),
)


_GENOTYPE = re.compile(
    r"\b(?P<gene>ABCG2|ABCB1|CYP2D6|CYP2C19|CYP3A5|UGT1A1)\s*"
    r"(?P<variant>(?:c\.)?\d+[A-Z]?>?[A-Z]*|\*\d+(?:/\*\d+)?)\s*"
    r"(?:genotype|allele)?\b",
    re.IGNORECASE,
)
_GENE_METABOLIZER = re.compile(
    r"\b(?P<gene>CYP2D6|CYP2C19|CYP3A4|CYP3A5|NAT2)\b[^;,.]{0,40}?"
    r"\b(?P<phenotype>poor|intermediate|extensive|ultra[- ]?rapid|slow) "
    r"(?:metaboli[sz]ers?|acetylators?)\b",
    re.IGNORECASE,
)
_UNSPECIFIED_METABOLIZER = re.compile(
    r"^(?P<phenotype>poor|intermediate|extensive|ultra[- ]?rapid) "
    r"metaboli[sz]ers?(?: \([A-Z]+s?\))?$",
    re.IGNORECASE,
)


def normalize_condition_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip(" .;,\t\n").lower()


def classify_external_condition(value: object) -> ConditionClassification:
    text = normalize_condition_text(value)
    if not text or text in {"nan", "none", "null", "n/a", "na"}:
        return ConditionClassification(
            scope="none_reported",
            signature=NO_REPORTED_CONDITION,
            atoms=(),
            reason="no_reported_external_condition",
            normalized_text=text,
        )

    mechanism_flags = tuple(
        name for name, pattern in _MECHANISM_PATTERNS if pattern.search(text)
    )
    if _RESULT_OR_COMPARISON.search(text):
        return ConditionClassification(
            scope="excluded",
            signature=None,
            atoms=(),
            reason="comparison_or_effect_statement_not_condition_arm",
            normalized_text=text,
            mechanism_flags=mechanism_flags,
        )

    atoms: set[ConditionAtom] = set()
    for atom, pattern in _ATOM_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if atom.family == "co_treatment" and _negated_or_comparator(text, match.start()):
            continue
        atoms.add(atom)

    for match in _GENOTYPE.finditer(text):
        gene = match.group("gene").lower()
        variant = re.sub(r"[^a-zA-Z0-9*]+", "", match.group("variant")).lower()
        atoms.add(ConditionAtom("genotype", f"{gene}_{variant}"))
    for match in _GENE_METABOLIZER.finditer(text):
        gene = match.group("gene").lower()
        phenotype = match.group("phenotype").lower().replace("-", "_").replace(" ", "_")
        atoms.add(ConditionAtom("metabolizer_phenotype", f"{gene}_{phenotype}"))
    unspecified_metabolizer = _UNSPECIFIED_METABOLIZER.match(text)
    if unspecified_metabolizer:
        phenotype = unspecified_metabolizer.group("phenotype").lower().replace("-", "_")
        atoms.add(ConditionAtom("metabolizer_phenotype", f"unspecified_{phenotype}"))

    atoms = _suppress_redundant_atoms(atoms)
    incompatible = _incompatible_family_values(atoms)
    if incompatible:
        return ConditionClassification(
            scope="excluded",
            signature=None,
            atoms=tuple(sorted(atoms, key=lambda atom: atom.key)),
            reason=f"incompatible_values_in_family:{incompatible}",
            normalized_text=text,
            mechanism_flags=mechanism_flags,
        )
    if not atoms:
        return ConditionClassification(
            scope="mechanism_only" if mechanism_flags else "unresolved",
            signature=None,
            atoms=(),
            reason=(
                "intrinsic_mechanism_not_external_condition"
                if mechanism_flags
                else "no_canonical_external_condition_match"
            ),
            normalized_text=text,
            mechanism_flags=mechanism_flags,
        )

    ordered = tuple(sorted(atoms, key=lambda atom: atom.key))
    return ConditionClassification(
        scope="external",
        signature="+".join(atom.key for atom in ordered),
        atoms=ordered,
        reason="canonical_external_condition",
        normalized_text=text,
        mechanism_flags=mechanism_flags,
    )


def _negated_or_comparator(text: str, start: int) -> bool:
    prefix = text[max(0, start - 35) : start]
    return bool(
        re.search(
            r"\b(?:without|before|no|not taking|placebo(?: for)?|absence of)\b[^;,.]*$",
            prefix,
            re.IGNORECASE,
        )
    )


def _suppress_redundant_atoms(atoms: set[ConditionAtom]) -> set[ConditionAtom]:
    output = set(atoms)
    keys = {atom.key for atom in output}
    if any(key.startswith("prandial_state=fed_") and key != "prandial_state=fed_unspecified" for key in keys):
        output.discard(ConditionAtom("prandial_state", "fed_unspecified"))
    if "disease=cirrhosis" in keys:
        output.discard(ConditionAtom("disease", "hepatic_impairment_noncirrhotic"))
    if any(atom.family == "surgery_status" for atom in output):
        output.discard(ConditionAtom("surgery", "bariatric_surgery"))
    if "transplant_status=pre_kidney_transplant" in keys:
        output.discard(ConditionAtom("transplant", "kidney"))
    if "physiologic_state=nonpregnant" in keys:
        output.discard(ConditionAtom("physiologic_state", "pregnancy"))
    if any(atom.family == "release_profile" for atom in output):
        output.discard(ConditionAtom("dosage_form", "tablet"))
        output.discard(ConditionAtom("dosage_form", "capsule"))
    return output


def _incompatible_family_values(atoms: set[ConditionAtom]) -> str | None:
    by_family: dict[str, set[str]] = {}
    for atom in atoms:
        by_family.setdefault(atom.family, set()).add(atom.value)
    for family, values in sorted(by_family.items()):
        if len(values) > 1 and family in {
            "prandial_state",
            "release_profile",
            "renal_function",
            "sex",
        }:
            return family
    return None
