"""Strict Ames gold voting with inclusive, record-level retrieval families.

Gold eligibility and retrieval eligibility are separate decisions. Predictions,
uncertain context, other genetic damage and modifier effects remain evidence.
Any passage carrying a bacterial/unspecified direct outcome is confined to L2
unless the unchanged gold contract accepts it as a candidate L1 voter.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

VERSION = "ames_bacterial_reverse_mutation.v1"
RETRIEVAL_VERSION = "ames_inclusive_retrieval.v3"
DIRECT = "Direct.ames"
NEAR = "Observed.nonvoter_ames"
MUTATION = "Mechanism.other_genetic_damage"
DAMAGE = "Mechanism.dna_damage_response"
MECHANISM = "Mechanism.genotoxicity_mechanisms"
LABELS = {"positive": 1, "strong_positive": 1, "negative": 0}
ACTIVATION = {"present": "present", "absent": "absent", "both": "both_reported"}

# Discovered in frozen raw_v1 (commit 03e4c7c...) base rows 1470 and 1473.
# Quarantine PMID + offending raw SMILES across files/paragraphs/aliases; an
# independently repaired structure is not caught. These are exclusions, never
# identity fixes. The builder must still validate every gold identity.
KNOWN_IDENTITY_QUARANTINE = {
    ("24275315", "CC(=O)SC[C@H](C)C(=O)O"),
    ("24275315", "NCc1ccccc1O"),
}
LOCAL_COMPOUND_CODE = re.compile(
    r"^(?:\d+[a-z]?|compound\s+(?:no\.?\s*)?\d+[a-z]?(?:\s.*)?)$",
    re.I,
)

PREDICTION = re.compile(
    r"\bin[ -]silico\b|\bqsar\b|\bpredicted\b|\bprediction\b|"
    r"\bcomputational\b|\bmolecular docking\b",
    re.I,
)
SPECIAL = re.compile(
    r"photo(?:mutagen|activat)|\bultraviolet\b|\buv[abc]?\b|irradiat|"
    r"host[- ]mediated|lactam test|\blac[zt]\b|\brifampicin\b|"
    r"antimutagen|anti[- ]mutagen|co[- ]mutagen|protect(?:ive|ant)|"
    r"co[- ]expos|co[- ]treat|combined (?:with|treatment)|in combination|"
    r"in the presence of.{0,45}(?:inhibitor|antioxidant)|"
    r"plant (?:homogenate|cell)|dietary|\bknockout\b|\bknockdown\b|"
    r"engineered|transfect|overexpress|photodegrad|chlorinat|ozonat",
    re.I,
)
RELATIONAL = re.compile(
    r"anti[- ]mutagen|protect(?:ant|ive|ion|ed)|\brescu|\bpretreat|"
    r"co[- ](?:treat|expos|incubat)|\bcotreat|\bcoexpos|\bcoincubat|"
    r"antioxidant|scaveng|in combination|combined treatment|"
    r"(?:inhibit|suppress|reduc|prevent)\w*.{0,65}(?:induced|mutagenicity)|"
    r"(?:addition|adding).{0,60}(?:reduc|inhibit|enhanc|potentiat)|"
    r"(?:inhibition|inhibitor).{0,65}(?:potentiat|enhanc)|"
    r"\b(?:probe|model|reporter) substrate|drug[- ]drug interaction|"
    r"used as (?:a |the )?(?:positive|negative|reference) control",
    re.I,
)
PRIMARY_CONTEXT_REVIEW = re.compile(
    r"\bdiets?\b|\bdietary\b|\bfed\b|\bfast(?:ed|ing)\b|"
    r"\bkidney\b|\brenal\b|\bmicrosom|\barachidonic\b|"
    r"\b(?:human|hamster|mouse|mice)\b.{0,45}(?:liver|s[- ]?9)|"
    r"(?:liver|s[- ]?9).{0,45}\b(?:human|hamster|mouse|mice)\b|"
    r"instead of|replac\w*.{0,30}(?:nadp|cofactor)|"
    r"(?:different|various|multiple|several).{0,35}(?:preparations|activation systems)|"
    r"\bversus\b|\bwhereas\b|\bin contrast\b|\bexcept\b|"
    r"only (?:in|with|without|at)|\b(?:bile|urine|nitrosat)|"
    r"\b(?:TA[ -]*\d+[A-Z0-9]*|WP2[A-Z0-9]*|S[ -]?9)[-‐‑–—]\s*only\b|"
    r"\b(?:ames ii|tami[xs]|fluctuation|microtiter|microtitre)\b|"
    r"\b(?:pkm\s*\d+|plasmid|derivative|deficient|proficient)\b",
    re.I,
)
PRIMARY_CITATION = re.compile(
    r"\b(?:cited|citing|references|RIFM)\b|\bref(?:erence)?s?\.?\s*[\[(]?\s*\d+|"
    r"\b(?:previous|prior|earlier)\s+(?:studies|study|reports?|work)\b|"
    r"\bpreviously\s+(?:reported|published|shown|tested)\b|"
    r"\bet\s+al\b\.?|\[\d+(?:\s*[-,;]\s*\d+)*\]",
    re.I,
)
# Mentioning both methods cannot establish one protocol-specific outcome. This
# intentionally also queues concordant comparisons until their arms are resolved.
PRIMARY_METHODS = (
    re.compile(r"\bpre[ -]?incubation\b", re.I),
    re.compile(r"\bplate[ -]incorporation\b", re.I),
)
PRIMARY_SUBJECT_UNCERTAINTY = re.compile(
    r"\b(?:probabl[ey]|possibly|presumably|suspected|inferred|postulated|hypothesized)\b|"
    r"\b(?:may|might|could)\s+(?:be|explain|contribute|account)\b",
    re.I,
)
PRIMARY_SPECIAL_EXPOSURE = re.compile(
    r"\bvapou?rs?\b|\bdesiccators?\b|\bsealed[ -]+chambers?\b|"
    r"\bgas[ -]phase\b|\bfumes?\b|\b(?:anaerobic|microaer(?:obic|ophilic))\b",
    re.I,
)
# Only detect a missing stereochemical representation. An @ marker is not proof
# of the correct stereoisomer; that remains the builder's identity-review job.
NAMED_STEREOCHEMISTRY = re.compile(
    r"\(\s*[+−-]\s*\)\s*[- ]?\s*[A-Z0-9]|" r"\(\s*\d*[RS](?:\s*,\s*\d*[RS])*\s*\)\s*-",
    re.I,
)
STANDARD_STRAINS = {
    "TA97",
    "TA97A",
    "TA98",
    "TA100",
    "TA102",
    "TA1535",
    "TA1537",
    "TA1538",
    "WP2UVRA",
    "WP2",
}
STRAIN_TOKEN = re.compile(r"\bTA[ -]*\d+[A-Z0-9]*\b|\bWP2(?:[ /-]*UVRA)?\b")
PANEL_WORDS = re.compile(
    r"\b(?:SALMONELLA(?:\s+ENTERICA)?(?:\s+SEROVAR)?(?:\s+TYPHIMURIUM)?|"
    r"S\.?\s*TYPHIMURIUM|ESCHERICHIA\s+COLI|E\.?\s*COLI|"
    r"TESTER|STRAINS?|AND|PLATE[ -]INCORPORATION|PRE[ -]?INCUBATION|"
    r"AMES|BACTERIAL|REVERSE|MUTATION|ASSAY|TEST)\b"
)


def text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "null", "none", "nan", "<na>"} else value


def context(row: Mapping[str, Any]) -> str:
    return " | ".join(
        text(row.get(k))
        for k in (
            "test_system",
            "biological_test_system",
            "biological_system",
            "qualifying_conditions",
            "exposure_and_mechanistic_conditions",
            "metabolic_activation",
            "metabolic_activation_status",
            "metabolic_activation_presence",
            "metabolic_activation_system",
            "assay_version",
            "assay_method_and_endpoint",
            "endpoint_subtype",
            "cytotoxicity_status",
            "extra_details",
            "support_text",
        )
    )


@dataclass(frozen=True)
class Decision:
    group: str
    reason: str
    label: int | None = None
    condition_atoms: tuple[str, ...] = ()


def strain_panel(value: Any) -> tuple[str, ...]:
    """Consume an explicit panel completely; never discard unknown genotypes.

    pKM variants remain pending review even when fully spelled out. Count-only
    panels, abbreviated numbers, responding-strain prose and unknown organisms
    are deliberately not reconstructed.
    """
    value = text(value).upper()
    matches = STRAIN_TOKEN.findall(value)
    strains = {re.sub(r"[ /-]", "", match) for match in matches}
    residual = PANEL_WORDS.sub("", STRAIN_TOKEN.sub("", value))
    if (
        not strains
        or not strains.issubset(STANDARD_STRAINS)
        or re.search(r"[^\s,;/&+().:-]", residual)
    ):
        return ()
    return tuple(sorted(strains))


def gold_candidate(source: str, row: Mapping[str, Any]) -> Decision:
    """Frozen v1 direct-vote eligibility; retrieval may retain rejected records."""
    if not text(row.get("SMILES")) or not text(row.get("support_text")):
        return Decision("", "missing_structure_or_support")
    identity_key = (text(row.get("pmid")), text(row.get("SMILES")))
    if identity_key in KNOWN_IDENTITY_QUARANTINE:
        return Decision("", "known_source_identity_mismatch")
    if source == "ames_base" and LOCAL_COMPOUND_CODE.fullmatch(
        text(row.get("molecule_name"))
    ):
        return Decision("", "unresolved_local_compound_code")
    family = text(row.get("assay_family"))
    if source == "ames_base" and family == "bacterial_reverse_mutation":
        if text(row.get("experimental_context")) not in {"in_vitro", "in vitro"}:
            return Decision("", "outside_in_vitro_bacterial_target")
        if text(row.get("chemical_entity_type")) != "defined_chemical":
            return Decision("", "unresolved_chemical_entity")
        body = context(row)
        if SPECIAL.search(body):
            return Decision("", "special_context_requires_semantic_review")
        if RELATIONAL.search(body):
            return Decision("", "effect_attribution_requires_relational_review")
        if PRIMARY_CONTEXT_REVIEW.search(body):
            return Decision("", "activation_or_experiment_arms_require_review")
        if PRIMARY_CITATION.search(body):
            return Decision(
                "", "cited_or_previous_experiment_requires_provenance_review"
            )
        if all(pattern.search(body) for pattern in PRIMARY_METHODS):
            return Decision("", "assay_method_comparison_requires_review")
        if PRIMARY_SUBJECT_UNCERTAINTY.search(body):
            return Decision("", "inferred_subject_or_outcome_requires_review")
        if "@" not in text(row.get("SMILES")) and NAMED_STEREOCHEMISTRY.search(
            text(row.get("molecule_name")) + " | " + body
        ):
            return Decision("", "named_stereochemistry_missing_from_structure")
        if PRIMARY_SPECIAL_EXPOSURE.search(body):
            return Decision("", "special_exposure_requires_condition_review")
        # No prediction or mixed record may escape through a NEAR fallback for
        # missing panel/activation. Shared hard scope gates still run first.
        if PREDICTION.search(body):
            return Decision("", "prediction_or_mixed_experiment_requires_review")
        panel = strain_panel(row.get("test_system"))
        if not panel:
            return Decision(NEAR, "unresolved_or_nonstandard_strain_panel")
        activation = ACTIVATION.get(text(row.get("metabolic_activation")))
        if activation is None:
            return Decision(NEAR, "unresolved_metabolic_activation")
        basis = text(row.get("evidence_basis"))
        if basis != "current_study_experiment":
            return Decision(NEAR, "nonprimary_or_reference_use")
        if row.get("needs_more_context") is not False:
            return Decision(NEAR, "source_requests_more_context")
        if text(row.get("qualifying_conditions")):
            return Decision(NEAR, "unreviewed_qualifying_conditions")
        if text(row.get("mutagenicity_result")) == "weak_positive":
            return Decision(NEAR, "weak_positive_requires_outcome_review")
        label = LABELS.get(text(row.get("mutagenicity_result")))
        if label is None:
            return Decision(NEAR, "nonbinary_source_result")
        atoms = tuple(
            sorted(
                (
                    f"metabolic_activation={activation}",
                    "strain_panel=" + ",".join(panel),
                )
            )
        )
        # The source-positive enum means panel-any-positive, not every strain
        # positive. Keep one summary even for a responding subset / both S9 arms;
        # never synthesize strain-level labels from this record.
        return Decision(DIRECT, "explicit_primary_bacterial_outcome", label, atoms)
    return Decision("", "not_a_direct_gold_candidate")


# Scan the full supplied content before indirect assignment. Broad bacterial
# markers intentionally route mixed/background mentions to L2 rather than expose
# incidental outcomes at a later level. Bacterial DDR alone is not an Ames call.
BACTERIAL_CONTEXT = re.compile(
    r"(?<![a-z])ames(?:[_ -]?(?:test|assay|prediction|mutagenicity|toxicity))?(?![a-z])|"
    r"\bsalmonella\b|\bs\.?\s*typhi(?:murium)?\b|"
    r"\bTA[- ‐‑–—]?\d{2,4}[a-z0-9]*\b|\bWP\s*2|"
    r"\brevers(?:e|ion)[_ -]+mutat|\brevertant|"
    r"bacteri(?:a|al)[_ -]+mutagen",
    re.I,
)
GENETIC_ENDPOINT = re.compile(
    r"mutat|chromosom|chromatid|micronucl|malsegreg|aneuploid|polyploid|"
    r"recombin|gene.?conversion|heritable|dominant.?lethal|specific.?locus|"
    r"genetic|genotox|clastogen|sequence.level.gene|micronuclear|mouse.lymphoma",
    re.I,
)
DAMAGE_ENDPOINT = re.compile(
    r"dna|comet|strand|adduct|crosslink|oxidized.base|oxidative.base|"
    r"damage|repair|h2ax|ddr|p53|gadd45|postlabell?ing|abasic|toxtracker|multiflow",
    re.I,
)
MECHANISTIC_ENDPOINT = re.compile(
    r"metabol|detox|conjugat|reactiv|covalent|electrophil|nucleophil|"
    r"oxid|antioxid|redox|\bros\b|thiol|repair|dna|topoisom|replicat|"
    r"spindle|kinetochore|segregat|tubulin|microtubul|genotox|mechanis",
    re.I,
)
GENERIC_MUTAGENICITY = {"unspecified_mutagenicity", "multiple_mutation_assays"}
BACTERIAL_TOKEN = re.compile(r"\bbacteri(?:al|a|um)\b", re.I)
MUTATION_TOKEN = re.compile(
    r"\b(?:mutat|mutagen|revertant|reversion\b|reverse.mutat)", re.I
)
ECOLI_TOKEN = re.compile(r"\bescherichia\b|\be\.?\s*coli\b", re.I)
REVERSION_TOKEN = re.compile(r"\breverse[\s_-]+mutat|\breversions?\b|\brevertant", re.I)


def has_direct_context(value: str) -> bool:
    value = value.replace("_", " ")
    return bool(
        BACTERIAL_CONTEXT.search(value)
        or (BACTERIAL_TOKEN.search(value) and MUTATION_TOKEN.search(value))
        or (ECOLI_TOKEN.search(value) and REVERSION_TOKEN.search(value))
    )


def retrieval_text(row: Mapping[str, Any]) -> str:
    """All source-supplied semantic fields; never audit reasons/source filenames."""
    return " | ".join(
        text(value)
        for key, value in row.items()
        if key
        not in {
            "SMILES",
            "pmid",
            "extraction_id",
            "paragraph_idx",
            "confidence",
            "molecule_name",
        }
    ).replace("_", " ")


# Concrete assay/readout fields take precedence over broad acquisition categories.
# In particular, protein/GSH adducts are not DNA lesions, and spindle machinery
# is not a measured chromosome-number change. No rule below uses the run name.
FIXED_READOUT = re.compile(
    r"micronucl|micromucl|chromatid.exchange|chromosom\w*.(?:aberr|abnorm|count)|"
    r"(?:structural|numerical).chromosom|aneuploid|polyploid|"
    r"(?:gene|somatic|heritable|forward|reversion|reverse|induced).mutat|"
    r"mutation.(?:frequency|spectrum)|mutant.frequency|"
    r"specific.locus|dominant.lethal|recombin\w*.conversion|"
    r"sequence.level.gene|mouse.lymphoma|drosophila.mutat|"
    r"heritable.lethal|sex.linked.recessive.lethal",
    re.I,
)
DNA_READOUT = re.compile(
    r"comet|strand.break|alkali.labile|dna.crosslink|dna.protein.crosslink|"
    r"dna.adduct|adduct.{0,35}(?:dna|deoxy|nucleotid)|"
    r"(?:dna|deoxy|guanine).{0,35}(?:adduct|alkylat)|"
    r"(?:dna|base).{0,25}(?:damage|lesion|oxidat|repair)|oxidized.(?:dna.)?base|"
    r"abasic|ap.site|apyrimidinic|h2ax|53bp1|gadd45|ddr|"
    r"repair.(?:activity|capacity|kinetics|deficient|intermediate|synthesis)|"
    r"premutagenic.damage|damage.or.repair.product|"
    r"dna.(?:integrity|fragmentation|uracil)|"
    r"umu.{0,15}(?:gene|induct|express)|sos.response|postlabell?ing",
    re.I,
)


def indirect_family(row: Mapping[str, Any]) -> Decision:
    """Source-independent endpoint assignment; uncertainty remains on the card.

    Mixed-passage exceptions and false bacterial hits require payload-pinned
    reading decisions in reviewed_source, rather than unsafe global negation.
    """
    native = " | ".join(
        text(row.get(k)) for k in ("assay_family", "endpoint_class")
    ).replace("_", " ")
    # A damage-specific endpoint class resolves broad terminology in method
    # prose (e.g. comet called 'clastogenicity', PAR measured inside micronuclei).
    # Mixed records with an additional fixed outcome still have explicit reviews.
    if FIXED_READOUT.search(native):
        return Decision(MUTATION, "source_independent_fixed_genetic_readout")
    endpoint_class = text(row.get("endpoint_class"))
    if endpoint_class.endswith("_dna_base") or DNA_READOUT.search(
        endpoint_class.replace("_", " ")
    ):
        return Decision(DAMAGE, "source_independent_dna_damage_endpoint_class")
    assay = " | ".join(
        text(row.get(k))
        for k in (
            "assay_family",
            "endpoint_class",
            "assay_method_and_endpoint",
            "assay_version",
            "endpoint_subtype",
        )
    ).replace("_", " ")
    # A repair reporter can measure HR capacity without measuring a heritable
    # recombination event; a generic mutation word is not sufficient here.
    if FIXED_READOUT.search(assay):
        return Decision(MUTATION, "source_independent_fixed_genetic_readout")
    if DNA_READOUT.search(assay):
        return Decision(DAMAGE, "source_independent_dna_damage_or_response_readout")
    category = text(row.get("mechanism_category")).replace("_", " ")
    if MECHANISTIC_ENDPOINT.search(assay + " | " + category):
        return Decision(MECHANISM, "source_independent_mechanistic_readout")
    if GENETIC_ENDPOINT.search(assay):
        return Decision(MUTATION, "source_independent_other_genetic_endpoint")
    # Some incomplete records supply only a native endpoint/category. Their
    # meaning can support retrieval, but never supplies a missing outcome.
    if DAMAGE_ENDPOINT.search(assay):
        return Decision(DAMAGE, "source_independent_damage_endpoint")
    return Decision("", "unresolved_endpoint_without_family_support")


def classify(source: str, row: Mapping[str, Any]) -> Decision:
    gold = gold_candidate(source, row)
    if gold.label is not None:
        return gold
    if gold.reason in {
        "missing_structure_or_support",
        "known_source_identity_mismatch",
        "unresolved_local_compound_code",
        "named_stereochemistry_missing_from_structure",
        "unresolved_chemical_entity",
    }:
        return gold
    body = retrieval_text(row)
    family = text(row.get("assay_family"))
    if (
        has_direct_context(body)
        or family in GENERIC_MUTAGENICITY
        or family == "bacterial_reverse_mutation"
    ):
        return Decision(NEAR, "nonvoter_direct_or_mixed_bacterial_context")
    if not family and text(row.get("endpoint_class")) == "mixed_fixed_damage_endpoint":
        # This coarse enum does not identify a DNA lesion assay. A named
        # nonbacterial host supports L3; an unspecified system stays near-direct.
        system = " | ".join(
            text(row.get(k))
            for k in (
                "test_system",
                "biological_test_system",
                "biological_system",
            )
        )
        if re.search(
            r"homo sapiens|\bhuman\b|mammal|\b(?:mouse|mice|rat|rats|rodent|hamster|"
            r"CHO|CHL|TK6|V79)\b|drosophila|saccharomyces|\byeast\b|\bplants?\b",
            system,
            re.I,
        ):
            return Decision(
                MUTATION, "nonbacterial_fixed_outcome_without_specific_assay"
            )
        return Decision(NEAR, "unspecified_mutagenicity_without_specific_assay")
    # Endpoint families describe the evidence, not a binary outcome or its
    # certainty. Retain the supplied prediction/role/context in the raw cards.
    return indirect_family(row)
