#!/usr/bin/env python3
"""Deterministically reconcile the six original auxiliary-value mappings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[9]
DATA_ROOT = REPO_ROOT / "data/raw/starling/bioavailability_ma"
MAPPING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MAPPING_ROOT / "globally_reconciled_auxiliary_value_mapping.json"
MAPPING_VERSION = "starling_auxiliary.globally_reconciled.v2"
ORAL_EXTRACTION_PATH = MAPPING_ROOT / "oral_study_context_extractions.json"
ORAL_SOURCE_ID = "oral_exposure"
ORAL_OUTPUT_FIELDS = {
    "global_species_context": "canonical_species_context",
    "global_biological_matrix": "canonical_biological_matrix",
}

NULL_LIKE = {
    "",
    "-",
    "n/a",
    "na",
    "nan",
    "none",
    "not specified",
    "not stated",
    "null",
    "unknown",
    "unspecified",
}
FORBIDDEN_OUTPUT = re.compile(
    r"(?<![a-z0-9])(?:unknown|unmapped|null|none|n/?a|not stated|not specified|unspecified)(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OutputSpec:
    source_columns: tuple[str, ...]
    output_name: str
    current_input_column: str
    current_output_column: str


@dataclass(frozen=True)
class SourceSpec:
    parquet_path: Path
    outputs: tuple[OutputSpec, ...]


SOURCE_SPECS = {
    "fa": SourceSpec(
        parquet_path=DATA_ROOT / "Fa/extractions.parquet",
        outputs=(
            OutputSpec(("assay_system",), "global_context", "assay_system", "canonical_context"),
            OutputSpec(
                ("biological_context", "assay_system"),
                "global_species_context",
                "biological_context",
                "canonical_species",
            ),
        ),
    ),
    "fg": SourceSpec(
        parquet_path=DATA_ROOT / "Fg/extractions.parquet",
        outputs=(
            OutputSpec(("assay_system",), "global_context", "assay_system", "canonical_context"),
            OutputSpec(
                ("assay_system",),
                "global_species_context",
                "assay_system",
                "canonical_species",
            ),
        ),
    ),
    "fh": SourceSpec(
        parquet_path=DATA_ROOT / "Fh/extractions.parquet",
        outputs=(
            OutputSpec(("assay_system",), "global_context", "assay_system", "canonical_context"),
            OutputSpec(
                ("species", "assay_system"),
                "global_species_context",
                "species",
                "canonical_species",
            ),
        ),
    ),
}


# Only reviewed aliases belong here. Labels absent from this registry pass through
# unchanged rather than being merged by an unsafe fuzzy rule.
GLOBAL_CONTEXT_ALIASES = {
    # Formatting-only variants.
    "2_4_a1": "2/4/a1",
    "caco_2 mdck": "caco_2/mdck",
    "caco_2 and mdck": "caco_2/mdck",
    "caco_2_mdck": "caco_2/mdck",
    "caco_2_uptake": "caco_2",
    "cpt_m1": "cpt-m1",
    "cpt_p1": "cpt-p1",
    "d_p system": "d/p system",
    "dissolution-permeation": "dissolution/permeation",
    "dissolution_permeation": "dissolution/permeation",
    "everted_gut_sac": "everted gut sac",
    "gastrointestinal_absorption": "gastrointestinal absorption",
    "ht29_mtx": "ht29-mtx",
    "iec_18": "iec-18",
    "in_situ_intestinal": "in situ intestine",
    "in_vivo_absorption": "in vivo absorption",
    "intestinal_absorption": "intestinal absorption",
    "intestinal_loop": "intestinal loop",
    "intestinal_perfusion": "intestinal perfusion",
    "intrinsic_dissolution": "intrinsic dissolution",
    "pet_imaging": "pet imaging",
    "pharmacokinetic_study": "pharmacokinetics",
    "stomach_absorption": "stomach absorption",
    # Reviewed lexical aliases.
    "clinical ddi study": "clinical ddi",
    "clinical interaction study": "clinical ddi",
    "clinical interaction": "clinical ddi",
    "flow-through dissolution": "usp_4 dissolution",
    "gi contents": "gastrointestinal contents",
    "gi stability": "gastrointestinal stability",
    "gastric intestinal stability": "gastrointestinal stability",
    "in vitro-in vivo scaling": "ivive",
    "in vitro/in vivo scaling": "ivive",
    "intestinal recirculation": "recirculating intestinal perfusion",
    "ivive scaling": "ivive",
    "mass balance": "mass balance study",
    "mass-balance study": "mass balance study",
    "oral administration": "oral dosing",
    "oral pharmacokinetic study": "oral pharmacokinetics",
    "oral pk": "oral pharmacokinetics",
    "paddle dissolution": "usp_2 dissolution",
    "pbpk simulation": "pbpk model",
    "poppk analysis": "population pk model",
    "population pharmacokinetic model": "population pk model",
    "population pk": "population pk model",
    "reciprocating cylinder": "usp_3 dissolution",
    "well-stirred liver model": "well-stirred model",
    "allometry": "allometric scaling",
    "cell-based assay": "cell assay",
    "cell lines": "cell line",
    "clinical/in vitro study": "in vitro and clinical study",
    "cyp enzyme": "cytochrome p450",
    "estimated": "estimate",
    "hepatocytes/liver microsomes": "hepatocytes and liver microsomes",
    "hepatocytes/recombinant enzyme": "hepatocytes and recombinant enzyme",
    "in vitro and in vivo study": "in vitro and in vivo",
    "in vitro/in vivo study": "in vitro and in vivo",
    "in vivo and in vitro study": "in vitro and in vivo",
    "in vivo/in vitro study": "in vitro and in vivo",
    "intraportal dosing": "intraportal administration",
    "isolated enzyme": "purified enzyme",
    "liver and intestinal microsomes": "intestinal and liver microsomes",
    "liver microsomes and hepatocytes": "hepatocytes and liver microsomes",
    "liver microsomes and liver cytosol": "liver microsomes and cytosol",
    "liver microsomes and s9": "liver microsomes and s9 fraction",
    "liver microsomes/recombinant enzyme": "liver microsomes and recombinant enzyme",
    "microsomal fraction": "microsomes",
    "microsomes and hepatocytes": "hepatocytes and microsomes",
    "microsomes and liver slices": "liver slices and microsomes",
    "mitochondrial fraction": "mitochondria",
    "mitochondrial preparations": "mitochondria",
    "p450 enzyme": "cytochrome p450",
    "p450 enzymes": "cytochrome p450",
    "pk model": "pharmacokinetic model",
    "recombinant enzyme/liver microsomes": "liver microsomes and recombinant enzyme",
    "reconstituted enzyme": "reconstituted system",
    "s9 and hepatocytes": "hepatocytes and s9",
    "s9 fraction and liver microsomes": "liver microsomes and s9 fraction",
    "substrate loss": "substrate loss method",
    "cytosol/microsomes": "microsomes and cytosol",
    "abcb1 atpase": "atpase assay",
    "caco_2 and 2/4/a1": "caco_2/2_4_a1",
    "caco_2 and cpt-p1": "caco_2/cpt-p1",
    "caco_2 and intestinal perfusion": "caco_2/intestinal_perfusion",
    "caco_2 and ussing chamber": "caco_2/ussing chamber",
    "caco_2 cpt-p1": "caco_2/cpt-p1",
    "caco_2 uptake": "caco_2",
    "caco_2_cpt_p1": "caco_2/cpt-p1",
    "caco_2_hepg2": "caco_2/hepg2",
    "caco_2_ht29": "caco_2/ht29",
    "caco_2/ht29-mtx coculture": "caco_2/ht29-mtx",
    "cell monolayer transport": "cell monolayer",
    "closed-loop intestinal perfusion": "intestinal perfusion",
    "co-administration study": "coadministration study",
    "cyp3a4 caco_2": "caco_2",
    "enzyme incubations": "enzyme incubation",
    "everted intestine": "everted gut sac",
    "ex vivo intestinal tissue": "intestinal tissue",
    "flow cytometry assay": "flow cytometry",
    "franz diffusion cell": "franz cell",
    "hipsc iec": "hipsc-iec",
    "hipsc_iec": "hipsc-iec",
    "in situ intestinal": "in situ intestine",
    "in vitro/in vivo": "in vitro and in vivo",
    "in_vitro_intestinal": "in vitro intestine",
    "intestinal biopsy study": "intestinal biopsy",
    "intestinal biopsies": "intestinal biopsy",
    "intestinal closed loop": "intestinal loop",
    "intestinal content": "intestinal contents",
    "intestinal flora": "intestinal microflora",
    "intestinal fraction": "intestinal fractions",
    "intestinal homogenate assay": "intestinal homogenate",
    "intestinal membrane assay": "intestinal membrane",
    "intestinal organoids": "intestinal organoid",
    "intestinal sac": "gut sac",
    "intestinal_sac": "gut sac",
    "intestinal_segment": "intestinal segment",
    "intestinal studies": "intestinal study",
    "intestinal tissue assay": "intestinal tissue",
    "intrinsic clearance assay": "intrinsic clearance",
    "iv pharmacokinetics": "intravenous pharmacokinetics",
    "iv/oral pharmacokinetics": "oral + intravenous pharmacokinetics",
    "jejunal loop": "intestinal loop",
    "jejunal perfusion": "intestinal perfusion",
    "knockout mice": "knockout mouse",
    "knockout mouse model": "knockout mouse",
    "llc-pk1 and mdck": "llc-pk1/mdck",
    "llc-pk1_mdr1": "llc-pk1",
    "ls180": "ls-180",
    "mdck caco_2": "caco_2/mdck",
    "mdck uptake": "mdck",
    "mdck/caco_2": "caco_2/mdck",
    "mdck_bcrp": "mdck",
    "mdck_hpept1": "mdck",
    "mdck_mdr1": "mdck",
    "mdck_mdr1/caco_2": "caco_2/mdck",
    "mdck_mdr1_mrp2": "mdck",
    "mdck_mrp1": "mdck",
    "mdck_mrp2": "mdck",
    "mdck_mrp3": "mdck",
    "mdck_pept1": "mdck",
    "mdr1-transfected cells": "mdr1 transfected cells",
    "oral intravenous pharmacokinetics": "oral + intravenous pharmacokinetics",
    "oral iv pharmacokinetics": "oral + intravenous pharmacokinetics",
    "oral vs intravenous pharmacokinetics": "oral + intravenous pharmacokinetics",
    "organ chip": "organ-on-chip",
    "organ-on-a-chip": "organ-on-chip",
    "perfused intestine": "intestinal perfusion",
    "perfused intestine-liver": "intestine-liver perfusion",
    "pharmacokinetic study": "pharmacokinetics",
    "polarized monolayers": "polarized monolayer",
    "recirculating intestinal perfusion": "intestinal perfusion",
    "single-pass intestinal perfusion": "intestinal perfusion",
    "t-84": "t84",
    "transwell assay": "transwell",
    "transport studies": "transport study",
    # Cell/platform naming variants.
    "caco_2 co-culture": "caco_2 coculture",
    "caco_2 co culture": "caco_2 coculture",
    "caco_2_ht29_mtx": "caco_2/ht29-mtx",
}


GLOBAL_SPECIES_ALIASES = {
    "bovine": "cattle",
    "cow": "cattle",
    "spodoptera frugiperda": "fall armyworm",
    "common brush-tailed possum": "brushtail possum",
    "common brushtail possum": "brushtail possum",
    "brush-tailed possum": "brushtail possum",
    "trichoplusia ni": "cabbage looper",
}


HOST_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:xenopus\s+laevis|x\.?\s*laevis)\b", re.I), "african clawed frog"),
    (re.compile(r"\bxenopus\b(?!\s+laevis)|\bfrog\s+oocytes?\b", re.I), "frog"),
    (re.compile(r"\b(?:insect\s+cells?|baculovirus)\b", re.I), "insect"),
    (re.compile(r"\bspodoptera(?:\s+frugiperda)?\b", re.I), "fall armyworm"),
    (re.compile(r"\bsaccharomyces\s+cerevisiae\b|\bs\.?\s*cerevisiae\b", re.I), "saccharomyces cerevisiae"),
    (re.compile(r"\byeast\b", re.I), "yeast"),
    (re.compile(r"\b(?:e\.?\s*coli|escherichia\s+coli|bacterial\s+(?:cell\s+)?expression)\b", re.I), "escherichia coli"),
    (re.compile(r"\b(?:dog|canine)\b.{0,45}\b(?:mdck\w*|kidney\s+cells?)\b|\b(?:mdck\w*|kidney\s+cells?)\b.{0,45}\b(?:dog|canine)\b", re.I), "dog"),
    (re.compile(r"\b(?:mouse|mice)\b.{0,35}\b(?:humanized|transgenic)\b|\b(?:humanized|transgenic)\b.{0,35}\b(?:mouse|mice)\b", re.I), "mouse"),
    (re.compile(r"\b(?:pig|porcine)\b.{0,35}\bipec\b|\bipec\b.{0,35}\b(?:pig|porcine)\b", re.I), "pig"),
)

HETEROLOGOUS_CUE = re.compile(
    r"\b(?:baculovirus|cDNA|express(?:ed|es|ing|ion)?|heterologous|humanized|"
    r"oocytes?|recombinant|sf[- ]?9|sf[- ]?21|supersomes?|transfect(?:ed|ion)?)\b",
    re.I,
)
RECOMBINANT_CONTEXT = re.compile(
    r"\b(?:baculovirus|cDNA|express(?:ed|es|ing|ion)?|heterologous|recombinant|supersomes?|transfect(?:ed|ion)?)\b",
    re.I,
)
GENE_CUE = re.compile(
    r"\b(?:abcb\w*|abcg\w*|bcrp|cDNA|cRNA|cyp\w*|enzyme|expressed|expressing|gene|"
    r"mdr\w*|mrp\w*|oatp\w*|p[- ]?gp|pept\w*|protein|recombinant|transfected|"
    r"transgenic|transporter|ugt\w*)\b",
    re.I,
)
PREPARATION_CUE = re.compile(
    r"\b(?:blood|cells?|cytosol|hepatocytes?|homogenate|intestin\w*|kidney|liver|"
    r"membranes?|microsomes?|oocytes?|organs?|patients?|perfusion|plasma|s9|"
    r"subjects?|tissue|vesicles?|volunteers?)\b",
    re.I,
)

# These patterns are intentionally literal. Cell-line names do not imply species.
EXPLICIT_SPECIES_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:human|humans)\b", re.I), "human"),
    (re.compile(r"\b(?:rat|rats)\b", re.I), "rat"),
    (re.compile(r"\b(?:mouse|mice|murine)\b", re.I), "mouse"),
    (re.compile(r"\b(?:dog|dogs|canine|canines|beagle|beagles)\b", re.I), "dog"),
    (re.compile(r"\b(?:pig|pigs|porcine|swine|minipigs?)\b", re.I), "pig"),
    (re.compile(r"\b(?:rabbit|rabbits)\b", re.I), "rabbit"),
    (re.compile(r"\b(?:guinea\s+pig|guinea\s+pigs)\b", re.I), "guinea pig"),
    (re.compile(r"\b(?:cynomolgus|macaca\s+fascicularis)\b", re.I), "cynomolgus monkey"),
    (re.compile(r"\b(?:rhesus|macaca\s+mulatta)\b", re.I), "rhesus monkey"),
    (re.compile(r"\b(?:monkey|monkeys|macaque|macaques)\b", re.I), "monkey"),
    (re.compile(r"\b(?:sheep|ovine)\b", re.I), "sheep"),
    (re.compile(r"\b(?:cattle|bovine|cows?)\b", re.I), "cattle"),
    (re.compile(r"\b(?:frog|frogs|xenopus(?:\s+laevis)?|x\.?\s*laevis)\b", re.I), "frog"),
    (re.compile(r"\b(?:hamster|hamsters)\b", re.I), "hamster"),
    (re.compile(r"\b(?:chicken|chickens)\b", re.I), "chicken"),
    (re.compile(r"\b(?:horse|horses|equine)\b", re.I), "horse"),
    (re.compile(r"\b(?:cat|cats|feline)\b", re.I), "cat"),
    (re.compile(r"\b(?:goat|goats|caprine)\b", re.I), "goat"),
    (re.compile(r"\b(?:insect|insects)\b", re.I), "insect"),
    (re.compile(r"\b(?:yeast|saccharomyces(?:\s+cerevisiae)?)\b", re.I), "yeast"),
    (re.compile(r"\b(?:e\.?\s*coli|escherichia\s+coli)\b", re.I), "escherichia coli"),
)


def _clean_source_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    if cleaned.casefold() in NULL_LIKE:
        return None
    return cleaned


def _tuple_key(values: Iterable[str | None]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _decode_tuple_key(value: str) -> tuple[str | None, ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list) or any(item is not None and not isinstance(item, str) for item in decoded):
        raise ValueError(f"invalid tuple key: {value}")
    return tuple(decoded)


def _clean_label(value: str | None) -> str | None:
    if value is None:
        return None
    label = re.sub(r"\s+", " ", value.strip()).casefold()
    if not label or FORBIDDEN_OUTPUT.search(label):
        raise ValueError(f"invalid globally reconciled label: {value!r}")
    return label


def _resolve_alias(label: str | None, aliases: dict[str, str]) -> str | None:
    label = _clean_label(label)
    seen: set[str] = set()
    while label is not None and label in aliases:
        if label in seen:
            raise ValueError(f"alias cycle at {label!r}")
        seen.add(label)
        label = _clean_label(aliases[label])
    return label


def _current_value(
    current: dict[str, Any],
    output: OutputSpec,
    source_values: tuple[str | None, ...],
) -> str | None:
    primary_index = output.source_columns.index(output.current_input_column)
    primary_value = source_values[primary_index]
    if primary_value is None:
        return None
    mapping = current[output.current_input_column][output.current_output_column]
    if primary_value not in mapping:
        raise KeyError(
            f"current mapping lacks {output.current_input_column}={primary_value!r}"
        )
    return mapping[primary_value]


def _context_override(raw_value: str | None, current_label: str | None) -> str | None:
    if raw_value is None or current_label is None:
        return current_label
    text = raw_value.casefold()
    if current_label == "ddi study":
        if re.search(r"\b(?:clinical|human|patients?|subjects?|volunteers?)\b", text):
            return "clinical ddi"
        return "drug interaction study"
    # Repair only concrete preparations hidden inside known broad Fh labels.
    if current_label in {"recombinant cells", "recombinant enzyme", "microsomes"}:
        compound_microsomes = re.search(
            r"\b(?:and|plus|with)\b.{0,50}\bmicrosom|\bmicrosom.{0,50}\b(?:and|plus|with)\b",
            text,
        )
        if "microsom" in text and RECOMBINANT_CONTEXT.search(text):
            return "recombinant microsomes"
        if re.search(r"\b(?:liver|hepatic)\s+microsom", text) and not compound_microsomes:
            return "liver microsomes"
        if re.search(r"\bintestinal\s+microsom", text) and not compound_microsomes:
            return "intestinal microsomes"
        if "supersome" in text:
            return "recombinant microsomes"
        if re.search(r"\b(?:membrane|membranes|membranous|vesicle|vesicles)\b", text):
            return "recombinant membranes"
        if re.search(r"\blysates?\b", text):
            return "recombinant cell lysate"
        if re.search(r"\bhomogenates?\b", text):
            return "recombinant cell homogenate"
        if (
            re.search(r"\b(?:cell|cells|cellular)\b", text)
            and RECOMBINANT_CONTEXT.search(text)
            and not re.search(r"\b(?:enzyme|enzymes|protein|proteins|purified)\b", text)
        ):
            return "recombinant cells"
    if (
        current_label == "gut sac"
        and "everted" in text
        and not re.search(r"\bnon[- ]?everted\b|/|\bperfusion\b", text)
    ):
        return "everted gut sac"
    if current_label == "intestinal perfusion" and re.search(r"single[- ]pass", text):
        return "single-pass intestinal perfusion"
    if current_label in {"caco_2", "caco_2 ht29-mtx coculture"} and re.search(
        r"(?:co[- ]?culture|coculture).*(?:ht[- ]?29)|(?:ht[- ]?29).*(?:co[- ]?culture|coculture)",
        text,
    ):
        return "caco_2 ht29-mtx coculture"
    return current_label


def _split_species_label(label: str | None) -> set[str]:
    if label is None:
        return set()
    return {
        _resolve_alias(part.strip(), GLOBAL_SPECIES_ALIASES) or part.strip()
        for part in label.split("+")
        if part.strip()
    }


def _explicit_species(text: str) -> set[str]:
    found: set[str] = set()
    for pattern, species in EXPLICIT_SPECIES_PATTERNS:
        if pattern.search(text):
            found.add(species)
    if "cynomolgus monkey" in found:
        found.discard("monkey")
    if "rhesus monkey" in found:
        found.discard("monkey")
    return found


def _gene_species(text: str) -> set[str]:
    found: set[str] = set()
    for clause in text.split(" | "):
        for pattern, species in EXPLICIT_SPECIES_PATTERNS:
            for match in pattern.finditer(clause):
                after = clause[match.end() : match.end() + 35]
                before = clause[max(0, match.start() - 35) : match.start()]
                after_gene = GENE_CUE.search(after)
                after_prep = PREPARATION_CUE.search(after)
                before_gene = list(GENE_CUE.finditer(before))
                before_prep = list(PREPARATION_CUE.finditer(before))
                gene_precedes_prep = after_gene is not None and (
                    after_prep is None or after_gene.start() < after_prep.start()
                )
                gene_follows_prep = bool(before_gene) and (
                    not before_prep or before_gene[-1].start() > before_prep[-1].start()
                )
                if gene_precedes_prep or (after_prep is None and gene_follows_prep):
                    found.add(species)
                    break
    if re.search(
        r"(?:\b(?:cyp\w*|ugt\w*|abc\w*|mdr\w*|mrp\w*|oatp\w*|pept\w*)\b.{0,30}\bhumanized\b|"
        r"\bhumanized\b.{0,30}\b(?:cyp\w*|ugt\w*|abc\w*|mdr\w*|mrp\w*|oatp\w*|pept\w*)\b)",
        text,
        re.I,
    ):
        found.add("human")
    if "cynomolgus monkey" in found:
        found.discard("monkey")
    if "rhesus monkey" in found:
        found.discard("monkey")
    return found


def _preparation_species(text: str) -> set[str]:
    found: set[str] = set()
    for clause in text.split(" | "):
        for pattern, species in EXPLICIT_SPECIES_PATTERNS:
            for match in pattern.finditer(clause):
                after = clause[match.end() : match.end() + 45]
                before = clause[max(0, match.start() - 45) : match.start()]
                after_prep = PREPARATION_CUE.search(after)
                after_gene = GENE_CUE.search(after)
                before_prep = list(PREPARATION_CUE.finditer(before))
                before_gene = list(GENE_CUE.finditer(before))
                prep_precedes_gene = after_prep is not None and (
                    after_gene is None or after_prep.start() < after_gene.start()
                )
                prep_follows_gene = bool(before_prep) and (
                    not before_gene or before_prep[-1].start() > before_gene[-1].start()
                )
                if prep_precedes_gene or prep_follows_gene:
                    found.add(species)
                    break
    if "cynomolgus monkey" in found:
        found.discard("monkey")
    if "rhesus monkey" in found:
        found.discard("monkey")
    return found


def _explicit_hosts(text: str) -> set[str]:
    found = {species for pattern, species in HOST_PATTERNS if pattern.search(text)}
    if "african clawed frog" in found:
        found.discard("frog")
    if "fall armyworm" in found:
        found.discard("insect")
    if "saccharomyces cerevisiae" in found:
        found.discard("yeast")
    return found


def _primary_expression_hosts(text: str) -> set[str]:
    hosts = _explicit_hosts(text)
    if re.search(r"\binsects?\b", text):
        hosts.add("insect")
    if re.search(r"\byeast\b", text):
        hosts.add("yeast")
    if "saccharomyces cerevisiae" in hosts:
        hosts.discard("yeast")
    return hosts


def _drop_broader_host_aliases(hosts: set[str]) -> set[str]:
    hosts = set(hosts)
    if "african clawed frog" in hosts:
        hosts.discard("frog")
    if "fall armyworm" in hosts:
        hosts.discard("insect")
    if "saccharomyces cerevisiae" in hosts:
        hosts.discard("yeast")
    return hosts


def _drop_host_equivalent_genes(genes: set[str], hosts: set[str]) -> set[str]:
    genes = set(genes)
    equivalents = {
        "african clawed frog": {"frog"},
        "fall armyworm": {"insect"},
        "saccharomyces cerevisiae": {"yeast"},
    }
    for specific_host, broader_genes in equivalents.items():
        if specific_host in hosts:
            genes -= broader_genes
    genes -= hosts & {
        "african clawed frog",
        "escherichia coli",
        "fall armyworm",
        "frog",
        "insect",
        "saccharomyces cerevisiae",
        "yeast",
    }
    return genes


def _format_species_set(species: Iterable[str]) -> str:
    return " + ".join(sorted(set(species)))


def _species_context(
    source: str,
    source_values: tuple[str | None, ...],
    baseline: str | None,
) -> str | None:
    baseline_species = _split_species_label(baseline)
    text = " | ".join(value for value in source_values if value).casefold()
    if not text:
        return None

    if (
        source == "fh"
        and source_values[0] is not None
        and re.search(r"\bsf[- ]?(?:9|21)\b", source_values[0], re.I)
        and not re.search(r"\bspodoptera(?:\s+frugiperda)?\b", source_values[0], re.I)
    ):
        # Sf9/Sf21 alone is a cell-line code, not explicit species evidence.
        baseline_species.discard("fall armyworm")

    # Repair two reviewed overly broad aliases using explicit source wording.
    if baseline_species == {"monkey"} and re.search(r"\b(?:cynomolgus|macaca\s+fascicularis)\b", text):
        baseline_species = {"cynomolgus monkey"}
    if baseline_species == {"frog"} and re.search(r"\bxenopus\s+laevis\b|\bx\.?\s*laevis\b", text):
        baseline_species = {"frog"}

    role_text = re.sub(r"[‐‑‒–—−]", "-", text)
    role_text = re.sub(r"\bexpressed\s+as\b", "", role_text)
    preparation_species = _preparation_species(role_text)
    hek_only_human = bool(
        source == "fh"
        and re.search(r"\bhek[- ]?[a-z0-9]*\b", role_text)
        and not re.search(
            r"\bhuman\b.{0,35}\b(?:liver|microsomes?|subjects?|tissue|volunteers?)\b",
            role_text,
        )
    )
    if hek_only_human:
        preparation_species.discard("human")
    hosts = _explicit_hosts(role_text) | preparation_species
    heterologous = bool(HETEROLOGOUS_CUE.search(role_text))
    if heterologous:
        gene_species = _gene_species(role_text)

        if source in {"fa", "fg"}:
            # Their original species contract identifies the experimental
            # subject/tissue/cell source. Preserve any baseline atom not
            # explicitly assigned to the introduced gene.
            hosts |= baseline_species - gene_species
            hosts = _drop_broader_host_aliases(hosts)

        # In Fh the dedicated species value often supplies the donor species while
        # assay_system states that the preparation is recombinant.
        if (
            source == "fh"
            and source_values[0] is not None
            and RECOMBINANT_CONTEXT.search(role_text)
        ):
            primary_text = (source_values[0] or "").casefold()
            primary_hosts = _primary_expression_hosts(primary_text)
            hosts |= primary_hosts
            donor_species = baseline_species - primary_hosts
            donor_hosts = donor_species & preparation_species
            hosts |= donor_hosts
            gene_species |= donor_species - donor_hosts
            hosts = _drop_broader_host_aliases(hosts)

        if hek_only_human:
            # HEK293 is a cell-line name, not sufficient evidence for a human host.
            hosts.discard("human")

        gene_species = _drop_host_equivalent_genes(gene_species, hosts)

        # Do not promote a species inferred only from a cell-line abbreviation.
        if hosts or gene_species:
            clauses: list[str] = []
            if hosts:
                clauses.append(f"{_format_species_set(hosts)} host")
            if gene_species:
                clauses.append(f"{_format_species_set(gene_species)} gene")
            return "; ".join(clauses)

    if baseline_species:
        return _format_species_set(baseline_species)

    # Recover only explicit species from a dedicated Fh species field or from
    # strong in-vivo subject wording. This avoids treating food/reagent organisms
    # in free assay text as experimental species.
    if source == "fh" and source_values[0] is not None:
        explicit = _explicit_species(source_values[0].casefold())
        return _format_species_set(explicit) if explicit else None
    if re.search(r"\b(?:in vivo|subjects?|patients?|volunteers?|dosed|dosing|administ(?:ered|ration))\b", text):
        explicit = _explicit_species(text)
        return _format_species_set(explicit) if explicit else None
    return None


def _distinct_source_tuples(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[str | None, ...]]:
    cleaned = frame.loc[:, columns].copy()
    for column in columns:
        cleaned[column] = cleaned[column].map(_clean_source_value)
    values = {
        tuple(_clean_source_value(value) for value in row)
        for row in cleaned.itertuples(index=False, name=None)
    }
    return sorted(values, key=lambda row: tuple("" if value is None else value.casefold() for value in row))


def _oral_source_mapping() -> dict[str, Any]:
    payload = json.loads(ORAL_EXTRACTION_PATH.read_text(encoding="utf-8"))
    if payload.get("artifact_version") != "oral_study_context_extraction.v1":
        raise ValueError("oral study-context extraction version mismatch")
    extracted = payload.get("mapping")
    if not isinstance(extracted, dict):
        raise ValueError("oral study-context extraction lacks mapping")
    outputs: dict[str, Any] = {}
    for output_name, extracted_field in ORAL_OUTPUT_FIELDS.items():
        values = {_tuple_key((None,)): None}
        for study_context, result in extracted.items():
            if not isinstance(result, dict):
                raise ValueError("invalid oral study-context extraction result")
            cleaned_context = _clean_source_value(study_context)
            if cleaned_context is None:
                raise ValueError("oral study-context extraction contains null-like key")
            values[_tuple_key((cleaned_context,))] = _clean_label(
                result.get(extracted_field)
            )
        outputs[output_name] = {
            "source_columns": ["study_context"],
            "mapping": dict(sorted(values.items())),
        }
    return outputs


def build_mapping(current_mappings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconcile the six first-stage mappings into the shared global vocabulary."""
    if set(current_mappings) != set(SOURCE_SPECS):
        raise ValueError("first-stage source mapping inventory mismatch")
    sources: dict[str, Any] = {}
    for source, source_spec in SOURCE_SPECS.items():
        current = current_mappings[source]
        required_columns = sorted(
            {column for output in source_spec.outputs for column in output.source_columns}
        )
        frame = pd.read_parquet(source_spec.parquet_path, columns=required_columns)
        source_outputs: dict[str, Any] = {}
        for output in source_spec.outputs:
            result: dict[str, str | None] = {}
            for source_values in _distinct_source_tuples(frame, output.source_columns):
                baseline = _current_value(current, output, source_values)
                if output.output_name == "global_context":
                    baseline = _context_override(source_values[0], baseline)
                    reconciled = _resolve_alias(baseline, GLOBAL_CONTEXT_ALIASES)
                else:
                    reconciled = _species_context(source, source_values, baseline)
                    reconciled = _clean_label(reconciled)
                result[_tuple_key(source_values)] = reconciled
            source_outputs[output.output_name] = {
                "source_columns": list(output.source_columns),
                "mapping": dict(sorted(result.items())),
            }
        sources[source] = source_outputs
    sources[ORAL_SOURCE_ID] = _oral_source_mapping()
    return {"mapping_version": MAPPING_VERSION, "sources": sources}


def validate_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("mapping_version") != MAPPING_VERSION or set(payload) != {"mapping_version", "sources"}:
        raise ValueError("invalid mapping root")
    if set(payload["sources"]) != {*SOURCE_SPECS, ORAL_SOURCE_ID}:
        raise ValueError("source inventory mismatch")

    audit: dict[str, Any] = {}
    for source, source_spec in SOURCE_SPECS.items():
        outputs = payload["sources"][source]
        if set(outputs) != {item.output_name for item in source_spec.outputs}:
            raise ValueError(f"output inventory mismatch for {source}")
        required_columns = sorted(
            {column for output in source_spec.outputs for column in output.source_columns}
        )
        source_frame = pd.read_parquet(
            source_spec.parquet_path,
            columns=required_columns,
        )
        source_audit: dict[str, Any] = {}
        for output in source_spec.outputs:
            section = outputs[output.output_name]
            if section.get("source_columns") != list(output.source_columns):
                raise ValueError(f"source-column contract mismatch for {source}/{output.output_name}")
            mapping = section.get("mapping")
            if not isinstance(mapping, dict):
                raise ValueError(f"mapping is not an object for {source}/{output.output_name}")
            expected_keys = {
                _tuple_key(values)
                for values in _distinct_source_tuples(source_frame, output.source_columns)
            }
            if set(mapping) != expected_keys:
                raise ValueError(
                    f"source tuple coverage mismatch for {source}/{output.output_name}: "
                    f"missing={len(expected_keys - set(mapping))} "
                    f"extra={len(set(mapping) - expected_keys)}"
                )
            values: list[str | None] = []
            for key, value in mapping.items():
                if len(_decode_tuple_key(key)) != len(output.source_columns):
                    raise ValueError(f"tuple width mismatch for {source}/{output.output_name}")
                values.append(_clean_label(value))
            source_audit[output.output_name] = {
                "tuples": len(mapping),
                "non_null": sum(value is not None for value in values),
                "null": sum(value is None for value in values),
                "distinct_labels": len({value for value in values if value is not None}),
                "role_bearing": sum(
                    value is not None and (" host" in value or " gene" in value)
                    for value in values
                ),
            }
        audit[source] = source_audit
    expected_oral = _oral_source_mapping()
    if payload["sources"][ORAL_SOURCE_ID] != expected_oral:
        raise ValueError("oral study-context mapping differs from frozen extraction")
    audit[ORAL_SOURCE_ID] = {
        output_name: {
            "tuples": len(section["mapping"]),
            "non_null": sum(value is not None for value in section["mapping"].values()),
            "null": sum(value is None for value in section["mapping"].values()),
            "distinct_labels": len(
                {value for value in section["mapping"].values() if value is not None}
            ),
            "role_bearing": 0,
        }
        for output_name, section in expected_oral.items()
    }
    return audit
