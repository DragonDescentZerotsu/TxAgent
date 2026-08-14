#!/usr/bin/env python3
"""Reconcile Skin_Reaction embedding-bucket labels into the final tuple map.

``build_embedding_bucket_mapping.py`` owns API execution and passes its
in-memory first-stage mappings here.  This module owns the task semantics:
reviewed aliases, species recovery, endpoint-concept boundaries, severity
grades, exact source-tuple coverage, and the final mapping version.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_ROOT = REPO_ROOT / "data/starling_data/skin_reaction"
MAPPING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MAPPING_ROOT / "globally_reconciled_auxiliary_value_mapping.json"
MAPPING_VERSION = "starling_auxiliary.skin_reaction.globally_reconciled.v4"

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
    # "reviewed_table" reads a frozen hand-mapped file; "llm" reads the extraction pass.
    label_source: str = "llm"
    reviewed_mapping_path: Path | None = None
    null_sentinel: str | None = None
    bucket_pattern: str | None = None
    clustering: str = "lloyd"
    cluster_target_size: int = 100
    max_labels_per_call: int | None = None
    # Most historical extractions normalize one source column.  Composite
    # extractions keep the same first-stage container shape but build each
    # item from this complete source-field tuple.
    current_input_columns: tuple[str, ...] | None = None

    @property
    def extraction_input_columns(self) -> tuple[str, ...]:
        return self.current_input_columns or (self.current_input_column,)


@dataclass(frozen=True)
class SourceSpec:
    parquet_path: Path
    outputs: tuple[OutputSpec, ...]


STUDY_DESIGN_REVIEWED_MAPPING = MAPPING_ROOT / "study_design_reviewed_mapping.json"

SOURCE_SPECS = {
    "direct_skin_reaction": SourceSpec(
        parquet_path=DATA_ROOT / "direct_skin_reaction/extractions.parquet",
        outputs=(
            OutputSpec(
                ("assay_or_test",),
                "global_context",
                "assay_or_test",
                "canonical_context",
            ),
            OutputSpec(
                ("species_or_population",),
                "global_species_context",
                "species_or_population",
                "canonical_species",
                null_sentinel="no species",
            ),
            OutputSpec(
                ("effect_metric",),
                "global_severity_grade",
                "effect_metric",
                "canonical_severity_grade",
                null_sentinel="no explicit grade",
                bucket_pattern=r"[0-4]",
            ),
        ),
    ),
    "sensitization_aop": SourceSpec(
        parquet_path=DATA_ROOT / "sensitization_aop/extractions.parquet",
        outputs=(
            OutputSpec(("assay_type",), "global_context", "assay_type", "canonical_context"),
            OutputSpec(
                ("assay_type", "experimental_conditions", "support_text"),
                "global_species_context",
                "species_evidence_packet",
                "canonical_species",
                null_sentinel="no species",
                clustering="minibatch",
                max_labels_per_call=50,
                current_input_columns=(
                    "assay_type",
                    "experimental_conditions",
                    "support_text",
                ),
            ),
            OutputSpec(
                ("endpoint_or_target",),
                "global_endpoint_context",
                "endpoint_or_target",
                "canonical_endpoint_context",
            ),
        ),
    ),
    "phototoxicity_irritation_local_damage": SourceSpec(
        parquet_path=DATA_ROOT / "phototoxicity_irritation_local_damage/extractions.parquet",
        outputs=(
            OutputSpec(
                ("assay_method",),
                "global_context",
                "assay_method",
                "canonical_context",
                clustering="minibatch",
                cluster_target_size=250,
                max_labels_per_call=250,
            ),
            OutputSpec(
                ("evidence_system",),
                "global_species_context",
                "evidence_system",
                "canonical_species",
                null_sentinel="no species or system",
            ),
        ),
    ),
    "skin_exposure": SourceSpec(
        parquet_path=DATA_ROOT / "skin_exposure/extractions.parquet",
        outputs=(
            OutputSpec(
                ("study_design",),
                "global_context",
                "study_design",
                "canonical_context",
                label_source="reviewed_table",
                reviewed_mapping_path=STUDY_DESIGN_REVIEWED_MAPPING,
            ),
            OutputSpec(
                ("skin_source",),
                "global_species_context",
                "skin_source",
                "canonical_species",
                null_sentinel="no species",
            ),
        ),
    ),
}


# Only reviewed aliases belong here. Labels absent from this registry pass through
# unchanged rather than being merged by an unsafe fuzzy rule.
GLOBAL_CONTEXT_ALIASES = {
    # Formatting-only variants of the reviewed study_design labels.
    "calculated_or_estimated": "calculated estimate",
    "calculated_estimate": "calculated estimate",
    "ex_vivo": "ex vivo",
    "human_exposure_monitoring": "human exposure monitoring",
    "in_vitro": "in vitro",
    "in_vitro_cell_culture": "in vitro cell culture",
    "in_vitro_diffusion_cell": "in vitro diffusion cell",
    "in_vitro_skin_model": "in vitro skin model",
    "in_vivo": "in vivo",
    "in_vivo_animal": "in vivo animal",
    "in_vivo_human": "in vivo human",
    "franz diffusion cell": "in vitro diffusion cell",
    "diffusion cell": "in vitro diffusion cell",
    # Sensitization assay names: spelled-out form folds onto the regulatory acronym.
    "local lymph node assay": "llna",
    "local lymph node assay (llna)": "llna",
    "murine local lymph node assay": "llna",
    "llna assay": "llna",
    "guinea pig maximization test": "gpmt",
    "guinea pig maximisation test": "gpmt",
    "magnusson-kligman maximization test": "gpmt",
    "magnusson and kligman maximization test": "gpmt",
    "human repeated insult patch test": "hript",
    "human repeat insult patch test": "hript",
    "repeated insult patch test": "ript",
    "repeat insult patch test": "ript",
    "ript": "hript",
    "human maximisation test": "human maximization test",
    "hmt": "human maximization test",
    "mouse ear swelling test": "mest",
    "murine ear swelling test": "mest",
    "repeated open application test": "roat",
    "repeat open application test": "roat",
    "open epicutaneous test": "oet",
    "direct peptide reactivity assay": "dpra",
    "amino acid derivative reactivity assay": "adra",
    "kinetic direct peptide reactivity assay": "kdpra",
    "kinetic dpra": "kdpra",
    "h_clat": "h-clat",
    "hclat": "h-clat",
    "human cell line activation test": "h-clat",
    "keratinosens assay": "keratinosens",
    "lusens assay": "lusens",
    "u_sens": "u-sens",
    "u-sens assay": "u-sens",
    "myeloid u937 skin sensitization test": "musst",
    "sens_is": "sens-is",
    "buehler assay": "buehler test",
    "buhler test": "buehler test",
    "draize test": "draize test",
    "epicutaneous patch test": "patch test",
    "human epicutaneous patch test": "human patch test",
    "human diagnostic patch test": "diagnostic patch test",
    "photopatch test": "photopatch test",
    # Readout platforms.
    "real-time rt-pcr": "real-time pcr",
    "quantitative real-time rt-pcr": "real-time pcr",
    "quantitative rt-pcr": "real-time pcr",
    "qrt-pcr": "real-time pcr",
    "qpcr": "real-time pcr",
    "rt-pcr": "real-time pcr",
    "immunoblot": "western blot",
    "flow cytometry analysis": "flow cytometry",
    "in silico model": "in silico prediction",
    "in silico qsar prediction": "in silico prediction",
    "qsar model": "in silico prediction",
    "qsar prediction": "in silico prediction",
    "glutathione depletion assay": "gsh depletion assay",
    "glutathione depletion": "gsh depletion assay",
    "gsh depletion chemoassay": "gsh depletion assay",
    "peptide depletion assay": "peptide reactivity assay",
    "peptide reactivity": "peptide reactivity assay",
}


# These are narrow, reviewed synonym folds.  The LLM may choose any concise
# label inside one embedding cluster, but cross-cluster reconciliation must not
# use fuzzy matching: biomarker and nucleophile identity is load-bearing.
GLOBAL_ENDPOINT_ALIASES = {
    "cysteine peptide depletion": "cysteine depletion",
    "cysteine-containing peptide depletion": "cysteine depletion",
    "cysteine-containing peptide reactivity": "cysteine reactivity",
    "glutathione depletion": "gsh depletion",
    "glutathione (gsh) depletion": "gsh depletion",
    "cd86 expression": "cd86",
    "cd86 upregulation": "cd86",
    "cd54 expression": "cd54",
    "cd54 upregulation": "cd54",
    "il8": "il-8",
    "il-8 expression": "il-8",
    "interleukin-8": "il-8",
}


GLOBAL_SPECIES_ALIASES = {
    "bovine": "cattle",
    "cow": "cattle",
    "cows": "cattle",
    "murine": "mouse",
    "mice": "mouse",
    "mus musculus": "mouse",
    "porcine": "pig",
    "swine": "pig",
    "minipig": "pig",
    "micropig": "pig",
    "sus scrofa": "pig",
    "guinea-pig": "guinea pig",
    "cavia porcellus": "guinea pig",
    "canine": "dog",
    "feline": "cat",
    "equine": "horse",
    "ovine": "sheep",
    "caprine": "goat",
    "rattus norvegicus": "rat",
    "oryctolagus cuniculus": "rabbit",
    "homo sapiens": "human",
    "humans": "human",
    "macaca fascicularis": "cynomolgus monkey",
    "macaca mulatta": "rhesus monkey",
    "xenopus laevis": "frog",
    "snakes": "snake",
}


# These patterns are intentionally literal. Cell-line and assay-acronym names do
# not by themselves imply a species.
EXPLICIT_SPECIES_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:human|humans)\b", re.I), "human"),
    (re.compile(r"\b(?:rat|rats)\b", re.I), "rat"),
    (re.compile(r"\b(?:mouse|mice|murine)\b", re.I), "mouse"),
    (re.compile(r"\b(?:dog|dogs|canine|canines|beagle|beagles)\b", re.I), "dog"),
    (re.compile(r"\b(?:guinea\s*-?\s*pig|guinea\s*-?\s*pigs)\b", re.I), "guinea pig"),
    (re.compile(r"(?<!guinea\s)(?<!guinea-)\b(?:pig|pigs|porcine|swine|minipigs?|micropigs?)\b", re.I), "pig"),
    (re.compile(r"\b(?:rabbit|rabbits)\b", re.I), "rabbit"),
    (re.compile(r"\b(?:cynomolgus|macaca\s+fascicularis)\b", re.I), "cynomolgus monkey"),
    (re.compile(r"\b(?:rhesus|macaca\s+mulatta)\b", re.I), "rhesus monkey"),
    (re.compile(r"\b(?:monkey|monkeys|macaque|macaques)\b", re.I), "monkey"),
    (re.compile(r"\b(?:sheep|ovine)\b", re.I), "sheep"),
    (re.compile(r"\b(?:cattle|bovine|cows?)\b", re.I), "cattle"),
    (re.compile(r"\b(?:frog|frogs|xenopus(?:\s+laevis)?|x\.?\s*laevis)\b", re.I), "frog"),
    (re.compile(r"\b(?:snake|snakes|snakeskin)\b", re.I), "snake"),
    (re.compile(r"\b(?:hamster|hamsters)\b", re.I), "hamster"),
    (re.compile(r"\b(?:chicken|chickens)\b", re.I), "chicken"),
    (re.compile(r"\b(?:horse|horses|equine)\b", re.I), "horse"),
    (re.compile(r"\b(?:cat|cats|feline)\b", re.I), "cat"),
    (re.compile(r"\b(?:goat|goats|caprine)\b", re.I), "goat"),
)

# Only these wordings license recovering a species from a free assay description
# when the extraction pass returned nothing. Without them a reagent organism such
# as bovine serum albumin would be read as the experimental species.
IN_VIVO_SUBJECT_CUE = re.compile(
    r"\b(?:in vivo|subjects?|patients?|volunteers?|dosed|dosing|"
    r"administ(?:ered|ration)|sensiti[sz]ation study|challenge)\b",
    re.I,
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
    if not isinstance(decoded, list) or any(
        item is not None and not isinstance(item, str) for item in decoded
    ):
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
        resolved = _clean_label(aliases[label])
        if resolved == label:
            return label
        label = resolved
    return label


def load_reviewed_mapping(path: Path) -> dict[str, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"reviewed mapping is not an object: {path}")
    cleaned: dict[str, str | None] = {}
    for key, value in mapping.items():
        cleaned[str(key)] = _clean_label(value)
    return cleaned


def _source_first_stage(
    first_stage: Mapping[str, Any], source: str, spec: SourceSpec
) -> dict[str, Any]:
    current = first_stage.get(source)
    if not isinstance(current, dict):
        raise ValueError(f"first-stage mapping is missing source {source!r}")
    for output in spec.outputs:
        by_column = current.get(output.current_input_column)
        mapping = (
            by_column.get(output.current_output_column)
            if isinstance(by_column, dict)
            else None
        )
        if not isinstance(mapping, dict):
            raise ValueError(
                f"first-stage mapping is missing {source}/{output.output_name}"
            )
    return current


def _current_value(
    current: dict[str, Any],
    output: OutputSpec,
    source_values: tuple[str | None, ...],
) -> str | None:
    mapping = current[output.current_input_column][output.current_output_column]
    if output.current_input_column in output.source_columns:
        primary_index = output.source_columns.index(output.current_input_column)
        primary_value = source_values[primary_index]
        if primary_value is None:
            return None
        lookup_key = primary_value
    else:
        lookup_key = _tuple_key(source_values)
    if lookup_key not in mapping:
        raise KeyError(
            f"current mapping lacks {output.current_input_column}={lookup_key!r}"
        )
    return mapping[lookup_key]


def _context_override(raw_value: str | None, current_label: str | None) -> str | None:
    """Reviewed repairs that only fire on explicit wording in the raw value."""
    if raw_value is None or current_label is None:
        return current_label
    text = raw_value.casefold()
    if current_label in {"patch test", "diagnostic patch test", "maximization test"}:
        human = re.search(r"\b(?:human|patients?|subjects?|volunteers?)\b", text)
        if current_label == "maximization test":
            if re.search(r"\bguinea[\s-]*pig\b", text):
                return "gpmt"
            if human:
                return "human maximization test"
        elif human:
            return f"human {current_label}"
    if current_label == "llna" and re.search(r"\bllna[\s:-]*da\b|\bllna[- ]da\b", text):
        # LLNA-DA is a separate OECD test guideline, not an LLNA readout variant.
        return "llna-da"
    if current_label == "in vitro diffusion cell" and re.search(r"\bex[\s_-]*vivo\b", text):
        return "ex vivo + in vitro diffusion cell"
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
    if "guinea pig" in found and not re.search(r"(?<!guinea[\s-])\b(?:pig|pigs|porcine|swine)\b", text, re.I):
        found.discard("pig")
    return found


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

    # Repair two reviewed overly broad aliases using explicit source wording.
    if baseline_species == {"monkey"}:
        if re.search(r"\b(?:cynomolgus|macaca\s+fascicularis)\b", text):
            baseline_species = {"cynomolgus monkey"}
        elif re.search(r"\b(?:rhesus|macaca\s+mulatta)\b", text):
            baseline_species = {"rhesus monkey"}

    if baseline_species:
        return _format_species_set(baseline_species)

    if source == "sensitization_aop":
        # The v3 classifier has already seen the complete row-level species
        # packet.  A null is an intentional abstention and must not be replaced
        # by a regex match over incidental prose or reagent organisms.
        return None

    if source in {"skin_exposure", "direct_skin_reaction"}:
        # These are dedicated tissue/population fields; a species named there is
        # experimental evidence, not incidental assay prose.
        explicit = _explicit_species(text)
        return _format_species_set(explicit) if explicit else None

    # assay_type is a free assay description. Recover a species only from in vivo
    # subject wording so a reagent or donor protein is never read as the species.
    if IN_VIVO_SUBJECT_CUE.search(text):
        explicit = _explicit_species(text)
        return _format_species_set(explicit) if explicit else None
    return None


def _distinct_source_tuples(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> list[tuple[str | None, ...]]:
    cleaned = frame.loc[:, columns].copy()
    for column in columns:
        cleaned[column] = cleaned[column].map(_clean_source_value)
    values = {
        tuple(_clean_source_value(value) for value in row)
        for row in cleaned.itertuples(index=False, name=None)
    }
    return sorted(
        values,
        key=lambda row: tuple("" if value is None else value.casefold() for value in row),
    )


def _reconcile_output(
    source: str,
    output: OutputSpec,
    source_values: tuple[str | None, ...],
    baseline: str | None,
) -> str | None:
    if output.output_name == "global_context":
        baseline = _context_override(source_values[0], baseline)
        return _resolve_alias(baseline, GLOBAL_CONTEXT_ALIASES)
    if output.output_name == "global_species_context":
        return _clean_label(_species_context(source, source_values, baseline))
    if output.output_name == "global_endpoint_context":
        return _resolve_alias(baseline, GLOBAL_ENDPOINT_ALIASES)
    if output.output_name == "global_severity_grade":
        grade = _clean_label(baseline)
        if grade is not None and grade not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"invalid reconciled severity grade: {grade!r}")
        return grade
    raise ValueError(f"unsupported reconciled output {output.output_name!r}")


def build_mapping(first_stage: Mapping[str, Any]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for source, source_spec in SOURCE_SPECS.items():
        current = _source_first_stage(first_stage, source, source_spec)
        required_columns = sorted(
            {column for output in source_spec.outputs for column in output.source_columns}
        )
        frame = pd.read_parquet(source_spec.parquet_path, columns=required_columns)
        source_outputs: dict[str, Any] = {}
        for output in source_spec.outputs:
            result: dict[str, str | None] = {}
            for source_values in _distinct_source_tuples(frame, output.source_columns):
                baseline = _current_value(current, output, source_values)
                reconciled = _reconcile_output(
                    source, output, source_values, baseline
                )
                result[_tuple_key(source_values)] = reconciled
            source_outputs[output.output_name] = {
                "source_columns": list(output.source_columns),
                "mapping": dict(sorted(result.items())),
            }
        sources[source] = source_outputs
    return {"mapping_version": MAPPING_VERSION, "sources": sources}


def build_output_section(
    first_stage_source: Mapping[str, Any],
    *,
    source: str,
    output_name: str,
) -> dict[str, Any]:
    """Build one final tuple-keyed section for a focused resumable run."""
    source_spec = SOURCE_SPECS[source]
    output = next(
        (item for item in source_spec.outputs if item.output_name == output_name),
        None,
    )
    if output is None:
        raise ValueError(f"unknown auxiliary output {source}/{output_name}")
    current = {str(key): value for key, value in first_stage_source.items()}
    _source_first_stage({source: current}, source, SourceSpec(source_spec.parquet_path, (output,)))
    frame = pd.read_parquet(source_spec.parquet_path, columns=list(output.source_columns))
    result: dict[str, str | None] = {}
    for source_values in _distinct_source_tuples(frame, output.source_columns):
        baseline = _current_value(current, output, source_values)
        result[_tuple_key(source_values)] = _reconcile_output(
            source, output, source_values, baseline
        )
    return {
        "source_columns": list(output.source_columns),
        "mapping": dict(sorted(result.items())),
    }


def validate_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("mapping_version") != MAPPING_VERSION or set(payload) != {
        "mapping_version",
        "sources",
    }:
        raise ValueError("invalid mapping root")
    if set(payload["sources"]) != set(SOURCE_SPECS):
        raise ValueError("source inventory mismatch")

    audit: dict[str, Any] = {}
    for source, source_spec in SOURCE_SPECS.items():
        outputs = payload["sources"][source]
        if set(outputs) != {item.output_name for item in source_spec.outputs}:
            raise ValueError(f"output inventory mismatch for {source}")
        required_columns = sorted(
            {column for output in source_spec.outputs for column in output.source_columns}
        )
        source_frame = pd.read_parquet(source_spec.parquet_path, columns=required_columns)
        source_audit: dict[str, Any] = {}
        for output in source_spec.outputs:
            section = outputs[output.output_name]
            if section.get("source_columns") != list(output.source_columns):
                raise ValueError(
                    f"source-column contract mismatch for {source}/{output.output_name}"
                )
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
            }
        audit[source] = source_audit
    return audit


def run(args: argparse.Namespace) -> None:
    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {destination}; pass --overwrite to replace it")
    first_stage = json.loads(Path(args.first_stage_json).read_text(encoding="utf-8"))
    payload = build_mapping(first_stage)
    audit = validate_mapping(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"wrote {destination}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--first-stage-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
