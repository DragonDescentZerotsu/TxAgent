from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_measurement_candidates as grammar,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    candidate_set_sha256,
    load_candidates,
)

REPO = Path(__file__).resolve().parents[4]
GOLD = (
    REPO / "tests/chembl_tool/common/measurement_resolution_quality/gold/ames.v10.jsonl"
)


def _pairs(row: dict[str, object]) -> set[tuple[str, str]]:
    return {
        (str(candidate["measurement"]), str(candidate["unit"]))
        for candidate in grammar.candidates_for_record(row)
    }


def _same_number(left: object, right: object) -> bool:
    try:
        return Decimal(str(left).replace(",", "")) == Decimal(
            str(right).replace(",", "")
        )
    except (InvalidOperation, ValueError):
        return False


def test_candidate_json_is_stable_across_python_hash_seeds():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.00001",
        "support_text": (
            "Frequency was measured in an intentionally extended assay description "
            "containing enough contextual words to separate the two equally long "
            "evidence anchors by far more than one excerpt window. A resistant mutant "
            "frequency was observed at 1 x 10^-5."
        ),
    }
    code = (
        "import json,sys; from data.processing.evidence_library.versions.v10.tasks.ames."
        "starling_measurement_candidates import candidate_json_for_record as f; "
        "print(f(json.loads(sys.argv[1])))"
    )
    outputs = []
    for seed in ("0", "1", "101"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code, json.dumps(row)], env=env, text=True
            )
        )
    assert len(set(outputs)) == 1
    candidates = json.loads(outputs[0])
    assert candidates
    assert all(candidate["evidence"] for candidate in candidates)


def test_scientific_product_canonicalizes_a_source_projected_readout():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.00006",
        "support_text": "The MMS-induced mutation rate was 6 × 10^-5.",
    }
    pairs = _pairs(row)
    assert ("0.00006", "MMS-induced mutation rate") in pairs
    assert all(not unit.startswith("10^-") for _, unit in pairs)


def test_authoritative_unit_text_and_scale_transform_are_candidates():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "1.95 × 10^-3",
        "unit_text": "µmol J^-1 (G value for SSB)",
    }
    pairs = _pairs(row)
    assert ("1.95", "µmol J^-1 (G value for SSB)") in pairs
    assert ("1.95", "10^-3 µmol J^-1 (G value for SSB)") in pairs


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "12.41",
                "assay_method_and_endpoint": "hepatic GSH:GSSG redox ratio",
                "support_text": "The GSH:GSSG ratio was 12.41.",
            },
            ("12.41", "GSH/GSSG ratio"),
        ),
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "1",
                "support_text": "MNNG produced 1 resistant mutant",
            },
            ("1", "resistant mutant"),
        ),
        (
            {
                "source_id": "premutagenic_damage",
                "measurement_text": "r = 0.384",
                "support_text": "Spearman correlation gave r = 0.384",
            },
            ("0.384", "Spearman correlation coefficient (r)"),
        ),
    ],
)
def test_generic_named_metric_grammar(row, expected):
    assert expected in _pairs(row)


def test_candidates_are_deterministic_and_hash_compatible():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "1",
        "support_text": "produced 1 resistant mutant",
    }
    first = grammar.candidates_for_record(row)
    second = grammar.candidates_for_record(dict(reversed(list(row.items()))))
    assert first == second
    assert all(candidate["evidence"] for candidate in first)
    assert all(len(candidate["evidence"]) <= 2 for candidate in first)
    payload = grammar.candidate_json_for_record(row)
    assert load_candidates(payload) == first
    assert grammar.candidate_set_sha256_for_record(row) == candidate_set_sha256(payload)


def test_more_than_five_values_is_an_atomic_ambiguity():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "1; 2; 3; 4; 5; 6",
        "unit_text": "foci per cell",
    }
    assert grammar.candidate_generation_disposition(row) == "ambiguous_multiple_values"
    assert grammar.candidates_for_record(row) == []


def test_candidate_set_over_cap_is_rejected_without_slicing(monkeypatch):
    row = {"source_id": "fixed_mutation", "measurement_text": "1"}
    monkeypatch.setattr(
        grammar,
        "_direct_pairs",
        lambda *_: {("1", f"source unit {index}") for index in range(41)},
    )
    assert grammar.candidate_generation_disposition(row) == (
        "ambiguous_candidate_cross_product"
    )
    assert grammar.candidates_for_record(row) == []


def test_equivalent_numeric_spellings_share_named_units():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "1",
        "support_text": "Polyploidy cells were 1.0 plus or minus 0.3.",
    }
    assert ("1", "polyploid cells") in _pairs(row)


def test_removed_reference_does_not_crash_slash_ratio_parser():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.4",
        "support_text": "The [GSSG/GSH+GSSG]/AUC ratio was 0.4.",
    }
    assert isinstance(grammar.candidates_for_record(row), list)


def test_oversized_candidate_text_is_rejected_atomically():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "1",
        "unit_text": "x" * (grammar.MAX_CANDIDATE_FIELD_BYTES + 1),
    }
    assert grammar.candidate_generation_disposition(row) == "candidate_text_too_large"
    assert grammar.candidates_for_record(row) == []


def test_unrelated_scientific_dose_does_not_create_mutation_frequency():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "60",
        "support_text": "60 micronuclei were observed after exposure at 10^-6 M.",
    }
    pairs = _pairs(row)
    assert ("60", "micronuclei") in pairs
    assert all("mutation frequency" not in unit for _, unit in pairs)


def test_percent_candidate_requires_a_value_linked_percent_span():
    unrelated = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.00004",
        "support_text": "The resistance frequency was 0.00004 at 1% survival.",
    }
    linked = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.21",
        "support_text": "Aneuploidy was 0.21 versus 0% in controls.",
    }
    assert ("0.00004", "%") not in _pairs(unrelated)
    assert ("0.21", "%") in _pairs(linked)


def test_correlation_candidate_requires_a_value_linked_correlation_span():
    unrelated = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "8 of 16 samples positive",
        "support_text": "Adduct level did not correlate with therapy.",
    }
    linked = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.384",
        "support_text": "Spearman correlation was positive; r = 0.384.",
    }
    assert all("correlation coefficient" not in unit for _, unit in _pairs(unrelated))
    assert ("0.384", "correlation coefficient (r)") in _pairs(linked)


def test_physical_unit_candidate_requires_value_adjacency():
    unrelated = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "27.8",
        "support_text": "Comet frequency was 27.8; tail length was 20.63 µm.",
    }
    linked = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "20.63",
        "support_text": "Tail length was 20.63 µm.",
    }
    assert ("27.8", "µm") not in _pairs(unrelated)
    assert ("20.63", "µm") in _pairs(linked)


def test_value_linked_units_do_not_cross_multiple_values():
    correlation = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "5-fold increase; correlation r = 0.97",
    }
    percent = {
        "source_id": "fixed_mutation",
        "measurement_text": "12.5 and 50%",
    }
    physical = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "27.8; 20.63",
        "support_text": "Comet frequency 27.8 and tail length 20.63 µm.",
    }
    assert ("5", "correlation coefficient") not in _pairs(correlation)
    assert ("0.97", "correlation coefficient") in _pairs(correlation)
    assert ("12.5", "%") not in _pairs(percent)
    assert ("50", "%") in _pairs(percent)
    assert ("27.8", "µm") not in _pairs(physical)
    assert ("20.63", "µm") in _pairs(physical)
    assert ("20.63", "Comet frequency") not in _pairs(physical)
    assert ("20.63", "frequency") not in _pairs(physical)

    focal_tail = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "20.63",
        "support_text": "Comet frequency was 27.8; tail length was 20.63 µm.",
    }
    assert ("20.63", "Comet frequency") not in _pairs(focal_tail)
    assert ("20.63", "frequency") not in _pairs(focal_tail)
    assert ("20.63", "µm") in _pairs(focal_tail)


def test_canonical_endpoint_cannot_supply_missing_scientific_unit_terms():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.01",
        "support_text": "The reversion rate was 1 × 10^-2.",
        "canonical_endpoint_name": "non_salmonella_reverse_mutation",
    }
    pairs = _pairs(row)
    assert ("0.01", "reversion rate") in pairs
    assert ("0.01", "mutation rate") not in pairs


def test_scientific_stems_follow_the_nearest_named_metric():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.0001; 0.0002",
        "support_text": (
            "The mutation frequency was 1 × 10^-4 and the mutation rate was 2 × 10^-4."
        ),
    }
    pairs = _pairs(row)
    assert ("0.0001", "mutation frequency") in pairs
    assert ("0.0002", "mutation rate") in pairs
    assert ("0.0001", "mutation rate") not in pairs
    assert ("0.0002", "mutation frequency") not in pairs


def test_ratio_parser_rejects_prose_and_reported_values():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "4",
        "support_text": (
            "The ratio of R- to S-enantiomers of styrene oxide produced by Clara "
            "cells is approximately 4 in mice."
        ),
    }
    assert all("4 in mice" not in unit for _, unit in _pairs(row))
    assert all("produced by Clara" not in unit for _, unit in _pairs(row))


def test_slash_ratio_parser_stops_at_fields_and_drops_clause_prefixes():
    competing = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.004",
        "support_text": "The DMS ratio was 0.004; for MNNG the O6/N7 ratio is 0.1.",
        "assay_method_and_endpoint": "Ratio of O6-methylguanine to N7-methylguanine",
    }
    prefixed = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.1",
        "support_text": "Compound 13 had GSH-adduct/IS ratio reported as 0.1.",
    }
    assert all("0.1" not in unit for _, unit in _pairs(competing))
    assert all(not unit.startswith("had ") for _, unit in _pairs(prefixed))


def test_ratio_candidates_reject_unbalanced_and_result_bearing_fragments():
    rows = [
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "1.13",
            "support_text": "The low-dose RBE (ratio of alpha coefficients) was 1.13.",
        },
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "0.5",
            "support_text": "Relative Vmax (F/N) of 0.5 and Km ratio of 0.8.",
        },
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "0.95",
            "support_text": "Specific activity (cpm/µg DNA) and ratio 0.95.",
        },
    ]
    for row in rows:
        units = {unit for _, unit in _pairs(row)}
        assert all(grammar._balanced_grouping(unit) for unit in units)
        assert all(" and ratio" not in unit for unit in units)
        assert all(" of 0.5 and " not in unit for unit in units)


def test_ratio_of_parser_stops_before_reported_result():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "6",
        "support_text": "The ratios of incubation period to control are 17.1 and 6.0.",
    }
    pairs = _pairs(row)
    assert ("6", "incubation period/control ratio") in pairs
    assert all("17" not in unit for _, unit in pairs)


def test_bracketed_slash_ratio_is_unwrapped_without_crashing():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "40",
        "support_text": "The [GSH]/[GSSG] ratio dropped from 300 to 40.",
    }
    assert ("40", "GSH/GSSG ratio") in _pairs(row)


def test_ratio_candidates_do_not_cross_sentence_or_numeric_suffix_boundaries():
    sentence = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.98",
        "support_text": "Cells grew to 22.7 x 10^5/ml. The ratio was 0.98.",
    }
    numbered = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "19.5",
        "support_text": "The peak 1/peak 2 area ratio was 19.5.",
    }
    assert all("ml. The" not in unit for _, unit in _pairs(sentence))
    assert all(unit != "2 area ratio" for _, unit in _pairs(numbered))


def test_ratio_normalization_keeps_complete_grouped_sides():
    rows = [
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "3.6",
            "support_text": "The (R)/(S) ratio was 3.6.",
        },
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "0.0182",
            "support_text": "The M2/(M1+M2) ratio was 0.0182.",
        },
    ]
    for row in rows:
        assert all(not unit.endswith("/ratio") for _, unit in _pairs(row))


def test_open_ratio_parsers_reject_clause_and_sentence_fragments():
    rows = [
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "1.88",
            "support_text": (
                "The ratio of GSH/GSSG decreased after oral exposure to TiO2 NPs "
                "was 1.88."
            ),
        },
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "1",
            "support_text": "The ratio of reduced AZQ to parent compound remained at 1.",
        },
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": "9.246",
            "support_text": "Detected by LC/MS/MS. Mean relative area ratio was 9.246.",
        },
    ]
    forbidden = ("decreased", "remained", "MS/MS.")
    for row in rows:
        units = {unit for _, unit in _pairs(row)}
        assert all(not any(term in unit for term in forbidden) for unit in units)


def test_suffix_metric_phrases_reject_orphan_operators_and_clause_words():
    phrases = grammar._suffix_metric_phrases(
        "MI/MI + MII ratio; hepatocytes this ratio; Table 2 lists cisplatin ratio; "
        "rather stable ratio; HCK+SVP vs SVP ratio; similar ratio."
    )
    assert "+ MII ratio" not in phrases
    assert "this ratio" not in phrases
    assert "hepatocytes this ratio" not in phrases
    assert "lists cisplatin ratio" not in phrases
    assert "rather stable ratio" not in phrases
    assert "vs SVP ratio" not in phrases
    assert "similar ratio" not in phrases


def test_suffix_metric_phrases_reject_punctuation_delimited_prose():
    phrases = grammar._suffix_metric_phrases(
        "(MMR-deficient par), ratio; (Y141), ratio; cells, ratio; µM, ratio; "
        "protein, ratio; control cells: ratio; effect: ratio; effect:ratio."
    )
    assert not phrases


def test_suffix_metric_phrases_preserve_source_symbols_and_delimiters():
    text = "2MI:CYP3A4 ratio; PS:PG ratio; R:S-SO ratio."
    phrases = grammar._suffix_metric_phrases(text)
    assert "2MI:CYP3A4 ratio" in phrases
    assert "PS:PG ratio" in phrases
    assert "R:S-SO ratio" in phrases
    assert "CYP3A4 ratio" not in phrases
    assert "PG ratio" not in phrases
    assert "S-SO ratio" not in phrases


def test_slash_ratio_parser_preserves_greek_prefixes():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "1.2",
        "support_text": "The β-hydroxybutyrate/acetoacetate ratio was 1.2.",
    }
    assert ("1.2", "β-hydroxybutyrate/acetoacetate ratio") in _pairs(row)


def test_suffix_metric_windows_reject_clause_verbs():
    phrases = grammar._suffix_metric_phrases(
        "Quantification showed MN frequency; 1-OHP had coefficient."
    )
    assert "MN frequency" in phrases
    assert "showed MN frequency" not in phrases
    assert "had coefficient" not in phrases
    assert "1-OHP had coefficient" not in phrases


def test_singleton_values_do_not_cross_unrelated_fraction_or_ratio_units():
    fraction = {
        "source_id": "fixed_mutation",
        "measurement_text": "9",
        "support_text": (
            "The mutant yield was 9. Only a fraction of deletions was large enough "
            "to be easily scored by cytogenetics."
        ),
    }
    partition = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "3",
        "assay_method_and_endpoint": "Partition ratio derived from [FLT]/[P450 3A5] ratio",
        "support_text": (
            "FLT concentrations were plotted against P450 3A5; the partition ratio "
            "was calculated to be 3."
        ),
    }
    assert not _pairs(fraction)
    assert ("3", "partition ratio") in _pairs(partition)
    assert all("FLT" not in unit for _, unit in _pairs(partition))


def test_suffix_metric_does_not_drop_comparison_numerator():
    texts = (
        "HCK+SVP vs SVP ratio 0.226.",
        "The reported 1- to 4-thioether ratio was 1.7.",
        "The gamma-H2AX to H2AX ratio was 0.44.",
        "The lactate to pyruvate ratio was 55.",
    )
    forbidden = {"SVP ratio", "4-thioether ratio", "H2AX ratio", "pyruvate ratio"}
    for text in texts:
        phrases = grammar._suffix_metric_phrases(text)
        assert not phrases.intersection(forbidden)


def test_slash_ratio_parser_rejects_truncated_nested_formula():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "2.2",
        "support_text": "The D2/E2-IsoPs/F2-IsoPs ratio fell to 2.2.",
    }
    assert all("E2-IsoPs/F2-IsoPs" not in unit for _, unit in _pairs(row))


@pytest.mark.parametrize(
    ("measurement", "text", "expected", "forbidden"),
    [
        (
            "4.68",
            "The form II/form I ratio was 4.68.",
            "form II/form I ratio",
            "I ratio",
        ),
        (
            "3.5",
            "The 405/470 nm fluorescence ratio rose to 3.5.",
            "405/470 nm fluorescence ratio",
            "nm fluorescence ratio",
        ),
        (
            "0.85",
            "The 1R,2S:1S,2R enantiomer ratio was 0.85.",
            "1R,2S:1S,2R enantiomer ratio",
            "2R ratio",
        ),
    ],
)
def test_complete_formula_metrics_replace_truncated_suffixes(
    measurement, text, expected, forbidden
):
    pairs = _pairs(
        {
            "source_id": "mutagenicity_mechanism",
            "measurement_text": measurement,
            "support_text": text,
        }
    )
    assert (measurement, expected) in pairs
    assert all(unit != forbidden for _, unit in pairs)


def test_metabolite_roman_numeral_ratio_is_not_truncated():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "1.7",
        "support_text": "The Metabolite II to Metabolite I ratio was 1.7.",
    }
    assert all(unit != "I ratio" for _, unit in _pairs(row))


def test_formula_numbers_do_not_steal_the_linked_measurement():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "15.5",
        "support_text": "The 2-OH/4-OH ratio is 15.5 for CYP1A1.4.",
    }
    assert ("15.5", "2-OH/4-OH ratio") in _pairs(row)


def test_one_named_ratio_can_link_a_coordinated_result_series():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "4.81",
        "support_text": "The 8-OH-dG/dG ratio values were 1.2, 2.4, and 4.81.",
    }
    assert ("4.81", "8-OH-dG/dG ratio") in _pairs(row)


def test_named_ratio_can_link_explicit_anaphoric_change():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.41",
        "support_text": (
            "The I405/I488 fluorescence ratio was 0.10, whereas diamide raised "
            "the ratio to 0.41."
        ),
    }
    assert ("0.41", "I405/I488 fluorescence ratio") in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "assay", "support", "forbidden"),
    [
        (
            "7.31",
            "Total glutathione and GSH/GSSG ratio",
            "Total glutathione 7.31, GSH 5.31, GSSG 2.20, and GSH/GSSG 2.525.",
            "GSH/GSSG ratio",
        ),
        (
            "0.8",
            "GSH/GSSG ratio in hepatocytes",
            "The GSH/GSSG ratio decreased from 4 in control to 1 after exposure.",
            "GSH/GSSG ratio",
        ),
        (
            "2.03",
            "2OHEt:Et ratio",
            "Mixed 1:1 molar ratio with GSH gave a 2OHEt:Et ratio of 2.03.",
            "molar ratio",
        ),
        (
            "11",
            "log metabolic ratio reproducibility",
            "The mean absolute difference in the log metabolic ratio was 11%.",
            "log metabolic ratio",
        ),
    ],
)
def test_inferred_ratios_require_their_own_value_link(
    measurement, assay, support, forbidden
):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": assay,
        "support_text": support,
    }
    assert all(unit != forbidden for _, unit in _pairs(row))


@pytest.mark.parametrize(
    ("measurement", "assay", "support", "forbidden"),
    [
        (
            "0.97",
            "Correlation between BCIG formation rate and estradiol formation rate",
            "The two rates had a correlation coefficient of 0.97.",
            "BCIG formation rate",
        ),
        (
            "3.69",
            "NNK-induced chromosome aberration frequency",
            "Risk was associated with an odds ratio of OR = 3.69.",
            "chromosome aberration frequency",
        ),
        (
            "14.3",
            "Vmax/Km catalytic efficiency and substrate inhibition constant",
            "Catalytic efficiency was Vmax/Km = 14.3; Ki = 117.7.",
            "substrate inhibition constant",
        ),
        (
            "8.09",
            "G2-M phase fraction and G2-M/G1 ratio",
            "The G2-M/G1 ratio was 8.09.",
            "G2-M phase fraction",
        ),
        (
            "20",
            "Reversion frequency and recombination events",
            "The recombination index (RI) was 20.",
            "reversion frequency",
        ),
        (
            "1",
            "Second-order rate constant; log k_GSH reported",
            "The measured log k_GSH was 1.00.",
            "Second-order rate constant",
        ),
        (
            "216",
            "Enzyme activity measured in liver cytosolic fraction",
            "The enzyme activity was 216 in liver cytosol.",
            "liver cytosolic fraction",
        ),
    ],
)
def test_assay_labels_are_not_shared_without_a_value_link(
    measurement, assay, support, forbidden
):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": assay,
        "support_text": support,
    }
    assert all(unit != forbidden for _, unit in _pairs(row))


def test_condition_and_method_fragments_are_not_units():
    malformed = {
        "change ratio",
        "hr control ratio",
        "h ratio",
        "control ratio",
        "in-vitro ratio",
        "measuring synergist ratio",
        "the ratio",
        "vitro ratio",
        "yielding toxic ratio",
        "method (partition ratio)",
        "detoxication index (ratio of CYP2B1/2)",
        "its DNA damage/itself ratio",
    }
    assert not any(grammar._valid_unit_syntax(unit) for unit in malformed)


def test_inferred_relative_ratio_is_filtered_but_authoritative_is_retained():
    inferred = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "11.6",
        "support_text": "The relative ratio of M1 to M4 was 11.6.",
    }
    authoritative = {
        "source_id": "premutagenic_damage",
        "measurement_text": "0.11",
        "unit_text": "relative ratio",
    }
    assert all(unit != "relative ratio" for _, unit in _pairs(inferred))
    assert ("0.11", "relative ratio") in _pairs(authoritative)


@pytest.mark.parametrize(
    ("measurement", "assay", "support", "forbidden"),
    [
        (
            "3",
            "Titration method (partition ratio), derived from FLT/P450 3A5 ratio",
            "The partition ratio was calculated to be 3.",
            "FLT",
        ),
        (
            "290",
            "NC/CYP2D6 molar ratio to estimate the partition ratio",
            "The partition ratio P was around 290.",
            "NC",
        ),
        (
            "280",
            "Partition-ratio study using a 3MI/CYP2F1 molar-ratio plot",
            "The partition ratio was found to be 280.",
            "3MI",
        ),
        (
            "5.4",
            "Partition-ratio titration as a function of the 2MI:CYP3A4 ratio",
            "The partition ratio was calculated to be 5.4.",
            "2MI",
        ),
        (
            "27",
            "Partition-ratio estimation from the BG/P450 molar ratio",
            "The extrapolated partition ratio was approximately 27.",
            "BG",
        ),
    ],
)
def test_partition_result_does_not_inherit_titration_axis(
    measurement, assay, support, forbidden
):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": assay,
        "support_text": support,
    }
    pairs = _pairs(row)
    assert (measurement, "partition ratio") in pairs
    assert all(forbidden not in unit for _, unit in pairs)


def test_secondary_partition_ratio_is_not_bound_to_percent_decrease():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "16",
        "assay_method_and_endpoint": (
            "Decrease in relative % P450 bound after GSH; partition ratio measured as well"
        ),
        "support_text": "GSH led to a 16, 2, and 31% decrease in P450 bound.",
    }
    assert ("16", "partition ratio") not in _pairs(row)


def test_turnover_number_is_not_relabelled_as_partition_ratio():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "4",
        "assay_method_and_endpoint": (
            "Partition-ratio determination from residual CYP2D6 activity"
        ),
        "support_text": (
            "The turnover number (partition ratio + 1) was 4, so the partition "
            "ratio was 3."
        ),
    }
    assert ("4", "partition ratio") not in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "assay", "support", "expected"),
    [
        (
            "32",
            "Partition ratio (molecules of nPX metabolized per molecule of P450 2B1 inactivated)",
            "The partition ratio was found to be 32.",
            "nPX molecules metabolized per P450 2B1 molecule inactivated",
        ),
        (
            "3.8",
            "Partition ratio determination: ratio of total furafylline metabolic consumption (substrate depletion) to loss of P450 1A2 inactivation events",
            "Furafylline was consumed and the partition ratio was 3.8.",
            "furafylline consumed per P450 1A2 lost",
        ),
    ],
)
def test_partition_filter_retains_specific_result_units(
    measurement, assay, support, expected
):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": assay,
        "support_text": support,
    }
    assert (measurement, expected) in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "phrase"),
    [
        ("0.03", "O-6:N-7"),
        ("0.11", "O-6 versus N-7"),
    ],
)
def test_guanine_alkylation_ratio_has_specific_semantic_candidate(measurement, phrase):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "support_text": (
            f"The in-vitro ratio of alkylation at {phrase} of guanine in DNA "
            f"of {measurement}."
        ),
    }
    assert (measurement, "O-6/N-7 guanine alkylation ratio") in _pairs(row)


def test_tr_gsh_formula_recovers_named_ratio_without_assay_prose():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "4.4",
        "assay_method_and_endpoint": "strains yielding toxic ratio TR_GSH",
        "support_text": "TR_GSH = 4.4, where TR_GSH is an EC50 ratio.",
    }
    pairs = _pairs(row)
    assert ("4.4", "TR_GSH ratio") in pairs
    assert all(unit != "yielding toxic ratio" for _, unit in pairs)


@pytest.mark.parametrize(
    ("measurement", "text", "expected"),
    [
        ("40", "The [GSH]/[GSSG] ratio dropped from 300 to 40.", "GSH/GSSG ratio"),
        (
            "0.83",
            "The JC-1 monomer:aggregate ratio increased from 0.64 to 0.83.",
            "JC-1 monomer/aggregate ratio",
        ),
    ],
)
def test_named_ratio_series_links_the_selected_tail(measurement, text, expected):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "support_text": text,
    }
    assert (measurement, expected) in _pairs(row)


def test_named_change_series_keeps_the_selected_tissue_ratio():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "1051",
        "support_text": (
            "For acute 3-day sodium salicylate, liver ratio fell from 1552 to "
            "1051 and brain ratio from 5144 to 2631."
        ),
    }
    assert ("1051", "liver ratio") in _pairs(row)
    assert all(unit != "brain ratio" for _, unit in _pairs(row))


def test_sensor_excitation_value_is_not_a_biochemical_ratio():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.36",
        "assay_method_and_endpoint": (
            "GRX1-roGFP2 sensor measuring GSH:GSSG as the IR405/488 "
            "fluorescence excitation ratio"
        ),
        "support_text": "YS cells had IR405/488 of 0.36 versus 0.22 in controls.",
    }
    assert all("GSH/GSSG" not in unit for _, unit in _pairs(row))


def test_uppercase_is_ratio_abbreviation_is_not_a_prose_stopword():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.1",
        "support_text": "Minimal GSH-adduct formation had GSH-adduct/IS ratio = 0.1.",
    }
    assert ("0.1", "GSH-adduct/IS ratio") in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "text", "forbidden"),
    [
        (
            "6.27",
            "MNPCE frequency was 6.27, with a PCE/NCE ratio of 0.94.",
            "PCE/NCE ratio",
        ),
        (
            "-0.009",
            "TDCCA had coefficient -0.009, unrelated to the sperm Y:X ratio.",
            "sperm Y:X ratio",
        ),
        (
            "258",
            "The 4-ipomeanol/P450 molar ratio gave a turnover number of 258.",
            "4-ipomeanol/P450 molar ratio",
        ),
        (
            "0.1",
            "The O6/7 ratio was 0.1 and the O4/O6 ratio was 0.01.",
            "O4/O6 ratio",
        ),
        (
            "3.1",
            "The measured ratio was 3.1 and the predicted ratio was 3.4.",
            "predicted ratio",
        ),
    ],
)
def test_ratio_units_do_not_cross_metrics(measurement, text, forbidden):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "support_text": text,
    }
    assert all(unit != forbidden for _, unit in _pairs(row))


def test_literal_log_units_are_a_closed_metric_vocabulary():
    prose = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.000402",
        "support_text": "Log phase cells had a mutation frequency of 0.000402.",
    }
    metric = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "4.12",
        "support_text": "The log tail moment was 4.12.",
    }
    assert all(unit.lower() != "log phase" for _, unit in _pairs(prose))
    assert ("4.12", "log tail moment") in _pairs(metric)


def test_k_ratio_list_links_only_values_after_the_named_formula():
    support = (
        "Equation 1 reported n (3.9). Using eq. 2, k_inh/k_p declined in the "
        "order eugenol (7.8) > magnolol (6.8) > curcumin (3.7)."
    )
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "3.7",
        "assay_method_and_endpoint": "k_inh/k_p inhibition/propagation ratio",
        "support_text": support,
    }
    assert ("3.7", "k_inh/k_p ratio") in _pairs(row)
    row["measurement_text"] = "3.9"
    assert ("3.9", "k_inh/k_p ratio") not in _pairs(row)


def test_ice_ratio_requires_the_local_reported_value():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "2.2",
        "assay_method_and_endpoint": "ICE-Δ ratio; values above 2 indicate poisoning",
        "support_text": "Exposure to BQ gave an ICE-Δ of 2.2.",
    }
    assert ("2.2", "ICE-Δ ratio") in _pairs(row)


def test_ratio_of_a_rate_does_not_emit_the_raw_rate():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.96",
        "assay_method_and_endpoint": (
            "DNA synthesis rate expressed as treated-to-control ratio"
        ),
        "support_text": (
            "The treated-to-control ratio for DNA synthesis rate was 0.96."
        ),
    }
    pairs = _pairs(row)
    assert ("0.96", "treated/control DNA-synthesis-rate ratio") in pairs
    assert all(
        unit not in {"DNA synthesis rate", "synthesis rate"} for _, unit in pairs
    )


def test_multivalue_authoritative_unit_stays_with_its_local_value():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "IF 2.54, reduced by 19.6%",
        "unit_text": "induction factor (IF)",
        "support_text": (
            "The induction factors (IF=2.54) were reduced by 19.6% against MNNG."
        ),
    }
    pairs = _pairs(row)
    assert ("2.54", "induction factor") in pairs
    assert ("19.6", "%") in pairs
    assert all(
        "induction factor" not in unit for value, unit in pairs if value == "19.6"
    )


def test_sos_dose_and_scientific_denominator_are_not_results():
    dose = {
        "source_id": "premutagenic_damage",
        "measurement_text": "IF 550 µM: 2.9 (PBS) vs 6.4 (LB)",
        "unit_text": "SOS induction factor",
        "support_text": "IF at 550 µM H2O2 = 2.9; IF = 6.4.",
    }
    binding = {
        "source_id": "premutagenic_damage",
        "measurement_text": "20 for binding of 0.65 molecules per 10^6 bp; 2.5 for binding of 0.1 molecule per 10^6 bp",
        "unit_text": "SOS induction factor",
        "support_text": "The induction factor was about 20; an induction factor of 2.5 was found.",
    }
    assert _pairs(dose) == {
        ("2.9", "SOS induction factor"),
        ("6.4", "SOS induction factor"),
    }
    pairs = _pairs(binding)
    assert {("20", "SOS induction factor"), ("2.5", "SOS induction factor")} <= pairs
    assert all(unit != "10^6 SOS induction factor" for _, unit in pairs)


def test_percent_repaired_requires_a_timed_repair_result():
    supported = {
        "source_id": "premutagenic_damage",
        "measurement_text": "≈10% of εA repaired in 24 h",
        "unit_text": "% repaired",
        "support_text": "Only about 10% of εA was repaired in 24 h.",
    }
    unrelated = {
        "source_id": "premutagenic_damage",
        "measurement_text": "25% persisted; totally repaired by 2 weeks",
        "unit_text": "% repaired",
        "support_text": "25% of DNA SSB persisted and was later totally repaired.",
    }
    assert ("10", "% repaired") in _pairs(supported)
    assert ("25", "% repaired") not in _pairs(unrelated)


def test_unprinted_mutation_rate_is_not_inferred_from_measurement_text():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.000000024",
        "support_text": "Mutation rates were estimated and reported in the source.",
    }
    assert all(unit != "mutation rate" for _, unit in _pairs(row))


@pytest.mark.parametrize(
    ("measurement", "assay", "unit"),
    [
        (
            "0.03",
            "ratio of 203Hg bound to DNA to total cellular 203Hg(II)",
            "203Hg bound to DNA/total cellular 203Hg(II) ratio",
        ),
        (
            "0.01",
            "ratio of 203Hg bound to DNA to total cellular 203Hg(II)",
            "203Hg bound to DNA / total cellular 203Hg ratio",
        ),
        ("6.48", "stability constant log K", "log K"),
    ],
)
def test_declared_metric_without_a_printed_value_is_not_a_pair(
    measurement, assay, unit
):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": assay,
        "support_text": "The named endpoint was discussed without a numeric result.",
    }
    assert (measurement, unit) not in _pairs(row)


def test_bare_scale_and_exponent_scale_are_distinct():
    exponent = {
        "source_id": "premutagenic_damage",
        "measurement_text": "1.95×10^-3",
        "unit_text": "ARP sites",
    }
    bare = {
        "source_id": "premutagenic_damage",
        "measurement_text": "8.12 (×10)",
        "unit_text": "ARP sites",
    }
    assert ("1.95", "10^-3 ARP sites") in _pairs(exponent)
    assert ("1.95", "×10 ARP sites") not in _pairs(exponent)
    assert ("8.12", "×10 ARP sites") in _pairs(bare)
    assert all(measurement != "10" for measurement, _ in _pairs(bare))


def test_inferred_unit_has_its_own_rule_and_source_span():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "1",
        "unit_text": "resistant mutants per viable cell",
        "support_text": "Vancomycin gave a mutation frequency of 1.",
    }
    candidates = grammar.candidates_for_record(row)
    inferred = next(item for item in candidates if item["unit"] == "mutation frequency")
    assert inferred["rule_id"] == "source_phrase_metric.v1"
    assert "mutation frequency" in inferred["evidence"][1]
    assert all(
        "Vancomycin gave a mutation frequency" != item["unit"] for item in candidates
    )


def test_apparent_pka_links_only_a_same_field_reported_value():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "3.8",
        "support_text": (
            "The midpoint represented the apparent pKa of the sulfhydryl. "
            "The value was estimated to be 3.8."
        ),
    }
    assert ("3.8", "apparent pKa") in _pairs(row)
    row["support_text"] = "The selected value was estimated to be 3.8."
    row["assay_method_and_endpoint"] = "Apparent pKa of the sulfhydryl"
    assert ("3.8", "apparent pKa") not in _pairs(row)


def test_dual_declared_formula_ratio_preserves_the_exact_formula():
    formula = "1U/(1X+1U)"
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.44",
        "assay_method_and_endpoint": f"Caffeine metabolite molar ratio {formula}",
        "support_text": f"The metabolite ratio {formula}; means were 0.43 and 0.44.",
    }
    assert ("0.44", formula) in _pairs(row)
    assert all(unit != formula[:-1] for _, unit in _pairs(row))
    row["support_text"] = "The reported mean was 0.44."
    assert ("0.44", formula) not in _pairs(row)


def test_simple_half_life_descriptor_uses_the_adjacent_time_unit():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "about 4 h",
        "unit_text": "half-life",
        "support_text": "The lesion half-life was about 4 h.",
    }
    assert ("4", "h") in _pairs(row)
    row["measurement_text"] = ">4 h"
    assert ("4", "h") not in _pairs(row)
    row["measurement_text"] = "20 min versus 4 h"
    assert ("4", "h") not in _pairs(row)


def test_grouped_focal_count_keeps_normalized_value_and_source_unit():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "60,000 (saturation)",
        "unit_text": "protein-associated SSBs per cell",
        "support_text": "Protein-associated SSBs saturated at 60,000 per cell.",
    }
    candidate = next(
        item
        for item in grammar.candidates_for_record(row)
        if item["unit"] == row["unit_text"]
    )
    assert candidate["measurement"] == "60000"
    assert "60,000" in candidate["evidence"][0]


def test_grouped_focal_count_rejects_lists_bounds_and_inexact_units():
    base = {
        "source_id": "premutagenic_damage",
        "unit_text": "tail moment",
        "support_text": "Values were 128, 191, 213, 129, 185 and 132.",
    }
    for measurement in ("128,191,213,129,185,132", "1,2", "1,000, 2,000"):
        assert not grammar.candidates_for_record(
            {**base, "measurement_text": measurement}
        )
    bounded = {
        **base,
        "measurement_text": ">2,000",
        "support_text": "The value was greater than 2,000.",
    }
    assert not grammar.candidates_for_record(bounded)
    assert not grammar._numeric_evidence(
        {
            "measurement_text": "60,000",
            "unit_text": "source exact unit",
            "support_text": "The result was 60,000.",
        },
        "60000",
        "different unit",
    )


@pytest.mark.parametrize(
    ("measurement", "unit", "support"),
    [
        ("from 3,200 µM", "% tail DNA", "Tail DNA increased from 3,200 µM."),
        ("1.57-fold (1,750 ppm)", "fold of control", "The dose was 1,750 ppm."),
    ],
)
def test_grouped_authoritative_value_rejects_exposure_doses(measurement, unit, support):
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": measurement,
        "unit_text": unit,
        "support_text": support,
    }
    assert not grammar.candidates_for_record(row)


@pytest.mark.parametrize(
    ("measurement", "unit", "support"),
    [
        ("exceeding 100,000", "molecules per genome", "Amounts exceeded 100,000."),
        ("over 400,000", "SSBs per cell", "There were over 400,000 SSBs per cell."),
        (
            "≈6,000% higher than control",
            "percent of control",
            "The result was ≈6,000% higher than control.",
        ),
    ],
)
def test_grouped_authoritative_value_rejects_bounds_and_relative_transforms(
    measurement, unit, support
):
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": measurement,
        "unit_text": unit,
        "support_text": support,
    }
    assert not grammar.candidates_for_record(row)


@pytest.mark.parametrize(
    ("measurement", "unit", "support"),
    [
        ("1/130,000", "bases", "The frequency was 1/130,000 bases."),
        ("1 per 10,000", "nucleotides", "There was 1 per 10,000 nucleotides."),
        (
            "2 lesions per 100,000 bp",
            "lesions per 100,000 bp",
            "There were 2 lesions per 100,000 bp.",
        ),
    ],
)
def test_grouped_authoritative_value_rejects_denominators(measurement, unit, support):
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": measurement,
        "unit_text": unit,
        "support_text": support,
    }
    assert all(value not in {"10000", "100000", "130000"} for value, _ in _pairs(row))


def test_grouped_authoritative_value_rejects_cross_metric_fold_link():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "3-fold; 4,700 adducts per 2-fold",
        "unit_text": "fold increase",
        "support_text": "The response had 4,700 adducts per 2-fold increase.",
    }
    assert ("4700", "fold increase") not in _pairs(row)


@pytest.mark.parametrize(
    "unit",
    [
        "adducts per cell (denominator not explicitly stated)",
        "adducts (scale unresolved)",
    ],
)
def test_grouped_authoritative_value_rejects_explicitly_incomplete_unit(unit):
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "14,000 adducts",
        "unit_text": unit,
        "support_text": "The sample contained 14,000 adducts.",
    }
    assert all(candidate[1] != unit for candidate in _pairs(row))


def test_grouped_authoritative_value_rejects_table_list_token():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "126,135",
        "unit_text": "cross-link index x 10^3",
        "support_text": "Table values 126,135 at low dose and 268,410 at high dose.",
    }
    assert not grammar.candidates_for_record(row)


def test_dna_length_prefix_links_only_the_exact_authoritative_unit():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "peak about 0.5 lesions/10 kb after 1 h",
        "unit_text": "lesions/10 kb of DNA",
    }
    assert ("0.5", "lesions/10 kb of DNA") in _pairs(row)
    assert ("0.5", "lesions/10 kb") not in _pairs(row)
    row["measurement_text"] = ">0.5 lesions/10 kb"
    assert not grammar._dna_length_pairs(row, ["0.5"])


def test_stereochemical_ratio_requires_a_direct_coefficient_to_one():
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "1.29",
        "assay_method_and_endpoint": "Epoxidation measured as the R:S ratio",
        "support_text": "Rat values were 1.56:1 (males) and 1.29:1 (females).",
    }
    assert ("1.29", "R:S ratio") in _pairs(row)
    row["measurement_text"] = "1.2"
    assert ("1.2", "R:S ratio") not in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "support", "unit"),
    [
        (
            "0.26",
            "The recombination rate for condition A was 0.26 (baseline 0.36).",
            "recombination rate",
        ),
        (
            "0.59",
            "The recombination rate was 2.5 without inhibitor and fell to 0.59 with inhibitor.",
            "recombination rate",
        ),
        (
            "0.0001",
            "Rates of reversion to marker negativity were approximately 10^-4.",
            "rates of reversion to marker negativity",
        ),
        (
            "1.6",
            "The SCE-inducing potency (SCEIP) in the test cells was 1.6.",
            "SCE-inducing potency (SCEIP)",
        ),
        ("0.000091", "The GCR rate was 9.1 [4.8-13.4] x 10^-5.", "GCR rate"),
        (
            "0.000503",
            "The glutamine-tRNA-suppressor mutation-frequency constant m = 5.03 x 10^-4.",
            "glutamine-tRNA-suppressor mutation-frequency constant m",
        ),
        (
            "0.000002",
            "The highest 6FT-resistant mutant frequency was 2 x 10^-6.",
            "6FT-resistant mutant frequency",
        ),
        (
            "0.0000025",
            "The mean hprt mutation frequency in exposed cells was 2.5 x 10^-6.",
            "mean hprt mutation frequency",
        ),
        (
            "0.0000025",
            "The mean hprt mutation frequency was 2.5 x 10^-6 versus 1.03 x 10^-6 in controls.",
            "mean hprt mutation frequency",
        ),
        (
            "0.0000041",
            "The spontaneous rate was 5.6 x 10^-7 and the MMS-induced mutation rate was 4.1 x 10^-6.",
            "MMS-induced mutation rate",
        ),
        (
            "0.0000028",
            "The frequency of double recombinants was 2 x 10^-7, 1.2 x 10^-6, and 2.8 x 10^-6 for A, B, and C, respectively.",
            "frequency of double recombinants",
        ),
        (
            "2.1",
            "The AFX-inducing potency (AFXIP) in hamster cells was 2.1.",
            "AFX-inducing potency (AFXIP)",
        ),
        (
            "0.0000003",
            "The 8AG-resistant mutant frequency being 3 x 10^-7.",
            "8AG-resistant mutant frequency",
        ),
        (
            "0.00000007",
            "The mean can1 mutation frequency was 7 x 10^-8.",
            "mean can1 mutation frequency",
        ),
        (
            "0.0000008",
            "The UV-induced mutation rate was 8 x 10^-7.",
            "UV-induced mutation rate",
        ),
        (
            "0.00042",
            "The lysine-tRNA-suppressor mutation-frequency constant k = 4.2 x 10^-4.",
            "lysine-tRNA-suppressor mutation-frequency constant k",
        ),
        (
            "0.0000072",
            "The GCR rate after treatment was 7.2 [6.1-8.3] x 10^-6.",
            "GCR rate",
        ),
        (
            "0.000004",
            "Rates of mutation to rifampicin resistance were about 4 x 10^-6.",
            "rates of mutation to rifampicin resistance",
        ),
        (
            "0.000003",
            "The frequency of triple recombinants was 1 x 10^-6 and 3 x 10^-6 for A and B, respectively.",
            "frequency of triple recombinants",
        ),
        (
            "0.74",
            "The recombination rate for condition Z was 0.74 (baseline 0.2).",
            "recombination rate",
        ),
    ],
)
def test_v4_fixed_mutation_shapes_are_exact_and_grounded(measurement, support, unit):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": measurement,
        "support_text": support,
    }
    candidates = grammar.candidates_for_record(row)
    match = next(
        item
        for item in candidates
        if (item["measurement"], item["unit"]) == (measurement, unit)
    )
    assert len(match["evidence"]) == 2
    if "10" in support:
        assert not any(item["unit"].startswith("10^") for item in candidates)


@pytest.mark.parametrize("verb", ["fell", "dropped", "rose", "increased", "decreased"])
def test_v4_recombination_change_verbs_link_only_the_changed_value(verb):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.4",
        "support_text": f"The recombination rate was 2.0 and {verb} to 0.4.",
    }
    assert ("0.4", "recombination rate") in _pairs(row)


def test_v4_keeps_many_to_one_scientific_spellings_unchanged():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.00008",
        "support_text": (
            "The reversion frequency was 8.0 x 10^-5 initially and "
            "8 x 10^-5 after 2 days."
        ),
    }
    pairs = _pairs(row)
    assert ("8.0", "10^-5 reversion frequency") in pairs
    assert ("8", "10^-5 reversion frequency") in pairs
    assert ("0.00008", "reversion frequency") not in pairs


@pytest.mark.parametrize(
    ("measurement", "row", "unit"),
    [
        (
            "1",
            {
                "support_text": "The ratio of R- to S-enantiomers of epoxide was 4 in mice and 1 in rats."
            },
            "ratio of R- to S-enantiomers of epoxide",
        ),
        (
            "7",
            {"support_text": "The reaction produced seven different peptide adducts."},
            "different peptide adducts",
        ),
        (
            "0.16",
            {"support_text": "The cytochrome P-450 induction index was 0.16."},
            "cytochrome P-450 induction index",
        ),
        (
            "1200",
            {"support_text": "The competition factor was 1.2 x 10^3."},
            "competition factor",
        ),
        (
            "0.1193",
            {
                "support_text": "Extinction coefficient was 119.3; absorbance A = 0.1193 at 560 nm."
            },
            "absorbance A",
        ),
        (
            "0.9",
            {
                "assay_method_and_endpoint": "QPCR measuring DNA lesions in a 2.7 kb fragment",
                "support_text": "QPCR measured DNA lesions in a 2.7 kb fragment.",
            },
            "DNA lesions in a 2.7 kb fragment",
        ),
        (
            "0.539",
            {"support_text": "An association was observed, with R = 0.539."},
            "R",
        ),
    ],
)
def test_v4_mechanism_shapes_are_exact_and_grounded(measurement, row, unit):
    source = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        **row,
    }
    candidate = next(
        item for item in grammar.candidates_for_record(source) if item["unit"] == unit
    )
    assert candidate["measurement"] == measurement
    assert candidate["evidence"] == grammar._evidence(source, measurement, unit)


@pytest.mark.parametrize("measurement", ["3.9", "1.4"])
def test_v4_vmax_comparison_links_each_reported_group(measurement):
    unit = "Vmax ratio of epoxide hydrolase (detoxifying) versus monooxygenase (bioactivating) activity"
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": unit,
        "support_text": "Vmax ratios for epoxide hydrolase versus monooxygenase are 3.9 in rat and 1.4 in mouse.",
    }
    assert (measurement, unit) in _pairs(row)


@pytest.mark.parametrize(
    ("row", "forbidden"),
    [
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "0.000001",
                "support_text": "The mutation rate was 0.2 after exposure at 1 x 10^-6 M.",
            },
            ("0.000001", "mutation rate"),
        ),
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "0.000001",
                "support_text": "The mutation frequency was 0.2 and exposure was 1 x 10^-6.",
            },
            ("0.000001", "mutation frequency"),
        ),
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "1200",
                "support_text": "The competition factor was 2.1. A dose of 1.2 x 10^3 mg/kg was used.",
            },
            ("1200", "competition factor"),
        ),
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "1200",
                "support_text": "The competition factor was >1.2 x 10^3.",
            },
            ("1200", "competition factor"),
        ),
    ],
)
def test_v4_scientific_links_reject_non_results(row, forbidden):
    assert forbidden not in _pairs(row)


@pytest.mark.parametrize(
    ("row", "forbidden"),
    [
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "7.2",
                "support_text": "At pH 7.2, the recombination rate was 0.26.",
            },
            ("7.2", "recombination rate"),
        ),
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "0.36",
                "support_text": "The recombination rate was 0.26 (baseline 0.36).",
            },
            ("0.36", "recombination rate"),
        ),
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "4.8",
                "support_text": "The GCR rate was 9.1 [4.8-13.4] x 10^-5.",
            },
            ("4.8", "GCR rate"),
        ),
        (
            {
                "source_id": "fixed_mutation",
                "measurement_text": "0.2",
                "support_text": "Mutation rate was 0.1 and recombination frequency was 0.2.",
            },
            ("0.2", "mutation rate"),
        ),
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "560",
                "support_text": "Absorbance at 560 nm was measured.",
            },
            ("560", "absorbance A"),
        ),
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "7",
                "support_text": "The exposure lasted seven days.",
            },
            ("7", "different peptide adducts"),
        ),
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "0.5",
                "assay_method_and_endpoint": "Association assay",
                "support_text": "Group R = 0.5.",
            },
            ("0.5", "R"),
        ),
        (
            {
                "source_id": "mutagenicity_mechanism",
                "measurement_text": "0.427",
                "support_text": "Regression R2 = 0.427; the correlation was R = 0.654.",
            },
            ("0.427", "correlation coefficient"),
        ),
    ],
)
def test_v4_named_links_reject_conditions_and_cross_metrics(row, forbidden):
    assert forbidden not in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "support", "forbidden"),
    [
        (
            "2.6",
            (
                "Vulpis (1984) exposed human lymphocytes to HTO at subacute dose rates "
                "(doses delivered within 2.5 h) and reported a tritium RBE of 2.6 at "
                "the smallest dose (0.25 Gy), decreasing with increasing dose."
            ),
            r"dose\s+rates?",
        ),
        (
            "1.49",
            (
                "Morimoto et al. (1989) obtained an RBE for chromosome aberrations in "
                "lymphocytes exposed to HTO at subacute dose rates (1-2 Gy per hour) "
                "of 1.49 ± 0.21."
            ),
            r"dose\s+rates?",
        ),
        (
            "0.427",
            (
                "Regression gave a correlation coefficient R=0.654 (R2=0.427), "
                "P=0.0002, the strongest association."
            ),
            r"(?:correlation\s+)?coefficient",
        ),
    ],
)
def test_v4_actual_rows_do_not_cross_condition_or_statistic_values(
    measurement, support, forbidden
):
    row = {
        "source_id": "fixed_mutation" if "RBE" in support else "mutagenicity_mechanism",
        "measurement_text": measurement,
        "support_text": support,
    }
    assert all(not re.search(forbidden, unit, re.IGNORECASE) for _, unit in _pairs(row))


@pytest.mark.parametrize(
    ("measurement", "support", "forbidden"),
    [
        (
            "1.04",
            (
                "Dry cleaners had an association between CA frequency, employment "
                "duration, and frequency of exposure, as well as increased MN and "
                "DNA damage compared with controls (CA: 1.04 vs. 0.59, P = 0.005)."
            ),
            ("1.04", "frequency of exposure"),
        ),
        (
            "0.33",
            (
                "Micronucleus (MN) frequency per 1000 cells was elevated in "
                "attendants. Across all subjects (n=71), Spearman's correlation "
                "showed MN frequency was associated with benzene exposure "
                "(r = 0.33, p < 0.01)."
            ),
            ("0.33", "MN frequency"),
        ),
        (
            "6.7",
            (
                "At 100 mg/kg the average frequency of micronucleated reticulocytes "
                "induced by compound 3b was 6.7 ± 1.8, while HU showed 33.7 ± 10.7."
            ),
            ("6.7", "frequency of micronucleated"),
        ),
        (
            "0.00000875",
            "A burst of 875 x 10^-8 mutations occurred. Later, mutation rate was compared.",
            ("0.00000875", "mutation rate"),
        ),
    ],
)
def test_v4_rejected_fixed_row_shapes_stay_excluded(measurement, support, forbidden):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": measurement,
        "support_text": support,
    }
    assert forbidden not in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "support", "unit"),
    [
        (
            "24",
            "The recombination rate was assessed and time was 24 hours.",
            "recombination rate",
        ),
        (
            "20",
            "The UV-induced mutation rate was 20-fold above control.",
            "UV-induced mutation rate",
        ),
        ("0.000091", "The GCR rate was >9.1 x 10^-5.", "GCR rate"),
        (
            "1.03",
            "The mean hprt mutation frequency was 2.5 versus 1.03 in controls.",
            "mean hprt mutation frequency",
        ),
        (
            "0.2",
            "The recombination rate was 0.1 and the GCR rate was 0.2.",
            "recombination rate",
        ),
    ],
)
def test_v4_fixed_readouts_reject_conditions_relatives_and_cross_metrics(
    measurement, support, unit
):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": measurement,
        "support_text": support,
    }
    assert (measurement, unit) not in _pairs(row)


@pytest.mark.parametrize(
    "separator",
    [
        ". ",
        "; ",
        "! ",
        "? ",
        ": ",
        " — ",
        "\n",
        ".",
        ";",
        "!",
        "?",
        '." ',
        ".) ",
    ],
)
@pytest.mark.parametrize(
    ("prefix", "unit", "assay"),
    [
        (
            "The ratio of R- to S-enantiomers of epoxide was 4 in mice",
            "ratio of R- to S-enantiomers of epoxide",
            "",
        ),
        (
            "Vmax ratios for epoxide hydrolase versus monooxygenase are 3.9 in rat",
            (
                "Vmax ratio of epoxide hydrolase (detoxifying) versus "
                "monooxygenase (bioactivating) activity"
            ),
            (
                "Vmax ratio of epoxide hydrolase (detoxifying) versus "
                "monooxygenase (bioactivating) activity"
            ),
        ),
    ],
)
def test_v4_comparison_scan_stops_at_nondecimal_boundaries(
    separator, prefix, unit, assay
):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": "0.5",
        "assay_method_and_endpoint": assay,
        "support_text": f"{prefix}{separator}Absorbance was 0.5 in cells.",
    }
    assert ("0.5", unit) not in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "support"),
    [
        (
            "0.5",
            "The ratio of R- to S-enantiomers of epoxide was 4 in mice and 0.5 in rats.",
        ),
        (
            "0.5",
            "The ratio of R- to S-enantiomers of epoxide was 4 in mice; and 0.5 in rats.",
        ),
        (
            "0.5",
            "The ratio of R- to S-enantiomers of epoxide was 4 in mice;and 0.5 in rats.",
        ),
        (
            "0.5",
            (
                "The ratio of R- to S-enantiomers of epoxide was 4 in mice; "
                "and about 0.5 in rats."
            ),
        ),
        (
            "0.5",
            "The ratio of R- to S-enantiomers of epoxide was 4 in mice and\n0.5 in rats.",
        ),
        (
            "0.5",
            '"The ratio of R- to S-enantiomers of epoxide was 4 in mice and 0.5 in rats."',
        ),
        (
            "0.5",
            (
                "The ratio of R- to S-enantiomers of epoxide was 4 in mice "
                "(control) and 0.5 in rats (treated)."
            ),
        ),
        (
            "0.6",
            "The ratio of R- to S-enantiomers of epoxide was .5 in mice and .6 in rats.",
        ),
        (
            "0.5",
            (
                "The ratio of R- to S-enantiomers of epoxide was 4 in "
                "S. cerevisiae and 0.5 in E. coli."
            ),
        ),
        (
            "0.5",
            (
                "The ratio of R- to S-enantiomers of epoxide was 4 in "
                "Smith et al. controls and 0.5 in treated rats."
            ),
        ),
        (
            "0.5",
            (
                "The ratio of R- to S-enantiomers of epoxide was 4 in mice "
                "(i.e. controls) and 0.5 in rats."
            ),
        ),
        (
            "0.5",
            (
                "The ratio of R- to S-enantiomers of epoxide was 4 in mice;\n"
                " and 0.5 in rats."
            ),
        ),
    ],
)
def test_v4_comparison_scan_preserves_one_source_series(measurement, support):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "support_text": support,
    }
    assert (measurement, "ratio of R- to S-enantiomers of epoxide") in _pairs(row)


@pytest.mark.parametrize(
    "subject",
    [
        "cell survival",
        "viability",
        "cytotoxicity",
        "absorbance",
        "growth",
        "cell death",
        "optical density",
        "temperature",
        "pH",
        "dose",
        "the cytotoxic response",
        "cell proliferation",
        "the percentage of viable bacterial cells",
    ],
)
def test_v4_fixed_readout_rejects_a_new_result_subject(subject):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.4",
        "support_text": f"The recombination rate was assessed and {subject} was 0.4.",
    }
    assert ("0.4", "recombination rate") not in _pairs(row)


@pytest.mark.parametrize(
    "bridge",
    [
        "was assessed, but",
        "was assessed while",
        "was assessed whereas",
        "was assessed;",
    ],
)
def test_v4_fixed_readout_rejects_new_result_clauses(bridge):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.4",
        "support_text": f"The recombination rate {bridge} cytotoxicity was 0.4.",
    }
    assert ("0.4", "recombination rate") not in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "support", "unit"),
    [
        ("0.4", "The recombination rate was 0.4.", "recombination rate"),
        (
            "0.4",
            "The recombination rate for strains A and B was 0.4.",
            "recombination rate",
        ),
        (
            "0.4",
            "The recombination rate was assessed and the rate was 0.4.",
            "recombination rate",
        ),
        (
            "0.4",
            "The recombination rate was assessed and it was 0.4.",
            "recombination rate",
        ),
        (
            "0.4",
            "The recombination rate was assessed and its value was 0.4.",
            "recombination rate",
        ),
        (
            "0.4",
            "The recombination rate was assessed and its rate was 0.4.",
            "recombination rate",
        ),
        (
            "0.000003",
            (
                "The frequency of triple recombinants was 1 x 10^-6 and 3 x 10^-6 "
                "for A and B, respectively."
            ),
            "frequency of triple recombinants",
        ),
    ],
)
def test_v4_fixed_readout_preserves_direct_and_series_links(measurement, support, unit):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": measurement,
        "support_text": support,
    }
    assert (measurement, unit) in _pairs(row)


@pytest.mark.parametrize(
    "suffix",
    [
        "°C",
        "° C",
        "C",
        "F",
        "K",
        "degrees C",
        "degrees Celsius",
        "percent",
        "%",
        "-fold",
        "fold",
        "times",
        "of control",
        "percent of control",
        "% of control",
        "fold of control",
        "times control",
        "of the control",
        "mg/kg",
        "hours",
        "days",
        "pH units",
        "dose units",
    ],
)
def test_v4_scientific_carriers_are_not_readout_results(suffix):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "37",
        "support_text": f"The mutation rate was recorded at 3.7 x 10^1 {suffix}.",
    }
    assert not _pairs(row)


@pytest.mark.parametrize(
    "support",
    [
        "The mutation rate was 2.0 ± 3.7 x 10^1.",
        "The mutation rate was 2.0; standard error was 3.7 x 10^1.",
        "The mutation rate was 2.0; SE = 3.7 x 10^1.",
        "The mutation rate was 2.0; the 95% confidence interval was 3.7 x 10^1 to 4.2 x 10^1.",
        "The mutation rate was 2.0 with a lower bound of 3.7 x 10^1.",
        "The mutation rate was 2.0 with upper confidence limit 3.7 x 10^1.",
        "The mutation rate was 2.0 (95% CI 3.7 x 10^1 to 4.2 x 10^1).",
        "The mutation rate was <3.7 x 10^1.",
        "The mutation rate was greater than 3.7 x 10^1.",
        "The mutation rate was lower than 3.7 x 10^1.",
        "The mutation rate was higher than 3.7 x 10^1.",
        "The mutation rate was below 3.7 x 10^1.",
        "The mutation rate was above 3.7 x 10^1.",
        "The mutation rate was up to 3.7 x 10^1.",
        "The mutation rate ranged from 2.0 to 3.7 x 10^1.",
        "The mutation rate was between 2.0 and 3.7 x 10^1.",
        "The mutation rate was 2.0 (baseline 3.7 x 10^1).",
    ],
)
def test_v4_scientific_bounds_and_errors_are_not_results(support):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "37",
        "support_text": support,
    }
    assert not _pairs(row)


def test_v4_actual_lower_than_bound_is_not_an_exact_point():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "0.00000001",
        "support_text": (
            "No resistant mutants were selected, with mutation frequency "
            "lower than 1e-8, as tabulated in Table 5."
        ),
    }
    assert not _pairs(row)


def test_v4_scientific_range_rejects_its_first_endpoint():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "20",
        "support_text": "The mutation rate ranged from 2.0 x 10^1 to 3.7 x 10^1.",
    }
    assert not _pairs(row)


def test_v4_measurement_bound_overrides_a_safe_scientific_support_spelling():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "<37",
        "support_text": "The mutation rate was 3.7 x 10^1.",
    }
    assert not _pairs(row)


def test_v4_scientific_condition_rejects_a_coefficient_target():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "3.7",
        "support_text": "The mutation rate was recorded at 3.7 x 10^-1 mg/L.",
    }
    assert not _pairs(row)


@pytest.mark.parametrize(
    "support",
    [
        "The mutation rate was determined after 3.7 x 10^-1 passages.",
        "The mutation rate was quantified at 3.7 x 10^-1 cells.",
    ],
)
def test_v4_scientific_source_role_rejects_coefficient_target(support):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "3.7",
        "support_text": support,
    }
    assert not _pairs(row)


@pytest.mark.parametrize(
    "support",
    [
        "At 3.7 x 10^-1 mg/L, the mutation rate was 3.7.",
        "The mutation rate was 3.7. Exposure was 3.7 x 10^-1 mg/L.",
    ],
)
def test_v4_scientific_condition_preserves_a_distinct_plain_result(support):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "3.7",
        "support_text": support,
    }
    pairs = _pairs(row)
    assert ("3.7", "mutation rate") in pairs
    assert all(not unit.startswith("10^") for _, unit in pairs)


def test_v4_scientific_result_keeps_a_coefficient_target_scale():
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": "3.7",
        "support_text": "The mutation rate was 3.7 x 10^-1.",
    }
    assert ("3.7", "10^-1 mutation rate") in _pairs(row)


@pytest.mark.parametrize(
    ("measurement", "support"),
    [
        ("37", "The mutation rate was 3.7 x 10^1 (95% CI 32-42)."),
        ("37", "The mutation rate was 3.7 x 10^1 ± 4.2."),
        ("0.37", "The mutation rate was 3.7 x 10^-1 at 37 °C."),
        ("0.37", "The mutation rate was 3.7 x 10^-1 at pH 7."),
        ("0.37", "The mutation rate was 3.7 x 10^-1 after 24 hours."),
        ("0.37", "The mutation rate was 3.7 x 10^-1 at 5 mg/kg."),
    ],
)
def test_v4_scientific_results_allow_following_context(measurement, support):
    row = {
        "source_id": "fixed_mutation",
        "measurement_text": measurement,
        "support_text": support,
    }
    assert (measurement, "mutation rate") in _pairs(row)


def test_v4_r_evidence_is_mechanism_scoped():
    row = {
        "source_id": "premutagenic_damage",
        "measurement_text": "0.23",
        "unit_text": "R",
        "support_text": "A regression reported R = 0.23.",
    }
    evidence = grammar._evidence(row, "0.23", "R")
    assert evidence[1].startswith("authoritative_unit_text.v1:")
    mechanism = {**row, "source_id": "mutagenicity_mechanism"}
    assert grammar._evidence(mechanism, "0.23", "R")[1].startswith(
        "source_phrase_metric.v1:"
    )


@pytest.mark.parametrize(
    "measurement",
    ["0.9 to 1.1", "relative 0.9", "0.9; 1.1"],
)
def test_v4_mechanism_dna_descriptor_requires_one_absolute_scalar(measurement):
    row = {
        "source_id": "mutagenicity_mechanism",
        "measurement_text": measurement,
        "assay_method_and_endpoint": "DNA lesions in a 2.7 kb fragment",
        "support_text": "The assay measured DNA lesions in a 2.7 kb fragment.",
    }
    assert all(item[1] != "DNA lesions in a 2.7 kb fragment" for item in _pairs(row))


def test_new_semantic_rules_are_limited_to_their_reviewed_sources():
    wrong = {"source_id": "fixed_mutation", "measurement_text": "1"}
    assert not grammar._apparent_pka_pairs(wrong, ["1"])
    assert not grammar._formula_ratio_pairs(wrong, ["1"])
    assert not grammar._half_life_time_pairs(wrong, ["1"])
    assert not grammar._dna_length_pairs(wrong, ["1"])
    assert not grammar._stereochemical_ratio_pairs(wrong, ["1"])


def test_frozen_gold_pair_oracle_meets_source_gates():
    cases = [json.loads(line) for line in GOLD.read_text().splitlines()][1:]
    totals, hits = Counter(), Counter()
    for case in cases:
        if case["expected"]["status"] != "ok":
            continue
        source = case["source_id"]
        expected = case["expected"]["measurements"][0]
        totals[source] += 1
        pairs = _pairs(case["input"])
        measurement, unit = expected["measurement"], expected["unit"]
        scale = re.fullmatch(r"10\^([+-]?\d+) (.+)", unit)
        targets = {Decimal(value) for value in grammar._value_variants(case["input"])}
        if scale and Decimal(measurement) not in targets:
            measurement = grammar._plain(
                Decimal(measurement) * Decimal(10) ** int(scale.group(1))
            )
            unit = scale.group(2)
        hits[source] += any(
            _same_number(candidate_measurement, measurement) and candidate_unit == unit
            for candidate_measurement, candidate_unit in pairs
        )
    assert sum(hits.values()) >= 146
    assert all(hits[source] / total >= 0.90 for source, total in totals.items())


def test_every_gold_candidate_set_is_bounded_and_contract_valid():
    cases = [json.loads(line) for line in GOLD.read_text().splitlines()][1:]
    for case in cases:
        candidates = grammar.candidates_for_record(case["input"])
        assert len(candidates) <= grammar.MAX_CANDIDATES
        assert load_candidates(json.dumps(candidates)) == candidates
