"""Corpus-level correctness invariants for the unit normalizer.

Self-contained and hermetic: the corpus is a curated set of real Starling unit forms
(distilled from the audit over data/starling_data/**) plus hand-written equivalence and
ambiguity sets. These assert the properties that guarantee we never *incorrectly* normalize
-- idempotency, structural cleanliness, no cross-dimension collisions, correct must-merge
equivalences, and preserved must-NOT-merge distinctions.
"""

from tools.chembl_tool.common.units import canonicalize_unit, clean_unit

# Real Starling forms spanning every unit family seen across the six on-disk unit columns.
CORPUS = [
    # bioavailability: exposure / AUC / concentration / time
    "ng/mL", "ng/ml", "ng per mL", "ng·h/mL", "h·ng/mL", "ng·hr/mL", "ng.h/mL", "ng*h/ml",
    "µg/mL", "µg/ml", "μg/ml", "mcg/mL", "µg·h/mL", "µg·h/L", "mg/L", "mg/liter", "µg/L",
    "h", "hr", "hours", "min", "minutes", "s", "sec", "%", "fold", "µM", "nM", "mM", "M",
    # clearance / metabolic
    "mL/min/kg", "ml/min/kg", "µL/min/mg", "µl/min/mg", "µL/min/mg protein",
    "µL/min/10^6 cells", "mL/min", "L/h", "L/h/kg", "h^-1", "h-1", "nmol/min/mg protein",
    # permeability (bbb + fa) with every scientific-notation variant
    "cm/s", "cm/sec", "cm·s-1", "cm s-1", "cm s^-1", "cm s⁻¹", "nm/s", "nm/sec", "cm/h", "cm/hr",
    "10^-6 cm/s", "×10^-6 cm/s", "x10^-6 cm/s", "10⁻⁶ cm/s", "10^{-6} cm/s",
    "10^-6 cm s^-1", "×10^{-6} cm s^{-1}", "×10-6 cm/s",
    # skin exposure: areic dose / rate / area
    "µg/cm²", "µg/cm2", "μg/cm2", "µg/cm²/h", "µg cm-2 h-1", "µg cm⁻² h⁻¹", "µg/cm2/h",
    "ng/cm²", "mg/cm²", "nmol/cm²/h", "µg/g", "cm/h", "µg", "mg",
    # sensitization / dimensionless
    "ppm", "dimensionless", "unitless", "ratio", "% w/v",
]


def test_clean_unit_is_idempotent():
    for raw in CORPUS:
        cleaned = clean_unit(raw)
        assert cleaned is not None
        assert clean_unit(cleaned) == cleaned


def test_canonicalize_agrees_with_clean_and_is_stable_through_the_cleaned_form():
    for raw in CORPUS:
        result = canonicalize_unit(raw)
        assert result.cleaned == clean_unit(raw)
        # Re-canonicalizing the cleaned form must yield the same key and dimension.
        recanon = canonicalize_unit(result.cleaned)
        assert recanon.canonical == result.canonical
        assert recanon.dimension == result.dimension


def test_cleaned_output_is_structurally_clean():
    for raw in CORPUS:
        cleaned = clean_unit(raw)
        assert cleaned  # never returns ""
        assert "μ" not in cleaned  # Greek mu (U+03BC) is always folded to the micro sign
        assert cleaned == cleaned.strip()
        assert "  " not in cleaned
        for operator in ("/", "·", "*"):
            assert f" {operator}" not in cleaned and f"{operator} " not in cleaned


def test_canonical_key_maps_to_a_single_dimension_across_the_corpus():
    # The automated over-merge guard: no canonical key may carry two dimensions.
    by_canonical: dict[str, set] = {}
    for raw in CORPUS:
        result = canonicalize_unit(raw)
        if not result.canonical:
            continue
        by_canonical.setdefault(result.canonical, set()).add(result.dimension)
    collisions = {c: dims for c, dims in by_canonical.items() if len(dims) > 1}
    assert not collisions, f"canonical keys with multiple dimensions: {collisions}"


# Forms that MUST collapse to one canonical key + one dimension.
MUST_MERGE = [
    ["ng/ml", "ng/mL", "ng·mL-1", "ng.ml-1", "ng per mL"],
    ["cm/s", "cm/sec", "cm·s-1", "cm s-1", "cm s^-1"],
    ["ng·h/mL", "h·ng/mL", "ng.hr.ml-1", "ng*h/ml"],
    ["10^-6 cm/s", "×10^-6 cm/s", "x10^-6 cm/s", "10⁻⁶ cm/s", "10^{-6} cm/s"],
    ["µg/cm²/h", "µg cm-2 h-1", "µg cm⁻² h⁻¹", "µg/cm2/h"],
]


def test_must_merge_equivalence_classes():
    for group in MUST_MERGE:
        canonicals = {canonicalize_unit(f).canonical for f in group}
        dimensions = {canonicalize_unit(f).dimension for f in group}
        assert len(canonicals) == 1, f"{group} -> {canonicals}"
        assert len(dimensions) == 1, f"{group} -> {dimensions}"


# Pairs that MUST stay distinct -- the micro / molar / metre ambiguity guard.
MUST_NOT_MERGE = [
    ("mg", "mM"), ("nm", "nM"), ("mm", "mM"), ("mL", "mM"),
    ("ng", "nM"), ("ml", "mM"), ("µg", "µM"), ("min", "M"), ("µm", "µM"),
]


def test_must_not_merge_pairs_stay_distinct():
    for left, right in MUST_NOT_MERGE:
        left_result = canonicalize_unit(left)
        right_result = canonicalize_unit(right)
        # Distinct in cleaned display AND in dimension.
        assert clean_unit(left) != clean_unit(right), (left, right)
        assert left_result.dimension != right_result.dimension, (left, right)
