import json
import hashlib
from pathlib import Path

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers import (
    reconciliation as mapping,
)


@pytest.fixture(scope="module")
def built_mapping():
    return json.loads(mapping.DEFAULT_OUTPUT.read_text(encoding="utf-8"))


def test_global_mapping_is_the_only_persisted_mapping():
    assert "globally_reconciled" in mapping.DEFAULT_OUTPUT.name
    data_processing = mapping.DEFAULT_OUTPUT.parent
    assert not list(data_processing.glob("[fF][aAhHgG]_auxiliary_value_to_bucket_mapping.json"))
    assert not (data_processing / "pair_bucket_auxiliary_mapping.json.gz").exists()


def test_prompt_registry_contains_only_the_original_six_prompts():
    path = Path(__file__).parents[4] / (
        "tools/chembl_tool/tasks/bioavailability_ma/data_processing/"
        "auxiliary_value_prompts.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        (source_id, output_name)
        for source_id, outputs in payload["prompts"].items()
        for output_name in outputs
    }
    assert actual == {
        ("fa", "global_context"),
        ("fa", "global_species_context"),
        ("fg", "global_context"),
        ("fg", "global_species_context"),
        ("fh", "global_context"),
        ("fh", "global_species_context"),
    }


def test_tuple_keys_round_trip_strings_and_nulls():
    key = mapping._tuple_key(("human", None, "Caco-2"))
    assert key == '["human",null,"Caco-2"]'
    assert mapping._decode_tuple_key(key) == ("human", None, "Caco-2")


def test_reviewed_context_aliases_merge_without_crossing_protected_boundaries():
    resolve = lambda value: mapping._resolve_alias(value, mapping.GLOBAL_CONTEXT_ALIASES)
    assert resolve("intestinal_perfusion") == "intestinal perfusion"
    assert resolve("single-pass intestinal perfusion") == "intestinal perfusion"
    assert resolve("caco_2 uptake") == "caco_2"
    assert resolve("mdck_mdr1") == "mdck"
    assert resolve("oral administration") == "oral dosing"
    assert resolve("population pharmacokinetic model") == "population pk model"
    assert resolve("basket dissolution") == "basket dissolution"

    assert resolve("caco_2") != resolve("mdck")
    assert resolve("liver microsomes") != resolve("s9 fraction")
    assert resolve("everted gut sac") != resolve("gut sac")
    assert resolve("pbpk model") != resolve("population pk model")


def test_species_roles_use_explicit_host_and_gene_evidence():
    assert mapping._species_context(
        "fa",
        (
            "MDCK/MDR1 canine kidney cells transfected with human MDR1",
            "MDCK/MDR1 cell monolayer transport assay",
        ),
        "dog + human",
    ) == "dog host; human gene"
    assert mapping._species_context(
        "fg",
        ("Xenopus laevis oocytes expressing human PEPT1",),
        "frog + human",
    ) == "african clawed frog host; human gene"
    assert mapping._species_context(
        "fh",
        ("human", "recombinant human CYP3A4"),
        "human",
    ) == "human gene"
    assert mapping._species_context(
        "fh",
        ("human", "human liver microsomes and recombinant human CYP3A4"),
        "human",
    ) == "human host; human gene"
    assert mapping._species_context(
        "fh",
        ("human", "recombinant human CYP3A4 expressed in E. coli membranes"),
        "human",
    ) == "escherichia coli host; human gene"


def test_species_roles_do_not_infer_species_from_cell_line_names():
    assert mapping._species_context(
        "fg",
        ("MDR1-MDCKII bidirectional transport assay",),
        None,
    ) is None
    assert mapping._species_context(
        "fa",
        ("Caco-2 cells", "Caco-2 monolayer"),
        None,
    ) is None
    assert mapping._species_context(
        "fh",
        ("human", "human liver microsomes"),
        "human",
    ) == "human"
    assert mapping._species_context(
        "fh",
        ("Sf9", "recombinant CYP3A4 expressed in Sf9 cells"),
        "fall armyworm",
    ) is None
    assert mapping._species_context(
        "fh",
        ("Sf9 (insect)", "recombinant CYP3A4 expressed in Sf9 cells"),
        "fall armyworm",
    ) == "insect host"


def test_reviewed_mixed_contexts_preserve_explicit_preparations():
    assert mapping._context_override(
        "ex vivo everted and non-everted gut sac",
        "gut sac",
    ) == "gut sac"
    assert mapping._context_override(
        "everted gut sac / single-pass intestinal perfusion",
        "gut sac",
    ) == "gut sac"
    assert mapping._context_override(
        "membrane fractions from Sf9 cells expressing UGT2B15",
        "recombinant enzyme",
    ) == "recombinant membranes"
    assert mapping._context_override(
        "recombinant UGT1A enzymes (cell lysate)",
        "recombinant enzyme",
    ) == "recombinant cell lysate"
    assert mapping._context_override(
        "human CYP2D6 variant proteins expressed in 293FT cells",
        "recombinant enzyme",
    ) == "recombinant enzyme"
    assert mapping._context_override(
        "transgenic hamster cell line expressing CYP2B1 (cell homogenates)",
        "recombinant cells",
    ) == "recombinant cell homogenate"


def test_role_assignment_respects_expression_and_tissue_precedence():
    assert mapping._species_context(
        "fg",
        ("transfected HEK293 cells expressing human OCT1",),
        "human",
    ) == "human gene"
    assert mapping._species_context(
        "fg",
        ("MRP2-deficient rat intestinal perfusion; MDCKII cells transfected with Bcrp1",),
        "rat",
    ) == "rat host"
    assert mapping._species_context(
        "fg",
        ("MDCK cells expressing human BCRP or murine Bcrp1",),
        "human + mouse",
    ) == "human + mouse gene"
    assert mapping._species_context(
        "fg",
        ("OATP-A-injected Xenopus laevis oocytes",),
        "frog",
    ) == "african clawed frog host"
    assert mapping._species_context(
        "fh",
        ("human", "recombinant human CYP2D6 expressed in insect microsomes"),
        "human",
    ) == "insect host; human gene"
    assert mapping._species_context(
        "fh",
        ("human (HEK293 cells)", "recombinant UGT1A9 expressed in HEK293 cells"),
        "human",
    ) == "human gene"
    assert mapping._species_context(
        "fh",
        ("rat and human", "rat and human intestinal microsomes; recombinant human UGT1A"),
        "human + rat",
    ) == "human + rat host; human gene"


def test_full_mapping_has_complete_deterministic_source_tuple_coverage(built_mapping):
    audit = mapping.validate_mapping(built_mapping)
    assert set(audit) == {"fa", "fg", "fh"}
    assert built_mapping["mapping_version"] == mapping.MAPPING_VERSION

    for source, source_spec in mapping.SOURCE_SPECS.items():
        required_columns = sorted(
            {column for output in source_spec.outputs for column in output.source_columns}
        )
        frame = mapping.pd.read_parquet(source_spec.parquet_path, columns=required_columns)
        for output in source_spec.outputs:
            expected = {
                mapping._tuple_key(values)
                for values in mapping._distinct_source_tuples(frame, output.source_columns)
            }
            section = built_mapping["sources"][source][output.output_name]
            assert section["source_columns"] == list(output.source_columns)
            assert set(section["mapping"]) == expected

    serialized = json.dumps(built_mapping, ensure_ascii=False, sort_keys=True)
    assert serialized == json.dumps(built_mapping, ensure_ascii=False, sort_keys=True)


def test_full_mapping_contains_expected_real_source_examples(built_mapping):
    fg_species = built_mapping["sources"]["fg"]["global_species_context"]["mapping"]
    assert fg_species[
        mapping._tuple_key(("Xenopus laevis oocytes expressing human PEPT1",))
    ] == "african clawed frog host; human gene"

    fh_species = built_mapping["sources"]["fh"]["global_species_context"]["mapping"]
    assert fh_species[
        mapping._tuple_key(("human", "human liver microsomes and recombinant human CYP enzymes"))
    ] == "human host; human gene"


def test_persisted_mapping_hash_and_complete_validation_are_frozen(built_mapping):
    assert hashlib.sha256(mapping.DEFAULT_OUTPUT.read_bytes()).hexdigest() == (
        "a0929ef4cf48717b0968f78a9a726b6a198cda3233334593b5dba67c59f17304"
    )
    assert mapping.validate_mapping(built_mapping)
